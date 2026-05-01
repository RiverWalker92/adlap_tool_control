#include "adlap_tool_control/instrument_controller.hpp"
#include "adlap_tool_control/keyboard_reader.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"



InstrumentController::InstrumentController(
    MotorController& motor_controller,
    rclcpp::Logger logger,
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr task_publisher) 
    : motor_controller_(motor_controller),
      logger_(logger),
      task_publisher_(task_publisher)
{
}

void InstrumentController::publish_task(const std::string& task)
{
  if (!task_publisher_) {
    return;
  }

  auto task_msg = std_msgs::msg::String();
  task_msg.data = task;
  task_publisher_->publish(task_msg);
}

/// @brief Allow manual adjustment of the lower motors using keyboard input
void InstrumentController::manual_adjustment(){
  ///// Workaround  - manual adjustment of the lower motors to get the shaft straight and do some testing ////
  KeyboardReader input;
  std::array<int, 4> duty_cycle_array = {7, 7, 7, 7};
  char c;
  bool initialized = false;
  bool ready_for_angles = false;
  for (;;)
  {
    // get the next event from the keyboard
    try
    {
      input.read_one(&c);
    }
    catch (const std::runtime_error&)
    {
      perror("read():");
      return;
    }
    double angle_rad = 5.0 * M_PI / 180.0;
    int step_size = motor_controller_.get_pulses_per_degree(false) * 5; // Step size of 5 degrees in pulses
    std::array<int, 4> current_positions = motor_controller_.get_positions();
    switch (c)
    {
      case KEYCODE_C:
        RCLCPP_INFO(logger_, "C -> Couple sequence");
        publish_task("coupling_sequence");
        
        motor_controller_.couple_sequence();
        break;          
      case KEYCODE_U:
        RCLCPP_INFO(logger_, "U -> Update starting positions");
        publish_task("update_starting_positions");
        if (!initialized) {
          RCLCPP_WARN(logger_, "Motors not initialized yet, press 'I' to initialize before updating starting positions");
          break;
        }
        // ofset gripper
        current_positions[3] -= 50 * motor_controller_.get_pulses_per_degree(true); // Add some extra to compensate for backlash
        motor_controller_.send_motor_positions(current_positions);
        rclcpp::sleep_for(std::chrono::milliseconds(200));
        motor_controller_.update_starting_positions(); 
        smoothed_roll_ = update_history_and_get_mean(roll_history_, 0.0, 1, -M_PI, M_PI); 
        smoothed_pitch_ = update_history_and_get_mean(pitch_history_, 0.0, 1, -M_PI, M_PI);  
        smoothed_yaw_ = update_history_and_get_mean(yaw_history_, 0.0, 1, -M_PI, M_PI);  
        smoothed_gripper_ = update_history_and_get_mean(gripper_history_, 0.0, 1, -M_PI, M_PI); 
        RCLCPP_INFO(logger_, "Smoothed angles - roll: '%f', pitch: '%f', yaw: '%f', gripper: '%f'", smoothed_roll_, smoothed_pitch_, smoothed_yaw_, smoothed_gripper_);
        absolute_omega_ = 0.0; // Reset absolute omega when updating starting positions
        bend_play_compensation_ = 0; // Reset bend play compensation when updating starting positions  
        play_comp_position_m1_ = current_positions[1]; // Reset play compensation position for motor 1
        play_comp_position_m2_ = current_positions[2]; // Reset play compensation position for motor 2
        ready_for_angles = true;       
        break;
      case KEYCODE_I:
        RCLCPP_INFO(logger_, "I -> Initialize lower motors");
        publish_task("initialization_sequence");
        motor_controller_.setup_motors();
        initialized = true;
        break;
      case KEYCODE_R:
        RCLCPP_INFO(logger_, "R -> Reset to starting positions");
        motor_controller_.send_motor_positions(motor_controller_.get_starting_positions());
        break;
      case KEYCODE_PLUS:
        RCLCPP_INFO(logger_, "Increase duty cycle");
        for (int i = 0; i < 4; i++) {
          duty_cycle_array[i] += 1;
        }
        motor_controller_.send_duty_cycle(duty_cycle_array);
        break;
      case KEYCODE_0:
        RCLCPP_INFO(logger_, "0 -> Send all 0 angles");
        if (!ready_for_angles) {
          RCLCPP_WARN(logger_, "Not ready for angle control yet, press 'I' to initialize motors and 'U' to update starting positions");
          break;
        }
        set_angles(0, 0, 0, 0);
        break;
      case KEYCODE_1:
        RCLCPP_INFO(logger_, "1 -> Roll left (bend angle 10 degrees)");
        if (!ready_for_angles) {
          RCLCPP_WARN(logger_, "Not ready for angle control yet, press 'I' to initialize motors and 'U' to update starting positions");
          break;
        }
        set_angles(smoothed_roll_ - angle_rad, smoothed_pitch_, smoothed_yaw_, smoothed_gripper_);
        break;
      case KEYCODE_2:
        RCLCPP_INFO(logger_, "2 -> Pitch down");        
        if (!ready_for_angles) {
          RCLCPP_WARN(logger_, "Not ready for angle control yet, press 'I' to initialize motors and 'U' to update starting positions");
          break;
        }
        set_angles(smoothed_roll_, smoothed_pitch_ - angle_rad, smoothed_yaw_, smoothed_gripper_);
        break;
      case KEYCODE_3:
        RCLCPP_INFO(logger_, "3 -> Roll right (bend angle -10 degrees)");
        if (!ready_for_angles) {
          RCLCPP_WARN(logger_, "Not ready for angle control yet, press 'I' to initialize motors and 'U' to update starting positions");
          break;
        }
        set_angles(smoothed_roll_ + angle_rad, smoothed_pitch_, smoothed_yaw_, smoothed_gripper_);
        break;
      case KEYCODE_4:
        RCLCPP_INFO(logger_, "4 -> Yaw left");
        if (!ready_for_angles) {
          RCLCPP_WARN(logger_, "Not ready for angle control yet, press 'I' to initialize motors and 'U' to update starting positions");
          break;
        }
        set_angles(smoothed_roll_, smoothed_pitch_, smoothed_yaw_ - angle_rad, smoothed_gripper_);
        break;
      case KEYCODE_5:
        RCLCPP_INFO(logger_, "5 -> Full rotation");        
        motor_controller_.send_relative_motor_positions(0,motor_controller_.get_pulses_per_rotation(false),motor_controller_.get_pulses_per_rotation(false),0);
        break;
      case KEYCODE_6:
        RCLCPP_INFO(logger_, "6 -> Yaw right");
        if (!ready_for_angles) {
          RCLCPP_WARN(logger_, "Not ready for angle control yet, press 'I' to initialize motors and 'U' to update starting positions");
          break;
        }
        set_angles(smoothed_roll_, smoothed_pitch_, smoothed_yaw_ + angle_rad, smoothed_gripper_);
        break;
      case KEYCODE_7:
        RCLCPP_INFO(logger_, "7 -> Open gripper");
        if (!ready_for_angles) {
          RCLCPP_WARN(logger_, "Not ready for angle control yet, press 'I' to initialize motors and 'U' to update starting positions");
          break;
        }
        set_angles(smoothed_roll_, smoothed_pitch_, smoothed_yaw_, smoothed_gripper_ + angle_rad);
        break;
      case KEYCODE_8:
        RCLCPP_INFO(logger_, "8 -> Pitch up");
        if (!ready_for_angles) {
          RCLCPP_WARN(logger_, "Not ready for angle control yet, press 'I' to initialize motors and 'U' to update starting positions");
          break;
        }
        set_angles(smoothed_roll_, smoothed_pitch_ + angle_rad, smoothed_yaw_, smoothed_gripper_);
        break;
      case KEYCODE_9:
        RCLCPP_INFO(logger_, "9 -> Close gripper");
        if (!ready_for_angles) {
          RCLCPP_WARN(logger_, "Not ready for angle control yet, press 'I' to initialize motors and 'U' to update starting positions");
          break;
        }
        set_angles(smoothed_roll_, smoothed_pitch_, smoothed_yaw_, smoothed_gripper_ - angle_rad);
        break;
      case KEYCODE_RIGHT:
        RCLCPP_INFO(logger_, "RIGHT");
        motor_controller_.send_relative_motor_positions(0,step_size,0,0, true);
        break;
      case KEYCODE_LEFT:
        RCLCPP_INFO(logger_, "LEFT");
        motor_controller_.send_relative_motor_positions(0,-step_size,0,0, true);
        break;
      case KEYCODE_UP:
        RCLCPP_INFO(logger_, "UP");
        motor_controller_.send_relative_motor_positions(0,0,step_size,0, true);
        break;
      case KEYCODE_DOWN:
        RCLCPP_INFO(logger_, "DOWN");
        motor_controller_.send_relative_motor_positions(0,0,-step_size,0, true);
        break;
      case KEYCODE_W:
        RCLCPP_INFO(logger_, "W");
        motor_controller_.send_relative_motor_positions(step_size,0,0,0, true);
        break;
      case KEYCODE_S:
        RCLCPP_INFO(logger_, "S");
        motor_controller_.send_relative_motor_positions(-step_size,0,0,0, true);
        break;
      case KEYCODE_D:
        RCLCPP_INFO(logger_, "D");
        motor_controller_.send_relative_motor_positions(step_size,0,0,step_size, true);
        break;
      case KEYCODE_A:
        RCLCPP_INFO(logger_, "A");
        motor_controller_.send_relative_motor_positions(-step_size,0,0,-step_size, true);
        break;
      case KEYCODE_E:
        motor_controller_.send_relative_motor_positions(0,step_size,step_size,0, true);
        RCLCPP_INFO(logger_, "E");
        break;
      case KEYCODE_Q:
        motor_controller_.send_relative_motor_positions(0,-step_size,-step_size,0, true);
        RCLCPP_INFO(logger_, "Q");
        break;
      case KEYCODE_ENTER:
        RCLCPP_INFO(logger_, "ENTER");
        if (!ready_for_angles) {
          RCLCPP_WARN(logger_, "Not ready for angle control yet, press 'I' to initialize motors and 'U' to update starting positions");
          break;
        }
        input.shutdown();
        motor_controller_.update_starting_positions();
        return;
      default:
        RCLCPP_WARN(logger_, "Unknown key pressed: 0x%02X", c);
        continue;
    }
  }
}

double InstrumentController::update_history_and_get_mean(std::deque<double>& history,
  double sample,
  std::size_t max_size,
  double min_value,
  double max_value)
{
    // Clamp sample to provided range
    if (sample < min_value) sample = min_value;
    else if (sample > max_value) sample = max_value;
    history.push_front(sample);
    while (history.size() > max_size) {
        history.pop_back();
    }

    double sum = 0.0;
    for (const double& v : history) {
        sum += v;
    }
    return history.empty() ? 0.0 : (sum / static_cast<double>(history.size()));
}

void InstrumentController::set_angles(double roll, double pitch, double yaw, double gripper, bool verbose)
{
    // Add to history + compute smoothed values (same helper called 4 times)
    smoothed_roll_ = update_history_and_get_mean(roll_history_, roll, smoothing_factor_, -M_PI, M_PI);
    smoothed_pitch_ = update_history_and_get_mean(pitch_history_, pitch, smoothing_factor_, -M_PI/4, M_PI/4);
    smoothed_yaw_ = update_history_and_get_mean(yaw_history_, yaw, smoothing_factor_, -M_PI/4, M_PI/4);
    smoothed_gripper_ = update_history_and_get_mean(gripper_history_, gripper, smoothing_factor_, 0, M_PI/6);
    if (verbose) RCLCPP_DEBUG(logger_, "Smoothed angles - roll: '%f', pitch: '%f', yaw: '%f', gripper: '%f'", smoothed_roll_, smoothed_pitch_, smoothed_yaw_, smoothed_gripper_);
    motor_controller_.send_motor_positions(calculate_motor_positions_from_angles(verbose), true);

}


int InstrumentController::get_motor2_value_for_angle(double radians, bool verbose){
  double degrees = radians * 180.0 / M_PI;
  if (degrees > MAX_BEND_ANGLE_DEGREES) {
    degrees = MAX_BEND_ANGLE_DEGREES;
  } else if (degrees < -MAX_BEND_ANGLE_DEGREES) {
    degrees = -MAX_BEND_ANGLE_DEGREES;
  }

  int starting_difference = motor_controller_.get_starting_positions()[2] - motor_controller_.get_starting_positions()[1]; // Starting difference between motor 2 and motor 1, compensated for the initial offset found in setup
  int current_difference = motor_controller_.get_positions()[2] - motor_controller_.get_positions()[1] - starting_difference - bend_play_compensation_; // Current difference between motor 2 and motor 1, compensated for play
  int wanted_difference = static_cast<int>(std::round(degrees * motor_controller_.get_pulses_per_degree(false) * BEND_FACTOR));
  int relative_difference = wanted_difference - current_difference + bend_play_compensation_;


  if (abs(relative_difference) <= motor_controller_.get_pulses_per_degree(false) * 5) { // If the current difference is within 5 degrees of the target, don't adjust to prevent jitter
    if (verbose) RCLCPP_DEBUG(logger_, "Same angle, no adjustment needed");
    return current_difference; // Return the current difference without adjustment
  }

  int play_compensation = 0;
  if (relative_difference > 0){
    play_compensation = motor_controller_.get_pulses_lower_motors_play();
  }
  else{
    play_compensation = -motor_controller_.get_pulses_lower_motors_play();
  }
  bend_play_compensation_ = play_compensation; // Store the current compensation for use in the next calculation
  return wanted_difference + play_compensation;
}

std::array<int, 4> InstrumentController::calculate_motor_positions_from_angles(bool verbose)
{
  int tip_rotation = - static_cast<int>(std::round(smoothed_roll_ * motor_controller_.get_pulses_per_rotation(true) / (TWO_PI)));
  int gripper_position = static_cast<int>(std::round(GRIPPER_FACTOR * smoothed_gripper_ * motor_controller_.get_pulses_per_rotation(true) / (TWO_PI)));

  // Drive the lower motors with the pitch and yaw. 
  // Motor 2 angles the shaft.
  // Motor 1 and 2 together change the direction of the bend.

  double h = -std::tan(smoothed_pitch_);
  double w = -std::tan(smoothed_yaw_);
  double omega = std::atan2(w, h) + M_PI; // direction of the bend
  double l1 = std::sqrt(h*h + w*w);
  double theta = std::atan(l1); // angle of the bend

  // Calculate the shortest rotation for omega
  // We need to choose the new omega to be the closest to the current position to prevent unnecessary full rotations
  // The motors are driven with absolute positions, so we need to keep track of the absolute omega
  // TODO: implement bending the other way and add this to the shortest rotation calculation and solve singularity
  double delta_omega = omega - absolute_omega_;
  double diff = std::fmod(delta_omega, TWO_PI);
  if (diff < - M_PI) {
      diff += TWO_PI;
  } else if (diff >= M_PI) {
      diff -= TWO_PI;
  }
  absolute_omega_ += diff; // Update absolute_omega with the shortest rotation

  int m2_bend = get_motor2_value_for_angle(theta, verbose);
  int shaft_rot = absolute_omega_ * motor_controller_.get_pulses_per_rotation(false) / (TWO_PI);

  if (verbose) {
    RCLCPP_DEBUG(
      logger_,
      "### Calculated motor positions ###\n"
      "  shaft_rot: '%d', m2_bend: '%d'\n"
      "  theta: '%f', omega: '%f'\n"
      "  roll: '%d', gripper_position: '%d'",
      shaft_rot, m2_bend, theta, absolute_omega_, tip_rotation, gripper_position);
  }

  std::array<int, 4>  starting_positions = motor_controller_.get_starting_positions();
  std::array<int, 4>  new_positions = {
    static_cast<int>(std::round(starting_positions[0] + tip_rotation)),
    static_cast<int>(std::round(starting_positions[1] + shaft_rot - bend_play_compensation_)),
    static_cast<int>(std::round(starting_positions[2] + shaft_rot + m2_bend)), 
    static_cast<int>(std::round(starting_positions[3] + tip_rotation + gripper_position))    
  };
  return new_positions;
}

std::array<double, 4> InstrumentController::angles_from_motors(const std::array<int, 4>& m_array)
{
  std::array<int, 4>  starting_positions = motor_controller_.get_starting_positions();
  std::array<int, 4> current_positions = motor_controller_.get_positions();

  double roll_angle = (static_cast<double>(starting_positions[0] - current_positions[0]) * TWO_PI / motor_controller_.get_pulses_per_rotation(true)); // Invert the roll calculation
  double gripper_angle = (static_cast<double>(current_positions[3] - starting_positions[3] - current_positions[0] + starting_positions[0]) * TWO_PI / motor_controller_.get_pulses_per_rotation(true)) / GRIPPER_FACTOR; // Invert the gripper calculation

  int play =  motor_controller_.get_pulses_lower_motors_play();
  int starting_diff = starting_positions[2] - starting_positions[1]; // Starting difference between motor 2 and motor 1, compensated for the initial offset found in setup

  int diff = current_positions[2] - current_positions[1] - starting_diff; // Current difference between motor 2 and motor 1, not yet compensated for play
  if (diff > 0) {
    if (diff > m1_m2_offset_+ 2*play){
      m1_m2_offset_ = diff - 2*play; 
    }
    else if (diff < m1_m2_offset_ - 2*play) {
      m1_m2_offset_ = diff + 2*play; 
    }
  }
  else{
    if (diff < m1_m2_offset_ - 2*play){
      m1_m2_offset_ = diff + 2*play;
    }
    else if (diff > m1_m2_offset_ + 2*play) {
      m1_m2_offset_ = diff - 2*play; 
    }
  }

  int real_play = (diff - m1_m2_offset_)/2; // Calculate the real play based on the current offset
  double shaft_rot = (static_cast<double>(current_positions[1] - starting_positions[1] + real_play)); 
  double m2_bend = (static_cast<double>(current_positions[2] - starting_positions[2] - shaft_rot - real_play));

  double theta_degrees = m1_m2_offset_ / (motor_controller_.get_pulses_per_degree(false) * BEND_FACTOR);
  double theta = theta_degrees * M_PI / 180.0;
  double l1 = std::tan(theta);
  double omega = shaft_rot * TWO_PI / motor_controller_.get_pulses_per_rotation(false);
  double h = -l1 * std::cos(omega);
  double w = -l1 * std::sin(omega);
  double pitch_angle = -std::atan(h);
  double yaw_angle = -std::atan(w);
  
  std::array<double, 4> angles;
  angles[0] = roll_angle;
  angles[1] = pitch_angle;
  angles[2] = yaw_angle;
  angles[3] = gripper_angle;
  return angles;
}


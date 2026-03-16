#include "adlap_tool_control/instrument_controller.hpp"
#include "adlap_tool_control/keyboard_reader.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"



InstrumentController::InstrumentController(MotorController& motor_controller, rclcpp::Logger logger) 
    : motor_controller_(motor_controller), logger_(logger)
{
}


/// @brief Allow manual adjustment of the lower motors using keyboard input
void InstrumentController::manual_adjustment(){
  ///// Workaround  - manual adjustment of the lower motors to get the shaft straight and do some testing ////
  KeyboardReader input;
  std::array<int, 4> duty_cycle_array = {7, 7, 7, 7};
  char c;
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
    switch (c)
    {
      case KEYCODE_C:
        RCLCPP_INFO(logger_, "C -> Couple sequence");
        motor_controller_.couple_sequence();
        break;          
      case KEYCODE_U:
        RCLCPP_INFO(logger_, "U -> Update starting positions");
        motor_controller_.update_starting_positions(); 
        update_history_and_get_mean(roll_history_, 0.0, 1, -M_PI, M_PI); 
        update_history_and_get_mean(pitch_history_, 0.0, 1, -M_PI, M_PI);  
        update_history_and_get_mean(yaw_history_, 0.0, 1, -M_PI, M_PI);  
        update_history_and_get_mean(gripper_history_, 0.0, 1, -M_PI, M_PI); 
        absolute_omega_ = 0.0; // Reset absolute omega when updating starting positions         
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
        set_angles(0, 0, 0, 0);
        break;
      case KEYCODE_1:
        RCLCPP_INFO(logger_, "1 -> Roll left (bend angle 10 degrees)");
        set_angles(smoothed_roll_ - angle_rad, smoothed_pitch_, smoothed_yaw_, smoothed_gripper_);
        break;
      case KEYCODE_2:
        RCLCPP_INFO(logger_, "2 -> Pitch down");
        set_angles(smoothed_roll_, smoothed_pitch_ - angle_rad, smoothed_yaw_, smoothed_gripper_);
        break;
      case KEYCODE_3:
        RCLCPP_INFO(logger_, "3 -> Roll right (bend angle -10 degrees)");
        set_angles(smoothed_roll_ + angle_rad, smoothed_pitch_, smoothed_yaw_, smoothed_gripper_);
        break;
      case KEYCODE_4:
        RCLCPP_INFO(logger_, "4 -> Yaw left");
        set_angles(smoothed_roll_, smoothed_pitch_, smoothed_yaw_ - angle_rad, smoothed_gripper_);
        break;
      case KEYCODE_6:
        RCLCPP_INFO(logger_, "6 -> Yaw right");
        set_angles(smoothed_roll_, smoothed_pitch_, smoothed_yaw_ + angle_rad, smoothed_gripper_);
        break;
      case KEYCODE_7:
        RCLCPP_INFO(logger_, "7 -> open gripper");
        set_angles(smoothed_roll_, smoothed_pitch_, smoothed_yaw_, smoothed_gripper_ + angle_rad);
        break;
      case KEYCODE_8:
        RCLCPP_INFO(logger_, "8 -> Pitch up");
        set_angles(smoothed_roll_, smoothed_pitch_ + angle_rad, smoothed_yaw_, smoothed_gripper_);
        break;
      case KEYCODE_9:
        RCLCPP_INFO(logger_, "9 -> close gripper");
        set_angles(smoothed_roll_, smoothed_pitch_, smoothed_yaw_, smoothed_gripper_ - angle_rad);
        break;
      case KEYCODE_RIGHT:
        RCLCPP_INFO(logger_, "RIGHT");
        motor_controller_.send_relative_motor_positions(0,10,0,0);
        break;
      case KEYCODE_LEFT:
        RCLCPP_INFO(logger_, "LEFT");
        motor_controller_.send_relative_motor_positions(0,-10,0,0);
        break;
      case KEYCODE_UP:
        RCLCPP_INFO(logger_, "UP");
        motor_controller_.send_relative_motor_positions(0,0,10,0);
        break;
      case KEYCODE_DOWN:
        RCLCPP_INFO(logger_, "DOWN");
        motor_controller_.send_relative_motor_positions(0,0,-10,0);
        break;
      case KEYCODE_W:
        RCLCPP_INFO(logger_, "W");
        motor_controller_.send_relative_motor_positions(10,0,0,0);
        break;
      case KEYCODE_S:
        RCLCPP_INFO(logger_, "S");
        motor_controller_.send_relative_motor_positions(-10,0,0,0);
        break;
      case KEYCODE_D:
        RCLCPP_INFO(logger_, "D");
        motor_controller_.send_relative_motor_positions(10,0,0,10);
        break;
      case KEYCODE_A:
        RCLCPP_INFO(logger_, "A");
        motor_controller_.send_relative_motor_positions(-10,0,0,-10);
        break;
      case KEYCODE_E:
        motor_controller_.send_relative_motor_positions(0,10,10,0);
        RCLCPP_INFO(logger_, "E");
        break;
      case KEYCODE_Q:
        motor_controller_.send_relative_motor_positions(0,-10,-10,0);
        RCLCPP_INFO(logger_, "Q");
        break;
      case KEYCODE_ENTER:
        RCLCPP_INFO(logger_, "ENTER");
        input.shutdown();
        motor_controller_.update_starting_positions();
        return;
      default:
        RCLCPP_WARN(logger_, "Unknown key pressed: 0x%02X", c);
        continue;
    }
  }
}

void InstrumentController::set_angles(double roll, double pitch, double yaw, double gripper)
{
    // Add to history + compute smoothed values (same helper called 4 times)
    smoothed_roll_ = update_history_and_get_mean(roll_history_, roll, smoothing_factor_, -M_PI, M_PI);
    smoothed_pitch_ = update_history_and_get_mean(pitch_history_, pitch, smoothing_factor_, -M_PI/4, M_PI/4);
    smoothed_yaw_ = update_history_and_get_mean(yaw_history_, yaw, smoothing_factor_, -M_PI/4, M_PI/4);
    smoothed_gripper_ = update_history_and_get_mean(gripper_history_, gripper, smoothing_factor_, 0, M_PI/6);
    RCLCPP_DEBUG(logger_, "Smoothed angles - roll: '%f', pitch: '%f', yaw: '%f', gripper: '%f'", smoothed_roll_, smoothed_pitch_, smoothed_yaw_, smoothed_gripper_);
    drive_motors(calculate_motor_positions_from_angles());

}
std::array<double, 4> InstrumentController::angles_from_motors(const std::array<int, 4>& m_array)
{
  std::array<double, 4> angles;
  angles[0] = 0.0; // TODO: calculate roll from motor positions
  angles[1] = 0.0; // TODO: calculate pitch from motor positions
  angles[2] = 0.0; // TODO: calculate yaw from motor positions
  angles[3] = 0.0; // TODO: calculate gripper angle from motor positions
  return angles;
}


int InstrumentController::get_motor2_value_for_angle(double radians){
  double degrees = radians * 180.0 / M_PI;

  int starting_difference = motor_controller_.get_starting_positions()[2] - motor_controller_.get_starting_positions()[1];
  int current_difference = motor_controller_.get_positions()[2] - motor_controller_.get_positions()[1] - starting_difference - bend_play_compensation_; // Current difference between motor 2 and motor 1, compensated for play
  int wanted_difference = static_cast<int>(std::round(degrees * motor_controller_.get_pulses_per_degree(false) * BEND_FACTOR));
  int relative_difference = wanted_difference - current_difference + bend_play_compensation_;

  RCLCPP_DEBUG(logger_, "degrees: %f", degrees);
  RCLCPP_DEBUG(logger_, "Current difference: %d", current_difference);
  RCLCPP_DEBUG(logger_, "Wanted difference: %d", wanted_difference);
  RCLCPP_DEBUG(logger_, "Relative difference: %d", relative_difference);

  if (abs(relative_difference) <= motor_controller_.get_pulses_per_degree(false) * 5) { // If the current difference is within 5 degrees of the target, don't adjust to prevent jitter
    RCLCPP_DEBUG(logger_, "Same angle, no adjustment needed");
    return current_difference; // Return the current difference without adjustment
  }

  int play_compensation = 0;
  if (relative_difference > 0){
    play_compensation = motor_controller_.get_lower_motors_play() * motor_controller_.get_pulses_per_degree(false);
  }
  else{
    play_compensation = -motor_controller_.get_lower_motors_play() * motor_controller_.get_pulses_per_degree(false);
  }
  bend_play_compensation_ = play_compensation; // Store the current compensation for use in the next calculation

  return wanted_difference + play_compensation;
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

    

  

std::array<int, 4> InstrumentController::calculate_motor_positions_from_angles()
{
  int tip_rotation = static_cast<int>(std::round(smoothed_roll_ * motor_controller_.get_motor_pulses_per_rotation(true) / (2 * M_PI)));
  int gripper_offset = static_cast<int>(std::round(0.6f * motor_controller_.get_motor_pulses_per_rotation(true)));
  int gripper_factor = 16;
  int gripper_position = static_cast<int>(std::round(gripper_factor * smoothed_gripper_ * motor_controller_.get_motor_pulses_per_rotation(true) / (2 * M_PI)));

  // Drive the lower motors with the pitch and yaw. 
  // Motor 1 angles the shaft.
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

  int m2_bend = get_motor2_value_for_angle(theta);
  int shaft_rot = absolute_omega_ * motor_controller_.get_motor_pulses_per_rotation(false) / (2 * M_PI);

  RCLCPP_INFO(logger_, "### Calculated motor positions ### \n shaft_rot: '%d', m2_bend: '%d', ", shaft_rot, m2_bend);
  RCLCPP_INFO(logger_, "theta: '%f' omega: '%f'", theta, absolute_omega_);
  RCLCPP_INFO(logger_, "roll, gripper_offset, gripper_position: '%d' '%d' '%d'", tip_rotation, gripper_offset, gripper_position);

  /*TODO
  - Gripper distance is currently hardcoded as 0.6 times a full rotation, this is incorrect
  - setup motor controller by setting the type of motor -> pulses per rotation, pwm duty cycle, max current and possibly play compensation are motor dependent
  - limit angles correctly. ROll is to small and gripper too see first point.
  */ 

  std::array<int, 4>  start_positions = motor_controller_.get_starting_positions();
  std::array<int, 4>  new_positions = {
    static_cast<int>(std::round(start_positions[0] + tip_rotation)),
    static_cast<int>(std::round(start_positions[1] + shaft_rot - bend_play_compensation_)),
    static_cast<int>(std::round(start_positions[2] + shaft_rot + m2_bend)), 
    static_cast<int>(std::round(start_positions[3] + tip_rotation + gripper_position))    
  };
  return new_positions;
}


void InstrumentController::drive_motors(std::array<int, 4> m_array) {
  // Check differences to prevent too large movements

  // for (size_t i = 0; i < 4; ++i){
  //   int diff = m_array[i] - motor_controller_.get_target_positions()[i];
  //   if (std::abs(diff) > 30){
  //     if (diff > 0){
  //       m_array[i] = motor_controller_.get_target_positions()[i] + 30;
  //     }else{
  //       m_array[i] = motor_controller_.get_target_positions()[i] - 30;
  //     }
  //   }
  // }

  motor_controller_.send_motor_positions(m_array, true);
}

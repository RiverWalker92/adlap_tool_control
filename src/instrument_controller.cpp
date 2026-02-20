#include "adlap_tool_control/instrument_controller.hpp"
#include "adlap_tool_control/keyboard_reader.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"



InstrumentController::InstrumentController(MotorController& motor_controller, rclcpp::Logger logger) 
    : motor_controller_(motor_controller), logger_(logger)
{
}

//TODO return both motor1 and motor2 values for a given angle. 
int InstrumentController::get_relative_motor1_value_for_angle(double radians){
  double degrees = radians * 180.0 / M_PI;

  if (bend_angle_ == degrees) {
    RCLCPP_DEBUG(logger_, "Same angle, no adjustment needed");
    return 0;
  }
  int play_compensation = 0;
  if (degrees > bend_angle_) {
    play_compensation = motor_controller_.get_lower_motors_play();
  }
  else{
    play_compensation = -motor_controller_.get_lower_motors_play();
  }
  bend_angle_ = degrees;
  int starting_difference = motor_controller_.get_starting_positions()[1] - motor_controller_.get_starting_positions()[2];
  return int(std::round((LOWER_MOTORS_MAX_DIFFERENCE - play_compensation) / LOWER_MOTORS_MAX_BEND_ANGLE * degrees + play_compensation + starting_difference - motor_controller_.get_positions()[1] + motor_controller_.get_positions()[2]));
}

/// @brief Allow manual adjustment of the lower motors using keyboard input
void InstrumentController::manual_adjustment(){
  ///// Workaround  - manual adjustment of the lower motors to get the shaft straight and do some testing ////
  KeyboardReader input;
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
    RCLCPP_INFO(logger_, "value: 0x%02X\n", c);
    switch (c)
    {
      case KEYCODE_C:
        RCLCPP_DEBUG(logger_, "C -> Couple sequence");
        motor_controller_.couple_sequence();
        break;          
      case KEYCODE_U:
        RCLCPP_DEBUG(logger_, "U -> Update starting positions");
        motor_controller_.update_starting_positions();          
        break;
      case KEYCODE_R:
        RCLCPP_DEBUG(logger_, "R -> Reset to starting positions");
        motor_controller_.send_motor_positions(motor_controller_.get_starting_positions());
        break;
      case KEYCODE_0:
        RCLCPP_DEBUG(logger_, "0 -> Set angle of bend to 0 degrees");
        motor_controller_.send_relative_motor_positions(0, get_relative_motor1_value_for_angle(0), 0, 0); // starting offset + current position of motor 2
        break;
      case KEYCODE_1:
        RCLCPP_DEBUG(logger_, "1 -> Set angle of bend to 10 degrees");
        motor_controller_.send_relative_motor_positions(0, get_relative_motor1_value_for_angle(10), 0, 0);
        break;
      case KEYCODE_2:
        RCLCPP_DEBUG(logger_, "2 -> Set angle of bend to 20 degrees");
        motor_controller_.send_relative_motor_positions(0, get_relative_motor1_value_for_angle(20), 0, 0); 
        break;
      case KEYCODE_3:
        RCLCPP_DEBUG(logger_, "3 -> Set angle of bend to 30 degrees");
        motor_controller_.send_relative_motor_positions(0, get_relative_motor1_value_for_angle(30), 0, 0);
        break;
      case KEYCODE_MINUS:
        RCLCPP_DEBUG(logger_, "MINUS");
        motor_controller_.send_relative_motor_positions(0, -10, -10, 0);
        break;
      case KEYCODE_EQUAL:
        RCLCPP_DEBUG(logger_, "EQUAL");
        motor_controller_.send_relative_motor_positions(0, 10, 10, 0);
        break;
      case KEYCODE_RIGHT:
        RCLCPP_DEBUG(logger_, "RIGHT");
        motor_controller_.send_relative_motor_positions(0,10,0,0);
        break;
      case KEYCODE_LEFT:
        RCLCPP_DEBUG(logger_, "LEFT");
        motor_controller_.send_relative_motor_positions(0,-10,0,0);
        break;
      case KEYCODE_UP:
        RCLCPP_DEBUG(logger_, "UP");
        motor_controller_.send_relative_motor_positions(0,0,10,0);
        break;
      case KEYCODE_DOWN:
        RCLCPP_DEBUG(logger_, "DOWN");
        motor_controller_.send_relative_motor_positions(0,0,-10,0);
        break;
      case KEYCODE_W:
        RCLCPP_DEBUG(logger_, "W");
        motor_controller_.send_relative_motor_positions(10,0,0,0);
        break;
      case KEYCODE_S:
        RCLCPP_DEBUG(logger_, "S");
        motor_controller_.send_relative_motor_positions(-10,0,0,0);
        break;
      case KEYCODE_D:
        RCLCPP_DEBUG(logger_, "D");
        motor_controller_.send_relative_motor_positions(10,0,0,10);
        break;
      case KEYCODE_A:
        RCLCPP_DEBUG(logger_, "A");
        motor_controller_.send_relative_motor_positions(-10,0,0,-10);
        break;
      case KEYCODE_E:
        motor_controller_.send_relative_motor_positions(0,10,10,0);
        RCLCPP_DEBUG(logger_, "E");
        break;
      case KEYCODE_Q:
        motor_controller_.send_relative_motor_positions(0,-10,-10,0);
        RCLCPP_DEBUG(logger_, "Q");
        break;
      case KEYCODE_ENTER:
        RCLCPP_DEBUG(logger_, "ENTER");
        input.shutdown();
        motor_controller_.update_starting_positions();
        return;
      default:
        RCLCPP_DEBUG(logger_, "Unknown key pressed: 0x%02X", c);
        continue;
    }
  }
}

double InstrumentController::update_history_and_get_mean(std::deque<double>& history,
  double sample,
  std::size_t max_size)
{
    history.push_front(sample);
    if (history.size() > max_size) {
        history.pop_back();
    }

    double sum = 0.0;
    for (const double& v : history) {
        sum += v;
    }
    return history.empty() ? 0.0 : (sum / static_cast<double>(history.size()));
}
    
void InstrumentController::set_angles(double roll, double pitch, double yaw, double gripper)
{
    // Add to history + compute smoothed values (same helper called 4 times)
    smoothed_roll_ = update_history_and_get_mean(roll_history_, roll, smoothing_factor_);
    smoothed_pitch_ = update_history_and_get_mean(pitch_history_, pitch, smoothing_factor_);
    smoothed_yaw_ = update_history_and_get_mean(yaw_history_, yaw, smoothing_factor_);
    smoothed_gripper_ = update_history_and_get_mean(gripper_history_, gripper, smoothing_factor_);

    drive_motors(calculate_motor_positions_based_on_smoothed_angles());

}
  

std::array<int, 4>& InstrumentController::calculate_motor_positions_based_on_smoothed_angles()
{
  int tip_rotation = static_cast<int>(std::round(smoothed_roll_ * MOTOR_PULSES_PER_ROTATION / (2 * UPPER_MOTOR_FACTOR * M_PI)));
  int gripper_offset = static_cast<int>(std::round(0.6f * MOTOR_PULSES_PER_ROTATION / UPPER_MOTOR_FACTOR));
  int gripper_factor = 8;
  int gripper_position = static_cast<int>(std::round(gripper_factor * smoothed_gripper_ * MOTOR_PULSES_PER_ROTATION / (2 * UPPER_MOTOR_FACTOR * M_PI)));

  // Drive the lower motors with the pitch and yaw. 
  // Motor 1 angles the shaft.
  // Motor 1 and 2 together change the direction of the bend.

  double h = std::tan(smoothed_pitch_);
  double w = std::tan(smoothed_yaw_);
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
  
  RCLCPP_INFO(logger_, "theta: '%f' omega: '%f'", theta, absolute_omega_);

  int m1_bend = get_m1_bend_pulses(theta);
  int shaft_rot = absolute_omega_ * MOTOR_PULSES_PER_ROTATION / (2 * M_PI);

  RCLCPP_INFO(logger_, "roll, gripper_offset, gripper_position: '%d' '%d' '%d'", tip_rotation, gripper_offset, gripper_position);
  std::array<int, 4>  new_positions = {
    static_cast<int>(std::round(starting_positions[0] + tip_rotation - gripper_offset - gripper_position)),
    static_cast<int>(std::round(starting_positions[1] + shaft_rot + m1_bend)), // TODO: drive this motor
    static_cast<int>(std::round(starting_positions[2] + shaft_rot)), // TODO: drive this motor
    static_cast<int>(std::round(starting_positions[3] + roll))
  };
  return new_positions;
}


void InstrumentController::drive_motors(const std::array<int, 4>& m_array) {
      // Check differences to prevent too large movements
      for (size_t i = 0; i < 4; ++i){
        int diff = m_array[i] - motor_controller_.get_current_position(i);
        if (std::abs(diff) > 30){
          if (diff > 0){
            m_array[i] = motor_controller_.get_current_position(i) + 30;
          }else{
            m_array[i] = motor_controller_.get_current_position(i) - 30;
          }
        }
      }

      motor_controller_.send_motor_positions(m_array);
    }

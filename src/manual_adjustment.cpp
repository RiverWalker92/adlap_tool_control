#include "adlap_tool_control/instrument_controller.hpp"
#include "adlap_tool_control/keyboard_reader.hpp"

#include "rclcpp/rclcpp.hpp"

/// @brief Allow manual adjustment of the lower motors using keyboard input
void InstrumentController::manual_adjustment()
{
  ///// Workaround  - manual adjustment of the lower motors to get the shaft straight and do some testing ////
  KeyboardReader input;
  std::array<int, 4> duty_cycle_array = {7, 7, 7, 7};
  char c;
  bool initialized = false;
  bool ready_for_angles = false;
  for (;;)
  {
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
    int step_size = gearbox.get_pulses_per_degree(1) * 5;
    std::array<int, 4> current_positions = gearbox.motor_controller.get_positions();
    RCLCPP_INFO(logger_, "Current positions: [%d, %d, %d, %d]", current_positions[0], current_positions[1], current_positions[2], current_positions[3]);
    switch (c)
    {
      case KEYCODE_C:
        RCLCPP_INFO(logger_, "C -> Couple sequence");
        {
        std_msgs::msg::String task_msg;
        task_msg.data = "coupling_sequence";
        task_publisher_->publish(task_msg);

        gearbox.couple_sequence();
        break;
        }
      case KEYCODE_U:
        RCLCPP_INFO(logger_, "U -> Update starting positions");
        {
        std_msgs::msg::String task_msg;
        task_msg.data = "updating_sequence";
        task_publisher_->publish(task_msg);
        if (!initialized) {
          RCLCPP_WARN(logger_, "Motors not initialized yet, press 'I' to initialize before updating starting positions");
          break;
        }
        current_positions[3] -= 50 * gearbox.get_pulses_per_degree(3); // Add some tension to close the jaws
        gearbox.motor_controller.send_motor_positions(current_positions);
        rclcpp::sleep_for(std::chrono::milliseconds(200));
        gearbox.motor_controller.update_starting_positions();
        smoothed_tip_rotation_ = update_history_and_get_mean(tip_rotation_history_, 0.0, 1, -M_PI, M_PI);
        smoothed_pitch_ = update_history_and_get_mean(pitch_history_, 0.0, 1, -M_PI, M_PI);
        smoothed_yaw_ = update_history_and_get_mean(yaw_history_, 0.0, 1, -M_PI, M_PI);
        smoothed_articulation_ = update_history_and_get_mean(articulation_history_, 0.0, 1, -M_PI, M_PI);
        smoothed_bend_ = update_history_and_get_mean(bend_history_, 0.0, 1, -M_PI, M_PI);
        smoothed_shaft_roll_ = update_history_and_get_mean(shaft_roll_history_, 0.0, 1, -M_PI, M_PI);
        RCLCPP_INFO(logger_, "Smoothed angles - roll: '%f', pitch: '%f', yaw: '%f', articulation: '%f'", smoothed_tip_rotation_, smoothed_pitch_, smoothed_yaw_, smoothed_articulation_);
        absolute_shaft_roll_ = 0.0;
        bend_play_compensation_ = 0;
        play_comp_position_m1_ = current_positions[1];
        play_comp_position_m2_ = current_positions[2];
        ready_for_angles = true;
        break;
      }
      case KEYCODE_I:
        RCLCPP_INFO(logger_, "I -> Initialize lower motors");
        {
        std_msgs::msg::String task_msg;
        task_msg.data = "initializing_sequence";
        task_publisher_->publish(task_msg);
        gearbox.setup_motors();
        initialized = true;
        break;
        }
      case KEYCODE_R:
        RCLCPP_INFO(logger_, "R -> Reset to starting positions");
        gearbox.motor_controller.send_motor_positions(gearbox.motor_controller.get_starting_positions());
        break;
      case KEYCODE_PLUS:
        RCLCPP_INFO(logger_, "Increase duty cycle");
        for (int i = 0; i < 4; i++) {
          duty_cycle_array[i] += 1;
        }
        gearbox.motor_controller.send_duty_cycle(duty_cycle_array);
        break;
      case KEYCODE_0:
        RCLCPP_INFO(logger_, "0 -> Send all 0 angles");
        if (!ready_for_angles) {
          RCLCPP_WARN(logger_, "Not ready for angle control yet, press 'I' to initialize motors and 'U' to update starting positions");
          break;
        }
        set_euler_angles(0, 0, 0, 0);
        break;
      case KEYCODE_1:
        RCLCPP_INFO(logger_, "1 -> Roll left (bend angle 10 degrees)");
        if (!ready_for_angles) {
          RCLCPP_WARN(logger_, "Not ready for angle control yet, press 'I' to initialize motors and 'U' to update starting positions");
          break;
        }
        set_euler_angles(smoothed_tip_rotation_ - angle_rad, smoothed_pitch_, smoothed_yaw_, smoothed_articulation_);
        break;
      case KEYCODE_2:
        RCLCPP_INFO(logger_, "2 -> Pitch down");
        if (!ready_for_angles) {
          RCLCPP_WARN(logger_, "Not ready for angle control yet, press 'I' to initialize motors and 'U' to update starting positions");
          break;
        }
        set_euler_angles(smoothed_tip_rotation_, smoothed_pitch_ - angle_rad, smoothed_yaw_, smoothed_articulation_);
        break;
      case KEYCODE_3:
        RCLCPP_INFO(logger_, "3 -> Roll right (bend angle -10 degrees)");
        if (!ready_for_angles) {
          RCLCPP_WARN(logger_, "Not ready for angle control yet, press 'I' to initialize motors and 'U' to update starting positions");
          break;
        }
        set_euler_angles(smoothed_tip_rotation_ + angle_rad, smoothed_pitch_, smoothed_yaw_, smoothed_articulation_);
        break;
      case KEYCODE_4:
        RCLCPP_INFO(logger_, "4 -> Yaw left");
        if (!ready_for_angles) {
          RCLCPP_WARN(logger_, "Not ready for angle control yet, press 'I' to initialize motors and 'U' to update starting positions");
          break;
        }
        set_euler_angles(smoothed_tip_rotation_, smoothed_pitch_, smoothed_yaw_ - angle_rad, smoothed_articulation_);
        break;
      case KEYCODE_5:
        RCLCPP_INFO(logger_, "5 -> Full rotation");
        gearbox.motor_controller.send_relative_motor_positions(0, gearbox.get_pulses_per_rotation(1), gearbox.get_pulses_per_rotation(2), 0);
        break;
      case KEYCODE_6:
        RCLCPP_INFO(logger_, "6 -> Yaw right");
        if (!ready_for_angles) {
          RCLCPP_WARN(logger_, "Not ready for angle control yet, press 'I' to initialize motors and 'U' to update starting positions");
          break;
        }
        set_euler_angles(smoothed_tip_rotation_, smoothed_pitch_, smoothed_yaw_ + angle_rad, smoothed_articulation_);
        break;
      case KEYCODE_7:
        RCLCPP_INFO(logger_, "7 -> open articulation");
        if (!ready_for_angles) {
          RCLCPP_WARN(logger_, "Not ready for angle control yet, press 'I' to initialize motors and 'U' to update starting positions");
          break;
        }
        set_euler_angles(smoothed_tip_rotation_, smoothed_pitch_, smoothed_yaw_, smoothed_articulation_ + angle_rad);
        break;
      case KEYCODE_8:
        RCLCPP_INFO(logger_, "8 -> Pitch up");
        if (!ready_for_angles) {
          RCLCPP_WARN(logger_, "Not ready for angle control yet, press 'I' to initialize motors and 'U' to update starting positions");
          break;
        }
        set_euler_angles(smoothed_tip_rotation_, smoothed_pitch_ + angle_rad, smoothed_yaw_, smoothed_articulation_);
        break;
      case KEYCODE_9:
        RCLCPP_INFO(logger_, "9 -> close articulation");
        if (!ready_for_angles) {
          RCLCPP_WARN(logger_, "Not ready for angle control yet, press 'I' to initialize motors and 'U' to update starting positions");
          break;
        }
        set_euler_angles(smoothed_tip_rotation_, smoothed_pitch_, smoothed_yaw_, smoothed_articulation_ - angle_rad);
        break;
      case KEYCODE_RIGHT:
        RCLCPP_INFO(logger_, "RIGHT");
        gearbox.motor_controller.send_relative_motor_positions(0, step_size, 0, 0, true);
        break;
      case KEYCODE_LEFT:
        RCLCPP_INFO(logger_, "LEFT");
        gearbox.motor_controller.send_relative_motor_positions(0, -step_size, 0, 0, true);
        break;
      case KEYCODE_UP:
        RCLCPP_INFO(logger_, "UP");
        gearbox.motor_controller.send_relative_motor_positions(0, 0, step_size, 0, true);
        break;
      case KEYCODE_DOWN:
        RCLCPP_INFO(logger_, "DOWN");
        gearbox.motor_controller.send_relative_motor_positions(0, 0, -step_size, 0, true);
        break;
      case KEYCODE_W:
        RCLCPP_INFO(logger_, "W");
        gearbox.motor_controller.send_relative_motor_positions(step_size, 0, 0, 0, true);
        break;
      case KEYCODE_S:
        RCLCPP_INFO(logger_, "S");
        gearbox.motor_controller.send_relative_motor_positions(-step_size, 0, 0, 0, true);
        break;
      case KEYCODE_D:
        RCLCPP_INFO(logger_, "D");
        gearbox.motor_controller.send_relative_motor_positions(step_size, 0, 0, step_size, true);
        break;
      case KEYCODE_A:
        RCLCPP_INFO(logger_, "A");
        gearbox.motor_controller.send_relative_motor_positions(-step_size, 0, 0, -step_size, true);
        break;
      case KEYCODE_E:
        gearbox.motor_controller.send_relative_motor_positions(0, step_size, step_size, 0, true);
        RCLCPP_INFO(logger_, "E");
        break;
      case KEYCODE_Q:
        gearbox.motor_controller.send_relative_motor_positions(0, -step_size, -step_size, 0, true);
        RCLCPP_INFO(logger_, "Q");
        break;
      case KEYCODE_ENTER:
        RCLCPP_INFO(logger_, "ENTER");
        if (!ready_for_angles) {
          RCLCPP_WARN(logger_, "Not ready for angle control yet, press 'I' to initialize motors and 'U' to update starting positions");
          break;
        }
        input.shutdown();
        gearbox.motor_controller.update_starting_positions();
        return;
      default:
        RCLCPP_WARN(logger_, "Unknown key pressed: 0x%02X", c);
        continue;
    }
  }
}
#pragma once
#include "adlap_tool_control/adlap_gearbox.hpp"


// ------------------------------------------------------------
// Control sequences for specific operations like coupling and setup
// ------------------------------------------------------------


/**
 * @brief Couple the motors to the gearbox.
 * 
 * Rotate each motor back and forth until it is blocked. 
 * It's movement is purposefully jittery to make sure the pins fall in.  
 * 
 */
void Gearbox::couple_sequence()
{
  bool verbose = true;
  // Making some rotations to get the pins do drop into the holes of the gearbox
  for (int i : {0, 1, 2, 3})
  {
    int step_size = get_pulses_per_degree(is_upper_motor(i)) * 20; // Step size of 20 degrees in pulses
    // Move the motor till it is blocked (coupled and reached its end position)
    std::array<int, 4> forward = {0, 0, 0, 0};
    forward[i] = step_size;
    std::array<int, 4> backward = {0, 0, 0, 0};
    backward[i] = -step_size;
    do
    {
      motor_controller.send_relative_motor_positions(forward, verbose);
      motor_controller.send_relative_motor_positions(forward, verbose);
      rclcpp::sleep_for(std::chrono::milliseconds(300));
      motor_controller.send_relative_motor_positions(backward, verbose);
      rclcpp::sleep_for(std::chrono::milliseconds(100));
      if (!rclcpp::ok()) throw std::runtime_error("Shutdown requested (rclcpp::ok() == false)");
    } while (!motor_controller.get_blocked()[i]);
    RCLCPP_DEBUG(logger_, "Motor %d blocked at position %d", i, motor_controller.get_positions()[i]);

    do
    {
      motor_controller.send_relative_motor_positions(backward, verbose);
      rclcpp::sleep_for(std::chrono::milliseconds(150));
      if (!rclcpp::ok()) throw std::runtime_error("Shutdown requested (rclcpp::ok() == false)");
    } while (!motor_controller.get_blocked()[i]);
    RCLCPP_DEBUG(logger_, "Motor %d blocked at position %d", i, motor_controller.get_positions()[i]);

    // Unblock the motor
    std::array<int, 4> unblock_values = motor_controller.get_positions();
    unblock_values[i] += step_size; // Move back a bit to unblock the motor
    motor_controller.send_motor_positions(unblock_values, verbose);
    rclcpp::sleep_for(std::chrono::milliseconds(500));
  }
  RCLCPP_DEBUG(logger_, "Coupling sequence completed");
}

/**
 * @brief Set up the motors in the correct position for instrument operation to start.
 * 
 * Move motor 1 to both extremes to find the bending limits and put it in the center.
 * Then move motor 1 and 2 together to find the hall sensor position and align the bending direction with that.
 * Finally, move motor 3 to the starting position for the gripper collet wheel to be thightened.
 * 
 *  */
void Gearbox::setup_motors()
{
  bool verbose = true;
  // Move motor 1 to both extremes and position it in the middle of that.
  int motor_nr = 1;

  // Temporarily set the duty cycle to a higher value to make sure the motor can move to the blocked position for calibration. 
  std::array<int, 4> original_duty_cycle = motor_controller.get_duty_cycles();
  // increace duty cycle for motor 1
  std::array<int, 4> new_duty_cycle = original_duty_cycle;
  new_duty_cycle[1] = new_duty_cycle[1] * 1.5; // Increase duty cycle by 50% for motor 1
  motor_controller.send_duty_cycle(new_duty_cycle, verbose);

  int step_size = get_pulses_per_degree(motor_nr); // Step size of a degree in pulses
  int max_position = 0;
  int min_position = 0;
  std::array<int, 4> driving_values;
  std::array<int, 4> setup_values;
  for (int j : {-step_size, step_size})
  {
    
    motor_controller.update_target_positions();  // reset target positions to current
    // Move the motor till it is blocked
    driving_values = {0, j, 0, 0};
    do
    {
      motor_controller.send_relative_motor_positions(driving_values, verbose);
      rclcpp::sleep_for(std::chrono::milliseconds(20));
      if (!rclcpp::ok()) throw std::runtime_error("Shutdown requested (rclcpp::ok() == false)");
    } while (!motor_controller.get_blocked()[motor_nr]);
    if (j > 0)
    {
      max_position = motor_controller.get_positions()[motor_nr];
      RCLCPP_DEBUG(logger_, "Motor %d blocked at max position %d", motor_nr, max_position);
    }
    else
    {
      min_position = motor_controller.get_positions()[motor_nr];
      RCLCPP_DEBUG(logger_, "Motor %d blocked at min position %d", motor_nr, min_position);
    }
  }
  // Place the motor in the middle between the two blocked positions
  setup_values = motor_controller.get_positions();
  setup_values[motor_nr] = (max_position + min_position) / 2 - lower_motor_start_offset_; // Add some extra to compensate for backlash
  motor_controller.send_motor_positions(setup_values, verbose);
  rclcpp::sleep_for(std::chrono::milliseconds(500));
  // Drive the motor back to remove tension and put it in the neutral starting position for the instrument operation
  setup_values[motor_nr] += 2 * get_pulses_lower_motors_play();
  motor_controller.send_motor_positions(setup_values, verbose);
  rclcpp::sleep_for(std::chrono::milliseconds(500));
  // Restore the original duty cycle after calibration
  motor_controller.send_duty_cycle(original_duty_cycle, verbose);


  //Move motor 1 and 2 together so motor 2 has the hall magnet down
  driving_values = {0, step_size, step_size, 0};
  int hall_up1 = NULL;
  int hall_up2 = NULL;
  int hall_down1 = NULL;
  int hall_down2 = NULL;
  bool prev_hall = motor_controller.get_hall_sensors()[1];
  do
  {
    motor_controller.send_relative_motor_positions(driving_values, verbose);
    rclcpp::sleep_for(std::chrono::milliseconds(20));
    std::array<bool, 2> hall_values = motor_controller.get_hall_sensors();
    if (hall_values[1] != prev_hall)
    {
      if (hall_values[1])
      {
        hall_up1 = motor_controller.get_positions()[1];
        hall_up2 = motor_controller.get_positions()[2];
        RCLCPP_DEBUG(logger_, "Hall sensor 2 up at position %d", hall_up2);
      }
      else
      {
        hall_down1 = motor_controller.get_positions()[1];
        hall_down2 = motor_controller.get_positions()[2];
        RCLCPP_DEBUG(logger_, "Hall sensor 2 down at position %d", hall_down2);
      }
    }
    prev_hall = hall_values[1];
    if (!rclcpp::ok()) throw std::runtime_error("Shutdown requested (rclcpp::ok() == false)");
  } while (hall_up1 == NULL || hall_down1 == NULL);
  // Place both motor in the middle between the hall edges
  setup_values = motor_controller.get_positions();
  if (hall_up1 > hall_down1) hall_down1 += get_pulses_per_rotation(1);
  setup_values[1] = (hall_up1 + hall_down1) / 2 + HALL2_OFFSET; // The pin is not exactly in the middle
  if (hall_up2 > hall_down2) hall_down2 += get_pulses_per_rotation(2);
  setup_values[2] = (hall_up2 + hall_down2) / 2 + HALL2_OFFSET; // The pin is not exactly in the middle
  motor_controller.send_motor_positions(setup_values, verbose);
  rclcpp::sleep_for(std::chrono::milliseconds(500));

  // Move motor 3 to the starting position so the collet can be tightened around the instrument shaft
  motor_nr = 3;
  driving_values = {0, 0, 0, (int)(-get_pulses_per_degree(motor_nr) * 5)};
  do
  {
    motor_controller.send_relative_motor_positions(driving_values, verbose);
    rclcpp::sleep_for(std::chrono::milliseconds(50));
    if (!rclcpp::ok()) throw std::runtime_error("Shutdown requested (rclcpp::ok() == false)");
  } while (!motor_controller.get_blocked()[motor_nr]);
  // Move back a bit to remove tension and put it in the neutral starting position for the instrument operation
  setup_values = motor_controller.get_positions();
  setup_values[motor_nr] += gripper_start_offset_; // Add some extra to compensate for backlash
  motor_controller.send_motor_positions(setup_values, verbose);
  rclcpp::sleep_for(std::chrono::milliseconds(500));

  RCLCPP_DEBUG(logger_, "Motor setup sequence completed with lower_motor_start_offset %d and gripper_start_offset %d", lower_motor_start_offset_, gripper_start_offset_);
}
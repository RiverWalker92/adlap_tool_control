#include "adlap_tool_control/instrument_controller.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"



InstrumentController::InstrumentController(Gearbox& gearbox, rclcpp::Logger logger) 
    : gearbox(gearbox), logger_(logger)
{
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

void InstrumentController::set_euler_angles(double roll, double pitch, double yaw, double articulation, bool verbose)
{
    // Add to history + compute smoothed values (same helper called 4 times)
    smoothed_tip_rotation_ = update_history_and_get_mean(tip_rotation_history_, roll, smoothing_factor_, -M_PI, M_PI);
    smoothed_pitch_ = update_history_and_get_mean(pitch_history_, pitch, smoothing_factor_, -M_PI/4, M_PI/4);
    smoothed_yaw_ = update_history_and_get_mean(yaw_history_, yaw, smoothing_factor_, -M_PI/4, M_PI/4);
    smoothed_articulation_ = update_history_and_get_mean(articulation_history_, articulation, smoothing_factor_, 0, M_PI/6);
    if (verbose) RCLCPP_DEBUG(logger_, "Smoothed angles - roll: '%f', pitch: '%f', yaw: '%f', articulation: '%f'", smoothed_tip_rotation_, smoothed_pitch_, smoothed_yaw_, smoothed_articulation_);
    gearbox.motor_controller.send_motor_positions(calculate_motor_positions_from_euler_angles(smoothed_tip_rotation_, smoothed_pitch_, smoothed_yaw_, smoothed_articulation_, verbose), true);
}

void InstrumentController::set_joint_angles(double shaft_roll, double bend, double tip_rotation, double articulation, bool verbose)
{
    // Add to history + compute smoothed values (same helper called 4 times)
    smoothed_shaft_roll_ = update_history_and_get_mean(shaft_roll_history_, shaft_roll, smoothing_factor_, -INFINITY, INFINITY);
    smoothed_bend_ = update_history_and_get_mean(bend_history_, bend, smoothing_factor_, -M_PI/4, M_PI/4);
    smoothed_tip_rotation_ = update_history_and_get_mean(tip_rotation_history_, tip_rotation, smoothing_factor_, -M_PI, M_PI);
    smoothed_articulation_ = update_history_and_get_mean(articulation_history_, articulation, smoothing_factor_, 0, M_PI/6);
    if (verbose) RCLCPP_DEBUG(logger_, "Smoothed angles - shaft_roll: '%f', bend: '%f', tip_rotation: '%f', articulation: '%f'", smoothed_shaft_roll_, smoothed_bend_, smoothed_tip_rotation_, smoothed_articulation_);
    gearbox.motor_controller.send_motor_positions(calculate_motor_positions_from_joint_angles(smoothed_shaft_roll_, smoothed_bend_, smoothed_tip_rotation_, smoothed_articulation_, verbose), true);
}

int InstrumentController::get_motor2_value_for_angle(double radians, bool verbose){
  double degrees = radians * 180.0 / M_PI;
  if (degrees > MAX_BEND_ANGLE_DEGREES) {
    degrees = MAX_BEND_ANGLE_DEGREES;
  } else if (degrees < -MAX_BEND_ANGLE_DEGREES) {
    degrees = -MAX_BEND_ANGLE_DEGREES;
  }

  int starting_difference = gearbox.motor_controller.get_starting_positions()[2] - gearbox.motor_controller.get_starting_positions()[1]; // Starting difference between motor 2 and motor 1, compensated for the initial offset found in setup
  //int current_difference = gearbox.motor_controller.get_positions()[2] - gearbox.motor_controller.get_positions()[1] - starting_difference - 2 * bend_play_compensation_; // Current difference between motor 2 and motor 1, compensated for play
  
  // Use real play to prevent incorrect position estimate if the motors havent reached target yet.
  std::array<int, 4> current_positions = gearbox.motor_controller.get_positions();
  int real_play = calculate_real_play(current_positions); // Calculate the real play based on the current motor positions and the current offset
  int current_difference = current_positions[2] - current_positions[1] - starting_difference - 2 * real_play; // Current difference between motor 2 and motor 1, compensated for play

  int wanted_difference = static_cast<int>(std::round(degrees * gearbox.get_pulses_per_degree(1) * BEND_FACTOR));
  int relative_difference = wanted_difference - current_difference;

  int deadband_threshold = static_cast<int>(std::round(gearbox.get_pulses_per_degree(1) * 2.5)); // deadband threshold in pulses
  if (verbose) RCLCPP_DEBUG(logger_, "relative_difference: '%d', real_play: '%d', deadband_threshold: '%d', bend_play_compensation_: '%d', current_difference: '%d', wanted_difference: '%d'", relative_difference, real_play, deadband_threshold, bend_play_compensation_, current_difference, wanted_difference);


  if ( bend_play_compensation_ > 0) {
    if (relative_difference < -deadband_threshold) {
      // Moving in the opposite direction, so apply deadband compensation
      bend_play_compensation_ = -gearbox.get_pulses_lower_motors_play();
    }
    else if (relative_difference <= 0) { 
      // In the deadband region
      if (verbose) RCLCPP_DEBUG(logger_, "Same angle, no adjustment needed");
      return wanted_difference; // Keep the commanded bend stable instead of chasing live feedback
    }
  }
  else{
    if (relative_difference > deadband_threshold) {
      // Moving in the opposite direction, so apply deadband compensation
      bend_play_compensation_ = gearbox.get_pulses_lower_motors_play();
    }
    else if (relative_difference >= 0) { 
      // In the deadband region
      if (verbose) RCLCPP_DEBUG(logger_, "Same angle, no adjustment needed");
      return wanted_difference; // Keep the commanded bend stable instead of chasing live feedback
    }
  }

  if (verbose) RCLCPP_DEBUG(logger_, "play_compensation_: '%d'", bend_play_compensation_);

  return wanted_difference;
}

std::array<int, 4> InstrumentController::calculate_motor_positions_from_euler_angles(double roll, double pitch, double yaw, double articulation, bool verbose)
{
  // Drive the lower motors with the pitch and yaw. 
  // Motor 2 angles the shaft.
  // Motor 1 and 2 together change the direction of the bend.

  double h = -std::tan(smoothed_pitch_);
  double w = -std::tan(smoothed_yaw_);
  double shaft_roll = std::atan2(w, h) + M_PI; // direction of the bend
  double l1 = std::sqrt(h*h + w*w);
  double bend = std::atan(l1); // angle of the bend
  // TODO: implement bending the other way and and solve singularity handling

  return calculate_motor_positions_from_joint_angles(shaft_roll, bend, roll, articulation, verbose);
}


std::array<int, 4> InstrumentController::calculate_motor_positions_from_joint_angles(double shaft_roll, double bend, double tip_rotation, double articulation, bool verbose)
{
  // Calculate the shortest rotation for shaft_roll
  // We need to choose the new shaft_roll to be the closest to the current position to prevent unnecessary full rotations
  // The motors are driven with absolute positions, so we need to keep track of the absolute shaft_roll
  double delta_shaft_roll = shaft_roll - absolute_shaft_roll_;
  double diff = std::fmod(delta_shaft_roll, TWO_PI);
  if (diff < - M_PI) {
      diff += TWO_PI;
  } else if (diff >= M_PI) {
      diff -= TWO_PI;
  }
  absolute_shaft_roll_ += diff; // Update absolute_shaft_roll_ with the shortest rotation
  
  int shaft_rotation_pulses = absolute_shaft_roll_ * gearbox.get_pulses_per_rotation(1) / (TWO_PI);
  int m2_bend_pulses = get_motor2_value_for_angle(bend, verbose);
  int tip_rotation_pulses = - static_cast<int>(std::round(tip_rotation * gearbox.get_pulses_per_rotation(0) / (TWO_PI)));
  int articulation_pulses = static_cast<int>(std::round(ARTICULATION_FACTOR * articulation * gearbox.get_pulses_per_rotation(3) / (TWO_PI)));

  if (verbose) {
    RCLCPP_DEBUG(
      logger_,
      "### Calculated motor positions ###\n"
      "  shaft_rotation_pulses: '%d'\n"
      "  m2_bend_pulses: '%d'\n"
      "  tip_rotation_pulses: '%d'\n"
      "  articulation_pulses: '%d'\n",
      shaft_rotation_pulses, m2_bend_pulses, tip_rotation_pulses, articulation_pulses);
  }

  std::array<int, 4>  starting_positions = gearbox.motor_controller.get_starting_positions();
  std::array<int, 4>  new_positions = {
    static_cast<int>(std::round(starting_positions[0] + tip_rotation_pulses)),
    static_cast<int>(std::round(starting_positions[1] + shaft_rotation_pulses - bend_play_compensation_)),
    static_cast<int>(std::round(starting_positions[2] + shaft_rotation_pulses + m2_bend_pulses + bend_play_compensation_)), 
    static_cast<int>(std::round(starting_positions[3] + tip_rotation_pulses + articulation_pulses))    
  };
  return new_positions;
}

// TODO: make this trigger on each recieved message in motor thread, otherwise we cannot reliably use m1_m2_offset_ variable.
// Now kinda fixed by calling this each time we recieve a new command in the node during motor position calculation.
int InstrumentController::calculate_real_play(const std::array<int, 4>& current_positions){
  std::array<int, 4>  starting_positions = gearbox.motor_controller.get_starting_positions();
  int play =  gearbox.get_pulses_lower_motors_play();
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
  return real_play;
}

std::array<double, 4> InstrumentController::joint_angles_from_motors(const std::array<int, 4>& current_positions)
{
  std::array<int, 4>  starting_positions = gearbox.motor_controller.get_starting_positions();

  double tip_rotation = (static_cast<double>(starting_positions[0] - current_positions[0]) * TWO_PI / gearbox.get_pulses_per_rotation(0)); // Invert the roll calculation
  double articulation_angle = (static_cast<double>(current_positions[3] - starting_positions[3] - current_positions[0] + starting_positions[0]) * TWO_PI / gearbox.get_pulses_per_rotation(3)) / ARTICULATION_FACTOR; // Invert the articulation calculation

  int real_play = calculate_real_play(current_positions); // Calculate the real play based on the current motor positions and the current offset
  double shaft_rot = (static_cast<double>(current_positions[1] - starting_positions[1] + real_play)); 
  double m2_bend = (static_cast<double>(current_positions[2] - starting_positions[2] - shaft_rot - real_play));

  double bend_degrees = m1_m2_offset_ / (gearbox.get_pulses_per_degree(1) * BEND_FACTOR);
  double bend_rad = bend_degrees * M_PI / 180.0;

  double shaft_roll = shaft_rot * TWO_PI / gearbox.get_pulses_per_rotation(1);

  return std::array<double, 4>{shaft_roll, bend_rad, tip_rotation, articulation_angle};
}

std::array<double, 4> InstrumentController::euler_angles_from_motors(const std::array<int, 4>& m_array)
{
  std::array<double, 4> joint_angles = joint_angles_from_motors(m_array);
  double shaft_roll = joint_angles[0];
  double bend = joint_angles[1];
  double tip_rotation = joint_angles[2];
  double articulation_angle = joint_angles[3];

  double l1 = std::tan(bend);
  double h = -l1 * std::cos(shaft_roll);
  double w = -l1 * std::sin(shaft_roll);
  double pitch_angle = -std::atan(h);
  double yaw_angle = -std::atan(w);

  return std::array<double, 4>{tip_rotation, pitch_angle, yaw_angle, articulation_angle};
}


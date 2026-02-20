#include "adlap_tool_control/serial_port.hpp"
#include "adlap_tool_control/keyboard_reader.hpp"

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

using namespace std::chrono_literals;

const std::string STATUS_TOPIC = "/tool_status";
const std::string TOOL_CONTROL_TOPIC = "/instrument_angles";

const float UPPER_MOTOR_FACTOR = 15.0f/25;
/* Motor gearbox is 150:1. The motors are equipped with a 12 pulse per rotation hall encoder.
This means 450 pulses equals 1 full rotation.*/
const int MOTOR_PULSES_PER_ROTATION = 1800; // pulses
/* The lower motors bend the shaft. The difference approximates the max bending.*/
const int LOWER_MOTORS_MAX_DIFFERENCE = 160;
const int LOWER_MOTORS_PLAY = 15; // in pulses. When changing directions, you need to move twice the play before the gearbox starts moving

const int HALL1_OFFSET = -120; // in pulses, to align the hall sensor with the physical 0 degree position
const int HALL2_OFFSET = -90; // in pulses, to align the hall sensor with the physical 0 degree position

const double LOWER_MOTORS_MAX_BEND_ANGLE = 50.0; // degrees
const int MIN_WAIT_TIME = 4; // Minimum wait time in milliseconds for the motors to respond
const double TWO_PI = 2.0 * M_PI;      

std::array<int, 4>  starting_positions = {0, 0, 0, 0}; // Starting positions for the motors
std::array<int, 4>  positions = {0, 0, 0, 0}; // Current positions of the motors
std::array<int, 6>  response_values = {0, 0, 0, 0, 0, 0}; // Response values from the motors

std::deque<double> roll_history; // History of roll values for smoothing
std::deque<double> pitch_history; // History of pitch values for smoothing
std::deque<double> yaw_history; // History of yaw values for smoothing
std::deque<double> gripper_history; // History of gripper values for smoothing

double smoothed_roll = 0.0;
double smoothed_pitch = 0.0;
double smoothed_yaw = 0.0;
double smoothed_gripper = 0.0;

int blocking_current = 30; // Blocking current threshold for the motors (used at initialization)
bool blocked[4] = {false, false, false, false}; // Flag to indicate if the motors are blocked
int max_current = 60; // Maximum current threshold for the motors
bool maxed[4] = {false, false, false, false}; // Flag to indicate if the motors are at maximum current

int bend_angle = 0; // Current articulation angles
double absolute_omega = 0.0; // Absolute omega value for shortest rotation calculation

class ToolController : public rclcpp::Node
{
  public:
  ToolController(std::shared_ptr<SerialPort> serial)
    : Node("tool_control_node"), serial_(serial)
    {
      // The publisher and subscriber topics are relative, so they are mapped to left or right with the node namespace
      publisher_ = this->create_publisher<std_msgs::msg::String>("~" + STATUS_TOPIC, 10);
      subscription_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
        "~" + TOOL_CONTROL_TOPIC, 10, std::bind(&ToolController::topic_callback, this, std::placeholders::_1));  

      // Test the serial port connection
      auto message = motor_message(positions);
      RCLCPP_INFO(this->get_logger(), "Writing: '%s'", message.c_str());
      serial_->write_data(message);

      // Wait for a short period to allow the device to respond
      std::this_thread::sleep_for(std::chrono::milliseconds(MIN_WAIT_TIME));
      std::string response = serial_->read_data();
      if (!response.empty()) {
        RCLCPP_INFO(this->get_logger(), "Received: '%s'", response.c_str());
      }else {
        throw std::runtime_error("Failed to read from serial port");
      }
      // Initialize the motors by coupling them to the gearbox
      //initialize();
      manual_adjustment();
    }

  private:
    std::string motor_message(const std::array<int, 4>& m_array){
      std::string response = 
        std::to_string(m_array[0]) + ", " 
        + std::to_string(m_array[1]) + ", " 
        + std::to_string(m_array[2]) + ", " 
        + std::to_string(m_array[3]) + "\n";
      return response;
    }

    void send_relative_motor_positions(const std::array<int, 4>& m_array, bool verbose=true) {
      send_relative_motor_positions(m_array[0], m_array[1], m_array[2], m_array[3], verbose);
    }

    void send_relative_motor_positions(int m0, int m1, int m2, int m3, bool verbose=true) {
      std::array<int, 4>  new_positions = {0, 0, 0, 0};
      new_positions[0] = positions[0] + m0;
      new_positions[1] = positions[1] + m1;
      new_positions[2] = positions[2] + m2;
      new_positions[3] = positions[3] + m3;
      
      send_motor_positions(new_positions, verbose);
    }

    void send_motor_positions(int m0, int m1, int m2, int m3, bool verbose=true) {
      std::array<int, 4>  new_positions = {m0, m1, m2, m3};
      send_motor_positions(new_positions, verbose);
    }

    /// @brief Send new absolute motor positions to the motor controller
    /// @param new_positions Array of 4 integers representing the new motor positions
    /// @throws std::runtime_error if the motor response is not ok after reversing movement
    void send_motor_positions(const std::array<int, 4>& new_positions, bool verbose=true) {
      std::string message = motor_message(new_positions);

      //TODO: Make steps smaller when large movements are commanded

      if (verbose) {
        RCLCPP_INFO(this->get_logger(), "Writing: '%s'", message.c_str());
      }
      serial_->write_data(message);
      rclcpp::sleep_for(std::chrono::milliseconds(MIN_WAIT_TIME));
      bool motor_status = set_response_values(verbose);

      // TODO: This is a workaround, cause the current return values are one loop behind
      serial_->write_data(message);
      rclcpp::sleep_for(std::chrono::milliseconds(MIN_WAIT_TIME));
      motor_status = set_response_values(verbose);
      // End of workaround

      if (!motor_status) {
        RCLCPP_ERROR(this->get_logger(), "Motor response not ok, reversing movement");
        serial_->write_data(motor_message(positions));
        rclcpp::sleep_for(std::chrono::milliseconds(500));
        motor_status = set_response_values(verbose);

        
        // TODO: This is a workaround, cause the current return values are one loop behind
        serial_->write_data(motor_message(positions));
        rclcpp::sleep_for(std::chrono::milliseconds(500));
        motor_status = set_response_values(verbose);
        // End of workaround

        if (!motor_status) {
          throw std::runtime_error("Motor response still not ok after reversing movement");
        }
      }else{
        positions = new_positions;
      }
      return;
    }

    /// @brief Read and set response values from the motor
    /// @return True if the currents are within limits, false otherwise
    bool set_response_values(bool verbose=true) {
      
      bool response_ok = true;
      std::string response = serial_->read_data();
      if (!response.empty()) {
        if (verbose){
          RCLCPP_INFO(this->get_logger(), "Raw response: '%s'", response.c_str());
        }
      }else {
        RCLCPP_ERROR(this->get_logger(), "Failed to read from serial port");
      }
      std::vector<int> values;
      std::stringstream ss(response);
      std::string item;
      while (std::getline(ss, item, ' ')) {
        values.push_back(std::stoi(item)); // Convert to int and store
      }
      if (values.size() < 6) {
        RCLCPP_ERROR(this->get_logger(), "Received less than 6 values from the motor: %zu", values.size());
        throw std::runtime_error("Received less than 6 values from the motor");
      }
      for (int i = 0; i < 6; ++i) {
        // Update the response values array
        response_values[i] = values[i];
      }
      for (int i = 0; i < 4; ++i) {
        // Check if the current value exceeds the blocking current
        if (response_values[i] > blocking_current){
          blocked[i] = true;
          RCLCPP_WARN(this->get_logger(), "Motor %d value out of range: %d", i, response_values[i]);
        }
        else {
          blocked[i] = false;
        }
        // Check if the current value exceeds the maximum current
        if(response_values[i] > max_current){
          maxed[i] = true;
          response_ok = false;
          RCLCPP_ERROR(this->get_logger(), "Motor %d current exceeded maximum: %d", i, response_values[i]);
        }
        else {
          maxed[i] = false;
        }
      }
      return response_ok;
    }

    /// @brief Couple the motors to the gearbox
    void couple_sequence(){
      // Making some rotations to get the pins do drop into the holes of the gearbox
      // Couple the upper motors
      for (int i : {0,1,2,3}){
        for (int j : {10, -10}){
          // Move the motor till it is blocked (coupled and reached its end position)
          std::array<int, 4> driving_values  = {0, 0, 0, 0};
          driving_values[i] = j;
          do{
            send_relative_motor_positions(driving_values);
          } while (!blocked[i]);
          // Unblock the motor
          driving_values[i] = -5*j;
          send_relative_motor_positions(driving_values);
        }
      }
    }

    
    void find_hall_sensor_positions() {
      // Couple the lower motors
      // These motors also control the front gears, which have to be aligned for the tool to be able to be inserted
      // The hall sensors can be used to find the correct position

      int hall1_value = response_values[4];
      int hall1_up = 0;
      int hall1_down = 0;
      int hall2_value = response_values[5];
      int hall2_up = 0;
      int hall2_down = 0;
      int step_size = 10;
      for (int i = 0; i < 3*MOTOR_PULSES_PER_ROTATION/step_size; i++) {
        send_relative_motor_positions(0,step_size,step_size,0, false);
        if (response_values[4] == 1 and hall1_value == 0) {
          hall1_up = positions[1] % MOTOR_PULSES_PER_ROTATION;
          RCLCPP_INFO(this->get_logger(), "Hall1 up: '%d'", hall1_up);
        }
        if (response_values[4] == 0 and hall1_value == 1) {
          hall1_down = positions[1] % MOTOR_PULSES_PER_ROTATION;
          RCLCPP_INFO(this->get_logger(), "Hall1 down: '%d'", hall1_down);
        }
        if (response_values[5] == 1 and hall2_value == 0) {
          hall2_up = positions[2] % MOTOR_PULSES_PER_ROTATION;
          RCLCPP_INFO(this->get_logger(), "Hall2 up: '%d'", hall2_up);
        }
        if (response_values[5] == 0 and hall2_value == 1) {
          hall2_down = positions[2] % MOTOR_PULSES_PER_ROTATION;
          RCLCPP_INFO(this->get_logger(), "Hall2 down: '%d'", hall2_down);
        }
        hall1_value = response_values[4];
        hall2_value = response_values[5];
      }  
      // Move back to the position where the hall sensor triggers (assumed to be the correct position for instrument insertion)
      if (hall1_up > hall1_down) {
        hall1_down += MOTOR_PULSES_PER_ROTATION;
      }
      if (hall2_up > hall2_down) {
        hall2_down += MOTOR_PULSES_PER_ROTATION;
      }
      int hall1_position = (hall1_up + hall1_down) / 2 + HALL1_OFFSET;
      int hall1_correction = hall1_position - positions[1] % MOTOR_PULSES_PER_ROTATION;
      RCLCPP_INFO(this->get_logger(), "Setting hall1 position: '%d'", hall1_correction);
      int hall2_position = (hall2_up + hall2_down) / 2 + HALL2_OFFSET;
      int hall2_correction = hall2_position - positions[2] % MOTOR_PULSES_PER_ROTATION;
      RCLCPP_INFO(this->get_logger(), "Setting hall2 position: '%d'", hall2_correction);
      send_relative_motor_positions(0,hall1_correction,hall2_correction,0);
    }

    /// @brief Initialize the tool by moving the motors through their ranges to find the limits
    void initialize(){
      // TODO: find the lower motor 1 limits
    }

    int get_relative_motor1_value_for_angle(int degrees){

      if (bend_angle == degrees) {
        RCLCPP_DEBUG(this->get_logger(), "Same angle, no adjustment needed");
        return 0;
      }
      int play_compensation = 0;
      if (degrees > bend_angle) {
        play_compensation = LOWER_MOTORS_PLAY;
      }
      else{
        play_compensation = -LOWER_MOTORS_PLAY;
      }
      bend_angle = degrees;
      int starting_difference = starting_positions[1] - starting_positions[2];
      return int((LOWER_MOTORS_MAX_DIFFERENCE - play_compensation) / LOWER_MOTORS_MAX_BEND_ANGLE * degrees + play_compensation + starting_difference - positions[1] + positions[2]);
    }

    /// @brief Allow manual adjustment of the lower motors using keyboard input
    void manual_adjustment(){
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
        RCLCPP_INFO(this->get_logger(), "value: 0x%02X\n", c);
        switch (c)
        {
          case KEYCODE_C:
            RCLCPP_DEBUG(this->get_logger(), "C -> Couple sequence");
            couple_sequence();
            break;          
          case KEYCODE_I:
            RCLCPP_DEBUG(this->get_logger(), "I -> Initialize");
            initialize();
            break;      
          case KEYCODE_U:
            RCLCPP_DEBUG(this->get_logger(), "U -> Update starting positions");
            starting_positions = positions;            
            break;
          case KEYCODE_R:
            RCLCPP_DEBUG(this->get_logger(), "R -> Reset to starting positions");
            send_motor_positions(starting_positions);
            break;
          case KEYCODE_0:
            RCLCPP_DEBUG(this->get_logger(), "0 -> Set angle of bend to 0 degrees");
            send_relative_motor_positions(0, get_relative_motor1_value_for_angle(0), 0, 0); // starting offset + current position of motor 2
            break;
          case KEYCODE_1:
            RCLCPP_DEBUG(this->get_logger(), "1 -> Set angle of bend to 10 degrees");
            send_relative_motor_positions(0, get_relative_motor1_value_for_angle(10), 0, 0); 
            break;
          case KEYCODE_2:
            RCLCPP_DEBUG(this->get_logger(), "2 -> Set angle of bend to 20 degrees");
            send_relative_motor_positions(0, get_relative_motor1_value_for_angle(20), 0, 0); 
            break;
          case KEYCODE_3:
            RCLCPP_DEBUG(this->get_logger(), "3 -> Set angle of bend to 30 degrees");
            send_relative_motor_positions(0, get_relative_motor1_value_for_angle(30), 0, 0);
            break;
          case KEYCODE_MINUS:
            RCLCPP_DEBUG(this->get_logger(), "MINUS");
            send_relative_motor_positions(0, -10, -10, 0);
            break;
          case KEYCODE_EQUAL:
            RCLCPP_DEBUG(this->get_logger(), "EQUAL");
            send_relative_motor_positions(0, 10, 10, 0);
            break;
          case KEYCODE_RIGHT:
            RCLCPP_DEBUG(this->get_logger(), "RIGHT");
            send_relative_motor_positions(0,10,0,0);
            break;
          case KEYCODE_LEFT:
            RCLCPP_DEBUG(this->get_logger(), "LEFT");
            send_relative_motor_positions(0,-10,0,0);
            break;
          case KEYCODE_UP:
            RCLCPP_DEBUG(this->get_logger(), "UP");
            send_relative_motor_positions(0,0,10,0);
            break;
          case KEYCODE_DOWN:
            RCLCPP_DEBUG(this->get_logger(), "DOWN");
            send_relative_motor_positions(0,0,-10,0);
            break;
          case KEYCODE_W:
            RCLCPP_DEBUG(this->get_logger(), "W");
            send_relative_motor_positions(10,0,0,0);
            break;
          case KEYCODE_S:
            RCLCPP_DEBUG(this->get_logger(), "S");
            send_relative_motor_positions(-10,0,0,0);
            break;
          case KEYCODE_D:
            RCLCPP_DEBUG(this->get_logger(), "D");
            send_relative_motor_positions(10,0,0,10);
            break;
          case KEYCODE_A:
            RCLCPP_DEBUG(this->get_logger(), "A");
            send_relative_motor_positions(-10,0,0,-10);
            break;
          case KEYCODE_E:
            send_relative_motor_positions(0,10,10,0);
            RCLCPP_DEBUG(this->get_logger(), "E");
            break;
          case KEYCODE_Q:
            send_relative_motor_positions(0,-10,-10,0);
            RCLCPP_DEBUG(this->get_logger(), "Q");
            break;
          case KEYCODE_ENTER:
            RCLCPP_DEBUG(this->get_logger(), "ENTER");
            input.shutdown();
            starting_positions = positions;
            return;
          default:
            RCLCPP_DEBUG(this->get_logger(), "Unknown key pressed: 0x%02X", c);
            continue;
        }
      }
    }

    int get_m1_bend_pulses(double radians){
      double degrees = radians * 180.0 / M_PI;
      int play_compensation = 0;
      if (bend_angle == static_cast<int>(std::round(degrees))) {
        play_compensation = 0;
      }
      else if (degrees > bend_angle) {
        play_compensation = LOWER_MOTORS_PLAY;
      }
      else{
        play_compensation = -LOWER_MOTORS_PLAY;
      }
      bend_angle = static_cast<int>(std::round(degrees));
      int starting_difference = starting_positions[1] - starting_positions[2];
      return static_cast<int>(std::round((LOWER_MOTORS_MAX_DIFFERENCE - play_compensation) / LOWER_MOTORS_MAX_BEND_ANGLE * degrees + play_compensation + starting_difference));
    }

    void topic_callback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
    {
      if (msg->data.size() < 4) {
        RCLCPP_WARN(this->get_logger(), "Received data size less than 4");
        return;
      }
      std::vector<double> angles = msg->data;
      // data values are: roll, pitch, yaw, gripper angle
      RCLCPP_INFO(this->get_logger(), "I heard array: '%f' '%f' '%f' '%f'", angles[0], angles[1], angles[2], angles[3]);

      double roll = angles[0];
      double pitch = angles[1];
      double yaw = angles[2] - TWO_PI; // Todo fix 2 pi offset in publishing node 
      double gripper = angles[3];

      // Add to history for smoothing
      int smoothing_factor = 20;
      roll_history.push_front(roll);
      pitch_history.push_front(pitch);
      yaw_history.push_front(yaw);
      gripper_history.push_front(gripper);
      if (roll_history.size() > smoothing_factor) {
          roll_history.pop_back();
      }
      if (pitch_history.size() > smoothing_factor) { 
          pitch_history.pop_back();
      }
      if (yaw_history.size() > smoothing_factor) {
          yaw_history.pop_back();
      }
      if (gripper_history.size() > smoothing_factor) {
          gripper_history.pop_back();
      } 
      // Calculate smoothed values
      smoothed_roll = 0.0;
      smoothed_pitch = 0.0;
      smoothed_yaw = 0.0;
      smoothed_gripper = 0.0;
      for (const double& val : roll_history) {
          smoothed_roll += val;
      }
      for (const double& val : pitch_history) {
          smoothed_pitch += val;
      }
      for (const double& val : yaw_history) {
          smoothed_yaw += val;
      }
      for (const double& val : gripper_history) {
          smoothed_gripper += val;
      }
      smoothed_roll /= roll_history.size();
      smoothed_pitch /= pitch_history.size();
      smoothed_yaw /= yaw_history.size();
      smoothed_gripper /= gripper_history.size();

      int tip_rotation = static_cast<int>(std::round(smoothed_roll * MOTOR_PULSES_PER_ROTATION / (2 * UPPER_MOTOR_FACTOR * M_PI)));
      int gripper_offset = static_cast<int>(std::round(0.6f * MOTOR_PULSES_PER_ROTATION / UPPER_MOTOR_FACTOR));
      int gripper_factor = 8;
      int gripper_position = static_cast<int>(std::round(gripper_factor * smoothed_gripper * MOTOR_PULSES_PER_ROTATION / (2 * UPPER_MOTOR_FACTOR * M_PI)));

      // Drive the lower motors with the pitch and yaw. 
      // Motor 1 angles the shaft.
      // Motor 1 and 2 together change the direction of the bend.

      double h = std::tan(smoothed_pitch);
      double w = std::tan(smoothed_yaw);
      double omega = std::atan2(w, h) + M_PI; // direction of the bend
      double l1 = std::sqrt(h*h + w*w);
      double theta = std::atan(l1); // angle of the bend

      // Calculate the shortest rotation for omega
      // We need to choose the new omega to be the closest to the current position to prevent unnecessary full rotations
      // The motors are driven with absolute positions, so we need to keep track of the absolute omega
      // TODO: implement bending the other way and add this to the shortest rotation calculation and solve singularity
      double delta_omega = omega - absolute_omega;
      double diff = std::fmod(delta_omega, TWO_PI);
      if (diff < - M_PI) {
          diff += TWO_PI;
      } else if (diff >= M_PI) {
          diff -= TWO_PI;
      }
      absolute_omega += diff; // Update absolute_omega with the shortest rotation

      RCLCPP_INFO(this->get_logger(), "theta: '%f' omega: '%f'", theta, absolute_omega);    

      int m1_bend = get_m1_bend_pulses(theta);
      int shaft_rot = absolute_omega * MOTOR_PULSES_PER_ROTATION / (2 * M_PI);      

      RCLCPP_INFO(this->get_logger(), "roll, gripper_offset, gripper_position: '%d' '%d' '%d'", tip_rotation, gripper_offset, gripper_position);
      std::array<int, 4>  new_positions = {
        static_cast<int>(std::round(starting_positions[0] + tip_rotation - gripper_offset - gripper_position)),
        static_cast<int>(std::round(starting_positions[1] + shaft_rot + m1_bend)), // TODO: drive this motor
        static_cast<int>(std::round(starting_positions[2] + shaft_rot)), // TODO: drive this motor
        static_cast<int>(std::round(starting_positions[3] + roll))
      };

      // Check differences to prevent too large movements
      for (size_t i = 0; i < 4; ++i){
        int diff = new_positions[i] - positions[i];
        if (std::abs(diff) > 30){
          if (diff > 0){
            new_positions[i] = positions[i] + 30;
          }else{
            new_positions[i] = positions[i] - 30;
          }
        }
      }

      send_motor_positions(new_positions);

      // Publish the response values for now, later publish the actual status of the tool
      auto status_msg = std_msgs::msg::String();
      status_msg.data = "Currents: ";
      for (size_t i = 0; i < 6; ++i){
        status_msg.data += std::to_string(response_values[i]) + (i < 5 ? ", " : ";");
      } 
      RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", status_msg.data.c_str());
      publisher_->publish(status_msg);
    }

    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr subscription_;
    std::shared_ptr<SerialPort> serial_;
};

int main(int argc, char * argv[])
{
  int ret = 0;
  rclcpp::init(argc, argv);

  try {
    std::string device_path = SerialPort::find_device_by_manufacturer_product("Raspberry Pi", "Pico");
    
    if (device_path.empty()) {
      throw std::runtime_error("Could not find Raspberry Pi Pico device. Please check if the device is connected.");
    }
    
    std::cout << "Using device: " << device_path << std::endl;
    
    auto serial = std::make_shared<SerialPort>(device_path, 115200);

    if (!serial->open_port()) {
      throw std::runtime_error("Failed to open serial port: " + device_path);
    }
    
    rclcpp::spin(std::make_shared<ToolController>(serial));
    serial->close_port();
    
  } catch (const std::exception & e) {
    RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Exception: %s", e.what());
    ret = 1;
  }  
  
  rclcpp::shutdown();
  return ret;
}
#include "adlap_tool_control/instrument_controller.hpp"

#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace
{

struct ExportOptions
{
  std::string output_path = "joint_angle_conversion_export.csv";
  std::size_t samples = 500; // configured in-file
};

bool parse_options(int argc, char* argv[], ExportOptions& options)
{
  for (int i = 1; i < argc; ++i)
  {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h")
    {
      return false;
    }
    else if (arg == "--output" && i + 1 < argc)
    {
      options.output_path = argv[++i];
    }
    else if (arg.rfind("--output=", 0) == 0)
    {
      options.output_path = arg.substr(std::string("--output=").size());
    }
    else
    {
      std::cerr << "Unknown argument: " << arg << '\n';
      return false;
    }
  }

  if (options.samples == 0)
  {
    std::cerr << "samples must be greater than 0 (check ExportOptions in the source)\n";
    return false;
  }

  return true;
}

void write_header(std::ofstream& out)
{
  out << "sweep_axis,input_shaft_roll,input_bend,input_tip_rotation,input_articulation,"
         "motor_0,motor_1,motor_2,motor_3,"
         "output_shaft_roll,output_bend,output_tip_rotation,output_articulation\n";
}

void export_sinusoid(std::ofstream& out,
                     InstrumentController& controller,
                     MotorController& motor,
                     const std::string& axis_name,
                     std::size_t axis_index,
                     std::size_t samples)
{
  // Sinusoid over two full cycles
  const double min_val = 0.2;
  const double max_val = 0.45;
  const double center = 0.5 * (min_val + max_val);
  const double amplitude = 0.5 * (max_val - min_val);

  const double denom = samples > 1 ? static_cast<double>(samples - 1) : 1.0;

  for (std::size_t i = 0; i < samples; ++i)
  {
    const double phase = 2.0 * M_PI * (2.0 * static_cast<double>(i) / denom);
    double value = center + amplitude * std::sin(phase);

    std::array<double, 4> input = {0.0, 0.0, 0.0, 0.0};
    input[axis_index] = value;

    // Drive by joint angles
    controller.set_joint_angles(input[0], input[1], input[2], input[3], false);

    const std::array<int, 4> motor_positions = motor.get_positions();
    const std::array<double, 4> output = controller.joint_angles_from_motors(motor_positions);

    out << axis_name << ','
        << input[0] << ',' << input[1] << ',' << input[2] << ',' << input[3] << ','
        << motor_positions[0] << ',' << motor_positions[1] << ','
        << motor_positions[2] << ',' << motor_positions[3] << ','
        << output[0] << ',' << output[1] << ',' << output[2] << ',' << output[3] << '\n';
  }
}

}  // namespace

int main(int argc, char* argv[])
{
  ExportOptions options;
  if (!parse_options(argc, argv, options))
  {
    std::cerr << "Usage: " << argv[0] << " [--output FILE]\n";
    return 1;
  }

  rclcpp::init(argc, argv);

  const rclcpp::Logger logger = rclcpp::get_logger("joint_angle_conversion_export");
  MotorController motor(std::shared_ptr<SerialPort>{}, logger, Motor::create_default());
  InstrumentController controller(motor, logger);

  std::ofstream out(options.output_path);
  if (!out)
  {
    std::cerr << "Failed to open output file: " << options.output_path << '\n';
    rclcpp::shutdown();
    return 1;
  }

  out << std::fixed << std::setprecision(10);
  write_header(out);

  const std::vector<std::pair<std::string, std::size_t>> axes = {
      {"shaft_roll", 0},
      {"bend", 1},
      {"tip_rotation", 2},
      {"articulation", 3},
  };

  for (const auto& a : axes)
  {
    export_sinusoid(out, controller, motor, a.first, a.second, options.samples);
  }

  rclcpp::shutdown();
  std::cout << "Wrote joint angle conversion export to " << options.output_path << '\n';
  return 0;
}
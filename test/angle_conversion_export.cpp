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

struct SweepAxis
{
  const char* name;
  double min_rad;
  double max_rad;
  std::size_t index;
};

struct ExportOptions
{
  std::string output_path = "angle_conversion_export.csv";
  std::size_t samples = 200;
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

  return true;
}

void write_header(std::ofstream& out)
{
  out << "sweep_axis,input_roll,input_pitch,input_yaw,input_gripper,"
         "motor_0,motor_1,motor_2,motor_3,"
         "output_roll,output_pitch,output_yaw,output_gripper\n";
}

void export_sweep(std::ofstream& out,
                  InstrumentController& controller,
                  MotorController& motor,
                  const SweepAxis& axis,
                  std::size_t samples)
{
  const double span = axis.max_rad - axis.min_rad;
  const double denominator = samples > 1 ? static_cast<double>(samples - 1) : 1.0;

  for (std::size_t i = 0; i < samples; ++i)
  {
    std::array<double, 4> input = {0.0, 0.0, 0.0, 0.0};
    const double fraction = samples > 1 ? static_cast<double>(i) / denominator : 0.0;
    input[axis.index] = axis.min_rad + fraction * span;

    controller.set_euler_angles(input[0], input[1], input[2], input[3], false);

    const std::array<int, 4> motor_positions = motor.get_positions();
    const std::array<double, 4> output = controller.euler_angles_from_motors(motor_positions);

    out << axis.name << ','
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
    std::cerr << "Usage: " << argv[0] << " [--output FILE] [--samples N]\n";
    return 1;
  }

  rclcpp::init(argc, argv);

  const rclcpp::Logger logger = rclcpp::get_logger("angle_conversion_export");
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

  const std::vector<SweepAxis> axes = {
      {"roll", -M_PI, M_PI, 0},
      {"pitch", -M_PI / 4.0, M_PI / 4.0, 1},
      {"yaw", -M_PI / 4.0, M_PI / 4.0, 2},
      {"gripper", 0.0, M_PI / 6.0, 3},
  };

  for (const auto& axis : axes)
  {
    export_sweep(out, controller, motor, axis, options.samples);
  }

  rclcpp::shutdown();
  std::cout << "Wrote angle conversion export to " << options.output_path << '\n';
  return 0;
}
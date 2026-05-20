#include <gtest/gtest.h>

#define private public
#include "adlap_tool_control/instrument_controller.hpp"
#undef private

#include <array>
#include <cmath>
#include <memory>
#include <vector>

namespace {

constexpr double kDegToRad = M_PI / 180.0;

struct TestAngles {
	double roll;
	double pitch;
	double yaw;
	double gripper;
};

class AngleConversionTest : public ::testing::Test {
protected:
	static void SetUpTestSuite()
	{
		if (!rclcpp::ok()) {
			int argc = 0;
			rclcpp::init(argc, nullptr);
		}
	}

	static void TearDownTestSuite()
	{
		if (rclcpp::ok()) {
			rclcpp::shutdown();
		}
	}

	AngleConversionTest()
			: logger_(rclcpp::get_logger("angle_conversion_test")),
				motor_(
						std::shared_ptr<SerialPort>{},
						logger_,
						Motor::create_default()),
				controller_(motor_, logger_)
	{
	}

	void run_round_trip_case(const TestAngles& in)
	{
		controller_.smoothed_tip_rotation_ = in.roll;
		controller_.smoothed_pitch_ = in.pitch;
		controller_.smoothed_yaw_ = in.yaw;
		controller_.smoothed_articulation_ = in.gripper;

		const std::array<int, 4> motor_positions = controller_.calculate_motor_positions_from_euler_angles(
			controller_.smoothed_tip_rotation_,
			controller_.smoothed_pitch_,
			controller_.smoothed_yaw_,
			controller_.smoothed_articulation_,
			false
		);
		const std::array<double, 4> out = controller_.euler_angles_from_motors(motor_positions);

		constexpr double tol = 0.01;
		EXPECT_NEAR(out[0], in.roll, tol);
		EXPECT_NEAR(out[1], in.pitch, tol);
		EXPECT_NEAR(out[2], in.yaw, tol);
		EXPECT_NEAR(out[3], in.gripper, tol);
	}

	rclcpp::Logger logger_;
	MotorController motor_;
	InstrumentController controller_;
};

TEST_F(AngleConversionTest, FourAngleSetsRoundTripThroughMotorConversion)
{
	const std::vector<TestAngles> cases = {
			{0.0, 0.0, 0.0, 0.0},
			{10.0 * kDegToRad, -8.0 * kDegToRad, 12.0 * kDegToRad, 2.0 * kDegToRad},
			{-15.0 * kDegToRad, 20.0 * kDegToRad, -18.0 * kDegToRad, 4.0 * kDegToRad},
			{25.0 * kDegToRad, -30.0 * kDegToRad, 30.0 * kDegToRad, 5.0 * kDegToRad},
	};

	for (const auto& tc : cases) {
		SCOPED_TRACE(::testing::Message() << "roll=" << tc.roll
																			<< ", pitch=" << tc.pitch
																			<< ", yaw=" << tc.yaw
																			<< ", gripper=" << tc.gripper);
		run_round_trip_case(tc);
	}
}

TEST_F(AngleConversionTest, OppositeM1M2MotionWithinPlayKeepsAnglesConstant)
{
	auto backlash_motor = MotorController(
				std::shared_ptr<SerialPort>{},
				logger_,
				Motor::create_default());
	InstrumentController backlash_controller(backlash_motor, logger_);

	const std::array<double, 4> baseline = backlash_controller.euler_angles_from_motors(backlash_motor.get_positions());
	const int max_play = backlash_motor.get_pulses_lower_motors_play();
	ASSERT_GT(max_play, 0);

	constexpr double tol = 1e-9;
	for (int step = 0; step <= max_play; ++step) {
		for (int direction : {-1, 1}) {
			const int m1 = direction * step;
			const int m2 = -direction * step;

			backlash_motor.send_motor_positions({0, m1, m2, 0});
			const std::array<double, 4> out = backlash_controller.euler_angles_from_motors(backlash_motor.get_positions());

			SCOPED_TRACE(::testing::Message() << "step=" << step << ", direction=" << direction);
			EXPECT_NEAR(out[0], baseline[0], tol);
			EXPECT_NEAR(out[1], baseline[1], tol);
			EXPECT_NEAR(out[2], baseline[2], tol);
			EXPECT_NEAR(out[3], baseline[3], tol);
		}
	}
}

TEST_F(AngleConversionTest, SinusoidalPathPerDofRoundTrip)
{
    constexpr int num_points = 20;
    constexpr double tol = 0.01;  // Tolerance for accumulated numerical error in forward/inverse conversion

    struct DofConfig {
        const char* name;
        double min_rad;
        double max_rad;
        int dof_index;
    };

    const std::vector<DofConfig> dofs = {
        {"roll", -M_PI, M_PI, 0},
        {"pitch", -M_PI / 4, M_PI / 4, 1},
        {"yaw", -M_PI / 4, M_PI / 4, 2},
        {"gripper", 0, M_PI / 6, 3},
    };

    for (const auto& dof : dofs) {
        SCOPED_TRACE(::testing::Message() << "DOF: " << dof.name);

        const double amplitude = (dof.max_rad - dof.min_rad) / 2.0;
        const double center = (dof.max_rad + dof.min_rad) / 2.0;

        for (int i = 0; i < num_points; ++i) {
            const double phase = 2.0 * M_PI * i / num_points;
            const double value = center + amplitude * std::sin(phase);

            ASSERT_GE(value, dof.min_rad - 1e-9) << "Value out of range at point " << i;
            ASSERT_LE(value, dof.max_rad + 1e-9) << "Value out of range at point " << i;

            TestAngles in{0.0, 0.0, 0.0, 0.0};
            in.roll = (dof.dof_index == 0) ? value : 0.0;
            in.pitch = (dof.dof_index == 1) ? value : 0.0;
            in.yaw = (dof.dof_index == 2) ? value : 0.0;
            in.gripper = (dof.dof_index == 3) ? value : 0.0;

            controller_.smoothed_tip_rotation_ = in.roll;
            controller_.smoothed_pitch_ = in.pitch;
            controller_.smoothed_yaw_ = in.yaw;
            controller_.smoothed_articulation_ = in.gripper;

            const std::array<int, 4> motor_positions = controller_.calculate_motor_positions_from_euler_angles(
                controller_.smoothed_tip_rotation_,
                controller_.smoothed_pitch_,
                controller_.smoothed_yaw_,
                controller_.smoothed_articulation_,
                false
            );
			motor_.send_motor_positions(motor_positions);  // Update internal state to reflect the new motor positions

            const std::array<double, 4> out = controller_.euler_angles_from_motors(motor_positions);

            SCOPED_TRACE(::testing::Message() << "point=" << i << ", phase=" << phase << ", input=" << value);
            EXPECT_NEAR(out[0], in.roll, tol) << "Roll mismatch at point " << i;
            EXPECT_NEAR(out[1], in.pitch, tol) << "Pitch mismatch at point " << i;
            EXPECT_NEAR(out[2], in.yaw, tol) << "Yaw mismatch at point " << i;
            EXPECT_NEAR(out[3], in.gripper, tol) << "Gripper mismatch at point " << i;
        }
    }
}

}  // namespace

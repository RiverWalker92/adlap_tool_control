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
						Motor::create(1, 360.0f, 1, 40, 30, 60, 1.0f, 0, 0, false)),
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
				Motor::create(1, 360.0f, 1, 40, 30, 60, 1.0f, 12, 0, false));
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

}  // namespace

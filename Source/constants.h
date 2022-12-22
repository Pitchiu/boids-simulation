#ifndef CONSTANTS_H
#define CONSTANTS_H

namespace
{
	constexpr size_t kUnitCount = 10000;

	constexpr unsigned CountBits(unsigned int number)
	{
		return (int)log2(number) + 1;
	}
	constexpr float kVisualRange = 100.0f;
	constexpr float kProtectedRange = 20.0f;

	constexpr float kTurnFactor = 0.017f;
	constexpr float kCenteringFactor = 0.000013f;
	constexpr float kAvoidFactor = 0.0015f;
	constexpr float kAlignFactor = 0.01f;
	constexpr float kMaxSpeed = 1.2f;
	constexpr float kMinSpeed = 0.9f;


	constexpr float kWindowWidth = 1600.0f;  // 1920
	constexpr float kWindowHeight = 900.0f;  // 1080

	constexpr int   kGridColsNum = (kWindowWidth / kVisualRange);
	constexpr int   kGridRowsNum = (kWindowHeight / kVisualRange);
	constexpr int   kCellsNumTotal = kGridColsNum * kGridRowsNum;

	constexpr float kMarginSize = 200.0f;
	constexpr float kLeftMarginSize = kMarginSize;
	constexpr float kRightMarginSize = kWindowWidth - kMarginSize;
	constexpr float kTopMarginSize = kWindowHeight - kMarginSize;
	constexpr float kBottomMarginSize = kMarginSize;
}
#endif
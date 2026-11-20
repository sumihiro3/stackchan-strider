#include "car_control.h"

namespace car {
namespace {

int clampStick(int v) { return v < -100 ? -100 : (v > 100 ? 100 : v); }

float clampFloat(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

}  // namespace

int driveFromStick(int y) {
    y = clampStick(y);
    const int magnitude = y < 0 ? -y : y;
    if (magnitude <= DEAD_ZONE) {
        return 0;
    }
    // 不感帯の外側を、SPEED_MIN〜SPEED_MAX へ直線で割り当てる
    const int speed = SPEED_MIN + (magnitude - DEAD_ZONE - 1) * (SPEED_MAX - SPEED_MIN) / (100 - DEAD_ZONE - 1);
    return y < 0 ? -speed : speed;
}

float pitchFromStick(int x, float neutral) {
    x = clampStick(x);
    neutral = clampFloat(neutral, 0.0f, PITCH_MAX_DEG);
    const int magnitude = x < 0 ? -x : x;
    if (magnitude <= DEAD_ZONE) {
        return neutral;
    }
    const float t = static_cast<float>(magnitude - DEAD_ZONE) / (100 - DEAD_ZONE);
    const float pitch = x > 0 ? neutral + t * (PITCH_MAX_DEG - neutral) : neutral - t * neutral;
    return clampFloat(pitch, 0.0f, PITCH_MAX_DEG);
}

float clampNeutral(float neutral) { return clampFloat(neutral, NEUTRAL_MIN_DEG, NEUTRAL_MAX_DEG); }

}  // namespace car

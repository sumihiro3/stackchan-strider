#include "car_control.h"

namespace car {
namespace {

int clampStick(int v) { return v < -100 ? -100 : (v > 100 ? 100 : v); }

float clampFloat(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

}  // namespace

int driveFromStick(int y, int speedMax) {
    y = clampStick(y);
    speedMax = clampSpeedMax(speedMax);
    const int magnitude = y < 0 ? -y : y;
    if (magnitude <= DEAD_ZONE) {
        return 0;
    }
    // 不感帯の外側を、SPEED_MIN〜speedMax へ直線で割り当てる
    const int speed = SPEED_MIN + (magnitude - DEAD_ZONE - 1) * (speedMax - SPEED_MIN) / (100 - DEAD_ZONE - 1);
    return y < 0 ? -speed : speed;
}

float pitchFromStick(int x, float neutral, bool reversed) {
    x = clampStick(x);
    if (reversed) {
        x = -x;
    }
    neutral = clampFloat(neutral, 0.0f, PITCH_MAX_DEG);
    const int magnitude = x < 0 ? -x : x;
    if (magnitude <= DEAD_ZONE) {
        return neutral;
    }
    const float t = static_cast<float>(magnitude - DEAD_ZONE) / (100 - DEAD_ZONE);
    const float pitch = x > 0 ? neutral + t * (PITCH_MAX_DEG - neutral) : neutral - t * neutral;
    return clampFloat(pitch, 0.0f, PITCH_MAX_DEG);
}

PivotCommand pivotFromStick(int x, int y, float straight, bool reversed, float spinPitch, int speedMax) {
    x = clampStick(reversed ? -x : x);
    straight = clampFloat(straight, 0.0f, PITCH_MAX_DEG);
    speedMax = clampSpeedMax(speedMax);
    const int magnitude = x < 0 ? -x : x;
    if (magnitude <= SPIN_THRESHOLD) {
        return {driveFromStick(y, speedMax), straight, false};
    }
    // しきい値の外側を、SPEED_MIN〜speedMax へ直線で割り当てる
    const int speed = SPEED_MIN + (magnitude - SPIN_THRESHOLD - 1) * (speedMax - SPEED_MIN) / (100 - SPIN_THRESHOLD - 1);
    // 直進の角度のほうが上に調整してあるときは、頭を下げない
    const float pitch = clampSpinPitch(spinPitch);
    return {x < 0 ? -speed : speed, pitch > straight ? pitch : straight, true};
}

float clampNeutral(float neutral) { return clampFloat(neutral, NEUTRAL_MIN_DEG, NEUTRAL_MAX_DEG); }

float clampSpinPitch(float degrees) { return clampFloat(degrees, SPIN_PITCH_MIN_DEG, PITCH_MAX_DEG); }

int clampSpeedMax(int percent) { return percent < SPEED_CAP_MIN ? SPEED_CAP_MIN : (percent > SPEED_CAP_MAX ? SPEED_CAP_MAX : percent); }

bool timedOut(uint32_t now, uint32_t since, uint32_t limitMs) {
    return static_cast<int32_t>(now - since) > static_cast<int32_t>(limitMs);
}

}  // namespace car

#include "yaw_tracker.h"

#include <cmath>

namespace yaw {
namespace {

constexpr float BIAS_SMOOTH = 0.005f;  // 止まっている間、1 回ごとにこの割合でバイアスを寄せる(100 Hz で約 2 秒)
constexpr float BIAS_MAX_STEP = 5.0f;  // これより速い回転は、止まっていても(手で回したなど)バイアスに入れない [°/s]

}  // namespace

float wrap180(float degrees) {
    float d = std::fmod(degrees + 180.0f, 360.0f);
    if (d < 0) {
        d += 360.0f;
    }
    return d - 180.0f;
}

float verticalRate(float gx, float gy, float gz, float ax, float ay, float az) {
    const float norm = std::sqrt(ax * ax + ay * ay + az * az);
    if (norm < 0.5f) {
        return gz;
    }
    return (gx * ax + gy * ay + gz * az) / norm;
}

void Integrator::add(float rateDegS, float dtS, bool still) {
    if (still && std::fabs(rateDegS - bias_) < BIAS_MAX_STEP) {
        bias_ += BIAS_SMOOTH * (rateDegS - bias_);
    }
    angle_ += (rateDegS - bias_) * dtS;
}

}  // namespace yaw

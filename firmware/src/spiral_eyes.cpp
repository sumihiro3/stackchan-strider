#include "spiral_eyes.h"

#include <cmath>

namespace spiral_eyes {
namespace {

constexpr float TURNS = 2.5f;  // 渦巻きの巻き数
constexpr int STEPS = 40;

SpiralEye rightSpiral(false);
SpiralEye leftSpiral(true);
m5avatar::Drawable* originalRight = nullptr;
m5avatar::Drawable* originalLeft = nullptr;

}  // namespace

void SpiralEye::draw(M5Canvas* canvas, m5avatar::BoundingRect rect, m5avatar::DrawContext* ctx) {
    const uint16_t color = ctx->getColorDepth() == 1 ? 1 : ctx->getColorPalette()->get(COLOR_PRIMARY);
    const float cx = rect.getCenterX();
    const float cy = rect.getCenterY();
    const float radius = strength_ >= 2 ? 26.0f : 20.0f;
    const int thick = strength_ >= 2 ? 3 : 2;
    const float speed = (strength_ >= 2 ? 3.0f : 1.5f) * (isLeft_ ? -1.0f : 1.0f);  // [回/秒]
    const float phase = static_cast<float>(millis() % 100000) / 1000.0f * speed * 2.0f * static_cast<float>(M_PI);
    const float thetaMax = TURNS * 2.0f * static_cast<float>(M_PI);
    float px = cx, py = cy;
    for (int i = 1; i <= STEPS; i++) {
        const float theta = thetaMax * i / STEPS;
        const float r = radius * theta / thetaMax;
        const float x = cx + r * std::cos(theta + phase);
        const float y = cy + r * std::sin(theta + phase);
        // 顔を描くタスクのスタックは 2 KB しかないので、浮動小数で太線を描く drawWideLine は使わず、1 px の線を並べて太くする
        for (int k = 0; k < thick; k++) {
            canvas->drawLine(static_cast<int>(px) + k, static_cast<int>(py), static_cast<int>(x) + k, static_cast<int>(y), color);
            canvas->drawLine(static_cast<int>(px), static_cast<int>(py) + k, static_cast<int>(x), static_cast<int>(y) + k, color);
        }
        px = x;
        py = y;
    }
}

void apply(m5avatar::Avatar& avatar, int strength) {
    m5avatar::Face* face = avatar.getFace();
    if (face == nullptr) {
        return;
    }
    if (originalRight == nullptr) {
        originalRight = face->getRightEye();
        originalLeft = face->getLeftEye();
    }
    if (strength <= 0) {
        face->setRightEye(originalRight);
        face->setLeftEye(originalLeft);
        return;
    }
    rightSpiral.setStrength(strength);
    leftSpiral.setStrength(strength);
    face->setRightEye(&rightSpiral);
    face->setLeftEye(&leftSpiral);
}

}  // namespace spiral_eyes

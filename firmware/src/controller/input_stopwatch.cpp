// M5Stack StopWatch: 画面のタッチで動かし、ボタンで回る(README「StopWatch」)。
//  - タッチ: 画面の中心からのずれがそのまま x・y。指を離すと止まる
//  - 黄のボタン: 押して離すと、くるっと 1 回転
//  - 青のボタン: 押して離すと、回り続ける(もう一度で止める)。3 秒押すと登録を消す
#include <M5Unified.h>

#include <cstdio>

#include "input.h"

namespace input {
namespace {

constexpr float TOUCH_SPAN = 0.38f;  // 画面の幅に対する、中心から端(±100)までの距離
constexpr uint32_t FORGET_HOLD_MS = 3000;

bool forgetHandled = false;
bool forgetRequested = false;
uint8_t oncePresses = 0;
uint8_t loopPresses = 0;

int clamp100(float v) { return v > 100 ? 100 : (v < -100 ? -100 : static_cast<int>(v)); }

}  // namespace

void begin() {}

void update() {
    if (M5.BtnB.pressedFor(FORGET_HOLD_MS) && !forgetHandled) {
        forgetHandled = true;
        forgetRequested = true;
    } else if (M5.BtnB.wasReleased()) {
        if (!forgetHandled) {
            loopPresses++;
        }
        forgetHandled = false;
    }
    if (M5.BtnA.wasReleased()) {
        oncePresses++;
    }
}

Reading read() {
    Reading r;
    r.oncePresses = oncePresses;
    r.loopPresses = loopPresses;
    const auto t = M5.Touch.getDetail();
    if (t.isPressed()) {
        const float span = M5.Display.width() * TOUCH_SPAN;
        r.x = clamp100((t.x - M5.Display.width() / 2) * 100 / span);
        r.y = clamp100((M5.Display.height() / 2 - t.y) * 100 / span);
    }
    return r;
}

bool takeForgetRequest() {
    const bool requested = forgetRequested;
    forgetRequested = false;
    return requested;
}

const char* modeLabel() { return ""; }

void debugText(char* left, char* right, int size) {
    // 画面の下に、ボタンの役目を出す(左 = 黄、右 = 青)
    snprintf(left, size, "Y:1 TURN");
    snprintf(right, size, "B:LOOP");
}

}  // namespace input

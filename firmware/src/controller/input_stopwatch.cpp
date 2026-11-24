// M5Stack StopWatch: 画面のタッチか、本体の傾きで操作する(README「Atom コントローラー」)。
//  - タッチ: 画面の中心からのずれがそのまま x・y。指を離すと止まる
//  - 傾き: 黄のボタンを押している間だけ、押した瞬間の姿勢からの傾きで進む。離すと止まる
//  - 青のボタン: 押して離すとタッチと傾きを切り替える(本体に保存)。3 秒押すと登録を消す
#include <M5Unified.h>
#include <Preferences.h>

#include <cmath>
#include <cstdio>

#include "input.h"

namespace input {
namespace {

constexpr float TOUCH_SPAN = 0.38f;     // 画面の幅に対する、中心から端(±100)までの距離
constexpr float TILT_FULL_DEG = 25.0f;  // この角度まで傾けると ±100
// 向きの合わせ。最初の試験で、右に傾けて x が正、前(奥)へ傾けて y が正になるように決める
constexpr bool TILT_INVERT_X = false;
constexpr bool TILT_INVERT_Y = false;
constexpr uint32_t FORGET_HOLD_MS = 3000;

enum class Mode : uint8_t { Touch, Tilt };

Preferences prefs;
Mode mode = Mode::Touch;
bool forgetHandled = false;
bool forgetRequested = false;
float rollRef = 0, pitchRef = 0;  // 黄のボタンを押した瞬間の姿勢
float roll = 0, pitch = 0;        // いまの傾き(押した瞬間からの差)[°]
bool tiltHeld = false;

int clamp100(float v) { return v > 100 ? 100 : (v < -100 ? -100 : static_cast<int>(v)); }

void readAngles(float& r, float& p) {
    float ax = 0, ay = 0, az = 0;
    M5.Imu.getAccel(&ax, &ay, &az);
    r = std::atan2(ax, az) * 180.0f / static_cast<float>(M_PI);
    p = std::atan2(ay, az) * 180.0f / static_cast<float>(M_PI);
}

}  // namespace

void begin() {
    prefs.begin("ctlmode", false);
    mode = prefs.getUChar("mode", 0) == 1 ? Mode::Tilt : Mode::Touch;
}

void update() {
    M5.Imu.update();
    if (M5.BtnB.pressedFor(FORGET_HOLD_MS) && !forgetHandled) {
        forgetHandled = true;
        forgetRequested = true;
    } else if (M5.BtnB.wasReleased()) {
        if (!forgetHandled) {
            mode = mode == Mode::Touch ? Mode::Tilt : Mode::Touch;
            prefs.putUChar("mode", mode == Mode::Tilt ? 1 : 0);
        }
        forgetHandled = false;
    }
    if (M5.BtnA.wasPressed()) {
        readAngles(rollRef, pitchRef);
    }
    tiltHeld = M5.BtnA.isPressed();
}

Reading read() {
    Reading r;
    if (mode == Mode::Touch) {
        const auto t = M5.Touch.getDetail();
        if (t.isPressed()) {
            const float span = M5.Display.width() * TOUCH_SPAN;
            r.x = clamp100((t.x - M5.Display.width() / 2) * 100 / span);
            r.y = clamp100((M5.Display.height() / 2 - t.y) * 100 / span);
        }
        return r;
    }
    float nowRoll = 0, nowPitch = 0;
    readAngles(nowRoll, nowPitch);
    roll = nowRoll - rollRef;
    pitch = nowPitch - pitchRef;
    if (tiltHeld) {
        r.x = clamp100((TILT_INVERT_X ? -roll : roll) * 100 / TILT_FULL_DEG);
        r.y = clamp100((TILT_INVERT_Y ? pitch : -pitch) * 100 / TILT_FULL_DEG);
    }
    return r;
}

bool takeForgetRequest() {
    const bool requested = forgetRequested;
    forgetRequested = false;
    return requested;
}

const char* modeLabel() { return mode == Mode::Touch ? "TOUCH" : (tiltHeld ? "TILT" : "TILT: hold yellow"); }

void debugText(char* left, char* right, int size) {
    if (mode == Mode::Tilt) {
        snprintf(left, size, "R%.0f", roll);  // 向きを合わせるための傾き [°]
        snprintf(right, size, "P%.0f", pitch);
    } else {
        left[0] = right[0] = '\0';
    }
}

}  // namespace input

// AtomS3 / M5StickC + Grove Thumb Joystick。ジョイスティックは 3.3V で動かす(README「Atom コントローラー」)。
// 読むピンは platformio.ini の CTL_PIN_X / CTL_PIN_Y で機器ごとに決める。
#include <M5Unified.h>

#include <cstdio>
#include <utility>

#include "../stick_link.h"
#include "input.h"

namespace input {
namespace {

constexpr int PIN_X = CTL_PIN_X;  // ジョイスティックの黄(ADC1。ADC2 は無線を使っている間は読めない)
constexpr int PIN_Y = CTL_PIN_Y;  // ジョイスティックの白
// 向きの合わせ。最初の試験で、右に倒して x が正、前に倒して y が正になるように決める
constexpr bool SWAP_XY = false;
constexpr bool INVERT_X = false;
constexpr bool INVERT_Y = false;

constexpr int VCC_MV = 3300;
constexpr int SPAN_MV = 850;  // 中心から端まで。3.3V × 0.29(仕様の振れ幅)の少し内側にして、端まで確実に届くようにする
constexpr uint32_t FORGET_HOLD_MS = 3000;

int centreX = VCC_MV / 2, centreY = VCC_MV / 2;
int rawX = 0, rawY = 0;
bool forgetHandled = false;
bool forgetRequested = false;

// 生の値(12 bit)を 0〜VCC_MV に比例で直す。analogReadMilliVolts は、校正値を持たない古い ESP32(初期の M5StickC)では失敗する。
// 中心は起動時に測り、押し込みは電源に対する割合で見るので、比例で直すだけで足りる
int readMillivolts(int pin) {
    int total = 0;
    for (int i = 0; i < 8; i++) {
        total += analogRead(pin);
    }
    return total / 8 * VCC_MV / 4095;
}

void calibrate() {
    M5.Display.fillScreen(TFT_BLACK);
    M5.Display.setTextColor(TFT_WHITE);
    M5.Display.drawCenterString("Hands off", M5.Display.width() / 2, M5.Display.height() / 2 - 8);
    delay(300);
    long sx = 0, sy = 0;
    for (int i = 0; i < 20; i++) {
        sx += readMillivolts(PIN_X);
        sy += readMillivolts(PIN_Y);
        delay(10);
    }
    centreX = static_cast<int>(sx / 20);
    centreY = static_cast<int>(sy / 20);
}

}  // namespace

void begin() {
    analogRead(PIN_X);  // 先に 1 度読んでアナログ入力にしておかないと、減衰の設定が効かない
    analogRead(PIN_Y);
    analogSetPinAttenuation(PIN_X, ADC_11db);
    analogSetPinAttenuation(PIN_Y, ADC_11db);
    calibrate();
}

void update() {
    if (M5.BtnA.pressedFor(FORGET_HOLD_MS) && !forgetHandled) {
        forgetHandled = true;
        forgetRequested = true;
    } else if (M5.BtnA.wasReleased()) {
        if (!forgetHandled) {
            calibrate();
        }
        forgetHandled = false;
    }
}

Reading read() {
    rawX = readMillivolts(PIN_X);
    rawY = readMillivolts(PIN_Y);
    Reading r;
    r.stop = stick_link::isPressed(rawX, VCC_MV);
    r.x = stick_link::axisFromMillivolts(rawX, centreX, SPAN_MV);
    r.y = stick_link::axisFromMillivolts(rawY, centreY, SPAN_MV);
    if (SWAP_XY) {
        std::swap(r.x, r.y);
    }
    r.x = INVERT_X ? -r.x : r.x;
    r.y = INVERT_Y ? -r.y : r.y;
    if (r.stop) {
        r.x = r.y = 0;  // 押し込みは X が端まで上がるので、右いっぱいと取り違えないよう止める
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
    snprintf(left, size, "%d", rawX);  // 向きを合わせるための生の値 [mV 相当]
    snprintf(right, size, "%d", rawY);
}

}  // namespace input

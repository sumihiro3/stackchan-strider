// スティックの値(-100〜100)を、首(ヨー)の回転の速さと、うなずき(かじ、または旋回の足)の角度に直す。機器には依存しない。
#pragma once

#include <cstdint>

namespace car {

constexpr int DEAD_ZONE = 10;  // スティックのこの範囲(±)は中心とみなす
constexpr int SPEED_MIN = 10;  // 走り出しの速さ [%]
constexpr int SPEED_MAX = 60;  // 速さの上限 [%]。シミュレーションでは全開だと跳ねて進みが悪い
// うなずきの上限 [°]。0° で左いっぱい(前輪 +30°)、30° で右いっぱい(-30°)。
// これを超えると引き棒やフェンダーが当たる(README_CAR.md「ファームウェア」)
constexpr float PITCH_MAX_DEG = 30.0f;
constexpr float NEUTRAL_MIN_DEG = 0.0f;  // 直進の角度として受け付ける範囲(旋回車は頭が水平の 0° で直進する)
constexpr float NEUTRAL_MAX_DEG = 25.0f;
constexpr float NEUTRAL_CAR500_DEG = 12.6f;
constexpr float NEUTRAL_RETRO_DEG = 13.0f;
// トライク(後輪でかじを取る)。レバーの腕が、うなずき 15° で後輪がまっすぐになる向きに付いている(car/trike_design.py)
constexpr float NEUTRAL_TRIKE_DEG = 15.0f;
// 旋回車(car/pivot_design.py)。頭が水平のとき足が浮いて直進し、頭を上げきる(PITCH_MAX_DEG)と右の駆動輪が浮いて旋回する
constexpr float NEUTRAL_PIVOT_DEG = 0.0f;
// 旋回車で、スティックの左右がこれを超えたら旋回する。これ以下では頭を動かさない。
// 頭を途中まで上げると足が半端に下り(約 13° で地面に届く)、直進中に地面をこするため
constexpr int SPIN_THRESHOLD = 40;

/** 旋回車への命令。 */
struct PivotCommand {
    int drive;    // 首の回転の速さ [%]。正 = 前進
    float pitch;  // うなずきの角度 [°]
    bool spin;    // 旋回中(頭を上げきって足を下ろしている)
};

/**
 * 前後のスティックの値を、首の回転の速さに直す。
 * @param y -100(後ろ)〜100(前)
 * @return 0(止まる)、または ±SPEED_MIN〜±SPEED_MAX [%]
 */
int driveFromStick(int y);

/**
 * 左右のスティックの値を、うなずきの角度に直す。中心で直進の角度、右いっぱいで PITCH_MAX_DEG、左いっぱいで 0。
 * @param x -100(左)〜100(右)
 * @param neutral 直進の角度 [°]
 * @param reversed true なら左右を入れ替える(右いっぱいで 0)。後輪でかじを取るトライク用
 * @return 0〜PITCH_MAX_DEG [°]
 */
float pitchFromStick(int x, float neutral, bool reversed = false);

/**
 * 旋回車のスティックの値を、首の回転の速さとうなずきの角度に直す。
 * 左右が SPIN_THRESHOLD を超えたら旋回: 頭を上げきり、右なら前進、左なら後退(足を軸に、右なら右へ、左なら左へ回る)。前後は使わない。
 * それ以外は直進: 頭は straight のまま、前後で走る。
 * @param x -100(左)〜100(右)
 * @param y -100(後ろ)〜100(前)
 * @param straight 直進のうなずきの角度 [°]
 * @param reversed true なら旋回の向きを入れ替える(実物で左右が逆だったとき用)
 */
PivotCommand pivotFromStick(int x, int y, float straight, bool reversed = false);

/** 直進の角度を、受け付ける範囲に収める。 */
float clampNeutral(float neutral);

/**
 * 最後に値が届いた時刻 since から limitMs より長くたったか。
 * since が now より新しい(別のタスクが now を取ったあとに記録した)ときは、たっていないとみなす。millis() の一周にも耐える。
 */
bool timedOut(uint32_t now, uint32_t since, uint32_t limitMs);

}  // namespace car

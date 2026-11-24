// スティックの値(-100〜100)を、首(ヨー)の回転の速さと、うなずき(かじ、または旋回の足)の角度に直す。機器には依存しない。
#pragma once

#include <cstdint>

namespace car {

constexpr int DEAD_ZONE = 10;  // スティックのこの範囲(±)は中心とみなす
constexpr int SPEED_MIN = 10;  // 走り出しの速さ [%]
constexpr int SPEED_MAX = 60;  // 速さの上限の初期値 [%]。歩行機のシミュレーションでは全開だと跳ねて進みが悪かった
// 速さの上限として受け付ける範囲 [%]。車輪の車は跳ねないので、設定で上げられる(片輪で回るピボットは、旋回が遅い)
constexpr int SPEED_CAP_MIN = 30;
constexpr int SPEED_CAP_MAX = 100;
// うなずきの上限 [°]。0° で左いっぱい(前輪 +30°)、30° で右いっぱい(-30°)。
// これを超えると引き棒やフェンダーが当たる(README_CAR.md「ファームウェア」)
constexpr float PITCH_MAX_DEG = 30.0f;
constexpr float NEUTRAL_MIN_DEG = 0.0f;  // 直進の角度として受け付ける範囲(ピボットは頭が水平の 0° で直進する)
constexpr float NEUTRAL_MAX_DEG = 25.0f;
constexpr float NEUTRAL_CAR500_DEG = 12.6f;
constexpr float NEUTRAL_RETRO_DEG = 13.0f;
// トライク(後輪でかじを取る)。レバーの腕が、うなずき 15° で後輪がまっすぐになる向きに付いている(car/trike_design.py)
constexpr float NEUTRAL_TRIKE_DEG = 15.0f;
// ピボット(car/pivot_design.py)。頭が水平のとき足が浮いて直進し、頭を上げきる(PITCH_MAX_DEG)と右の駆動輪が浮いて旋回する
constexpr float NEUTRAL_PIVOT_DEG = 0.0f;
// ピボットで、スティックの左右がこれを超えたら旋回する。これ以下では頭を動かさない。
// 頭を途中まで上げると足が半端に下り(約 12° で地面に届く)、直進中に地面をこするため
constexpr int SPIN_THRESHOLD = 40;
// 旋回のときに頭を上げる角度として受け付ける下限 [°]。足が地面に届くのは約 12° なので、それより下では旋回しない。
// 角度が小さいほど足は浅く下り、浮かせる駆動輪の浮きも小さくなる(実物では、浅いほうがよく回った)
constexpr float SPIN_PITCH_MIN_DEG = 10.0f;

/** ピボットへの命令。 */
struct PivotCommand {
    int drive;    // 首の回転の速さ [%]。正 = 前進
    float pitch;  // うなずきの角度 [°]
    bool spin;    // 旋回中(頭を上げて足を下ろしている)
};

/**
 * 前後のスティックの値を、首の回転の速さに直す。
 * @param y -100(後ろ)〜100(前)
 * @param speedMax 速さの上限 [%](SPEED_CAP_MIN〜SPEED_CAP_MAX に収める)
 * @return 0(止まる)、または ±SPEED_MIN〜±speedMax [%]
 */
int driveFromStick(int y, int speedMax = SPEED_MAX);

/**
 * 左右のスティックの値を、うなずきの角度に直す。中心で直進の角度、右いっぱいで PITCH_MAX_DEG、左いっぱいで 0。
 * @param x -100(左)〜100(右)
 * @param neutral 直進の角度 [°]
 * @param reversed true なら左右を入れ替える(右いっぱいで 0)。後輪でかじを取るトライク用
 * @return 0〜PITCH_MAX_DEG [°]
 */
float pitchFromStick(int x, float neutral, bool reversed = false);

/**
 * ピボットのスティックの値を、首の回転の速さとうなずきの角度に直す。
 * 左右が SPIN_THRESHOLD を超えたら旋回: 頭を spinPitch まで上げ、右なら前進、左なら後退(足を軸に、右なら右へ、左なら左へ回る)。前後は使わない。
 * それ以外は直進: 頭は straight のまま、前後で走る。
 * @param x -100(左)〜100(右)
 * @param y -100(後ろ)〜100(前)
 * @param straight 直進のうなずきの角度 [°]
 * @param reversed true なら旋回の向きを入れ替える(実物で左右が逆だったとき用)
 * @param spinPitch 旋回のときに頭を上げる角度 [°](SPIN_PITCH_MIN_DEG〜PITCH_MAX_DEG に収める。straight より小さければ straight を使う)
 * @param speedMax 速さの上限 [%](直進にも旋回にも使う)
 */
PivotCommand pivotFromStick(int x, int y, float straight, bool reversed = false, float spinPitch = PITCH_MAX_DEG, int speedMax = SPEED_MAX);

/** 直進の角度を、受け付ける範囲に収める。 */
float clampNeutral(float neutral);

/** 旋回のときに頭を上げる角度を、受け付ける範囲に収める。 */
float clampSpinPitch(float degrees);

/** 速さの上限を、受け付ける範囲に収める。 */
int clampSpeedMax(int percent);

/**
 * 最後に値が届いた時刻 since から limitMs より長くたったか。
 * since が now より新しい(別のタスクが now を取ったあとに記録した)ときは、たっていないとみなす。millis() の一周にも耐える。
 */
bool timedOut(uint32_t now, uint32_t since, uint32_t limitMs);

}  // namespace car

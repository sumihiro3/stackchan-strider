// スティックの値(-100〜100)を、首(ヨー)の回転の速さと、うなずき(かじ)の角度に直す。機器には依存しない。
#pragma once

#include <cstdint>

namespace car {

constexpr int DEAD_ZONE = 10;  // スティックのこの範囲(±)は中心とみなす
constexpr int SPEED_MIN = 10;  // 走り出しの速さ [%]
constexpr int SPEED_MAX = 60;  // 速さの上限 [%]。シミュレーションでは全開だと跳ねて進みが悪い
// うなずきの上限 [°]。0° で左いっぱい(前輪 +30°)、30° で右いっぱい(-30°)。
// これを超えると引き棒やフェンダーが当たる(README_CAR.md「ファームウェア」)
constexpr float PITCH_MAX_DEG = 30.0f;
constexpr float NEUTRAL_MIN_DEG = 5.0f;  // 直進の角度として受け付ける範囲
constexpr float NEUTRAL_MAX_DEG = 25.0f;
constexpr float NEUTRAL_CAR500_DEG = 12.6f;
constexpr float NEUTRAL_RETRO_DEG = 13.0f;
// トライク(後輪でかじを取る)。レバーの腕が、うなずき 15° で後輪がまっすぐになる向きに付いている(car/trike_design.py)
constexpr float NEUTRAL_TRIKE_DEG = 15.0f;

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

/** 直進の角度を、受け付ける範囲に収める。 */
float clampNeutral(float neutral);

/**
 * 最後に値が届いた時刻 since から limitMs より長くたったか。
 * since が now より新しい(別のタスクが now を取ったあとに記録した)ときは、たっていないとみなす。millis() の一周にも耐える。
 */
bool timedOut(uint32_t now, uint32_t since, uint32_t limitMs);

}  // namespace car

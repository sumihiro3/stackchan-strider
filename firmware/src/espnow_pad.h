// Atom コントローラー(src/controller)から ESP-NOW で届くスティックの値を受け取る。
// 登録(組にした相手の MAC)は本体に保存し、登録した相手からの値だけを使う。
#pragma once

#include <cstdint>

namespace espnow_pad {

enum class State : uint8_t {
    Unpaired,   // 登録がない(組にする相手を待っている)
    Waiting,    // 登録はあるが、値が届いていない
    Connected,  // 値が届いている
};

/** Atom が登録してあるか。Wi-Fi を始める前でも呼べる(BLE のパッドを探すかどうかを決めるため)。 */
bool registered();

/** ESP-NOW を始める。Wi-Fi を始めたあとに 1 回呼ぶ。 */
void begin();

/** 組にする申し込みへの返事など、コールバックの中でできない処理をする。loop() で毎回呼ぶ。 */
void update(uint32_t now);

/** いまの状態。 */
State state(uint32_t now);

/**
 * 最新のスティックの値を取り出す。
 * @param x -100(左)〜100(右)
 * @param y -100(後ろ)〜100(前)
 * @param stop スティックを押し込んでいる(止める)
 * @return 前回呼んでから新しい値が届いていれば true
 */
bool takeStick(int& x, int& y, bool& stop);

/** 最後に値が届いた時刻 [ms](まだなら 0)。 */
uint32_t lastStickMs();

/** 登録を消し、しばらく(60 秒)新しい相手の申し込みを受け付ける。 */
void forget();

}  // namespace espnow_pad

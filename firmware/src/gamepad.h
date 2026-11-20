// BLE のゲームパッド(HID over GATT)を探してつなぎ、左スティックの値を受け取る。
// 機種ごとの作り込みはせず、Report Map(HID の記述子)から左スティックの位置を読む(hid_parser)。
// つなぐ・つなぎ直すのは専用のタスクで行うので、loop() は止まらない。
#pragma once

#include <cstdint>

namespace gamepad {

enum class State : uint8_t {
    Searching,   // 探している(ペアリング待ちのパッドか、登録済みのパッドの電源が入るのを待つ)
    Connecting,  // つないでいる途中
    Connected,   // つながっていて、スティックの値が届く
};

/** BLE を初期化し、探す・つなぐタスクを始める。setup() で 1 回呼ぶ。 */
void begin();

/** いまの状態。 */
State state();

/**
 * 最新の左スティックの値を取り出す。
 * @param x -100(左)〜100(右)
 * @param y -100(後ろ)〜100(前)。スティックを前に倒すと正
 * @return 前回呼んでから新しい値が届いていれば true
 */
bool takeStick(int& x, int& y);

/** つながっているパッドの名前(つながっていなければ空)。 */
const char* name();

/** 登録(ボンド)をすべて消し、つながっていれば切って、探し直す。 */
void forget();

}  // namespace gamepad

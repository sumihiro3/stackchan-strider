// Atom コントローラーとスタックチャンの間で ESP-NOW に載せる、短いパケットと、スティックの読み取り値の変換。
// 機器には依存しない(両方のファームと、native の単体テストで使う)。
#pragma once

#include <cstddef>
#include <cstdint>

namespace stick_link {

enum class Kind : uint8_t {
    Pair = 1,     // コントローラー → 一斉送信。組になる相手を探す
    PairAck = 2,  // スタックチャン → コントローラー。組になった
    Stick = 3,    // コントローラー → スタックチャン。スティックの値
};

constexpr uint8_t BUTTON_STOP = 0x01;  // スティックを押し込んでいる
constexpr size_t PACKET_SIZE = 9;      // 'S' 'L' 版 種類 番号 x y ボタン 和

struct Packet {
    Kind kind;
    uint8_t seq;      // 送るたびに 1 ずつ増える
    int8_t x;         // -100(左)〜100(右)
    int8_t y;         // -100(後ろ)〜100(前)
    uint8_t buttons;  // BUTTON_*
};

/**
 * パケットをバイト列にする。
 * @return 書いたバイト数(PACKET_SIZE)。cap が足りなければ 0
 */
size_t encode(const Packet& packet, uint8_t* out, size_t cap);

/**
 * バイト列をパケットに戻す。識別子・版・長さ・和・種類・x/y の範囲のどれかが合わなければ false。
 */
bool decode(const uint8_t* data, size_t len, Packet& out);

/**
 * スティックの軸の電圧を -100〜100 に直す。中心で 0、中心 ± spanMv で ±100。範囲外は端の値にそろえる。
 * @param spanMv 端までの電圧差。0 以下なら 0 を返す
 */
int axisFromMillivolts(int mv, int centreMv, int spanMv);

/**
 * Grove Thumb Joystick は、押し込むと X が電源の電圧まで上がる。普段の X は 0.78 × 電源までなので、0.85 × 電源を超えたら押し込みとみなす(ADC は 3.1V 付近で頭打ちになるので、0.9 だと余裕が少ない)。
 */
bool isPressed(int xMv, int vccMv);

}  // namespace stick_link

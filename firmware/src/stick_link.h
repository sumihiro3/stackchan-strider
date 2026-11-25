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
// ボタンを押した回数の下 2 ビット(押したときだけ増える)。受け手は回数が変わったときに 1 回動く。
// 押している間の状態ではなく回数を送るので、パケットが落ちても押しを取りこぼさず、同じ値が何回届いても 1 回しか動かない
constexpr uint8_t PRESS_ONCE_SHIFT = 1;  // くるっと 1 回転(bit 1-2)
constexpr uint8_t PRESS_LOOP_SHIFT = 3;  // 回り続ける・止める(bit 3-4)
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

/**
 * 組にする申し込みを受け付ける電波の強さの下限 [dBm]。3 m ほどまで離れたコントローラーだけを組にして(実測: 数 cm で -29〜-43、3 m で -57〜-61)、
 * 展示会などで離れた所にある同じファームの機器を登録しないようにする。下限より弱い申し込みも、シリアルに rssi が出る。
 */
constexpr int PAIR_RSSI_MIN = -65;

enum class PairDecision : uint8_t {
    Ignore,  // 返事をしない
    Ack,     // すぐ返事をする(登録した相手)
    Ask,     // 画面で人に確かめてから登録する
};

/**
 * スタックチャンが組にする申し込み(Pair、または知らない相手からの Stick)を受けたときにどうするか。
 * @param known 登録した相手から来た
 * @param windowOpen 新しい相手を受け付けている(登録がない、または登録を消して 60 秒以内)
 * @param rssi 受けた電波の強さ [dBm]
 */
PairDecision decidePairRequest(bool known, bool windowOpen, int rssi, int minRssi);

/**
 * コントローラーが PairAck を受け入れるか。組になったあとは受け入れない(相手を書き換えると、送る先が登録にない MAC になって途切れる)。
 */
bool acceptPairAck(bool paired, int rssi, int minRssi);

/** ボタンの回数を buttons に入れる。ほかのビット(BUTTON_STOP)はそのまま残す。 */
uint8_t packPresses(uint8_t buttons, uint8_t onceCount, uint8_t loopCount);

/** buttons から、くるっと 1 回転のボタンの回数(0〜3)を取り出す。 */
uint8_t onceCount(uint8_t buttons);

/** buttons から、回り続けるボタンの回数(0〜3)を取り出す。 */
uint8_t loopCount(uint8_t buttons);

/** 前に受けた回数 last から、いま受けた回数 now までに押された回数(0〜3)。 */
uint8_t pressesSince(uint8_t last, uint8_t now);

/** 人が見比べるための、MAC の下 2 バイト(画面に 4 桁の 16 進で出す)。 */
uint16_t shortId(const uint8_t mac[6]);

}  // namespace stick_link

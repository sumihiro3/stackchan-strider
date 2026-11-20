// HID の記述子(Report Map)から、ゲームパッドの左スティック(X / Y)の位置を読み取る。機器には依存しない。
#pragma once

#include <cstddef>
#include <cstdint>

namespace hid {

/// 入力レポートの中の 1 つの軸。bitOffset は、レポート ID を除いた中身の先頭からのビット位置。
struct AxisField {
    bool found = false;
    uint8_t reportId = 0;  // 0 = 記述子にレポート ID がない
    uint16_t bitOffset = 0;
    uint8_t bitSize = 0;
    int32_t logicalMin = 0;
    int32_t logicalMax = 0;
};

/// 左スティックの X(Usage 0x30)と Y(0x31)。
struct GamepadLayout {
    AxisField x;
    AxisField y;
};

/**
 * Report Map を読み、入力レポートの中で最初に出てくる X と Y の位置と範囲を取り出す。
 * @param descriptor Report Map(HID 記述子)のバイト列
 * @param length バイト数
 * @param out 取り出した位置
 * @return X と Y の両方が見つかれば true
 */
bool parseReportMap(const uint8_t* descriptor, size_t length, GamepadLayout& out);

/**
 * 入力レポートの中身(レポート ID を含まない)から軸の値を取り出し、-100〜100 に直す(範囲の中央が 0)。
 * @param field parseReportMap で得た軸
 * @param report 通知で届いたレポートの中身
 * @param length バイト数
 * @return -100〜100。レポートが短すぎるときは 0
 */
int readAxis(const AxisField& field, const uint8_t* report, size_t length);

}  // namespace hid

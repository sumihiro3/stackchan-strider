// hid_parser の単体テスト(pio test -e native)。
// 記述子は HID の仕様どおりに組んだ代表例。実機の HD-6500 の記述子も入れてある(HD6500_AS_XBOX)
#include <unity.h>

#include "hid_parser.h"

using hid::AxisField;
using hid::GamepadLayout;

void setUp() {}
void tearDown() {}

// 8 ビットの X/Y/Z/Rz(0〜255)とボタン 16 個。レポート ID 1。安価な汎用パッドに多い形
static const uint8_t GENERIC_8BIT[] = {
    0x05, 0x01,        // Usage Page (Generic Desktop)
    0x09, 0x05,        // Usage (Game Pad)
    0xA1, 0x01,        // Collection (Application)
    0x85, 0x01,        //   Report ID (1)
    0x05, 0x09,        //   Usage Page (Button)
    0x19, 0x01,        //   Usage Minimum (1)
    0x29, 0x10,        //   Usage Maximum (16)
    0x15, 0x00,        //   Logical Minimum (0)
    0x25, 0x01,        //   Logical Maximum (1)
    0x75, 0x01,        //   Report Size (1)
    0x95, 0x10,        //   Report Count (16)
    0x81, 0x02,        //   Input (Data, Var, Abs)
    0x05, 0x01,        //   Usage Page (Generic Desktop)
    0x09, 0x30,        //   Usage (X)
    0x09, 0x31,        //   Usage (Y)
    0x09, 0x32,        //   Usage (Z)
    0x09, 0x35,        //   Usage (Rz)
    0x15, 0x00,        //   Logical Minimum (0)
    0x26, 0xFF, 0x00,  //   Logical Maximum (255)
    0x75, 0x08,        //   Report Size (8)
    0x95, 0x04,        //   Report Count (4)
    0x81, 0x02,        //   Input (Data, Var, Abs)
    0xC0,              // End Collection
};

// Xbox ワイヤレス コントローラー(BLE)に倣った形: 物理コレクションの中に 16 ビットの X/Y と Rx/Ry(0〜65535)。
// 先にほかのレポート ID の出力レポートがあり、X/Y は入力レポート 1 の先頭
static const uint8_t XBOX_LIKE_16BIT[] = {
    0x05, 0x01,                    // Usage Page (Generic Desktop)
    0x09, 0x05,                    // Usage (Game Pad)
    0xA1, 0x01,                    // Collection (Application)
    0x85, 0x03,                    //   Report ID (3)
    0x06, 0x00, 0xFF,              //   Usage Page (Vendor)
    0x09, 0x01,                    //   Usage (1)
    0x15, 0x00, 0x26, 0xFF, 0x00,  //   Logical 0..255
    0x75, 0x08, 0x95, 0x04,        //   8 bit × 4
    0x91, 0x02,                    //   Output (出力はビット位置に数えない)
    0x85, 0x01,                    //   Report ID (1)
    0x05, 0x01,                    //   Usage Page (Generic Desktop)
    0x09, 0x01,                    //   Usage (Pointer)
    0xA1, 0x00,                    //   Collection (Physical)
    0x09, 0x30,                    //     Usage (X)
    0x09, 0x31,                    //     Usage (Y)
    0x15, 0x00,                    //     Logical Minimum (0)
    0x27, 0xFF, 0xFF, 0x00, 0x00,  //     Logical Maximum (65535)
    0x95, 0x02,                    //     Report Count (2)
    0x75, 0x10,                    //     Report Size (16)
    0x81, 0x02,                    //     Input
    0xC0,                          //   End Collection
    0x09, 0x01,                    //   Usage (Pointer)
    0xA1, 0x00,                    //   Collection (Physical)
    0x09, 0x33,                    //     Usage (Rx)
    0x09, 0x34,                    //     Usage (Ry)
    0x95, 0x02, 0x75, 0x10,        //     16 bit × 2
    0x81, 0x02,                    //     Input
    0xC0,                          //   End Collection
    0xC0,                          // End Collection
};

// 符号付き 8 ビット(-127〜127)。Logical Maximum を 2 バイトの 0xFFFF(= 65535 のつもり)と書く機器への備えも別に試す
static const uint8_t SIGNED_8BIT_NO_ID[] = {
    0x05, 0x01, 0x09, 0x04, 0xA1, 0x01,  // Joystick, Application
    0x09, 0x30, 0x09, 0x31,              // X, Y
    0x15, 0x81,                          // Logical Minimum (-127)
    0x25, 0x7F,                          // Logical Maximum (127)
    0x75, 0x08, 0x95, 0x02,              // 8 bit × 2
    0x81, 0x02,                          // Input
    0xC0,
};

static const uint8_t UNSIGNED_MAX_AS_2BYTES[] = {
    0x05, 0x01, 0x09, 0x05, 0xA1, 0x01,
    0x09, 0x30, 0x09, 0x31,
    0x15, 0x00,              // Logical Minimum (0)
    0x26, 0xFF, 0xFF,        // Logical Maximum: 仕様では -1 だが、機器は 65535 のつもり
    0x75, 0x10, 0x95, 0x02,  // 16 bit × 2
    0x81, 0x02,
    0xC0,
};

// 実機の HD-6500。Mac につないだときの記述子(ioreg の ReportDescriptor)。Xbox ワイヤレス コントローラー(VID 045E / PID 02E0)を名乗る
static const uint8_t HD6500_AS_XBOX[] = {
    0x05, 0x01, 0x09, 0x05, 0xA1, 0x01, 0x85, 0x01, 0x09, 0x01, 0xA1, 0x00, 0x09, 0x30, 0x09, 0x31,
    0x15, 0x00, 0x27, 0xFF, 0xFF, 0x00, 0x00, 0x95, 0x02, 0x75, 0x10, 0x81, 0x02, 0xC0, 0x09, 0x01,
    0xA1, 0x00, 0x09, 0x33, 0x09, 0x34, 0x15, 0x00, 0x27, 0xFF, 0xFF, 0x00, 0x00, 0x95, 0x02, 0x75,
    0x10, 0x81, 0x02, 0xC0, 0x05, 0x01, 0x09, 0x32, 0x15, 0x00, 0x26, 0xFF, 0x03, 0x95, 0x01, 0x75,
    0x0A, 0x81, 0x02, 0x15, 0x00, 0x25, 0x00, 0x75, 0x06, 0x95, 0x01, 0x81, 0x03, 0x05, 0x01, 0x09,
    0x35, 0x15, 0x00, 0x26, 0xFF, 0x03, 0x95, 0x01, 0x75, 0x0A, 0x81, 0x02, 0x15, 0x00, 0x25, 0x00,
    0x75, 0x06, 0x95, 0x01, 0x81, 0x03, 0x05, 0x01, 0x09, 0x39, 0x15, 0x01, 0x25, 0x08, 0x35, 0x00,
    0x46, 0x3B, 0x01, 0x66, 0x14, 0x00, 0x75, 0x04, 0x95, 0x01, 0x81, 0x42, 0x75, 0x04, 0x95, 0x01,
    0x15, 0x00, 0x25, 0x00, 0x35, 0x00, 0x45, 0x00, 0x65, 0x00, 0x81, 0x03, 0x05, 0x09, 0x19, 0x01,
    0x29, 0x0A, 0x15, 0x00, 0x25, 0x01, 0x75, 0x01, 0x95, 0x0A, 0x81, 0x02, 0x15, 0x00, 0x25, 0x00,
    0x75, 0x06, 0x95, 0x01, 0x81, 0x03, 0x05, 0x01, 0x09, 0x80, 0x85, 0x02, 0xA1, 0x00, 0x09, 0x85,
    0x15, 0x00, 0x25, 0x01, 0x95, 0x01, 0x75, 0x01, 0x81, 0x02, 0x15, 0x00, 0x25, 0x00, 0x75, 0x07,
    0x95, 0x01, 0x81, 0x03, 0xC0, 0x05, 0x0F, 0x09, 0x21, 0x85, 0x03, 0xA1, 0x02, 0x09, 0x97, 0x15,
    0x00, 0x25, 0x01, 0x75, 0x04, 0x95, 0x01, 0x91, 0x02, 0x15, 0x00, 0x25, 0x00, 0x75, 0x04, 0x95,
    0x01, 0x91, 0x03, 0x09, 0x70, 0x15, 0x00, 0x25, 0x64, 0x75, 0x08, 0x95, 0x04, 0x91, 0x02, 0x09,
    0x50, 0x66, 0x01, 0x10, 0x55, 0x0E, 0x15, 0x00, 0x26, 0xFF, 0x00, 0x75, 0x08, 0x95, 0x01, 0x91,
    0x02, 0x09, 0xA7, 0x15, 0x00, 0x26, 0xFF, 0x00, 0x75, 0x08, 0x95, 0x01, 0x91, 0x02, 0x65, 0x00,
    0x55, 0x00, 0x09, 0x7C, 0x15, 0x00, 0x26, 0xFF, 0x00, 0x75, 0x08, 0x95, 0x01, 0x91, 0x02, 0xC0,
    0x85, 0x04, 0x05, 0x06, 0x09, 0x20, 0x15, 0x00, 0x26, 0xFF, 0x00, 0x75, 0x08, 0x95, 0x01, 0x81,
    0x02, 0xC0,
};

static GamepadLayout parse(const uint8_t* d, size_t n) {
    GamepadLayout layout;
    TEST_ASSERT_TRUE(hid::parseReportMap(d, n, layout));
    return layout;
}

void test_generic_8bit_finds_axes_after_buttons() {
    const GamepadLayout layout = parse(GENERIC_8BIT, sizeof(GENERIC_8BIT));
    TEST_ASSERT_TRUE(layout.x.found);
    TEST_ASSERT_TRUE(layout.y.found);
    TEST_ASSERT_EQUAL_UINT8(1, layout.x.reportId);
    TEST_ASSERT_EQUAL_UINT16(16, layout.x.bitOffset);  // ボタン 16 ビットの後ろ
    TEST_ASSERT_EQUAL_UINT16(24, layout.y.bitOffset);
    TEST_ASSERT_EQUAL_UINT8(8, layout.x.bitSize);
    TEST_ASSERT_EQUAL_INT32(0, layout.x.logicalMin);
    TEST_ASSERT_EQUAL_INT32(255, layout.x.logicalMax);
}

void test_xbox_like_ignores_output_and_reads_16bit() {
    const GamepadLayout layout = parse(XBOX_LIKE_16BIT, sizeof(XBOX_LIKE_16BIT));
    TEST_ASSERT_EQUAL_UINT8(1, layout.x.reportId);
    TEST_ASSERT_EQUAL_UINT16(0, layout.x.bitOffset);
    TEST_ASSERT_EQUAL_UINT16(16, layout.y.bitOffset);
    TEST_ASSERT_EQUAL_UINT8(16, layout.y.bitSize);
    TEST_ASSERT_EQUAL_INT32(65535, layout.y.logicalMax);
}

void test_signed_axes_without_report_id() {
    const GamepadLayout layout = parse(SIGNED_8BIT_NO_ID, sizeof(SIGNED_8BIT_NO_ID));
    TEST_ASSERT_EQUAL_UINT8(0, layout.x.reportId);
    TEST_ASSERT_EQUAL_INT32(-127, layout.x.logicalMin);
    TEST_ASSERT_EQUAL_INT32(127, layout.x.logicalMax);
}

void test_two_byte_ffff_max_is_read_as_unsigned() {
    const GamepadLayout layout = parse(UNSIGNED_MAX_AS_2BYTES, sizeof(UNSIGNED_MAX_AS_2BYTES));
    TEST_ASSERT_EQUAL_INT32(0, layout.x.logicalMin);
    TEST_ASSERT_EQUAL_INT32(65535, layout.x.logicalMax);
}

void test_no_axes_returns_false() {
    static const uint8_t KEYBOARD_ONLY[] = {0x05, 0x01, 0x09, 0x06, 0xA1, 0x01, 0x05, 0x07, 0x19, 0xE0, 0x29, 0xE7,
                                            0x15, 0x00, 0x25, 0x01, 0x75, 0x01, 0x95, 0x08, 0x81, 0x02, 0xC0};
    GamepadLayout layout;
    TEST_ASSERT_FALSE(hid::parseReportMap(KEYBOARD_ONLY, sizeof(KEYBOARD_ONLY), layout));
}

// マウス(相対値の X/Y)は、近くにあってもゲームパッドとして使わない
void test_relative_mouse_axes_are_ignored() {
    static const uint8_t MOUSE[] = {0x05, 0x01, 0x09, 0x02, 0xA1, 0x01, 0x09, 0x01, 0xA1, 0x00,
                                    0x09, 0x30, 0x09, 0x31, 0x15, 0x81, 0x25, 0x7F, 0x75, 0x08,
                                    0x95, 0x02, 0x81, 0x06,  // Input (Data, Var, Rel)
                                    0xC0, 0xC0};
    GamepadLayout layout;
    TEST_ASSERT_FALSE(hid::parseReportMap(MOUSE, sizeof(MOUSE), layout));
}

void test_read_axis_normalizes_unsigned_8bit() {
    const GamepadLayout layout = parse(GENERIC_8BIT, sizeof(GENERIC_8BIT));
    // 通知の中身にレポート ID は含まない(BLE ではキャラクタリスティックごとに ID が決まる)
    const uint8_t centre[] = {0x00, 0x00, 0x80, 0x80, 0x80, 0x80};
    const uint8_t corner[] = {0x00, 0x00, 0xFF, 0x00, 0x80, 0x80};
    TEST_ASSERT_INT_WITHIN(1, 0, hid::readAxis(layout.x, centre, sizeof(centre)));
    TEST_ASSERT_EQUAL_INT(100, hid::readAxis(layout.x, corner, sizeof(corner)));
    TEST_ASSERT_EQUAL_INT(-100, hid::readAxis(layout.y, corner, sizeof(corner)));
}

void test_read_axis_normalizes_16bit_little_endian() {
    const GamepadLayout layout = parse(XBOX_LIKE_16BIT, sizeof(XBOX_LIKE_16BIT));
    const uint8_t report[] = {0xFF, 0xFF, 0x00, 0x80, 0, 0, 0, 0};  // X = 65535, Y = 32768
    TEST_ASSERT_EQUAL_INT(100, hid::readAxis(layout.x, report, sizeof(report)));
    TEST_ASSERT_INT_WITHIN(1, 0, hid::readAxis(layout.y, report, sizeof(report)));
}

void test_read_axis_signed() {
    const GamepadLayout layout = parse(SIGNED_8BIT_NO_ID, sizeof(SIGNED_8BIT_NO_ID));
    const uint8_t report[] = {0x81, 0x7F};  // X = -127, Y = 127
    TEST_ASSERT_EQUAL_INT(-100, hid::readAxis(layout.x, report, sizeof(report)));
    TEST_ASSERT_EQUAL_INT(100, hid::readAxis(layout.y, report, sizeof(report)));
}

void test_hd6500_left_stick() {
    const GamepadLayout layout = parse(HD6500_AS_XBOX, sizeof(HD6500_AS_XBOX));
    TEST_ASSERT_EQUAL_UINT8(1, layout.x.reportId);
    TEST_ASSERT_EQUAL_UINT16(0, layout.x.bitOffset);
    TEST_ASSERT_EQUAL_UINT16(16, layout.y.bitOffset);
    TEST_ASSERT_EQUAL_UINT8(16, layout.x.bitSize);
    TEST_ASSERT_EQUAL_INT32(0, layout.x.logicalMin);
    TEST_ASSERT_EQUAL_INT32(65535, layout.x.logicalMax);
}

void test_read_axis_short_report_returns_zero() {
    const GamepadLayout layout = parse(GENERIC_8BIT, sizeof(GENERIC_8BIT));
    const uint8_t tooShort[] = {0x00, 0x00};
    TEST_ASSERT_EQUAL_INT(0, hid::readAxis(layout.x, tooShort, sizeof(tooShort)));
}

int main() {
    UNITY_BEGIN();
    RUN_TEST(test_generic_8bit_finds_axes_after_buttons);
    RUN_TEST(test_xbox_like_ignores_output_and_reads_16bit);
    RUN_TEST(test_signed_axes_without_report_id);
    RUN_TEST(test_two_byte_ffff_max_is_read_as_unsigned);
    RUN_TEST(test_no_axes_returns_false);
    RUN_TEST(test_relative_mouse_axes_are_ignored);
    RUN_TEST(test_read_axis_normalizes_unsigned_8bit);
    RUN_TEST(test_read_axis_normalizes_16bit_little_endian);
    RUN_TEST(test_read_axis_signed);
    RUN_TEST(test_read_axis_short_report_returns_zero);
    RUN_TEST(test_hd6500_left_stick);
    return UNITY_END();
}

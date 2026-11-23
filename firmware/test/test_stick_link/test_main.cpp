// stick_link の単体テスト(pio test -e native)
#include <unity.h>

#include <cstring>

#include "stick_link.h"

using namespace stick_link;

void setUp() {}
void tearDown() {}

void test_round_trip() {
    const Packet in{Kind::Stick, 42, -73, 100, BUTTON_STOP};
    uint8_t buf[PACKET_SIZE];
    TEST_ASSERT_EQUAL_UINT(PACKET_SIZE, encode(in, buf, sizeof(buf)));
    Packet out{};
    TEST_ASSERT_TRUE(decode(buf, sizeof(buf), out));
    TEST_ASSERT_EQUAL_UINT8(static_cast<uint8_t>(Kind::Stick), static_cast<uint8_t>(out.kind));
    TEST_ASSERT_EQUAL_UINT8(42, out.seq);
    TEST_ASSERT_EQUAL_INT8(-73, out.x);
    TEST_ASSERT_EQUAL_INT8(100, out.y);
    TEST_ASSERT_EQUAL_UINT8(BUTTON_STOP, out.buttons);
}

void test_encode_needs_room() {
    uint8_t buf[PACKET_SIZE - 1];
    TEST_ASSERT_EQUAL_UINT(0, encode(Packet{Kind::Pair, 0, 0, 0, 0}, buf, sizeof(buf)));
}

void test_decode_rejects_broken_packets() {
    uint8_t buf[PACKET_SIZE];
    encode(Packet{Kind::Stick, 1, 10, -10, 0}, buf, sizeof(buf));
    Packet out{};
    TEST_ASSERT_FALSE(decode(buf, sizeof(buf) - 1, out));  // 短い
    uint8_t bad[PACKET_SIZE];
    for (size_t i = 0; i < PACKET_SIZE; i++) {  // どの 1 バイトが壊れても捨てる
        std::memcpy(bad, buf, sizeof(buf));
        bad[i] ^= 0x10;
        TEST_ASSERT_FALSE_MESSAGE(decode(bad, sizeof(bad), out), "corrupted byte accepted");
    }
}

void test_decode_rejects_unknown_kind_and_out_of_range() {
    uint8_t buf[PACKET_SIZE];
    Packet out{};
    encode(Packet{static_cast<Kind>(9), 0, 0, 0, 0}, buf, sizeof(buf));
    TEST_ASSERT_FALSE(decode(buf, sizeof(buf), out));
    encode(Packet{Kind::Stick, 0, -101, 0, 0}, buf, sizeof(buf));
    TEST_ASSERT_FALSE(decode(buf, sizeof(buf), out));
}

void test_axis_centre_ends_and_clamp() {
    TEST_ASSERT_EQUAL_INT(0, axisFromMillivolts(1650, 1650, 850));
    TEST_ASSERT_EQUAL_INT(100, axisFromMillivolts(1650 + 850, 1650, 850));
    TEST_ASSERT_EQUAL_INT(-100, axisFromMillivolts(1650 - 850, 1650, 850));
    TEST_ASSERT_EQUAL_INT(50, axisFromMillivolts(1650 + 425, 1650, 850));
    TEST_ASSERT_EQUAL_INT(100, axisFromMillivolts(3300, 1650, 850));  // 振れ幅を超えたら端の値
    TEST_ASSERT_EQUAL_INT(-100, axisFromMillivolts(0, 1650, 850));
    TEST_ASSERT_EQUAL_INT(0, axisFromMillivolts(2000, 1650, 0));  // 振れ幅 0 は 0 として扱う
}

void test_press_is_above_normal_range() {
    // 3.3V で動かすと、普段の X は 0.2〜0.78 × Vcc、押し込むと Vcc
    TEST_ASSERT_FALSE(isPressed(static_cast<int>(3300 * 0.78f), 3300));
    TEST_ASSERT_TRUE(isPressed(3100, 3300));  // ADC は 3.1V 付近で頭打ちになる
    TEST_ASSERT_FALSE(isPressed(1650, 3300));
}

int main() {
    UNITY_BEGIN();
    RUN_TEST(test_round_trip);
    RUN_TEST(test_encode_needs_room);
    RUN_TEST(test_decode_rejects_broken_packets);
    RUN_TEST(test_decode_rejects_unknown_kind_and_out_of_range);
    RUN_TEST(test_axis_centre_ends_and_clamp);
    RUN_TEST(test_press_is_above_normal_range);
    return UNITY_END();
}

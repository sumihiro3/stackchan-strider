// car_control の単体テスト(pio test -e native)
#include <unity.h>

#include "car_control.h"

void setUp() {}
void tearDown() {}

void test_drive_dead_zone_is_stop() {
    TEST_ASSERT_EQUAL_INT(0, car::driveFromStick(0));
    TEST_ASSERT_EQUAL_INT(0, car::driveFromStick(car::DEAD_ZONE));
    TEST_ASSERT_EQUAL_INT(0, car::driveFromStick(-car::DEAD_ZONE));
}

void test_drive_starts_at_min_and_ends_at_max() {
    TEST_ASSERT_EQUAL_INT(car::SPEED_MIN, car::driveFromStick(car::DEAD_ZONE + 1));
    TEST_ASSERT_EQUAL_INT(car::SPEED_MAX, car::driveFromStick(100));
    TEST_ASSERT_EQUAL_INT(-car::SPEED_MAX, car::driveFromStick(-100));
    TEST_ASSERT_EQUAL_INT(car::SPEED_MAX, car::driveFromStick(150));  // 範囲外は端にそろえる
}

void test_steer_centre_is_neutral() {
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 12.6f, car::pitchFromStick(0, 12.6f));
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 12.6f, car::pitchFromStick(car::DEAD_ZONE, 12.6f));
}

void test_steer_full_right_is_max_and_full_left_is_zero() {
    // うなずき 30° = 右いっぱい、0° = 左いっぱい
    TEST_ASSERT_FLOAT_WITHIN(0.01f, car::PITCH_MAX_DEG, car::pitchFromStick(100, 12.6f));
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 0.0f, car::pitchFromStick(-100, 12.6f));
}

void test_steer_never_leaves_limits() {
    for (int x = -200; x <= 200; x += 7) {
        const float pitch = car::pitchFromStick(x, 13.0f);
        TEST_ASSERT_TRUE(pitch >= 0.0f && pitch <= car::PITCH_MAX_DEG);
    }
    // 直進の角度の設定が範囲外でも、出す角度は範囲に収める
    TEST_ASSERT_TRUE(car::pitchFromStick(0, 45.0f) <= car::PITCH_MAX_DEG);
}

void test_neutral_clamp() {
    TEST_ASSERT_FLOAT_WITHIN(0.01f, car::NEUTRAL_MIN_DEG, car::clampNeutral(-5.0f));
    TEST_ASSERT_FLOAT_WITHIN(0.01f, car::NEUTRAL_MAX_DEG, car::clampNeutral(99.0f));
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 12.6f, car::clampNeutral(12.6f));
}

void test_steer_reversed_swaps_left_and_right() {
    // 後輪でかじを取るトライクは、うなずきと曲がる向きの関係が逆になる
    const float n = car::NEUTRAL_TRIKE_DEG;
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 0.0f, car::pitchFromStick(100, n, true));
    TEST_ASSERT_FLOAT_WITHIN(0.01f, car::PITCH_MAX_DEG, car::pitchFromStick(-100, n, true));
    TEST_ASSERT_FLOAT_WITHIN(0.01f, n, car::pitchFromStick(0, n, true));
    TEST_ASSERT_FLOAT_WITHIN(0.01f, car::pitchFromStick(-40, n), car::pitchFromStick(40, n, true));
}

void test_trike_neutral_is_accepted() {
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 15.0f, car::NEUTRAL_TRIKE_DEG);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, car::NEUTRAL_TRIKE_DEG, car::clampNeutral(car::NEUTRAL_TRIKE_DEG));
}

void test_timed_out_handles_newer_timestamp() {
    // 受信のコールバックは loop() の外で時刻を記録するので、loop() が先に取った now より新しいことがある
    TEST_ASSERT_FALSE(car::timedOut(1000, 1005, 500));  // now より 5 ms 新しい → 途切れていない
    TEST_ASSERT_FALSE(car::timedOut(1000, 600, 500));   // 400 ms 前 → まだ
    TEST_ASSERT_TRUE(car::timedOut(1000, 400, 500));    // 600 ms 前 → 途切れた
    TEST_ASSERT_FALSE(car::timedOut(5, 0xFFFFFFF0u, 500));  // millis() が一周した直後
}

int main() {
    UNITY_BEGIN();
    RUN_TEST(test_drive_dead_zone_is_stop);
    RUN_TEST(test_drive_starts_at_min_and_ends_at_max);
    RUN_TEST(test_steer_centre_is_neutral);
    RUN_TEST(test_steer_full_right_is_max_and_full_left_is_zero);
    RUN_TEST(test_steer_never_leaves_limits);
    RUN_TEST(test_neutral_clamp);
    RUN_TEST(test_steer_reversed_swaps_left_and_right);
    RUN_TEST(test_trike_neutral_is_accepted);
    RUN_TEST(test_timed_out_handles_newer_timestamp);
    return UNITY_END();
}

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

void test_timed_out_handles_newer_timestamp() {
    // 受信のコールバックは loop() の外で時刻を記録するので、loop() が先に取った now より新しいことがある
    TEST_ASSERT_FALSE(car::timedOut(1000, 1005, 500));  // now より 5 ms 新しい → 途切れていない
    TEST_ASSERT_FALSE(car::timedOut(1000, 600, 500));   // 400 ms 前 → まだ
    TEST_ASSERT_TRUE(car::timedOut(1000, 400, 500));    // 600 ms 前 → 途切れた
    TEST_ASSERT_FALSE(car::timedOut(5, 0xFFFFFFF0u, 500));  // millis() が一周した直後
}

void test_pivot_centre_stops_with_the_head_level() {
    const car::PivotCommand cmd = car::pivotFromStick(0, 0);
    TEST_ASSERT_EQUAL_INT(0, cmd.drive);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, car::LEVEL_DEG, cmd.pitch);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 0.0f, car::LEVEL_DEG);
    TEST_ASSERT_FALSE(cmd.spin);
}

void test_pivot_forward_and_back_keep_the_head_level() {
    // 前後は走るだけ。頭は水平のまま(足は浮いている)
    const car::PivotCommand forward = car::pivotFromStick(0, 100);
    TEST_ASSERT_EQUAL_INT(car::SPEED_MAX, forward.drive);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, car::LEVEL_DEG, forward.pitch);
    TEST_ASSERT_FALSE(forward.spin);
    TEST_ASSERT_EQUAL_INT(-car::SPEED_MAX, car::pivotFromStick(0, -100).drive);
}

void test_pivot_small_sideways_tilt_does_not_lower_the_foot() {
    // 左右が SPIN_THRESHOLD 以下なら旋回しない。足を半端に下ろすと、直進中に地面をこする
    const car::PivotCommand cmd = car::pivotFromStick(car::SPIN_THRESHOLD, 60);
    TEST_ASSERT_FALSE(cmd.spin);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, car::LEVEL_DEG, cmd.pitch);
    TEST_ASSERT_EQUAL_INT(car::driveFromStick(60), cmd.drive);
    TEST_ASSERT_FALSE(car::pivotFromStick(-car::SPIN_THRESHOLD, 0).spin);
}

void test_pivot_sideways_spins_with_the_head_fully_up() {
    // 右へ倒すと頭を上げきって前進(右の駆動輪が浮き、左の駆動輪が押して右へ回る)。左は後退
    const car::PivotCommand right = car::pivotFromStick(100, 0);
    TEST_ASSERT_TRUE(right.spin);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, car::PITCH_MAX_DEG, right.pitch);
    TEST_ASSERT_EQUAL_INT(car::SPEED_MAX, right.drive);
    const car::PivotCommand left = car::pivotFromStick(-100, 0);
    TEST_ASSERT_TRUE(left.spin);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, car::PITCH_MAX_DEG, left.pitch);
    TEST_ASSERT_EQUAL_INT(-car::SPEED_MAX, left.drive);
}

void test_pivot_spin_starts_at_min_speed_and_ignores_forward() {
    TEST_ASSERT_EQUAL_INT(car::SPEED_MIN, car::pivotFromStick(car::SPIN_THRESHOLD + 1, 0).drive);
    TEST_ASSERT_EQUAL_INT(-car::SPEED_MIN, car::pivotFromStick(-car::SPIN_THRESHOLD - 1, 0).drive);
    // 旋回中は前後を使わない(足を下ろしたまま前後に進むと、足を引きずる)
    TEST_ASSERT_EQUAL_INT(car::SPEED_MAX, car::pivotFromStick(100, -100).drive);
}

void test_pivot_reversed_swaps_the_spin_direction() {
    TEST_ASSERT_EQUAL_INT(-car::SPEED_MAX, car::pivotFromStick(100, 0, true).drive);
    TEST_ASSERT_EQUAL_INT(car::SPEED_MAX, car::pivotFromStick(-100, 0, true).drive);
    // 前後の向きは変えない
    TEST_ASSERT_EQUAL_INT(car::SPEED_MAX, car::pivotFromStick(0, 100, true).drive);
}

void test_pivot_never_leaves_limits() {
    for (int x = -200; x <= 200; x += 13) {
        for (int y = -200; y <= 200; y += 17) {
            const car::PivotCommand cmd = car::pivotFromStick(x, y, false, 45.0f, 250);
            TEST_ASSERT_TRUE(cmd.pitch >= 0.0f && cmd.pitch <= car::PITCH_MAX_DEG);
            TEST_ASSERT_TRUE(cmd.drive >= -car::SPEED_CAP_MAX && cmd.drive <= car::SPEED_CAP_MAX);
        }
    }
}

void test_drive_uses_the_given_speed_cap() {
    // 速さの上限は設定で変えられる。省くと SPEED_MAX
    TEST_ASSERT_EQUAL_INT(100, car::driveFromStick(100, 100));
    TEST_ASSERT_EQUAL_INT(-100, car::driveFromStick(-100, 100));
    TEST_ASSERT_EQUAL_INT(car::SPEED_MIN, car::driveFromStick(car::DEAD_ZONE + 1, 100));
    TEST_ASSERT_EQUAL_INT(40, car::driveFromStick(100, 40));
    TEST_ASSERT_EQUAL_INT(car::SPEED_CAP_MAX, car::driveFromStick(100, 250));  // 範囲外の上限は収める
}

void test_speed_cap_clamp() {
    TEST_ASSERT_EQUAL_INT(car::SPEED_CAP_MIN, car::clampSpeedMax(0));
    TEST_ASSERT_EQUAL_INT(car::SPEED_CAP_MAX, car::clampSpeedMax(150));
    TEST_ASSERT_EQUAL_INT(80, car::clampSpeedMax(80));
}

void test_pivot_spin_uses_the_given_pitch() {
    // 旋回のときに頭を上げる角度は設定で変えられる(足の深さが変わる)。直進は水平のまま
    const car::PivotCommand spin = car::pivotFromStick(100, 0, false, 20.0f);
    TEST_ASSERT_TRUE(spin.spin);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 20.0f, spin.pitch);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, car::LEVEL_DEG, car::pivotFromStick(0, 100, false, 20.0f).pitch);
}

void test_pivot_spin_pitch_stays_in_range() {
    TEST_ASSERT_FLOAT_WITHIN(0.01f, car::SPIN_PITCH_MIN_DEG, car::clampSpinPitch(3.0f));
    TEST_ASSERT_FLOAT_WITHIN(0.01f, car::PITCH_MAX_DEG, car::clampSpinPitch(45.0f));
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 22.5f, car::clampSpinPitch(22.5f));
    TEST_ASSERT_FLOAT_WITHIN(0.01f, car::PITCH_MAX_DEG, car::pivotFromStick(100, 0, false, 45.0f).pitch);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, car::SPIN_PITCH_MIN_DEG, car::pivotFromStick(100, 0, false, 3.0f).pitch);
}

void test_pivot_uses_the_speed_cap_for_both_spin_and_straight() {
    TEST_ASSERT_EQUAL_INT(100, car::pivotFromStick(100, 0, false, car::PITCH_MAX_DEG, 100).drive);
    TEST_ASSERT_EQUAL_INT(-100, car::pivotFromStick(-100, 0, false, car::PITCH_MAX_DEG, 100).drive);
    TEST_ASSERT_EQUAL_INT(100, car::pivotFromStick(0, 100, false, car::PITCH_MAX_DEG, 100).drive);
    TEST_ASSERT_EQUAL_INT(car::SPEED_MIN, car::pivotFromStick(car::SPIN_THRESHOLD + 1, 0, false, car::PITCH_MAX_DEG, 100).drive);
}

int main() {
    UNITY_BEGIN();
    RUN_TEST(test_drive_dead_zone_is_stop);
    RUN_TEST(test_drive_starts_at_min_and_ends_at_max);
    RUN_TEST(test_timed_out_handles_newer_timestamp);
    RUN_TEST(test_pivot_centre_stops_with_the_head_level);
    RUN_TEST(test_pivot_forward_and_back_keep_the_head_level);
    RUN_TEST(test_pivot_small_sideways_tilt_does_not_lower_the_foot);
    RUN_TEST(test_pivot_sideways_spins_with_the_head_fully_up);
    RUN_TEST(test_pivot_spin_starts_at_min_speed_and_ignores_forward);
    RUN_TEST(test_pivot_reversed_swaps_the_spin_direction);
    RUN_TEST(test_pivot_never_leaves_limits);
    RUN_TEST(test_drive_uses_the_given_speed_cap);
    RUN_TEST(test_speed_cap_clamp);
    RUN_TEST(test_pivot_spin_uses_the_given_pitch);
    RUN_TEST(test_pivot_spin_pitch_stays_in_range);
    RUN_TEST(test_pivot_uses_the_speed_cap_for_both_spin_and_straight);
    return UNITY_END();
}

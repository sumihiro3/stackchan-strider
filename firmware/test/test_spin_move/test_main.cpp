// spin_move と yaw_tracker の単体テスト(pio test -e native)
#include <unity.h>

#include "spin_move.h"
#include "yaw_tracker.h"

void setUp() {}
void tearDown() {}

namespace {
spin::Settings settings() {
    spin::Settings s;
    s.drive = 60;
    s.pitch = 25.0f;
    s.coastDeg = 10.0f;
    return s;
}
}  // namespace

void test_once_waits_then_turns_and_stops_before_full_turn() {
    spin::Mover m;
    m.startOnce(1000, settings());
    spin::Output o = m.update(1000, 0);
    TEST_ASSERT_TRUE(o.controls);
    TEST_ASSERT_EQUAL_INT(0, o.drive);  // ためている間は回らず、頭だけ上げる
    TEST_ASSERT_EQUAL_FLOAT(25.0f, o.pitch);
    TEST_ASSERT_EQUAL(spin::Face::Ready, o.face);
    o = m.update(1000 + spin::READY_MS, 50);  // 回り始めた向き(50°)から数える
    TEST_ASSERT_EQUAL_INT(60, o.drive);
    TEST_ASSERT_EQUAL(spin::Face::Spiral, o.face);
    o = m.update(3000, 50 + 349);
    TEST_ASSERT_EQUAL(spin::Phase::Turning, m.phase());
    o = m.update(3100, 50 + 350);  // 360 − 惰性 10° で止める
    TEST_ASSERT_FALSE(o.controls);
    TEST_ASSERT_EQUAL(spin::Phase::Idle, m.phase());
    TEST_ASSERT_FALSE(m.timedOut());
}

void test_once_counts_either_direction() {
    spin::Mover m;
    m.startOnce(0, settings());
    m.update(spin::READY_MS, 0);
    m.update(2000, -350);
    TEST_ASSERT_EQUAL(spin::Phase::Idle, m.phase());
}

void test_once_times_out() {
    spin::Mover m;
    m.startOnce(0, settings());
    m.update(spin::READY_MS, 0);
    m.update(spin::READY_MS + spin::ONCE_TIMEOUT_MS, 100);
    TEST_ASSERT_EQUAL(spin::Phase::Idle, m.phase());
    TEST_ASSERT_TRUE(m.timedOut());
}

void test_loop_toggle_stops_into_dizzy_then_idle() {
    spin::Mover m;
    m.toggleLoop(0, settings());
    m.update(spin::READY_MS, 0);
    m.update(5000, 2000);
    TEST_ASSERT_EQUAL(spin::Phase::Turning, m.phase());  // 何周しても止まらない
    m.toggleLoop(6000, settings());
    spin::Output o = m.update(6000, 2100);
    TEST_ASSERT_EQUAL(spin::Phase::Dizzy, m.phase());
    TEST_ASSERT_FALSE(o.controls);  // 目を回している間は、スティックで動かせる
    TEST_ASSERT_EQUAL(spin::Face::Dizzy, o.face);
    o = m.update(6000 + spin::DIZZY_MS, 2100);
    TEST_ASSERT_EQUAL(spin::Phase::Idle, m.phase());
    TEST_ASSERT_EQUAL(spin::Face::Normal, o.face);
}

void test_loop_times_out_for_safety() {
    spin::Mover m;
    m.toggleLoop(0, settings());
    m.update(spin::READY_MS, 0);
    m.update(spin::READY_MS + spin::LOOP_TIMEOUT_MS, 9000);
    TEST_ASSERT_EQUAL(spin::Phase::Dizzy, m.phase());
    TEST_ASSERT_TRUE(m.timedOut());
}

void test_stop_while_ready_goes_idle_without_dizzy() {
    spin::Mover m;
    m.toggleLoop(0, settings());
    m.update(100, 0);
    m.stop(200);
    TEST_ASSERT_EQUAL(spin::Phase::Idle, m.phase());
}

void test_cancel_by_stick_skips_dizzy() {
    spin::Mover m;
    m.toggleLoop(0, settings());
    m.update(spin::READY_MS, 0);
    m.cancel();
    TEST_ASSERT_EQUAL(spin::Phase::Idle, m.phase());
}

void test_once_button_while_looping_stops() {
    spin::Mover m;
    m.toggleLoop(0, settings());
    m.update(spin::READY_MS, 0);
    m.startOnce(2000, settings());
    TEST_ASSERT_EQUAL(spin::Phase::Dizzy, m.phase());
}

void test_wrap180() {
    TEST_ASSERT_FLOAT_WITHIN(1e-4, 0.0f, yaw::wrap180(360.0f));
    TEST_ASSERT_FLOAT_WITHIN(1e-4, -170.0f, yaw::wrap180(190.0f));
    TEST_ASSERT_FLOAT_WITHIN(1e-4, 170.0f, yaw::wrap180(-190.0f));
}

void test_vertical_rate_follows_tilted_up() {
    TEST_ASSERT_FLOAT_WITHIN(1e-3, 90.0f, yaw::verticalRate(0, 0, 90, 0, 0, 1));
    // 頭が 30° 上を向くと、上の向きは y と z に分かれる。上の向きまわりの回転だけを数える
    const float s = 0.5f, c = 0.8660254f;
    TEST_ASSERT_FLOAT_WITHIN(1e-3, 90.0f, yaw::verticalRate(0, 90 * s, 90 * c, 0, s, c));
    TEST_ASSERT_FLOAT_WITHIN(1e-3, 0.0f, yaw::verticalRate(90, 0, 0, 0, s, c));  // 横まわり(うなずき)は数えない
}

void test_integrator_learns_bias_only_while_still() {
    yaw::Integrator g;
    for (int i = 0; i < 2000; i++) {
        g.add(0.2f, 0.01f, true);  // 止まっていて、0.2°/s ずれている
    }
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 0.2f, g.bias());
    const float before = g.angle();
    for (int i = 0; i < 100; i++) {
        g.add(90.2f, 0.01f, false);  // 1 秒、90°/s で回る
    }
    TEST_ASSERT_FLOAT_WITHIN(0.5f, 90.0f, g.angle() - before);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 0.2f, g.bias());  // 回っている間は見積もり直さない
}

void test_integrator_ignores_fast_turns_while_marked_still() {
    yaw::Integrator g;
    for (int i = 0; i < 100; i++) {
        g.add(60.0f, 0.01f, true);  // 止めているつもりでも手で回された
    }
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 0.0f, g.bias());
}

int main() {
    UNITY_BEGIN();
    RUN_TEST(test_once_waits_then_turns_and_stops_before_full_turn);
    RUN_TEST(test_once_counts_either_direction);
    RUN_TEST(test_once_times_out);
    RUN_TEST(test_loop_toggle_stops_into_dizzy_then_idle);
    RUN_TEST(test_loop_times_out_for_safety);
    RUN_TEST(test_stop_while_ready_goes_idle_without_dizzy);
    RUN_TEST(test_cancel_by_stick_skips_dizzy);
    RUN_TEST(test_once_button_while_looping_stops);
    RUN_TEST(test_wrap180);
    RUN_TEST(test_vertical_rate_follows_tilted_up);
    RUN_TEST(test_integrator_learns_bias_only_while_still);
    RUN_TEST(test_integrator_ignores_fast_turns_while_marked_still);
    return UNITY_END();
}

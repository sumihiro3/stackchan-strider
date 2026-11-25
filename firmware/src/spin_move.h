// ボタンで回る動き(くるっと 1 回転、回り続ける)の順番を決める。機器には依存しない(native の単体テストで使う)。
//
//   Idle → Ready(頭を上げて「いくよ」、READY_MS)→ Turning(目ぐるぐる)→ Idle
//   回り続けたあとは Dizzy(目が回った、DIZZY_MS)を挟んで Idle
#pragma once

#include <cstdint>

namespace spin {

constexpr uint32_t READY_MS = 800;          // 回る前にためる時間
constexpr uint32_t DIZZY_MS = 4000;         // 回り続けたあと、目を回している時間(「めがまわる〜」の吹き出しもこの間出す)
constexpr uint32_t ONCE_TIMEOUT_MS = 15000;  // 1 回転がこの時間で終わらなければ止める
constexpr uint32_t LOOP_TIMEOUT_MS = 30000;  // 回り続けるのは長くてもこの時間まで(安全のため)
constexpr float TURN_DEG = 360.0f;
constexpr float COAST_DEG = 6.0f;  // 止める命令のあとに惰性で回る分。この分だけ手前で止める(2026-11-25 フローリングで約 4°)

enum class Phase : uint8_t { Idle, Ready, Turning, Dizzy };
enum class Kind : uint8_t { Once, Loop };
enum class Face : uint8_t { Normal, Ready, Spiral, Dizzy };

/** 回るときの設定。 */
struct Settings {
    int drive = 60;       // 首の回転の速さ [%]。符号で回る向きが決まる(呼ぶ側で決める)
    float pitch = 30.0f;  // 頭を上げる角度 [°]
    float coastDeg = COAST_DEG;
};

/** いまの命令。 */
struct Output {
    bool controls = false;  // true の間はスティックより優先して、drive と pitch を使う
    int drive = 0;          // 首の回転の速さ [%]
    float pitch = 0.0f;     // うなずきの角度 [°]
    Face face = Face::Normal;
};

/** 回る動きの順番。loop() から update() を毎回呼ぶ。 */
class Mover {
public:
    /** くるっと 1 回転を始める。動いている間は何もしない(回り続けている間は止める)。 */
    void startOnce(uint32_t now, const Settings& settings);

    /** 回り続けるのを始める。回り続けている(ためている)間なら止める。 */
    void toggleLoop(uint32_t now, const Settings& settings);

    /** 止める(止めるボタン、頭をさわった)。回り続けていたなら目を回す。 */
    void stop(uint32_t now);

    /** スティックが動かされた。すぐにやめて、スティックの操作に渡す(目は回さない)。 */
    void cancel();

    /**
     * 命令を更新する。
     * @param yawDeg 車体の向き [°](つないだ合計の角度。向きの符号は問わない)
     */
    Output update(uint32_t now, float yawDeg);

    Phase phase() const { return phase_; }
    Kind kind() const { return kind_; }

    /** 最後の 1 回転・回り続けるが、時間切れで止まったか。 */
    bool timedOut() const { return timedOut_; }

    /** Turning のときの、回り始めてから回った角度 [°]。 */
    float turned() const { return turned_; }

private:
    void begin(uint32_t now, Kind kind, const Settings& settings);
    void enter(Phase phase, uint32_t now);

    Phase phase_ = Phase::Idle;
    Kind kind_ = Kind::Once;
    Settings settings_;
    uint32_t phaseMs_ = 0;
    bool haveStartYaw_ = false;
    float startYaw_ = 0;
    float turned_ = 0;
    bool timedOut_ = false;
};

}  // namespace spin

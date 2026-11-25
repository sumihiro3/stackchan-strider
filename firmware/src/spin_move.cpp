#include "spin_move.h"

#include <cmath>

namespace spin {

void Mover::begin(uint32_t now, Kind kind, const Settings& settings) {
    kind_ = kind;
    settings_ = settings;
    timedOut_ = false;
    turned_ = 0;
    haveStartYaw_ = false;
    enter(Phase::Ready, now);
}

void Mover::enter(Phase phase, uint32_t now) {
    phase_ = phase;
    phaseMs_ = now;
}

void Mover::startOnce(uint32_t now, const Settings& settings) {
    if (phase_ == Phase::Idle || phase_ == Phase::Dizzy) {
        begin(now, Kind::Once, settings);
    } else if (kind_ == Kind::Loop) {
        stop(now);
    }
}

void Mover::toggleLoop(uint32_t now, const Settings& settings) {
    if (phase_ == Phase::Idle || phase_ == Phase::Dizzy) {
        begin(now, Kind::Loop, settings);
    } else {
        stop(now);
    }
}

void Mover::stop(uint32_t now) {
    if (phase_ == Phase::Turning && kind_ == Kind::Loop) {
        enter(Phase::Dizzy, now);
    } else if (phase_ != Phase::Dizzy) {
        phase_ = Phase::Idle;
    }
}

void Mover::cancel() { phase_ = Phase::Idle; }

Output Mover::update(uint32_t now, float yawDeg) {
    const uint32_t elapsed = now - phaseMs_;
    if (phase_ == Phase::Ready && elapsed >= READY_MS) {
        enter(Phase::Turning, now);
    }
    if (phase_ == Phase::Turning) {
        if (!haveStartYaw_) {
            startYaw_ = yawDeg;
            haveStartYaw_ = true;
        }
        turned_ = std::fabs(yawDeg - startYaw_);
        const uint32_t turning = now - phaseMs_;
        if (kind_ == Kind::Once) {
            const bool done = turned_ >= TURN_DEG - settings_.coastDeg;
            if (done || turning >= ONCE_TIMEOUT_MS) {
                timedOut_ = !done;
                phase_ = Phase::Idle;
            }
        } else if (turning >= LOOP_TIMEOUT_MS) {
            timedOut_ = true;
            enter(Phase::Dizzy, now);
        }
    }
    if (phase_ == Phase::Dizzy && now - phaseMs_ >= DIZZY_MS) {
        phase_ = Phase::Idle;
    }

    Output out;
    switch (phase_) {
        case Phase::Ready:
            out = {true, 0, settings_.pitch, Face::Ready};
            break;
        case Phase::Turning:
            out = {true, settings_.drive, settings_.pitch, Face::Spiral};
            break;
        case Phase::Dizzy:
            out = {false, 0, 0.0f, Face::Dizzy};  // 止まったあとは、スティックで動かせる
            break;
        case Phase::Idle:
            break;
    }
    return out;
}

}  // namespace spin

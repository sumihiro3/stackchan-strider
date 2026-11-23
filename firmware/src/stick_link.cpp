#include "stick_link.h"

namespace stick_link {
namespace {

constexpr uint8_t MAGIC0 = 'S';
constexpr uint8_t MAGIC1 = 'L';
constexpr uint8_t VERSION = 1;

uint8_t sum(const uint8_t* data, size_t len) {
    uint8_t s = 0;
    for (size_t i = 0; i < len; i++) {
        s = static_cast<uint8_t>(s + data[i]);
    }
    return s;
}

bool knownKind(uint8_t k) { return k >= static_cast<uint8_t>(Kind::Pair) && k <= static_cast<uint8_t>(Kind::Stick); }

bool inRange(int8_t v) { return v >= -100 && v <= 100; }

}  // namespace

size_t encode(const Packet& packet, uint8_t* out, size_t cap) {
    if (cap < PACKET_SIZE) {
        return 0;
    }
    out[0] = MAGIC0;
    out[1] = MAGIC1;
    out[2] = VERSION;
    out[3] = static_cast<uint8_t>(packet.kind);
    out[4] = packet.seq;
    out[5] = static_cast<uint8_t>(packet.x);
    out[6] = static_cast<uint8_t>(packet.y);
    out[7] = packet.buttons;
    out[8] = sum(out, PACKET_SIZE - 1);
    return PACKET_SIZE;
}

bool decode(const uint8_t* data, size_t len, Packet& out) {
    if (len != PACKET_SIZE || data[0] != MAGIC0 || data[1] != MAGIC1 || data[2] != VERSION ||
        data[8] != sum(data, PACKET_SIZE - 1) || !knownKind(data[3])) {
        return false;
    }
    const auto x = static_cast<int8_t>(data[5]);
    const auto y = static_cast<int8_t>(data[6]);
    if (!inRange(x) || !inRange(y)) {
        return false;
    }
    out = Packet{static_cast<Kind>(data[3]), data[4], x, y, data[7]};
    return true;
}

int axisFromMillivolts(int mv, int centreMv, int spanMv) {
    if (spanMv <= 0) {
        return 0;
    }
    const long v = static_cast<long>(mv - centreMv) * 100 / spanMv;
    return v > 100 ? 100 : (v < -100 ? -100 : static_cast<int>(v));
}

bool isPressed(int xMv, int vccMv) { return xMv * 100 > vccMv * 85; }

}  // namespace stick_link

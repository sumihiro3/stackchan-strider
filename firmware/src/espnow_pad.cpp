#include "espnow_pad.h"

#include <Arduino.h>
#include <Preferences.h>
#include <esp_now.h>
#include <esp_wifi.h>

#include <cstring>

#include "car_control.h"
#include "stick_link.h"

namespace espnow_pad {
namespace {

constexpr uint32_t CONNECTED_MS = 500;
constexpr uint32_t PAIR_WINDOW_MS = 60000;
constexpr uint32_t ACK_INTERVAL_MS = 300;  // 登録した相手から値が届いている間、この間隔で PairAck を返す(相手の画面の「Linked」)
const char* const PREFS_NS = "atom";

portMUX_TYPE lock = portMUX_INITIALIZER_UNLOCKED;
uint8_t peer[6] = {};
bool hasPeer = false;
uint32_t pairUntilMs = 0;  // 0 以外なら、この時刻まで新しい相手を受け付ける(登録を消したあと)
bool started = false;

// コールバック(Wi-Fi のタスク)から loop() へ渡す値。lock で守る
int lastX = 0, lastY = 0;
bool lastStop = false;
bool fresh = false;
uint32_t stickMs = 0;
bool pairPending = false;
uint8_t pairFrom[6];
int pairRssi = 0;

// loop() だけが触る
bool hasCandidate = false;
uint8_t candidate[6];
bool hasRejected = false;
uint8_t rejected[6];

bool sameMac(const uint8_t* a, const uint8_t* b) { return std::memcmp(a, b, 6) == 0; }

void addPeer(const uint8_t* mac) {
    if (esp_now_is_peer_exist(mac)) {
        return;
    }
    esp_now_peer_info_t info = {};
    std::memcpy(info.peer_addr, mac, 6);
    info.channel = 0;
    info.ifidx = WIFI_IF_STA;
    esp_now_add_peer(&info);
}

void onReceive(const esp_now_recv_info_t* info, const uint8_t* data, int len) {
    stick_link::Packet packet;
    if (!stick_link::decode(data, static_cast<size_t>(len), packet)) {
        return;
    }
    portENTER_CRITICAL(&lock);
    const bool fromPeer = hasPeer && sameMac(info->src_addr, peer);
    if (packet.kind == stick_link::Kind::Stick && fromPeer) {
        lastX = packet.x;
        lastY = packet.y;
        lastStop = (packet.buttons & stick_link::BUTTON_STOP) != 0;
        fresh = true;
        stickMs = millis();
    } else if ((packet.kind == stick_link::Kind::Pair || packet.kind == stick_link::Kind::Stick) && !pairPending) {
        // 申し込み。登録した相手(返事を取りこぼして申し込み直している)か、受け付けている間の新しい相手にだけ返事をする(update)。
        // 登録を消したあとも、Atom は前の登録のまま値(Stick)を送り続けるので、それも申し込みとして扱う
        std::memcpy(pairFrom, info->src_addr, 6);
        pairRssi = info->rx_ctrl ? info->rx_ctrl->rssi : -127;
        pairPending = true;
    }
    portEXIT_CRITICAL(&lock);
}

}  // namespace

bool registered() {
    Preferences prefs;
    prefs.begin(PREFS_NS, true);
    const bool found = prefs.getBytesLength("peer") == 6;
    prefs.end();
    return found;
}

void begin() {
    if (started) {
        return;
    }
    Preferences prefs;
    prefs.begin(PREFS_NS, true);
    hasPeer = prefs.getBytes("peer", peer, sizeof(peer)) == sizeof(peer);
    prefs.end();
    if (esp_now_init() != ESP_OK) {
        Serial.println("[atom] esp_now_init failed");
        return;
    }
    esp_now_register_recv_cb(onReceive);
    if (hasPeer) {
        addPeer(peer);
    }
    started = true;
}

namespace {

bool windowOpen(uint32_t now) { return !hasPeer || (pairUntilMs && static_cast<int32_t>(pairUntilMs - now) > 0); }

void registerPeer(const uint8_t* from) {
    Preferences prefs;
    prefs.begin(PREFS_NS, false);
    prefs.putBytes("peer", from, 6);
    prefs.end();
    if (hasPeer) {
        esp_now_del_peer(peer);
    }
    portENTER_CRITICAL(&lock);
    std::memcpy(peer, from, sizeof(peer));
    hasPeer = true;
    portEXIT_CRITICAL(&lock);
    pairUntilMs = 0;
    addPeer(from);
    Serial.printf("[atom] paired %02X:%02X:%02X:%02X:%02X:%02X\n", from[0], from[1], from[2], from[3], from[4], from[5]);
}

void sendAck(const uint8_t* to) {
    uint8_t buf[stick_link::PACKET_SIZE];
    const stick_link::Packet ack{stick_link::Kind::PairAck, 0, 0, 0, 0};
    esp_now_send(to, buf, stick_link::encode(ack, buf, sizeof(buf)));
}

}  // namespace

void update(uint32_t now) {
    if (!started) {
        return;
    }
    // ESP-NOW の届いた知らせ(MAC の ACK)は、登録していない相手にも無線の層で返る。
    // コントローラーが「登録されて値が使われている」と分かるよう、登録した相手にだけ返事を送る
    static uint32_t lastAckMs = 0;
    const uint32_t lastStick = lastStickMs();
    if (hasPeer && lastStick && !car::timedOut(now, lastStick, CONNECTED_MS) && car::timedOut(now, lastAckMs, ACK_INTERVAL_MS)) {
        lastAckMs = now;
        sendAck(peer);
    }
    if (hasCandidate && !windowOpen(now)) {
        hasCandidate = false;
    }
    uint8_t from[6];
    portENTER_CRITICAL(&lock);
    const bool pending = pairPending;
    std::memcpy(from, pairFrom, sizeof(from));
    const int rssi = pairRssi;
    pairPending = false;
    const bool known = hasPeer && sameMac(from, peer);
    portEXIT_CRITICAL(&lock);
    if (!pending) {
        return;
    }
    if (!known && hasRejected && sameMac(from, rejected) && windowOpen(now)) {
        return;
    }
    switch (stick_link::decidePairRequest(known, windowOpen(now), rssi, stick_link::PAIR_RSSI_MIN)) {
        case stick_link::PairDecision::Ignore:
            return;
        case stick_link::PairDecision::Ask:
            if (!hasCandidate) {
                std::memcpy(candidate, from, sizeof(candidate));
                hasCandidate = true;
                Serial.printf("[atom] pair request %04X rssi=%d\n", stick_link::shortId(from), rssi);
            }
            return;
        case stick_link::PairDecision::Ack:
            break;
    }
    addPeer(from);
    sendAck(from);
}

bool pairCandidate(uint16_t& id) {
    if (hasCandidate) {
        id = stick_link::shortId(candidate);
    }
    return hasCandidate;
}

void acceptCandidate() {
    if (hasCandidate) {
        hasCandidate = false;
        registerPeer(candidate);
    }
}

void rejectCandidate(bool remember) {
    if (hasCandidate) {
        hasCandidate = false;
        if (remember) {
            std::memcpy(rejected, candidate, sizeof(rejected));
            hasRejected = true;
        }
    }
}

State state(uint32_t now) {
    if (!hasPeer) {
        return State::Unpaired;
    }
    const uint32_t last = lastStickMs();
    return last && !car::timedOut(now, last, CONNECTED_MS) ? State::Connected : State::Waiting;
}

bool takeStick(int& x, int& y, bool& stop) {
    portENTER_CRITICAL(&lock);
    const bool got = fresh;
    x = lastX;
    y = lastY;
    stop = lastStop;
    fresh = false;
    portEXIT_CRITICAL(&lock);
    return got;
}

uint32_t lastStickMs() {
    portENTER_CRITICAL(&lock);
    const uint32_t ms = stickMs;
    portEXIT_CRITICAL(&lock);
    return ms;
}

void forget() {
    Preferences prefs;
    prefs.begin(PREFS_NS, false);
    prefs.remove("peer");
    prefs.end();
    uint8_t old[6];
    portENTER_CRITICAL(&lock);
    const bool had = hasPeer;
    std::memcpy(old, peer, sizeof(old));
    hasPeer = false;
    stickMs = 0;
    portEXIT_CRITICAL(&lock);
    if (had && started) {
        esp_now_del_peer(old);  // Wi-Fi の内部のロックを取るので、割り込みを止めた中では呼ばない
    }
    hasCandidate = false;
    hasRejected = false;
    pairUntilMs = millis() + PAIR_WINDOW_MS;
    if (pairUntilMs == 0) {
        pairUntilMs = 1;
    }
}

}  // namespace espnow_pad

// 手もとのコントローラー。入力(input.h)を 0.05 秒ごとに読み、ESP-NOW でスタックチャンへ送る。
// 入力は機器ごとに 1 つ: M5Stack StopWatch のタッチと、黄・青のボタン(input_stopwatch.cpp)。
#include <M5Unified.h>
#include <Preferences.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_system.h>
#include <esp_wifi.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>

#include "../stick_link.h"
#include "input.h"

namespace {

constexpr uint32_t SEND_INTERVAL_MS = 50;
constexpr int FAILS_BEFORE_HOP = 10;      // 続けて届かなければ、次のチャンネルへ
constexpr int PAIR_SENDS_PER_CHANNEL = 3;  // 組にする相手を探すとき、1 つのチャンネルで送る回数
constexpr uint32_t LINK_OK_MS = 1000;      // スタックチャンの返事(PairAck、0.3 秒ごと)が最後に来てからこの時間内なら「つながっている」
constexpr uint32_t DRAW_INTERVAL_MS = 100;  // StopWatch の大きい画面は送るのに時間がかかるので、描くのは送るより少なくする
constexpr int CHANNEL_MAX = 13;

const uint8_t BROADCAST[6] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff};

Preferences prefs;
M5Canvas canvas(&M5.Display);

uint8_t peer[6] = {};
bool paired = false;
int channel = 1;
uint8_t seq = 0;
int consecutiveFails = 0;
int pairSends = 0;

// ESP-NOW のコールバック(Wi-Fi のタスクで動く)から loop() へ渡す値
uint32_t lastAppAckMs = 0;  // 登録した相手から最後に PairAck が来た時刻(loop() だけが触る)
std::atomic<int> sendResults{0};  // 正 = 届いた回数、負 = 届かなかった回数(loop() が読むたびに 0 に戻す)
std::atomic<bool> pairAckPending{false};
uint8_t pairAckFrom[6];
int pairAckRssi = 0;
uint16_t ownId = 0;  // 自分の MAC の下 2 バイト。スタックチャンの確認画面に出る番号と見比べる

void setChannel(int ch) {
    if (esp_wifi_set_channel(static_cast<uint8_t>(ch), WIFI_SECOND_CHAN_NONE) == ESP_OK) {
        channel = ch;
    } else {
        Serial.printf("[ctl] set channel %d failed\n", ch);
    }
}

void nextChannel() {
    static int target = 1;
    target = target % CHANNEL_MAX + 1;
    setChannel(target);  // 失敗しても次は先のチャンネルを試す
}

void addPeer(const uint8_t* mac) {
    if (esp_now_is_peer_exist(mac)) {
        return;
    }
    esp_now_peer_info_t info = {};
    std::memcpy(info.peer_addr, mac, 6);
    info.channel = 0;  // いまのチャンネルで送る(チャンネルを変えるとついていく)
    info.ifidx = WIFI_IF_STA;
    info.encrypt = false;
    esp_now_add_peer(&info);
}

void onSent(const esp_now_send_info_t*, esp_now_send_status_t status) {
    if (status == ESP_NOW_SEND_SUCCESS) {
        sendResults++;
    } else {
        sendResults--;
    }
}

void onReceive(const esp_now_recv_info_t* info, const uint8_t* data, int len) {
    stick_link::Packet packet;
    if (!stick_link::decode(data, static_cast<size_t>(len), packet) || packet.kind != stick_link::Kind::PairAck) {
        return;
    }
    if (!pairAckPending) {
        std::memcpy(pairAckFrom, info->src_addr, 6);
        pairAckRssi = info->rx_ctrl ? info->rx_ctrl->rssi : -127;
        pairAckPending = true;
    }
}

void send(const uint8_t* to, stick_link::Kind kind, int x, int y, uint8_t buttons) {
    uint8_t buf[stick_link::PACKET_SIZE];
    const stick_link::Packet packet{kind, seq++, static_cast<int8_t>(x), static_cast<int8_t>(y), buttons};
    esp_now_send(to, buf, stick_link::encode(packet, buf, sizeof(buf)));
}

void forgetPeer() {
    if (paired) {
        esp_now_del_peer(peer);
    }
    paired = false;
    prefs.remove("peer");
    pairSends = 0;
}

void draw(const input::Reading& in) {
    const bool linked = paired && lastAppAckMs && millis() - lastAppAckMs < LINK_OK_MS;
    const int w = canvas.width(), h = canvas.height();
    const int scale = std::max(1, w / 128);  // StopWatch(466 px)では文字と点を大きくする
    const int line = 12 * scale;
    // 丸い画面でも欠けないよう、文字は中央寄りに置く
    const int top = w > 200 ? h / 8 : 2;
    const int bottom = w > 200 ? h - h / 8 - line : h - line;
    const int idTop = bottom - line;  // 自分の ID の行
    const int cx = w / 2, cy = (top + 2 * line + idTop) / 2;
    const int r = std::min(w * (w > 200 ? 6 : 10) / 20, (idTop - top - 2 * line) / 2) - 4;
    canvas.fillSprite(TFT_BLACK);
    canvas.setTextSize(scale);
    canvas.setTextColor(linked ? TFT_GREEN : TFT_YELLOW);
    canvas.drawCenterString(!paired ? "Pairing" : (linked ? "Linked" : "Search"), cx, top);
    canvas.setTextColor(TFT_DARKGREY);
    const int battery = M5.Power.getBatteryLevel();
    String sub = String("ch") + channel + (battery >= 0 ? String(" ") + battery + "%" : String());
    if (input::modeLabel()[0]) {
        sub += String(" ") + input::modeLabel();
    }
    canvas.drawCenterString(sub, cx, top + line);
    canvas.drawCircle(cx, cy, r, TFT_DARKGREY);
    canvas.fillCircle(cx + in.x * r / 100, cy - in.y * r / 100, 5 * scale, in.stop ? TFT_RED : TFT_CYAN);
    if (in.stop) {
        canvas.setTextColor(TFT_RED);
        canvas.drawCenterString("STOP", cx, cy - line / 2);
    }
    // スタックチャンの「登録しますか」の画面に同じ番号が出る。スタックチャン側だけ登録を消したときも見比べられるよう、いつも出す
    char id[12];
    std::snprintf(id, sizeof(id), "ID %04X", ownId);
    canvas.setTextColor(paired ? TFT_DARKGREY : TFT_WHITE);
    canvas.drawCenterString(id, cx, idTop);
    char left[16], right[16];
    input::debugText(left, right, sizeof(left));
    canvas.setTextColor(TFT_DARKGREY);
    const int margin = w > 200 ? w / 5 : 2;
    canvas.drawString(left, margin, bottom);  // ボタンの役目。左が黄、右が青
    canvas.drawRightString(right, w - margin, bottom);
    canvas.pushSprite(0, 0);
}

}  // namespace

void setup() {
    Serial.begin(115200);
    auto cfg = M5.config();
    M5.begin(cfg);
    // 電源ボタンで電源を入れた起動では、画面(CO5300)の初期化が効かず黒いままになる(タッチと送信は動く)。
    // CPU だけのリセットなら毎回映るので、電源を入れたときは 1 回だけ再起動する。電源は M5PM1 が保持しているので落ちない(2026-11-24 実機)
    if (esp_reset_reason() == ESP_RST_POWERON) {
        delay(100);
        esp_restart();
    }
    if (M5.Display.width() > 200) {
        canvas.setColorDepth(8);  // 466×466 を 16 bit で持つと 430 KB になる。8 bit で十分
        canvas.setPsram(true);
    }
    canvas.createSprite(M5.Display.width(), M5.Display.height());
    Serial.printf("[ctl] board=%d display=%dx%d\n", static_cast<int>(M5.getBoard()), M5.Display.width(), M5.Display.height());

    prefs.begin("atomctl", false);
    paired = prefs.getBytes("peer", peer, sizeof(peer)) == sizeof(peer);

    WiFi.mode(WIFI_STA);
    WiFi.disconnect();
    esp_wifi_set_country_code("JP", false);  // 既定では 1〜11ch しか使えない。日本のルーターは 12・13ch も使う
    uint8_t mac[6];
    esp_wifi_get_mac(WIFI_IF_STA, mac);
    ownId = stick_link::shortId(mac);
    esp_now_init();
    esp_now_register_send_cb(onSent);
    esp_now_register_recv_cb(onReceive);
    addPeer(BROADCAST);
    if (paired) {
        addPeer(peer);
    }
    setChannel(1);
    Serial.println("[ctl] radio ready");
    input::begin();
    Serial.println("[ctl] input ready");
}

void loop() {
    static uint32_t lastSendMs = 0, lastDrawMs = 0;
    M5.update();
    input::update();
    if (input::takeForgetRequest()) {
        forgetPeer();
    }

    if (pairAckPending && paired && std::memcmp(pairAckFrom, peer, sizeof(peer)) == 0) {
        lastAppAckMs = millis();  // 登録した相手が、こちらの値を使っている
        pairAckPending = false;
    } else if (pairAckPending) {
        const bool accepted = stick_link::acceptPairAck(paired, pairAckRssi, stick_link::PAIR_RSSI_MIN);
        if (accepted) {
            std::memcpy(peer, pairAckFrom, sizeof(peer));
            paired = true;
            prefs.putBytes("peer", peer, sizeof(peer));
            addPeer(peer);
            consecutiveFails = 0;
            lastAppAckMs = millis();
        }
        Serial.printf("[ctl] pair ack rssi=%d %s\n", pairAckRssi, accepted ? "paired" : "ignored");
        pairAckPending = false;
    }

    const uint32_t now = millis();
    if (now - lastSendMs < SEND_INTERVAL_MS) {
        return;
    }
    lastSendMs = now;

    const input::Reading in = input::read();
    const int x = in.stop ? 0 : in.x;
    const int y = in.stop ? 0 : in.y;

    // 送った結果を数え、届かない状態が続いたらチャンネルを変えて相手を探す
    const int results = sendResults.exchange(0);
    consecutiveFails = results > 0 ? 0 : consecutiveFails - results;
    if (paired) {
        if (consecutiveFails >= FAILS_BEFORE_HOP) {
            consecutiveFails = 0;
            nextChannel();
        }
        const uint8_t buttons = stick_link::packPresses(in.stop ? stick_link::BUTTON_STOP : 0, in.oncePresses, in.loopPresses);
        send(peer, stick_link::Kind::Stick, x, y, buttons);
    } else {
        if (++pairSends >= PAIR_SENDS_PER_CHANNEL) {
            pairSends = 0;
            nextChannel();
        }
        send(BROADCAST, stick_link::Kind::Pair, 0, 0, 0);
    }
    static uint32_t lastLogMs = 0;
    if (now - lastLogMs >= 2000) {  // USB につないだときの確かめ用
        lastLogMs = now;
        Serial.printf("[ctl] board=%d paired=%d ch=%d linked=%d x=%d y=%d stop=%d mode=%s batt=%d\n",
                      static_cast<int>(M5.getBoard()), paired ? 1 : 0, channel,
                      paired && lastAppAckMs && now - lastAppAckMs < LINK_OK_MS ? 1 : 0, x, y, in.stop ? 1 : 0, input::modeLabel(),
                      static_cast<int>(M5.Power.getBatteryLevel()));
    }
    if (now - lastDrawMs >= DRAW_INTERVAL_MS) {
        lastDrawMs = now;
        draw(in);
    }
}

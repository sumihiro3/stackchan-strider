#include "gamepad.h"

#include <Arduino.h>
#include <NimBLEDevice.h>

#include <atomic>
#include <cstring>

#include "hid_parser.h"

namespace gamepad {
namespace {

const NimBLEUUID HID_SERVICE(static_cast<uint16_t>(0x1812));
const NimBLEUUID REPORT_MAP(static_cast<uint16_t>(0x2A4B));
const NimBLEUUID REPORT(static_cast<uint16_t>(0x2A4D));
const NimBLEUUID REPORT_REFERENCE(static_cast<uint16_t>(0x2908));

// 外見(Appearance)。キーボードとマウスはつながない
constexpr uint16_t APPEARANCE_KEYBOARD = 0x03C1;
constexpr uint16_t APPEARANCE_MOUSE = 0x03C2;

constexpr uint32_t SCAN_MS = 4000;
constexpr uint32_t CONNECT_TIMEOUT_MS = 6000;
constexpr uint32_t REJECT_MS = 60000;  // 左スティックが読めなかった機器は、この間は選ばない
// 接続の間隔 15〜30 ms、切断の検出 0.5 秒(単位は 1.25 ms と 10 ms)。パッドが切れたら早く止めたい
constexpr uint16_t CONN_MIN_INTERVAL = 12, CONN_MAX_INTERVAL = 24, CONN_LATENCY = 0, CONN_TIMEOUT = 50;

std::atomic<State> currentState{State::Searching};
std::atomic<int> stickX{0}, stickY{0};
std::atomic<uint32_t> stickSeq{0};
uint32_t takenSeq = 0;
std::atomic<bool> forgetRequested{false};
portMUX_TYPE nameLock = portMUX_INITIALIZER_UNLOCKED;
char deviceName[32] = "";

NimBLEClient* client = nullptr;
hid::GamepadLayout layout;
NimBLEAddress rejected;
uint32_t rejectedAtMs = 0;

void setName(const std::string& value) {
    portENTER_CRITICAL(&nameLock);
    strncpy(deviceName, value.c_str(), sizeof(deviceName) - 1);
    deviceName[sizeof(deviceName) - 1] = '\0';
    portEXIT_CRITICAL(&nameLock);
}

// スティックを中心に戻した扱いにする(切れたときに、loop() が止まるように)
void releaseStick() {
    stickX = 0;
    stickY = 0;
    stickSeq++;
}

class ClientCallbacks : public NimBLEClientCallbacks {
    void onDisconnect(NimBLEClient*, int reason) override {
        Serial.printf("[pad] disconnected (reason %d)\n", reason);
        currentState = State::Searching;
        setName("");
        releaseStick();
    }
    // パッドは入出力のない Just Works でペアリングする。確認を求められたら受け入れる
    void onConfirmPasskey(NimBLEConnInfo& info, uint32_t) override { NimBLEDevice::injectConfirmPasskey(info, true); }
    void onAuthenticationComplete(NimBLEConnInfo& info) override {
        Serial.printf("[pad] encrypted=%d bonded=%d\n", info.isEncrypted(), info.isBonded());
    }
} clientCallbacks;

void onReport(NimBLERemoteCharacteristic*, uint8_t* data, size_t length, bool) {
    // HID の Y は手前に引くと大きくなる。前に倒すと正にする
    stickX = hid::readAxis(layout.x, data, length);
    stickY = -hid::readAxis(layout.y, data, length);
    stickSeq++;
}

bool isBonded(const NimBLEAddress& address) {
    for (int i = 0; i < NimBLEDevice::getNumBonds(); i++) {
        if (NimBLEDevice::getBondedAddress(i) == address) {
            return true;
        }
    }
    return false;
}

// 探した結果から、つなぐ相手を選ぶ。登録済みのパッドを優先し、なければ HID のゲームパッドらしいもの
const NimBLEAdvertisedDevice* chooseDevice(const NimBLEScanResults& results) {
    const NimBLEAdvertisedDevice* candidate = nullptr;
    for (const NimBLEAdvertisedDevice* device : results) {
        if (!device->isAdvertisingService(HID_SERVICE)) {
            continue;
        }
        if (device->getAddress() == rejected && millis() - rejectedAtMs < REJECT_MS) {
            continue;
        }
        if (isBonded(device->getAddress())) {
            return device;
        }
        if (device->haveAppearance()) {
            const uint16_t appearance = device->getAppearance();
            if (appearance == APPEARANCE_KEYBOARD || appearance == APPEARANCE_MOUSE) {
                continue;
            }
        }
        if (!candidate) {
            candidate = device;
        }
    }
    return candidate;
}

// HD-6500 など、実機の記述子をあとでテストに足せるように、16 進でシリアルに出す
void dumpReportMap(const NimBLEAttValue& map) {
    Serial.printf("[pad] report map (%u bytes):", static_cast<unsigned>(map.size()));
    for (size_t i = 0; i < map.size(); i++) {
        Serial.printf("%s%02X", i % 16 ? " " : "\n  ", map.data()[i]);
    }
    Serial.println();
}

// つないだあと: 暗号化 → Report Map を読む → 左スティックのあるレポートを購読する
bool setUpHid(NimBLEClient* c) {
    if (!c->secureConnection()) {
        Serial.println("[pad] pairing failed");
        return false;
    }
    NimBLERemoteService* service = c->getService(HID_SERVICE);
    NimBLERemoteCharacteristic* mapChr = service ? service->getCharacteristic(REPORT_MAP) : nullptr;
    if (!mapChr) {
        Serial.println("[pad] no HID report map");
        return false;
    }
    const NimBLEAttValue map = mapChr->readValue();
    dumpReportMap(map);
    if (!hid::parseReportMap(map.data(), map.size(), layout)) {
        Serial.println("[pad] no absolute X/Y in the report map");
        return false;
    }
    Serial.printf("[pad] X: id %u bit %u size %u range %ld..%ld\n", layout.x.reportId, layout.x.bitOffset, layout.x.bitSize,
                  static_cast<long>(layout.x.logicalMin), static_cast<long>(layout.x.logicalMax));
    bool subscribed = false;
    for (NimBLERemoteCharacteristic* chr : service->getCharacteristics(true)) {
        if (chr->getUUID() != REPORT || !chr->canNotify()) {
            continue;
        }
        // Report Reference = [レポート ID, 種類(1 = 入力)]。ない機器は、入力レポートが 1 つだけとみなす
        NimBLERemoteDescriptor* ref = chr->getDescriptor(REPORT_REFERENCE);
        if (ref) {
            const NimBLEAttValue value = ref->readValue();
            if (value.size() >= 2 && (value.data()[1] != 1 || value.data()[0] != layout.x.reportId)) {
                continue;
            }
        }
        if (chr->subscribe(true, onReport)) {
            subscribed = true;
        }
    }
    if (!subscribed) {
        Serial.println("[pad] could not subscribe to the input report");
    }
    return subscribed;
}

void tryConnect(NimBLEScan* scan) {
    currentState = State::Searching;
    const NimBLEScanResults results = scan->getResults(SCAN_MS, false);
    const NimBLEAdvertisedDevice* device = chooseDevice(results);
    if (!device) {
        scan->clearResults();
        return;
    }
    currentState = State::Connecting;
    const NimBLEAddress address = device->getAddress();
    const std::string advertisedName = device->getName();
    Serial.printf("[pad] connecting to %s (%s)\n", address.toString().c_str(), advertisedName.c_str());
    if (!client) {
        client = NimBLEDevice::createClient();
        client->setClientCallbacks(&clientCallbacks, false);
        client->setConnectTimeout(CONNECT_TIMEOUT_MS);
        client->setConnectionParams(CONN_MIN_INTERVAL, CONN_MAX_INTERVAL, CONN_LATENCY, CONN_TIMEOUT);
    }
    const bool connected = client->connect(device);
    scan->clearResults();
    if (!connected) {
        currentState = State::Searching;
        return;
    }
    if (!setUpHid(client)) {
        rejected = address;
        rejectedAtMs = millis();
        client->disconnect();
        currentState = State::Searching;
        return;
    }
    setName(advertisedName.empty() ? address.toString() : advertisedName);
    releaseStick();
    currentState = State::Connected;
    Serial.printf("[pad] connected: %s\n", deviceName);
}

void task(void*) {
    NimBLEScan* scan = NimBLEDevice::getScan();
    scan->setActiveScan(true);  // 名前は応答パケットに入っていることが多い
    // 電波は Wi-Fi と共用。探し続けるとスマホの操作が遅れるので、聞く時間を 3 割にする
    scan->setInterval(100);
    scan->setWindow(30);
    for (;;) {
        if (forgetRequested.exchange(false)) {
            if (client && client->isConnected()) {
                client->disconnect();
                delay(300);
            }
            NimBLEDevice::deleteAllBonds();
            Serial.println("[pad] bonds deleted");
        }
        if (client && client->isConnected()) {
            delay(200);
            continue;
        }
        tryConnect(scan);
        delay(100);
    }
}

}  // namespace

void begin() {
    NimBLEDevice::init("Stackchan Car");
    NimBLEDevice::setSecurityAuth(true, false, true);  // ボンドする、MITM なし、Secure Connections
    NimBLEDevice::setSecurityIOCap(BLE_HS_IO_NO_INPUT_OUTPUT);
    xTaskCreatePinnedToCore(task, "gamepad", 6144, nullptr, 1, nullptr, 0);
}

State state() { return currentState.load(); }

bool takeStick(int& x, int& y) {
    const uint32_t seq = stickSeq.load();
    x = stickX.load();
    y = stickY.load();
    if (seq == takenSeq) {
        return false;
    }
    takenSeq = seq;
    return true;
}

const char* name() {
    static char copy[sizeof(deviceName)];
    portENTER_CRITICAL(&nameLock);
    memcpy(copy, deviceName, sizeof(copy));
    portEXIT_CRITICAL(&nameLock);
    return copy;
}

void forget() { forgetRequested = true; }

}  // namespace gamepad

// スタックチャン・ピボット: M5Stack 製品版スタックチャン(K151)を、ピボット(car/pivot_design.py)として走らせるファームウェア。
//
//  - 首振り(ヨー)サーボを回し続けて駆動輪を回す。頭が水平(うなずき 0°)なら直進し、頭を上げると足が下りて、その場で回る
//  - 操作は 2 通り。どちらもスティック 1 本(上下 = 前進・後退と速さ、左右に大きく倒す = 旋回)
//    1. スマホのブラウザのジョイスティック。家の Wi-Fi 経由。触れている間は 0.1 秒ごとに値が来て、0.9 秒途切れたら止まる
//    2. 手に持つコントローラー(M5Stack StopWatch)。ESP-NOW で直接つなぐ(espnow_pad.cpp)
//  - うなずきは 0〜30° に必ず収める(car_control.h)
//  - ボタンで回る: くるっと 1 回転、回り続ける(操作ページのボタン、StopWatch の黄・青)。spin_move.h
//    回る前に「いくよ」を出して少しため、回っている間は目がぐるぐるになる。1 回転は頭のジャイロで数える(yaw_tracker.h)
//  - 画面はスタックチャンの顔(m5stack-avatar)。走っている間に笑顔にするかは設定で選べる
//  - はじめて起動すると設定用の Wi-Fi を出し、画面の QR をスマホで読んで家の Wi-Fi を設定する(StopWatch は設定前から使える)
//
// 注意: この機体では土台が回り続けるので、首(ヨー)を「角度指定」で動かす命令(goHome / stop / moveX)は使わない。
//       それらは原点の角度まで首を回そうとして、脚が勝手に動く。止めるときは出力 0 の連続回転(rotateX(0))にする。
#include <Arduino.h>
#include <initializer_list>
#include <Avatar.h>
#include <DNSServer.h>
#include <ESPmDNS.h>
#include <M5StackChan.h>
#include <Preferences.h>
#include <WebServer.h>
#include <WiFi.h>

#include <cmath>

#include "car_control.h"
#include "espnow_pad.h"
#include "pages.h"
#include "spin_move.h"
#include "spiral_eyes.h"
#include "yaw_tracker.h"

namespace {

constexpr uint32_t COMMAND_TIMEOUT_MS = 900;  // スマホからの値(0.1 秒ごと)がこの時間途切れたら止める
constexpr uint32_t CONNECT_TIMEOUT_MS = 20000;  // 家の Wi-Fi への接続を待つ時間
constexpr uint32_t QR_SHOW_MS = 15000;  // 操作ページの QR を出しておく時間
constexpr uint32_t FORGET_HOLD_MS = 5000;  // 頭の上をこの時間さわり続けると「Wi-Fi の設定を消しますか」を出す
constexpr uint32_t CONFIRM_SHOW_MS = 10000;  // 確認の画面を出しておく時間。過ぎたら何もせず顔に戻る
constexpr float PITCH_STEP_DEG = 0.5f;  // うなずきは、この角度以上変わったときだけ動かす
constexpr uint32_t PITCH_INTERVAL_MS = 50;  // うなずきの命令の最短の間隔
constexpr int PITCH_SPEED = 700;  // movePitch の速さ(0〜1000)
constexpr uint32_t YAW_READY_MS = 2000;  // 電源を入れてから、首のサーボのモードを切り替えるまで待つ時間
constexpr uint32_t ATOM_TIMEOUT_MS = 500;  // Atom の値(0.05 秒ごと)がこの時間途切れたら止める
constexpr uint32_t BUBBLE_MS = 3000;  // StopWatch がつながったときの吹き出しを出しておく時間
constexpr float DIZZY_SWAY_RAD = 0.18f;  // 目が回ったときに顔を左右に揺らす大きさ
const IPAddress SETUP_IP(192, 168, 4, 1);
const char* const MDNS_NAME = "strider";

enum class Mode { Setup, Ready };

Preferences prefs;
WebServer server(80);
DNSServer dns;
m5avatar::Avatar avatar;

Mode mode = Mode::Setup;
String setupSsid;  // 設定用 Wi-Fi の名前
bool invertDirection = false;
bool spinReversed = false;  // 旋回の向きを入れ替える(スティックの右で左へ回るとき用)
bool smileWhileWalking = false;  // 走っている間は笑顔にするか(設定で切り替える。初期状態は変えない)

// 操作。最後に動かしたほう(スマホか StopWatch)の値を使う
enum class Source { None, Phone, Atom };
Source activeSource = Source::None;
int stickX = 0, stickY = 0;  // -100〜100
int driveSigned = 0;  // 首の回転の速さ [%]。正 = 前進
float spinDeg = car::PITCH_MAX_DEG;  // 回るときに頭を上げる角度(小さいほど足が浅く下りる)
int speedMax = car::SPEED_MAX;  // 速さの上限 [%]
float pitchTarget = car::LEVEL_DEG;
float pitchSent = -1.0f;  // 最後に送ったうなずきの角度(-1 = まだ送っていない)
uint32_t pitchSentMs = 0;
uint32_t lastPhoneMs = 0;
String stoppedPress;  // 最後に止めた「ひと押し」の番号。これと同じ番号の値は、離したあとに遅れて届いたものなので捨てる
espnow_pad::State atomState = espnow_pad::State::Unpaired;
uint32_t bubbleUntilMs = 0;
uint32_t restartAtMs = 0;  // 0 以外なら、この時刻に再起動する
uint32_t qrUntilMs = 0;  // 0 以外なら、この時刻まで QR を表示している
int qrSecondsShown = -1;  // QR の画面にいま出している残り秒数(-1 = まだ出していない)
uint32_t confirmUntilMs = 0;  // 0 以外なら、この時刻まで「Wi-Fi の設定を消しますか」を表示している
bool avatarRunning = false;
uint32_t pairConfirmUntilMs = 0;  // 0 以外なら、この時刻まで「コントローラーを登録しますか」を表示している

// ボタンで回る動きと、車体の向き
spin::Mover mover;
bool buttonSpinLeft = false;  // ボタンで回る向き(初期は右。右はデスクマットでもびっこを引かない)
bool moverControlling = false;  // 前の loop() で、回る動きが首とうなずきを動かしていたか
spin::Face faceShown = spin::Face::Normal;
yaw::Integrator bodyYaw;  // 車体の向き [°](頭のジャイロ。頭は車体に留まっている)
uint32_t imuMs = 0;
uint32_t stillSinceMs = 0;  // 首を止めた時刻(止めてから少し待ってから、ジャイロのずれを見積もる)
int driveApplied = 0x7fffffff;  // 最後に首へ送った速さ

// ---------------------------------------------------------------- 走る・回る
void applyDrive() {
    const int sign = invertDirection ? -1 : 1;
    M5StackChan.Motion.rotateX(sign * driveSigned * 10);  // 速さ % → -1000〜1000。0 で止まる
    driveApplied = driveSigned;
}

// うなずきを目標の角度へ。小さな変化と、短い間隔での連続の命令は送らない(サーボの通信を詰まらせない)
void updatePitch(uint32_t now, bool force = false) {
    const float diff = std::fabs(pitchTarget - pitchSent);
    if (diff < 0.05f) {
        return;
    }
    if (!force && (now - pitchSentMs < PITCH_INTERVAL_MS || (diff < PITCH_STEP_DEG && now - pitchSentMs < 300))) {
        return;
    }
    const float degrees = constrain(pitchTarget, 0.0f, car::PITCH_MAX_DEG);  // 念のためここでも 0〜30° に収める
    M5StackChan.Motion.movePitch(static_cast<int>(std::lround(degrees * 10)), PITCH_SPEED);  // 角度の単位は 0.1°
    pitchSent = degrees;
    pitchSentMs = now;
}

// 顔の表情。設定が ON のときだけ、走っている間を笑顔にする
void applyExpression() {
    if (!avatarRunning) {
        return;
    }
    const bool smile = smileWhileWalking && driveSigned != 0;
    avatar.setExpression(smile ? m5avatar::Expression::Happy : m5avatar::Expression::Neutral);
}

// スティックの値から、首の回転の速さとうなずきの目標を決める
void updateTargets() {
    const car::PivotCommand command = car::pivotFromStick(stickX, stickY, spinReversed, spinDeg, speedMax);
    driveSigned = command.drive;
    pitchTarget = command.pitch;
}

void setControl(int x, int y, Source source) {
    stickX = constrain(x, -100, 100);
    stickY = constrain(y, -100, 100);
    activeSource = (stickX == 0 && stickY == 0) ? Source::None : source;
    const bool moved = abs(stickX) > car::DEAD_ZONE || abs(stickY) > car::DEAD_ZONE;
    if (moved && (mover.phase() == spin::Phase::Ready || mover.phase() == spin::Phase::Turning)) {
        mover.cancel();  // 回っている間にスティックを動かしたら、スティックの操作に戻る
    }
    if (moverControlling && mover.phase() != spin::Phase::Idle && mover.phase() != spin::Phase::Dizzy) {
        return;  // 回っている間は、スティックの値(中心)で止めない
    }
    updateTargets();
    applyDrive();
    updatePitch(millis());
    applyExpression();
}

void stopControl() { setControl(0, 0, Source::None); }

// ボタンで回るときの設定。右回りはスティックを右へ倒したときと同じ(前進)。「回る向きを入れ替える」も同じように効く
spin::Settings spinSettings() {
    spin::Settings settings;
    const int direction = (buttonSpinLeft ? -1 : 1) * (spinReversed ? -1 : 1);
    settings.drive = direction * speedMax;
    settings.pitch = spinDeg;
    return settings;
}

void showFace();

// ボタンで回る動きを始める前に、操作ページの QR などを閉じて顔を出す
void prepareSpin() {
    if (qrUntilMs) {
        showFace();
    }
}

void spinOnce(uint32_t now) {
    prepareSpin();
    mover.startOnce(now, spinSettings());
}

void spinLoopToggle(uint32_t now) {
    prepareSpin();
    mover.toggleLoop(now, spinSettings());
}

// 顔を、回る動きの段階に合わせる(いくよ、ぐるぐる、目が回った)
void applySpinFace(spin::Face face, uint32_t now) {
    if (!avatarRunning) {
        faceShown = spin::Face::Normal;
        return;
    }
    if (face == spin::Face::Dizzy) {
        avatar.setRotation(DIZZY_SWAY_RAD * std::sin(static_cast<float>(now) / 120.0f));
    }
    if (face == faceShown) {
        return;
    }
    faceShown = face;
    switch (face) {
        case spin::Face::Ready:
            spiral_eyes::apply(avatar, 0);
            avatar.setSpeechText("いくよ!");
            break;
        case spin::Face::Spiral:
            spiral_eyes::apply(avatar, 1);
            avatar.setSpeechText("");
            break;
        case spin::Face::Dizzy:
            spiral_eyes::apply(avatar, 2);
            avatar.setSpeechText("めがまわる〜");
            break;
        case spin::Face::Normal:
            spiral_eyes::apply(avatar, 0);
            avatar.setRotation(0);
            avatar.setSpeechText("");
            break;
    }
}

// 車体の向きを更新する。首を止めてから 1 秒たった間は、ジャイロのゼロ点のずれも見積もる
void updateYaw(uint32_t now) {
    if (driveApplied != 0) {
        stillSinceMs = now;
    }
    if (!M5.Imu.update()) {
        return;
    }
    float gx, gy, gz, ax, ay, az;
    M5.Imu.getGyro(&gx, &gy, &gz);
    M5.Imu.getAccel(&ax, &ay, &az);
    if (imuMs) {
        const bool still = driveApplied == 0 && now - stillSinceMs > 1000;
        bodyYaw.add(yaw::verticalRate(gx, gy, gz, ax, ay, az), static_cast<float>(now - imuMs) / 1000.0f, still);
    }
    imuMs = now;
}

// 回る動きを進め、首とうなずきに渡す。回る動きが終わったら、スティックの値に戻す
void updateSpin(uint32_t now) {
    const spin::Output out = mover.update(now, bodyYaw.angle());
    if (out.controls) {
        driveSigned = out.drive;
        pitchTarget = out.pitch;
    } else if (moverControlling) {
        updateTargets();
    }
    moverControlling = out.controls;
    if (driveSigned != driveApplied) {
        applyDrive();
    }
    applySpinFace(out.face, now);
}

// StopWatch のスティックが倒れている間は、スマホからの値を受け付けない
bool atomIsSteering() {
    return activeSource == Source::Atom && (abs(stickX) > car::DEAD_ZONE || abs(stickY) > car::DEAD_ZONE);
}

// ---------------------------------------------------------------- 画面
// 顔の描画を止める。suspend() は描画タスクを画面への通信の途中で凍結しうるので使わず、
// 描画ループが 1 コマ描き終えて自分で終わる stop() を使い、終わるまで少し待つ
void stopAvatar() {
    if (avatarRunning) {
        avatar.stop();
        avatarRunning = false;
        delay(200);
    }
    // 顔の吹き出し(M5Stack-Avatar の Balloon)は、画面そのものの字の大きさを変えたまま戻さない。顔以外の画面を描く前に戻す
    M5.Display.setTextSize(1);
}

void showFace() {
    qrUntilMs = 0;
    M5.Display.fillScreen(TFT_BLACK);
    avatar.start();
    avatarRunning = true;
    applyExpression();
}

void drawCaption(const char* text, int x, int y) {
    M5.Display.setFont(&fonts::lgfxJapanGothic_16);
    M5.Display.setTextColor(TFT_BLACK, TFT_WHITE);
    M5.Display.setTextDatum(textdatum_t::top_center);
    M5.Display.drawString(text, x, y);
}

// QR を左に大きく、右に「何をすればいいか」を文章で出す画面。lines は右側の文章(1 行ずつ)
void showQrScreen(const char* title, const String& qr, std::initializer_list<const char*> lines, const char* footer) {
    stopAvatar();
    M5.Display.fillScreen(TFT_WHITE);
    M5.Display.fillRect(0, 0, 320, 30, TFT_BLACK);
    M5.Display.setFont(&fonts::lgfxJapanGothic_16);
    M5.Display.setTextDatum(textdatum_t::middle_left);
    M5.Display.setTextColor(TFT_WHITE, TFT_BLACK);
    M5.Display.drawString(title, 10, 15);
    M5.Display.qrcode(qr.c_str(), 8, 40, 164, 4);
    M5.Display.setTextColor(TFT_BLACK, TFT_WHITE);
    M5.Display.setTextDatum(textdatum_t::top_left);
    int y = 46;
    for (const char* line : lines) {
        M5.Display.drawString(line, 182, y);
        y += 24;
    }
    M5.Display.setTextDatum(textdatum_t::bottom_center);
    M5.Display.drawString(footer, 160, 236);
}

// 設定の 1 歩目: スマホを設定用 Wi-Fi につないでもらう
void showSetupJoinScreen() {
    showQrScreen("Wi-Fi の設定  1 / 2", "WIFI:T:nopass;S:" + setupSsid + ";;",
                 {"スマホのカメラで", "左の QR を読み、", "出てきた Wi-Fi に", "つないでください"}, setupSsid.c_str());
}

// 設定の 2 歩目: スマホがつながったら、設定ページの開き方を案内する
void showSetupPageScreen() {
    const String url = "http://" + SETUP_IP.toString() + "/";
    showQrScreen("Wi-Fi の設定  2 / 2", url,
                 {"つながりました。", "設定のページが", "開かないときは", "左の QR を読んで", "ください"}, url.c_str());
}

// 操作ページの QR
void showControlQr() {
    const String url = "http://" + WiFi.localIP().toString() + "/";
    showQrScreen("操作ページ", url, {"家の Wi-Fi に", "つないだスマホで", "左の QR を読むと", "操作できます"}, url.c_str());
    qrUntilMs = millis() + QR_SHOW_MS;
    qrSecondsShown = -1;
}

// 操作ページの QR の画面に、顔へ戻るまでの残り秒数を出す(右の文章の下)。秒が変わったときだけ描き直す
void drawQrCountdown(uint32_t now) {
    const int seconds = static_cast<int>((qrUntilMs - now + 999) / 1000);
    if (seconds == qrSecondsShown) {
        return;
    }
    qrSecondsShown = seconds;
    char line[32];
    snprintf(line, sizeof(line), "あと %d 秒で", seconds);
    M5.Display.fillRect(180, 148, 140, 66, TFT_WHITE);
    M5.Display.setFont(&fonts::lgfxJapanGothic_16);
    M5.Display.setTextDatum(textdatum_t::top_left);
    M5.Display.setTextColor(TFT_RED, TFT_WHITE);
    M5.Display.drawString(line, 182, 150);
    M5.Display.setTextColor(TFT_BLACK, TFT_WHITE);
    M5.Display.drawString("元の画面に", 182, 172);
    M5.Display.drawString("戻ります", 182, 194);
}

void showMessage(const char* line1, const char* line2 = "") {
    stopAvatar();
    M5.Display.fillScreen(TFT_WHITE);
    drawCaption(line1, 160, 96);
    drawCaption(line2, 160, 124);
}

// 確認画面のボタンの位置(画面は 320×240)
constexpr int CONFIRM_BUTTON_TOP = 150, CONFIRM_BUTTON_BOTTOM = 220;
constexpr int CANCEL_LEFT = 16, CANCEL_RIGHT = 152, ERASE_LEFT = 168, ERASE_RIGHT = 304;

// 頭を長押ししたときの確認。画面の「消す」をタッチしたときだけ Wi-Fi の設定を消す
void showForgetConfirm() {
    stopAvatar();
    qrUntilMs = 0;
    M5.Display.fillScreen(TFT_WHITE);
    drawCaption("Wi-Fi の設定を消しますか?", 160, 36);
    drawCaption("消すと、最初の設定画面に戻ります", 160, 70);
    drawCaption("画面のボタンをタッチしてください", 160, 104);
    const int height = CONFIRM_BUTTON_BOTTOM - CONFIRM_BUTTON_TOP;
    M5.Display.drawRoundRect(CANCEL_LEFT, CONFIRM_BUTTON_TOP, CANCEL_RIGHT - CANCEL_LEFT, height, 10, TFT_BLACK);
    M5.Display.fillRoundRect(ERASE_LEFT, CONFIRM_BUTTON_TOP, ERASE_RIGHT - ERASE_LEFT, height, 10, TFT_RED);
    M5.Display.setFont(&fonts::lgfxJapanGothic_24);
    M5.Display.setTextDatum(textdatum_t::middle_center);
    M5.Display.setTextColor(TFT_BLACK, TFT_WHITE);
    M5.Display.drawString("やめる", (CANCEL_LEFT + CANCEL_RIGHT) / 2, (CONFIRM_BUTTON_TOP + CONFIRM_BUTTON_BOTTOM) / 2);
    M5.Display.setTextColor(TFT_WHITE, TFT_RED);
    M5.Display.drawString("消す", (ERASE_LEFT + ERASE_RIGHT) / 2, (CONFIRM_BUTTON_TOP + CONFIRM_BUTTON_BOTTOM) / 2);
    confirmUntilMs = millis() + CONFIRM_SHOW_MS;
}

constexpr uint32_t PAIR_CONFIRM_MS = 20000;

// 近くから申し込んできたコントローラーを登録するか、人に確かめる。展示会などで、ほかの人の機器を取り違えないため
void showPairConfirm(uint16_t id) {
    stopAvatar();
    qrUntilMs = 0;
    M5.Display.fillScreen(TFT_WHITE);
    // 16 px の字で 1 行に入るのは全角 19 字ほど(画面の幅 320 px)
    drawCaption("コントローラーを登録しますか?", 160, 24);
    char line[24];
    std::snprintf(line, sizeof(line), "ID %04X", id);
    drawCaption(line, 160, 56);
    drawCaption("コントローラーの画面の番号と", 160, 92);
    drawCaption("同じなら「登録」を押す", 160, 116);
    const int height = CONFIRM_BUTTON_BOTTOM - CONFIRM_BUTTON_TOP;
    M5.Display.drawRoundRect(CANCEL_LEFT, CONFIRM_BUTTON_TOP, CANCEL_RIGHT - CANCEL_LEFT, height, 10, TFT_BLACK);
    M5.Display.fillRoundRect(ERASE_LEFT, CONFIRM_BUTTON_TOP, ERASE_RIGHT - ERASE_LEFT, height, 10, TFT_DARKGREEN);
    M5.Display.setFont(&fonts::lgfxJapanGothic_24);
    M5.Display.setTextDatum(textdatum_t::middle_center);
    M5.Display.setTextColor(TFT_BLACK, TFT_WHITE);
    M5.Display.drawString("やめる", (CANCEL_LEFT + CANCEL_RIGHT) / 2, (CONFIRM_BUTTON_TOP + CONFIRM_BUTTON_BOTTOM) / 2);
    M5.Display.setTextColor(TFT_WHITE, TFT_DARKGREEN);
    M5.Display.drawString("登録", (ERASE_LEFT + ERASE_RIGHT) / 2, (CONFIRM_BUTTON_TOP + CONFIRM_BUTTON_BOTTOM) / 2);
    pairConfirmUntilMs = millis() + PAIR_CONFIRM_MS;
}

// 確認の画面を閉じて、いまのモードの画面に戻す
void restoreScreen() {
    if (mode == Mode::Setup && atomState != espnow_pad::State::Connected) {
        if (WiFi.softAPgetStationNum() > 0) {
            showSetupPageScreen();
        } else {
            showSetupJoinScreen();
        }
    } else {
        showFace();
    }
}

// ---------------------------------------------------------------- ページと API
void sendJsonState() {
    char json[400];
    snprintf(json, sizeof(json),
             "{\"drive\":%d,\"pitch\":%.1f,\"rev\":%d,\"spin\":%.1f,\"vmax\":%d,\"invert\":%d,\"smile\":%d,\"volt\":%.2f,\"atomBusy\":%d,\"atom\":%d,"
             "\"move\":%d,\"loop\":%d,\"bleft\":%d,\"turned\":%.0f,\"timeout\":%d,\"yaw\":%.1f,\"bias\":%.2f}",
             driveSigned, pitchTarget, spinReversed ? 1 : 0, spinDeg, speedMax, invertDirection ? 1 : 0, smileWhileWalking ? 1 : 0,
             M5StackChan.getBatteryVoltage(), atomIsSteering() ? 1 : 0, static_cast<int>(atomState), static_cast<int>(mover.phase()),
             mover.kind() == spin::Kind::Loop ? 1 : 0, buttonSpinLeft ? 1 : 0, mover.turned(), mover.timedOut() ? 1 : 0, bodyYaw.angle(),
             bodyYaw.bias());
    server.sendHeader("Cache-Control", "no-store");
    server.send(200, "application/json", json);
}

// スマホのジョイスティック。触れている間は 0.1 秒ごとに来る(止める見張りの合図も兼ねる)
void handleJoy() {
    const String press = server.arg("s");
    if (!press.isEmpty() && press == stoppedPress) {
        sendJsonState();
        return;
    }
    lastPhoneMs = millis();
    if (!atomIsSteering()) {
        setControl(server.arg("x").toInt(), server.arg("y").toInt(), Source::Phone);
    }
    sendJsonState();
}

// 止めるのは、どちらが操作していても受け付ける
void handleStop() {
    if (server.hasArg("s")) {
        stoppedPress = server.arg("s");
    }
    mover.stop(millis());
    stopControl();
    sendJsonState();
}

// ボタンで回る: m = once(くるっと 1 回転)/ loop(回り続ける。回っていれば止める)/ stop。left = ボタンで回る向き(0 = 右、1 = 左)
void handleSpin() {
    const uint32_t now = millis();
    if (server.hasArg("left")) {
        buttonSpinLeft = server.arg("left").toInt() != 0;
        prefs.putBool("bleft", buttonSpinLeft);
    }
    const String move = server.arg("m");
    if (move == "once") {
        spinOnce(now);
    } else if (move == "loop") {
        spinLoopToggle(now);
    } else if (move == "stop") {
        mover.stop(now);
    }
    sendJsonState();
}

// 旋回の角度(spin10 = 角度の 10 倍)、速さの上限(vmax [%])、旋回の向きの入れ替え(rev = 0 / 1)。送られたものだけ変える
void handleTune() {
    if (server.hasArg("rev")) {
        spinReversed = server.arg("rev").toInt() != 0;
        prefs.putBool("rev", spinReversed);
    }
    if (server.hasArg("spin10")) {
        spinDeg = car::clampSpinPitch(server.arg("spin10").toInt() / 10.0f);
        prefs.putInt("spin", static_cast<int>(std::lround(spinDeg * 10)));
    }
    if (server.hasArg("vmax")) {
        speedMax = car::clampSpeedMax(server.arg("vmax").toInt());
        prefs.putInt("vmax", speedMax);
    }
    updateTargets();
    applyDrive();
    updatePitch(millis(), true);
    sendJsonState();
}

void handleAtomForget() {
    espnow_pad::forget();
    if (activeSource == Source::Atom) {
        stopControl();
    }
    sendJsonState();
}

void handleInvert() {
    invertDirection = server.arg("on").toInt() != 0;
    prefs.putBool("invert", invertDirection);
    applyDrive();
    sendJsonState();
}

void handleSmile() {
    smileWhileWalking = server.arg("on").toInt() != 0;
    prefs.putBool("smile", smileWhileWalking);
    applyExpression();
    sendJsonState();
}

void handleForget() {
    stopControl();
    prefs.remove("ssid");
    prefs.remove("pass");
    server.send(200, "application/json", "{\"ok\":1}");
    restartAtMs = millis() + 800;
}

void handleScan() {
    const int count = WiFi.scanNetworks();
    String json = "[";
    for (int i = 0; i < count; i++) {
        String name = WiFi.SSID(i);
        if (name.isEmpty() || json.indexOf("\"" + name + "\"") >= 0) {
            continue;  // 名前のないものと重複は出さない
        }
        name.replace("\\", "\\\\");
        name.replace("\"", "\\\"");
        if (json.length() > 1) {
            json += ",";
        }
        json += "{\"ssid\":\"" + name + "\",\"lock\":" + (WiFi.encryptionType(i) == WIFI_AUTH_OPEN ? "0" : "1") + "}";
    }
    json += "]";
    WiFi.scanDelete();
    server.sendHeader("Cache-Control", "no-store");
    server.send(200, "application/json", json);
}

void handleSaveWifi() {
    const String ssid = server.arg("ssid");
    if (ssid.isEmpty()) {
        server.send(400, "application/json", "{\"error\":\"ssid\"}");
        return;
    }
    prefs.putString("ssid", ssid);
    prefs.putString("pass", server.arg("pass"));
    server.send(200, "application/json", "{\"ok\":1}");
    restartAtMs = millis() + 1200;
}

void startServer() {
    if (mode == Mode::Ready) {
        server.on("/", []() {
            if (qrUntilMs) {
                qrUntilMs = millis();  // 操作ページが開かれたら QR をしまって顔に戻す
            }
            server.send_P(200, "text/html; charset=utf-8", CONTROL_PAGE);
        });
        server.on("/api/joy", handleJoy);
        server.on("/api/stop", handleStop);
        server.on("/api/spin", handleSpin);
        server.on("/api/state", sendJsonState);
        server.on("/api/tune", handleTune);
        server.on("/api/atom/forget", handleAtomForget);
        server.on("/api/invert", handleInvert);
        server.on("/api/smile", handleSmile);
        server.on("/api/forget", handleForget);
        server.onNotFound([]() { server.send(404, "text/plain", "not found"); });
    } else {
        server.on("/", []() { server.send_P(200, "text/html; charset=utf-8", SETUP_PAGE); });
        server.on("/api/scan", handleScan);
        server.on("/api/wifi", HTTP_POST, handleSaveWifi);
        // スマホが「このWi-Fiはログインが必要か」を確かめに来る通信を、設定ページへ案内する
        server.onNotFound([]() {
            server.sendHeader("Location", "http://" + SETUP_IP.toString() + "/");
            server.send(302, "text/plain", "");
        });
    }
    server.begin();
}

// ---------------------------------------------------------------- Wi-Fi
// StopWatch の ESP-NOW を始める。Wi-Fi の省電力の待ち受けを止めて、受け取りの遅れと取りこぼしを減らす
void startEspNow() {
    WiFi.setSleep(false);
    espnow_pad::begin();
}

void startSetupMode() {
    mode = Mode::Setup;
    const uint64_t mac = ESP.getEfuseMac();
    char name[24];
    snprintf(name, sizeof(name), "Strider-%04X", static_cast<unsigned>((mac >> 32) & 0xFFFF));
    setupSsid = name;
    WiFi.mode(WIFI_AP_STA);  // STA は周囲の Wi-Fi を探すために残す
    WiFi.softAPConfig(SETUP_IP, SETUP_IP, IPAddress(255, 255, 255, 0));
    WiFi.softAP(setupSsid.c_str());
    dns.start(53, "*", SETUP_IP);
    startEspNow();
    startServer();
    showSetupJoinScreen();
}

bool connectHomeWifi(const String& ssid, const String& pass) {
    showMessage("Wi-Fi につないでいます", ssid.c_str());
    WiFi.mode(WIFI_STA);
    WiFi.setHostname(MDNS_NAME);
    WiFi.begin(ssid.c_str(), pass.c_str());
    const uint32_t started = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - started < CONNECT_TIMEOUT_MS) {
        delay(200);
    }
    return WiFi.status() == WL_CONNECTED;
}

void startReadyMode() {
    mode = Mode::Ready;
    startEspNow();
    MDNS.begin(MDNS_NAME);
    MDNS.addService("http", "tcp", 80);
    startServer();
    showControlQr();
}

}  // namespace

void setup() {
    M5StackChan.begin();
    M5.Display.setBrightness(120);
    avatar.setSpeechFont(&fonts::lgfxJapanGothic_16);
    Serial.begin(115200);
    Serial.printf("[strider] board=%d rotation=%d width=%d height=%d\n", static_cast<int>(M5.getBoard()),
                  M5.Display.getRotation(), M5.Display.width(), M5.Display.height());
    // 首(ヨー)を連続回転モードにして止めておく。電源を入れた直後はサーボがまだ応えず、モードの切り替え(初回の rotateX だけで行われる)が
    // 失敗したまま記録されて、あとで回せなくなる。起動から YAW_READY_MS 待ってから切り替える
    while (millis() < YAW_READY_MS) {
        delay(10);
    }
    M5StackChan.Motion.rotateX(0);
    prefs.begin("strider", false);
    invertDirection = prefs.getBool("invert", false);
    smileWhileWalking = prefs.getBool("smile", false);
    spinReversed = prefs.getBool("rev", false);
    buttonSpinLeft = prefs.getBool("bleft", false);
    spinDeg = car::clampSpinPitch(prefs.getInt("spin", static_cast<int>(std::lround(car::PITCH_MAX_DEG * 10))) / 10.0f);
    speedMax = car::clampSpeedMax(prefs.getInt("vmax", car::SPEED_MAX));
    pitchTarget = car::LEVEL_DEG;
    updatePitch(millis(), true);  // 起動したら、頭を水平にする(足が上がる)
    const String ssid = prefs.getString("ssid", "");
    if (!ssid.isEmpty() && connectHomeWifi(ssid, prefs.getString("pass", ""))) {
        startReadyMode();
    } else {
        if (!ssid.isEmpty()) {
            showMessage("Wi-Fi につながりません", "設定をやり直します");
            delay(2500);
        }
        startSetupMode();
    }
}

// StopWatch の値を取り込む。途切れたら、StopWatch が操作していれば止める
void handleAtom(uint32_t now) {
    espnow_pad::update(now);
    if (bubbleUntilMs && now >= bubbleUntilMs) {
        bubbleUntilMs = 0;
        if (avatarRunning) {
            avatar.setSpeechText("");
        }
    }
    uint16_t candidateId = 0;
    if (!pairConfirmUntilMs && !confirmUntilMs && !restartAtMs && espnow_pad::pairCandidate(candidateId)) {
        stopControl();
        showPairConfirm(candidateId);
    } else if (pairConfirmUntilMs) {
        const auto touch = M5.Touch.getDetail();
        const bool inButtonRow = touch.y >= CONFIRM_BUTTON_TOP && touch.y <= CONFIRM_BUTTON_BOTTOM;
        const bool accept = touch.wasPressed() && inButtonRow && touch.x >= ERASE_LEFT && touch.x <= ERASE_RIGHT;
        const bool cancel = touch.wasPressed() && inButtonRow && touch.x >= CANCEL_LEFT && touch.x <= CANCEL_RIGHT;
        const bool expired = static_cast<int32_t>(now - pairConfirmUntilMs) >= 0;
        if (accept || cancel || expired || !espnow_pad::pairCandidate(candidateId)) {
            if (accept) {
                espnow_pad::acceptCandidate();
            } else {
                espnow_pad::rejectCandidate(cancel);
            }
            pairConfirmUntilMs = 0;
            restoreScreen();
        }
    }
    const espnow_pad::State state = espnow_pad::state(now);
    if (state != atomState) {
        const bool wasConnected = atomState == espnow_pad::State::Connected;
        atomState = state;
        const bool busy = pairConfirmUntilMs || confirmUntilMs || restartAtMs;
        if (state == espnow_pad::State::Connected) {
            // StopWatch で操作できるなら、スマホ向けの QR(操作ページ・Wi-Fi の設定)は要らない。顔に戻す
            if (!avatarRunning && !busy) {
                showFace();
            }
            if (avatarRunning) {
                avatar.setSpeechText("StopWatch OK");
                bubbleUntilMs = now + BUBBLE_MS;
            }
        } else if (wasConnected && mode == Mode::Setup && avatarRunning && !busy) {
            restoreScreen();  // StopWatch が切れたら、Wi-Fi の設定の画面に戻す
        }
    }
    uint8_t once = 0, loopPresses = 0;
    espnow_pad::takePresses(once, loopPresses);
    if (once) {
        spinOnce(now);
    }
    for (uint8_t i = 0; i < loopPresses; i++) {
        spinLoopToggle(now);
    }
    int x = 0, y = 0;
    bool stop = false;
    if (espnow_pad::takeStick(x, y, stop)) {
        if (stop) {
            mover.stop(now);
        }
        if (stop) {
            x = y = 0;
        }
        const bool centred = abs(x) <= car::DEAD_ZONE && abs(y) <= car::DEAD_ZONE;
        if (activeSource == Source::Atom || !centred) {
            setControl(centred ? 0 : x, centred ? 0 : y, Source::Atom);  // 中心に戻したら、ほかの操作に譲る
        }
    }
    if (activeSource == Source::Atom && car::timedOut(now, espnow_pad::lastStickMs(), ATOM_TIMEOUT_MS)) {
        stopControl();
    }
}

void loop() {
    M5StackChan.update();
    server.handleClient();
    const uint32_t now = millis();
    handleAtom(now);
    updateYaw(now);
    updateSpin(now);
    updatePitch(now);

    if (mode == Mode::Setup) {
        dns.processNextRequest();
        // スマホが設定用 Wi-Fi につながったら 2 歩目の画面へ、切れたら 1 歩目へ戻す
        static bool phoneJoined = false;
        const bool joined = WiFi.softAPgetStationNum() > 0;
        if (joined != phoneJoined && !restartAtMs && !pairConfirmUntilMs) {
            phoneJoined = joined;
            if (joined) {
                showSetupPageScreen();
            } else {
                showSetupJoinScreen();
            }
        }
    } else {
        // スマホからの値が途切れたら止める(電波が切れた、ページを閉じた、など)
        if (activeSource == Source::Phone && car::timedOut(now, lastPhoneMs, COMMAND_TIMEOUT_MS)) {
            stopControl();
        }
        if (qrUntilMs && now >= qrUntilMs) {
            showFace();
        } else if (qrUntilMs) {
            drawQrCountdown(now);
        }
        // 頭の上をさわって離すと操作ページの QR を出す。さわり続けると「Wi-Fi の設定を消しますか」を出す
        static bool holdHandled = false;  // 長押しで確認を出したあと、指を離すまで次の操作を受け付けない
        if (confirmUntilMs) {
            const auto touch = M5.Touch.getDetail();
            const bool inButtonRow = touch.y >= CONFIRM_BUTTON_TOP && touch.y <= CONFIRM_BUTTON_BOTTOM;
            if (touch.wasPressed() && inButtonRow && touch.x >= ERASE_LEFT && touch.x <= ERASE_RIGHT && !restartAtMs) {
                confirmUntilMs = 0;
                prefs.remove("ssid");
                prefs.remove("pass");
                showMessage("Wi-Fi の設定を消しました", "再起動します");
                restartAtMs = now + 1500;
            } else if ((touch.wasPressed() && inButtonRow && touch.x >= CANCEL_LEFT && touch.x <= CANCEL_RIGHT) || now >= confirmUntilMs) {
                confirmUntilMs = 0;
                showFace();
            }
        } else if (!restartAtMs && !pairConfirmUntilMs) {
            const bool spinning = mover.phase() == spin::Phase::Ready || mover.phase() == spin::Phase::Turning;
            if (spinning && M5StackChan.TouchSensor.wasPressed()) {
                holdHandled = true;  // 回っている間に頭をさわったら止める(離したときに QR は出さない)
                mover.stop(now);
            } else if (M5StackChan.TouchSensor.pressedFor(FORGET_HOLD_MS) && !holdHandled) {
                holdHandled = true;
                stopControl();
                showForgetConfirm();
            } else if (M5StackChan.TouchSensor.wasReleased() && !holdHandled && !qrUntilMs) {
                showControlQr();
            }
        }
        if (holdHandled && !M5StackChan.TouchSensor.isPressed()) {
            holdHandled = false;
        }
    }

    if (restartAtMs && now >= restartAtMs) {
        M5StackChan.Motion.rotateX(0);
        delay(100);
        ESP.restart();
    }
    delay(5);
}

// コントローラーの入力。機器ごとに 1 つを選んでビルドする(platformio.ini の build_src_filter)。
//  - input_stopwatch.cpp: M5Stack StopWatch の画面のタッチと本体の傾き
#pragma once

namespace input {

struct Reading {
    int x = 0;          // -100(左)〜100(右)
    int y = 0;          // -100(後ろ)〜100(前)
    bool stop = false;  // 止める
};

/** M5.begin() のあとに 1 回呼ぶ。 */
void begin();

/** ボタンを見る。loop() で毎回、M5.update() のあとに呼ぶ。 */
void update();

/** いまの入力。0.05 秒ごとに呼ぶ。 */
Reading read();

/** 登録を消す操作(ボタンの長押し)があれば true を返し、取り消す。 */
bool takeForgetRequest();

/** 画面の上に出す、入力の方式の名前(なければ空)。 */
const char* modeLabel();

/** 画面の下に出す、確かめ用の値(左と右)。 */
void debugText(char* left, char* right, int size);

}  // namespace input

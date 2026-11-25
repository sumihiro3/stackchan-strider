// ぐるぐるの目。M5Stack-Avatar の目(Drawable)と差し替えて使う。
#pragma once

#include <Avatar.h>

namespace spiral_eyes {

/** 渦巻きの目。描くたびに時刻に合わせて回る。 */
class SpiralEye final : public m5avatar::Drawable {
public:
    /** @param isLeft 左目なら、右目と逆向きに回す */
    explicit SpiralEye(bool isLeft) : isLeft_(isLeft) {}

    /** 強さ。1 = 回っている間、2 = 回り続けたあと(速く、太く、大きく回す)。 */
    void setStrength(int strength) { strength_ = strength; }

    void draw(M5Canvas* canvas, m5avatar::BoundingRect rect, m5avatar::DrawContext* ctx) override;

private:
    bool isLeft_;
    int strength_ = 1;
};

/**
 * 顔の目を、ぐるぐるの目と元の目のあいだで切り替える。
 * @param strength 0 = 元の目、1 = ぐるぐる、2 = 強いぐるぐる
 */
void apply(m5avatar::Avatar& avatar, int strength);

}  // namespace spiral_eyes

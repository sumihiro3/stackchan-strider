// 車体の向き(上から見た回転)を数える計算。機器には依存しない(native の単体テストで使う)。
//
// ピボットでは、スタックチャンの頭(CoreS3、ジャイロ入り)はヨークで車体に留まっていて、首振りで回るのは土台のほう。
// そのため車体の向きは、頭のジャイロだけで数えられる(2026-11-25 実機: 首を 180° 回しても、ジャイロの向きは 0.1° しか変わらない)。
#pragma once

namespace yaw {

/** 角度を -180〜180° に収める。 */
float wrap180(float degrees);

/**
 * ジャイロの角速度を「上」の向きへ射影した、上から見た回転の速さ [°/s] を返す。
 * 頭が上を向いていても(うなずき 0〜30°)正しく出る。加速度がほぼ 0 のとき(落下中など)は gz を返す。
 * @param gx,gy,gz ジャイロ [°/s]
 * @param ax,ay,az 加速度 [g](静止時は上向きに 1 g)
 */
float verticalRate(float gx, float gy, float gz, float ax, float ay, float az);

/** ジャイロの回転の速さを積分して向きを出す。止まっている間に、ゼロ点のずれ(バイアス)を見積もって引く。 */
class Integrator {
public:
    /**
     * @param rateDegS 上から見た回転の速さ [°/s]
     * @param dtS 前の値からの時間 [s]
     * @param still 車が止まっている(首を回していない)ことが分かっているか。true の間だけバイアスを見積もり直す
     */
    void add(float rateDegS, float dtS, bool still);

    /** 向き [°](合計の角度)。 */
    float angle() const { return angle_; }

    /** いま見積もっているバイアス [°/s]。 */
    float bias() const { return bias_; }

private:
    float angle_ = 0;
    float bias_ = 0;
};

}  // namespace yaw

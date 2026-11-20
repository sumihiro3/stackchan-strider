#include "hid_parser.h"

namespace hid {
namespace {

constexpr uint16_t PAGE_GENERIC_DESKTOP = 0x01;
constexpr uint16_t USAGE_X = 0x30;
constexpr uint16_t USAGE_Y = 0x31;
constexpr int MAX_USAGES = 16;
constexpr int MAX_REPORT_IDS = 256;

uint32_t readUnsigned(const uint8_t* data, int size) {
    uint32_t value = 0;
    for (int i = 0; i < size; i++) {
        value |= static_cast<uint32_t>(data[i]) << (8 * i);
    }
    return value;
}

// 短いアイテムの値は、仕様では符号付き(Logical Minimum / Maximum)
int32_t readSigned(const uint8_t* data, int size) {
    const uint32_t value = readUnsigned(data, size);
    if (size == 0 || size == 4) {
        return static_cast<int32_t>(value);
    }
    const uint32_t sign = 1u << (8 * size - 1);
    return static_cast<int32_t>((value ^ sign) - sign);
}

}  // namespace

bool parseReportMap(const uint8_t* d, size_t length, GamepadLayout& out) {
    out = GamepadLayout{};
    // グローバルの状態
    uint16_t usagePage = 0;
    int32_t logicalMin = 0, logicalMax = 0;
    uint32_t logicalMaxRaw = 0;
    int logicalMaxSize = 0;
    uint32_t reportSize = 0, reportCount = 0;
    uint8_t reportId = 0;
    // ローカルの状態(Main アイテムのたびに消える)
    uint32_t usages[MAX_USAGES];
    int usageCount = 0;
    uint32_t usageMin = 0, usageMax = 0;
    bool hasRange = false;
    // 入力レポートの、レポート ID ごとのビット位置
    static uint16_t inputBits[MAX_REPORT_IDS];
    for (auto& bits : inputBits) {
        bits = 0;
    }

    size_t i = 0;
    while (i < length) {
        const uint8_t prefix = d[i];
        if (prefix == 0xFE) {  // 長いアイテム(使われないので読み飛ばす)
            if (i + 2 >= length) {
                break;
            }
            i += 3 + d[i + 1];
            continue;
        }
        const int size = (prefix & 0x03) == 3 ? 4 : (prefix & 0x03);
        if (i + 1 + size > length) {
            break;
        }
        const uint8_t* data = d + i + 1;
        const uint8_t type = (prefix >> 2) & 0x03;  // 0 = Main, 1 = Global, 2 = Local
        const uint8_t tag = prefix >> 4;
        const uint32_t value = readUnsigned(data, size);

        if (type == 1) {
            switch (tag) {
                case 0x0: usagePage = static_cast<uint16_t>(value); break;
                case 0x1: logicalMin = readSigned(data, size); break;
                case 0x2:
                    logicalMax = readSigned(data, size);
                    logicalMaxRaw = value;
                    logicalMaxSize = size;
                    break;
                case 0x7: reportSize = value; break;
                case 0x8: reportId = static_cast<uint8_t>(value); break;
                case 0x9: reportCount = value; break;
                default: break;
            }
        } else if (type == 2) {
            switch (tag) {
                case 0x0:
                    if (usageCount < MAX_USAGES) {
                        // 4 バイトの Usage は上位 16 ビットが Usage Page
                        usages[usageCount++] = size == 4 ? value : (static_cast<uint32_t>(usagePage) << 16) | value;
                    }
                    break;
                case 0x1: usageMin = value; hasRange = true; break;
                case 0x2: usageMax = value; hasRange = true; break;
                default: break;
            }
        } else if (type == 0) {
            if (tag == 0x8) {  // Input
                const bool constant = value & 0x01;
                const bool relative = value & 0x04;  // マウスの X/Y は相対値。スティックは絶対値
                // Logical Maximum を 0xFFFF のように書いて、符号付きでは負になってしまう機器がある。最小値より小さければ符号なしとみなす
                int32_t maxValue = logicalMax;
                if (maxValue < logicalMin && logicalMaxSize > 0 && logicalMaxSize < 4) {
                    maxValue = static_cast<int32_t>(logicalMaxRaw);
                }
                for (uint32_t n = 0; n < reportCount; n++) {
                    uint32_t usage = 0;
                    if (!constant) {
                        if (n < static_cast<uint32_t>(usageCount)) {
                            usage = usages[n];
                        } else if (hasRange && usageMin + n <= usageMax) {
                            usage = (static_cast<uint32_t>(usagePage) << 16) | (usageMin + n);
                        } else if (usageCount > 0) {
                            usage = usages[usageCount - 1];
                        }
                    }
                    AxisField* target = nullptr;
                    if ((usage >> 16) == PAGE_GENERIC_DESKTOP && !relative) {
                        if ((usage & 0xFFFF) == USAGE_X && !out.x.found) {
                            target = &out.x;
                        } else if ((usage & 0xFFFF) == USAGE_Y && !out.y.found) {
                            target = &out.y;
                        }
                    }
                    if (target && reportSize > 0 && reportSize <= 32) {
                        target->found = true;
                        target->reportId = reportId;
                        target->bitOffset = static_cast<uint16_t>(inputBits[reportId] + n * reportSize);
                        target->bitSize = static_cast<uint8_t>(reportSize);
                        target->logicalMin = logicalMin;
                        target->logicalMax = maxValue;
                    }
                }
                inputBits[reportId] = static_cast<uint16_t>(inputBits[reportId] + reportSize * reportCount);
            }
            usageCount = 0;
            hasRange = false;
        }
        i += 1 + size;
    }
    // X と Y が別のレポートに分かれていたら、使いにくいので失敗にする
    return out.x.found && out.y.found && out.x.reportId == out.y.reportId;
}

int readAxis(const AxisField& field, const uint8_t* report, size_t length) {
    if (!field.found || field.bitSize == 0) {
        return 0;
    }
    const uint32_t lastBit = static_cast<uint32_t>(field.bitOffset) + field.bitSize;
    if ((lastBit + 7) / 8 > length) {
        return 0;
    }
    uint32_t raw = 0;
    for (int b = 0; b < field.bitSize; b++) {
        const uint32_t bit = field.bitOffset + b;
        if (report[bit / 8] & (1u << (bit % 8))) {
            raw |= 1u << b;
        }
    }
    int64_t value = raw;
    if (field.logicalMin < 0 && field.bitSize < 32) {
        const uint32_t sign = 1u << (field.bitSize - 1);
        value = static_cast<int32_t>((raw ^ sign) - sign);
    }
    const int64_t lo = field.logicalMin, hi = field.logicalMax;
    if (hi <= lo) {
        return 0;
    }
    // 範囲の中央を 0 に、端を ±100 にする
    const int64_t scaled = ((value - lo) * 200 + (hi - lo) / 2) / (hi - lo) - 100;
    return static_cast<int>(scaled < -100 ? -100 : (scaled > 100 ? 100 : scaled));
}

}  // namespace hid

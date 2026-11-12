import cv2
import re
import numpy as np
import pytesseract
from ultralytics import YOLO
from collections import Counter

# ============================================================
# 馬拉松號碼布辨識：針對 11月6日.mov（960x720、30 FPS）設定
#
# 必要檔案（放在本程式同一個資料夾）：
#   11月6日.mov
#   bib-detector-nano.pt
#
# 輸出：
#   bib_detection_result.mp4
#   bib_numbers.txt
#
# 不使用人物偵測、不顯示人物 ID。
# 只在號碼布候選框通過外觀篩選，且數字辨識通過時才畫框。
# ============================================================

VIDEO_PATH = "11月6日.mov"
MODEL_PATH = "bib-detector-nano.pt"
OUTPUT_PATH = "bib_detection_result.mp4"
LOG_PATH = "bib_numbers.txt"

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 偵測門檻：提高門檻以減少衣服文字等誤框
DETECT_CONF = 0.45
IMAGE_SIZE = 1280

# 本段影片號碼布以黃／橘色為主；只作為輔助篩選，不單獨決定是否為號碼布
MIN_BIB_COLOR_RATIO = 0.035

# 號碼格式：本次影片以 3～5 位數為主
MIN_DIGITS = 3
MAX_DIGITS = 5

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


def has_bib_color(crop):
    """檢查候選區域是否含有一定比例的黃／橘色號碼布底色。"""
    if crop is None or crop.size == 0:
        return False, 0.0

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # 黃色到橘色範圍；排除低飽和度背景
    mask1 = cv2.inRange(hsv, np.array([8, 65, 65]), np.array([38, 255, 255]))
    ratio = float(np.count_nonzero(mask1)) / float(mask1.size)

    return ratio >= MIN_BIB_COLOR_RATIO, ratio


def preprocess_variants(crop):
    """產生多種 OCR 影像，處理遠距離、陰影及對比不足。"""
    h, w = crop.shape[:2]
    if h < 8 or w < 8:
        return []

    # 保留比例放大，讓小號碼有足夠像素
    scale = max(3.0, min(6.0, 220.0 / max(1, h)))
    enlarged = cv2.resize(
        crop, None, fx=scale, fy=scale,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)

    # 對比增強
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)

    # 二值化版本
    _, otsu = cv2.threshold(
        contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 黑字白底／白字黑底兩種方向都交給 OCR 判斷
    return [contrast, otsu, cv2.bitwise_not(otsu)]


def ocr_one(image):
    config = (
        "--oem 3 --psm 7 "
        "-c tessedit_char_whitelist=0123456789 "
        "-c classify_bln_numeric_mode=1"
    )

    data = pytesseract.image_to_data(
        image,
        config=config,
        output_type=pytesseract.Output.DICT
    )

    found = []
    for raw, conf in zip(data["text"], data["conf"]):
        digits = re.sub(r"[^0-9]", "", raw or "")
        try:
            confidence = float(conf)
        except (TypeError, ValueError):
            confidence = -1

        if (
            MIN_DIGITS <= len(digits) <= MAX_DIGITS
            and confidence >= 35
        ):
            found.append((digits, confidence))

    if not found:
        return "", 0.0

    # 取 OCR 信心分數最高的數字片段
    return max(found, key=lambda x: x[1])


def read_bib_number(crop):
    variants = preprocess_variants(crop)
    if not variants:
        return "", 0.0

    results = []
    for image in variants:
        number, confidence = ocr_one(image)
        if number:
            results.append((number, confidence))

    if not results:
        return "", 0.0

    # 優先採用至少兩種前處理結果一致的號碼
    counts = Counter(number for number, _ in results)
    best_number, votes = counts.most_common(1)[0]

    matching_scores = [
        score for number, score in results if number == best_number
    ]
    best_score = max(matching_scores)

    if votes >= 2:
        return best_number, best_score

    # 只有一種影像讀出號碼時，提高信心門檻
    if best_score >= 75:
        return best_number, best_score

    return "", 0.0


def clamp_box(box, frame_width, frame_height):
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, min(x1, frame_width - 1))
    y1 = max(0, min(y1, frame_height - 1))
    x2 = max(0, min(x2, frame_width))
    y2 = max(0, min(y2, frame_height))
    return x1, y1, x2, y2


def main():
    import os

    if not os.path.isfile(VIDEO_PATH):
        raise FileNotFoundError(f"找不到影片：{VIDEO_PATH}")

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"找不到模型：{MODEL_PATH}\n"
            "請把 bib-detector-nano.pt 放在本程式同一個資料夾。"
        )

    if not os.path.isfile(TESSERACT_PATH):
        raise FileNotFoundError(
            "找不到 Tesseract，請確認安裝路徑：\n" + TESSERACT_PATH
        )

    print("載入號碼布偵測模型……")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片：{VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("無法建立輸出影片，請檢查輸出資料夾權限。")

    log_rows = []
    frame_index = 0
    detected_count = 0

    print(f"影片尺寸：{width}x{height}，FPS：{fps:.2f}，總影格：{total}")
    print("開始辨識……")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_index += 1
        timestamp = frame_index / fps

        # 只呼叫號碼布專用模型，不使用人物模型
        prediction = model.predict(
            source=frame,
            imgsz=IMAGE_SIZE,
            conf=DETECT_CONF,
            iou=0.45,
            max_det=30,
            verbose=False
        )

        output_frame = frame.copy()

        for result in prediction:
            if result.boxes is None:
                continue

            for box in result.boxes:
                x1, y1, x2, y2 = clamp_box(
                    box.xyxy[0].tolist(), width, height
                )

                bw = x2 - x1
                bh = y2 - y1

                if bw < 18 or bh < 12:
                    continue

                # 號碼布通常是橫向長方形；排除過窄、過高的文字區域
                aspect = bw / float(bh)
                if aspect < 1.05 or aspect > 3.8:
                    continue

                crop = frame[y1:y2, x1:x2]
                color_ok, color_ratio = has_bib_color(crop)

                # 本影片的號碼布底色以黃／橘為主。
                # 顏色只作為輔助條件，避免把一般衣服印字當成號碼布。
                if not color_ok:
                    continue

                # OCR 時略微向內裁切，減少衣服邊緣及背景干擾
                pad_x = max(1, int(bw * 0.04))
                pad_y = max(1, int(bh * 0.04))
                ix1 = min(x2 - 1, x1 + pad_x)
                iy1 = min(y2 - 1, y1 + pad_y)
                ix2 = max(ix1 + 1, x2 - pad_x)
                iy2 = max(iy1 + 1, y2 - pad_y)

                number, ocr_conf = read_bib_number(
                    frame[iy1:iy2, ix1:ix2]
                )

                # 只輸出有辨識到有效數字的號碼布
                if not number:
                    continue

                detected_count += 1
                log_rows.append(
                    f"{timestamp:.2f}s\t{number}\t"
                    f"OCR={ocr_conf:.1f}\tBibColor={color_ratio:.3f}"
                )

                cv2.rectangle(
                    output_frame, (x1, y1), (x2, y2),
                    (0, 255, 0), 2
                )

                label = f"{number}  OCR:{ocr_conf:.0f}%"
                label_y = max(25, y1 - 9)

                (tw, th), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
                )
                cv2.rectangle(
                    output_frame,
                    (x1, max(0, label_y - th - 8)),
                    (min(width - 1, x1 + tw + 8), label_y + baseline),
                    (0, 180, 0), -1
                )
                cv2.putText(
                    output_frame, label,
                    (x1 + 4, label_y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2,
                    cv2.LINE_AA
                )

        writer.write(output_frame)

        if frame_index % 30 == 0:
            print(
                f"進度：{frame_index}/{total} "
                f"({frame_index / max(1, total) * 100:.1f}%)"
            )

    cap.release()
    writer.release()

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("時間\t辨識號碼\t辨識信心\t號碼布顏色比例\n")
        f.write("\n".join(log_rows))

    print("\n辨識完成")
    print(f"輸出影片：{OUTPUT_PATH}")
    print(f"辨識紀錄：{LOG_PATH}")
    print(f"通過篩選的辨識次數：{detected_count}")


if __name__ == "__main__":
    main()

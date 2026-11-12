
import cv2
import pytesseract
import re
import os
import time

# =========================
# 基本設定
# =========================

VIDEO_PATH = "OP.mp4"

# Tesseract 安裝位置
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# OCR 每隔幾秒執行一次
OCR_INTERVAL = 2

# 只接受 3～5 位數字
MIN_DIGITS = 3
MAX_DIGITS = 5

# 圖片放大倍率
SCALE = 1.25

# =========================
# 開啟影片
# =========================

if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(f"找不到影片：{VIDEO_PATH}")

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("影片開啟失敗")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_path = "OP_辨識結果.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height)
)

# 儲存最近一次辨識結果
detections = []

last_ocr_time = 0
frame_number = 0

print("開始影片數字辨識...")
print("按 Q 結束，按 SPACE 暫停 / 繼續")

# =========================
# 影片辨識主迴圈
# =========================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    current_time = frame_number / fps

    # =========================
    # 定時執行 OCR
    # =========================

    now = time.monotonic()

    if now - last_ocr_time >= OCR_INTERVAL:

        last_ocr_time = now

        # 影像前處理
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        enlarged = cv2.resize(
            gray,
            None,
            fx=SCALE,
            fy=SCALE,
            interpolation=cv2.INTER_CUBIC
        )

        # 二值化
        _, processed = cv2.threshold(
            enlarged,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # 只允許辨識數字
        config = (
            "--oem 3 --psm 11 "
            "-c tessedit_char_whitelist=0123456789"
        )

        data = pytesseract.image_to_data(
            processed,
            config=config,
            output_type=pytesseract.Output.DICT
        )

        detections = []

        for i, text in enumerate(data["text"]):

            text = text.strip()

            # 必須完全是數字
            if not re.fullmatch(r"\d+", text):
                continue

            # 過濾太短或太長的數字
            if not (MIN_DIGITS <= len(text) <= MAX_DIGITS):
                continue

            try:
                confidence = float(data["conf"][i])
            except ValueError:
                continue

            # 過濾低信心結果
            if confidence < 40:
                continue

            # 還原原始影片座標
            x = int(data["left"][i] / SCALE)
            y = int(data["top"][i] / SCALE)
            w = int(data["width"][i] / SCALE)
            h = int(data["height"][i] / SCALE)

            detections.append({
                "text": text,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "confidence": confidence
            })

        # 印出當次辨識結果
        numbers = [d["text"] for d in detections]

        print(
            f"\n時間：{current_time:.1f} 秒"
            f"\n辨識結果：{numbers}"
        )

    # =========================
    # 顯示辨識框
    # =========================

    display = frame.copy()

    for d in detections:

        x = d["x"]
        y = d["y"]
        w = d["w"]
        h = d["h"]
        text = d["text"]

        # 綠色框
        cv2.rectangle(
            display,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # 數字標籤
        cv2.putText(
            display,
            text,
            (x, max(y - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # 顯示影片時間
    cv2.putText(
        display,
        f"Time: {current_time:.1f}s",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2
    )

    cv2.imshow("Marathon Bib Number Recognition", display)

    # 寫入輸出影片
    writer.write(display)

    # 按鍵控制
    key = cv2.waitKey(max(1, int(1000 / fps))) & 0xFF

    if key == ord("q"):
        break

    elif key == ord(" "):
        while True:
            key = cv2.waitKey(0) & 0xFF

            if key == ord(" "):
                break

            elif key == ord("q"):
                cap.release()
                writer.release()
                cv2.destroyAllWindows()
                raise SystemExit

# =========================
# 結束
# =========================

cap.release()
writer.release()
cv2.destroyAllWindows()

print("\n影片辨識結束")
print(f"辨識結果影片：{output_path}")

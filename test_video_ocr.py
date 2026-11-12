
import cv2
import pytesseract
import re

# Tesseract 執行檔
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

VIDEO_PATH = "OP.mp4"

# 每隔 2 秒辨識一次
INTERVAL = 2

# 純數字 OCR 設定
CONFIG = (
    "--oem 3 "
    "--psm 11 "
    "-c tessedit_char_whitelist=0123456789"
)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("影片讀取失敗！")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

duration = total_frames / fps

print(f"影片長度：{duration:.2f} 秒")
print(f"FPS：{fps}")
print("-" * 40)

current_time = 0

while current_time < duration:

    # 跳到指定時間
    cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)

    ret, frame = cap.read()

    if not ret:
        break

    # 轉灰階
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 放大圖片
    gray = cv2.resize(
        gray,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    # OCR
    result = pytesseract.image_to_string(
        gray,
        config=CONFIG
    )

    # 只保留數字
    digits = re.findall(r"[0-9]+", result)

    print(f"\n時間：{current_time:.1f} 秒")
    print("辨識結果：", digits)

    # 顯示影片畫面
    cv2.imshow("Video OCR Test", frame)

    if cv2.waitKey(100) & 0xFF == ord("q"):
        break

    current_time += INTERVAL

cap.release()
cv2.destroyAllWindows()

print("\n測試結束")

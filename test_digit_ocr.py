import cv2
import pytesseract
import re

# Tesseract 執行檔位置
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# 讀取圖片（先放一張號碼布圖片）
IMAGE_PATH = "bib.jpg"

img = cv2.imread(IMAGE_PATH)

if img is None:
    print("圖片讀取失敗！")
    exit()

# 灰階處理
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 放大圖片
gray = cv2.resize(
    gray, None, fx=2, fy=2,
    interpolation=cv2.INTER_CUBIC
)

# Tesseract 純數字設定
config = (
    "--oem 3 "
    "--psm 7 "
    "-c tessedit_char_whitelist=0123456789"
)

# OCR 辨識
result = pytesseract.image_to_string(
    gray,
    config=config
)

# 只保留數字
digits = re.sub(r"[^0-9]", "", result)

print("原始辨識結果：", repr(result))
print("純數字結果：", digits)

cv2.imshow("Bib Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
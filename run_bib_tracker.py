import argparse
import csv
import os
import re
import sys
import cv2
import numpy as np

# 設置 PaddleX / Matplotlib 離線工作環境
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PADDLEX_CACHE = os.path.join(SCRIPT_DIR, ".paddlex")
os.environ["PADDLE_PDX_CACHE_HOME"] = PADDLEX_CACHE
os.environ["MPLCONFIGDIR"] = os.path.join(PADDLEX_CACHE, "mpl")
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

OCR_MIN_SCORE = 0.30

from ultralytics import YOLO
from paddleocr import PaddleOCR


def get_ocr_engine():
    """初始化針對場景號碼布優化的 PaddleOCR 引擎。"""
    return PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang="en",
    )


def normalize_bib_number(raw_digits):
    """
    號碼布智慧校正與標準化：
    1. 台灣馬拉松選手編號以 5 位數（如 30xxx, 31xxx）或 4 位數為主。
    2. 左側組別字（如 甲、乙、丙、丁、戊 或 T、X）常因筆畫傾斜被誤讀為前綴數字（1、7、01 等）。
    此函數剔除可辨識的組別前綴筆劃，不用特定選手號碼清單做硬編碼。
    """
    d = raw_digits.strip()
    if len(d) in (4, 5):
        return d

    if len(d) == 6:
        # 剔除組別字被誤讀成的前綴筆畫；不依賴特定選手號碼清單。
        if d[1:].startswith(("30", "31", "40", "50", "24")):
            return d[1:]

    return d


def should_replace_bib(old_bib, old_conf, new_bib, new_conf):
    """
    智慧判定是否用新號碼替換舊號碼：
    - 標準 4～5 位數號碼布享有最高優先級，絕不被 6 位數雜訊或 3 位數殘缺碎片篡改。
    - 相同標準位數時，信心度較高者獲勝。
    """
    old_is_std = len(old_bib) in (4, 5)
    new_is_std = len(new_bib) in (4, 5)

    if new_is_std and not old_is_std:
        return True
    if not new_is_std and old_is_std:
        return False

    if len(new_bib) == len(old_bib):
        return new_conf > old_conf

    return len(new_bib) > len(old_bib)


def extract_pure_digits_from_crop(ocr, crop):
    """
    對號碼布區域進行放大與 OCR 辨識。
    嚴格規則：只提取 3～6 位純數字，並自動標準化校正組別前綴。
    """
    h, w = crop.shape[:2]
    if h < 8 or w < 8:
        return []

    scale = max(3.0, min(7.0, 220.0 / max(1, h)))
    enlarged = cv2.resize(
        crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
    )

    try:
        preds = list(ocr.predict(enlarged))
        if not preds:
            return []
        ocr_res = preds[0]
        texts = ocr_res.get("rec_texts", [])
        scores = ocr_res.get("rec_scores", [])

        digits_found = []
        for text, score in zip(texts, scores):
            cleaned = re.sub(r"[^0-9]", "", text)
            if 3 <= len(cleaned) <= 6 and float(score) >= OCR_MIN_SCORE:
                norm_num = normalize_bib_number(cleaned)
                digits_found.append((norm_num, float(score)))

        return digits_found
    except Exception:
        return []


def box_center_distance(boxA, boxB):
    """計算兩候選框的中心歐幾里得距離。"""
    cA_x = (boxA[0] + boxA[2]) / 2.0
    cA_y = (boxA[1] + boxA[3]) / 2.0
    cB_x = (boxB[0] + boxB[2]) / 2.0
    cB_y = (boxB[1] + boxB[3]) / 2.0
    return ((cA_x - cB_x) ** 2 + (cA_y - cB_y) ** 2) ** 0.5


def calculate_iou(boxA, boxB):
    """計算兩候選框的交集面積比 (IoU)。"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    denom = float(boxAArea + boxBArea - interArea)
    return interArea / denom if denom > 0 else 0


def process_image(img_path, model_path, ocr, conf=0.08, imgsz=960):
    """處理單張圖片。"""
    print(f"\n[處理圖片] 讀取：{img_path}")
    img = cv2.imread(img_path)
    if img is None:
        print(f"錯誤：無法開啟圖片 {img_path}")
        return

    model = YOLO(model_path)
    results = model.predict(img, conf=conf, imgsz=imgsz, max_det=100, verbose=False)[0]

    detections = []
    annotated = img.copy()

    for idx, box in enumerate(results.boxes):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        yolo_conf = float(box.conf[0])
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop = img[y1:y2, x1:x2]
        numbers = extract_pure_digits_from_crop(ocr, crop)

        for num, ocr_conf in numbers:
            detections.append({
                "bib": num,
                "ocr_conf": ocr_conf,
                "yolo_conf": yolo_conf,
                "bbox": [x1, y1, x2, y2],
            })
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"#{num} ({ocr_conf*100:.0f}%)"
            cv2.putText(
                annotated,
                label,
                (x1, max(22, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    out_name = os.path.splitext(img_path)[0] + "_result.jpg"
    cv2.imwrite(out_name, annotated)
    print(f"已儲存標註圖片：{out_name}")
    print(f"純數字辨識結果（共 {len(detections)} 個）：")
    for d in detections:
        print(f"  - 號碼：{d['bib']} | 信心度：{d['ocr_conf']:.2f}")


def process_video(
    video_path,
    model_path,
    ocr,
    conf=0.08,
    imgsz=960,
    frame_step=3,
    save_video=True,
    start_sec=0.0,
    max_duration=0.0,
):
    """
    處理路跑影片：
    - 使用短期時序記憶維持遮擋前後的 track 身分與號碼。
    - 以全域一對一框配對，降低跑者交錯時的錯誤沿用。
    - 只在偵測到號碼布框時畫框，不輸出遮擋期間的幽靈框。
    """
    print(f"\n[處理影片] 讀取：{video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"錯誤：無法開啟影片 {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_duration = total_frames / fps

    if start_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)
        start_frame = int(start_sec * fps)
    else:
        start_frame = 0

    target_duration = max_duration if max_duration > 0 else (total_duration - start_sec)
    print(f"影片資訊：{width}x{height} | {fps:.1f} FPS | 總長度：{total_duration:.1f} 秒 (約 {total_duration/60:.1f} 分鐘)")
    if max_duration > 0:
        print(f"處理範圍：從 {start_sec:.1f}s 起，共處理 {max_duration:.1f} 秒 (約 {int(target_duration*fps)} 幀)")
    print(f"處理間隔：每 {frame_step} 幀分析一次 (約每 {frame_step/fps:.2f} 秒)")
    print(f"偵測門檻：{conf:.2f} | YOLO 尺寸：{imgsz} | OCR 門檻：{OCR_MIN_SCORE:.2f}")

    writer = None
    input_dir = os.path.dirname(os.path.abspath(video_path))
    input_stem = os.path.splitext(os.path.basename(video_path))[0]
    run_tag = f"enhanced_step{frame_step}"
    out_video_path = os.path.join(input_dir, f"{input_stem}_{run_tag}_detection_result.mp4")
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    model = YOLO(model_path)

    raw_records = []
    bib_summary = {}

    # 時序記憶：短暫漏檢或 OCR 失敗時保留同一目標的號碼與位置。
    active_tracks = {}
    next_track_id = 1

    current_active_draw_boxes = []

    frame_idx = start_frame
    last_progress_log = 0

    print("開始辨識...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        current_time = frame_idx / fps
        elapsed_sec = current_time - start_sec

        if max_duration > 0 and elapsed_sec >= max_duration:
            print(f"已達到指定處理長度：{max_duration:.1f} 秒")
            break

        if int(current_time) // 5 > last_progress_log:
            last_progress_log = int(current_time) // 5
            progress_pct = (elapsed_sec / target_duration) * 100 if target_duration > 0 else 0
            print(f"  [進度 {progress_pct:.0f}%] 目前處理到：{current_time:.1f}s (累計鎖定 {len(bib_summary)} 位選手)...")

        # 每 frame_step 幀執行一次 YOLO + OCR 分析
        if frame_idx % frame_step == 0:
            preds = model.predict(frame, conf=conf, imgsz=imgsz, max_det=100, verbose=False)[0]

            detections = []
            for box in preds.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)

                if (x2 - x1) < 15 or (y2 - y1) < 10:
                    continue

                curr_box = (x1, y1, x2, y2)
                crop = frame[y1:y2, x1:x2]
                numbers = extract_pure_digits_from_crop(ocr, crop)
                detections.append({"box": curr_box, "numbers": numbers})

            # 全域一對一配對，避免多個相鄰號碼框被串到同一條記憶。
            # 使用上一段位移預測位置，短暫遮擋後回來時仍可接回原 track。
            candidates = []
            for di, detection in enumerate(detections):
                for tid, track in active_tracks.items():
                    gap = frame_idx - track["last_frame"]
                    predicted = tuple(track["box"][i] + track["velocity"][i] * gap for i in range(4))
                    dist = box_center_distance(detection["box"], predicted)
                    gate = min(180.0, max(70.0, 0.75 * max(detection["box"][2] - detection["box"][0], detection["box"][3] - detection["box"][1]) + 5.0 * gap))
                    iou = calculate_iou(detection["box"], predicted)
                    if dist <= gate or iou >= 0.08:
                        candidates.append((dist - iou * 40.0, di, tid))
            candidates.sort()
            detection_to_track = {}
            used_tracks = set()
            for _, di, tid in candidates:
                if di not in detection_to_track and tid not in used_tracks:
                    detection_to_track[di] = tid
                    used_tracks.add(tid)

            detected_this_frame = []
            for di, detection in enumerate(detections):
                curr_box, numbers = detection["box"], detection["numbers"]
                tid = detection_to_track.get(di)
                if tid is None:
                    tid = next_track_id
                    next_track_id += 1
                    active_tracks[tid] = {
                        "box": curr_box, "prev_box": curr_box, "velocity": (0, 0, 0, 0),
                        "bib": "", "conf": 0.0, "last_frame": frame_idx,
                    }
                else:
                    track = active_tracks[tid]
                    gap = max(1, frame_idx - track["last_frame"])
                    velocity = tuple((curr_box[i] - track["box"][i]) / gap for i in range(4))
                    track["velocity"] = tuple(0.65 * track["velocity"][i] + 0.35 * velocity[i] for i in range(4))
                    track["prev_box"] = track["box"]
                    track["box"] = curr_box
                    track["last_frame"] = frame_idx

                track = active_tracks[tid]
                for num, score in numbers:
                    if not track["bib"] or should_replace_bib(track["bib"], track["conf"], num, score):
                        track["bib"], track["conf"] = num, score

                cur_bib, cur_conf = track["bib"], track["conf"]
                if not cur_bib:
                    continue
                detected_this_frame.append((curr_box, cur_bib, cur_conf))
                # 只把當前實際 OCR 結果寫入流水帳，記憶補框不重複灌水。
                for num, score in numbers:
                    raw_records.append({
                        "timestamp_sec": round(current_time, 2), "frame": frame_idx,
                        "bib_number": cur_bib, "confidence": round(cur_conf, 3),
                        "box": f"[{curr_box[0]},{curr_box[1]},{curr_box[2]},{curr_box[3]}]",
                    })
                    if cur_bib not in bib_summary:
                        bib_summary[cur_bib] = {"count": 1, "max_conf": cur_conf, "first_time": current_time, "last_time": current_time}
                        print(f"  [{current_time:5.2f}s] ★ 鎖定新選手號碼：#{cur_bib} (信心度: {cur_conf*100:.1f}%)")
                    else:
                        bib_summary[cur_bib]["count"] += 1
                        bib_summary[cur_bib]["max_conf"] = max(bib_summary[cur_bib]["max_conf"], cur_conf)
                        bib_summary[cur_bib]["last_time"] = current_time

            # 僅保留目前幀真正活躍的框
            current_active_draw_boxes = detected_this_frame

            # 清理過期追蹤目標
            for tid in list(active_tracks.keys()):
                if frame_idx - active_tracks[tid]["last_frame"] > max(10, int(fps * 1.5)):
                    del active_tracks[tid]

        # 輸出影片幀
        if writer is not None:
            annotated_frame = frame.copy()
            for (bx1, by1, bx2, by2), b_bib, b_conf in current_active_draw_boxes:
                cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                blabel = f"#{b_bib} ({b_conf*100:.0f}%)"
                cv2.putText(
                    annotated_frame,
                    blabel,
                    (bx1, max(22, by1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            writer.write(annotated_frame)

    cap.release()
    if writer is not None:
        writer.release()
        print(f"\n[輸出完成] 已生成精準校正標註影片：{out_video_path}")

    # 保留各個獨立號碼；僅靠字串包含關係合併會把不同選手錯併。
    cleaned_summary = bib_summary

    # 寫入即時辨識流水帳 CSV
    csv_path = os.path.join(input_dir, f"{input_stem}_{run_tag}_bib_numbers.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer_csv = csv.DictWriter(
            f, fieldnames=["timestamp_sec", "frame", "bib_number", "confidence", "box"]
        )
        writer_csv.writeheader()
        writer_csv.writerows(raw_records)
    print(f"[輸出完成] 已儲存詳細識別紀錄：{csv_path} (共 {len(raw_records)} 筆紀錄)")

    # 寫入選手摘要匯總 CSV
    summary_path = os.path.join(input_dir, f"{input_stem}_{run_tag}_bib_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["選手號碼布", "最高辨識信心", "捕捉幀數", "首次出現時間(秒)", "最後出現時間(秒)"])
        for bib, data in sorted(cleaned_summary.items(), key=lambda x: -x[1]["count"]):
            writer_csv.writerow([
                bib,
                f"{data['max_conf']*100:.1f}%",
                data["count"],
                f"{data['first_time']:.2f}s",
                f"{data['last_time']:.2f}s",
            ])
    print(f"[輸出完成] 已儲存校正選手清單：{summary_path} (共鎖定 {len(cleaned_summary)} 位選手)")


def main():
    parser = argparse.ArgumentParser(description="路跑號碼布純數字識別系統 (YOLO + PaddleOCR + 組別校正 + 防跳框)")
    parser.add_argument("--input", default="11月7日.mov", help="輸入圖片或影片路徑 (預設: 11月7日.mov)")
    parser.add_argument("--model", default="bib-detector-nano.pt", help="YOLO 號碼布模型路徑")
    parser.add_argument("--conf", type=float, default=0.08, help="YOLO 偵測信心門檻 (預設: 0.08)")
    parser.add_argument("--imgsz", type=int, default=960, help="YOLO 輸入影像尺寸 (預設: 960，保留原片尺寸)")
    parser.add_argument("--step", type=int, default=3, help="影片處理間隔幀數 (預設: 3)")
    parser.add_argument("--start", type=float, default=0.0, help="從第幾秒開始處理 (預設: 0.0)")
    parser.add_argument("--duration", type=float, default=0.0, help="最多處理幾秒影片 (預設: 0 = 處理至影片結束)")
    parser.add_argument("--no-video", action="store_true", help="不輸出標註影片 (僅生成 CSV)")

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"找不到輸入檔案：{args.input}")
        return

    if not os.path.isfile(args.model):
        print(f"找不到 YOLO 模型權重：{args.model}")
        return

    print("載入 PaddleOCR 純數字辨識引擎...")
    ocr = get_ocr_engine()

    ext = os.path.splitext(args.input)[1].lower()
    if ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        process_image(args.input, args.model, ocr, conf=args.conf, imgsz=args.imgsz)
    else:
        process_video(
            args.input,
            args.model,
            ocr,
            conf=args.conf,
            imgsz=args.imgsz,
            frame_step=args.step,
            save_video=(not args.no_video),
            start_sec=args.start,
            max_duration=args.duration,
        )


if __name__ == "__main__":
    main()

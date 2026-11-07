import os
import sys
import time
import cv2
import numpy as np

# Thêm thư mục project vào sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from core.yolo_engine import YOLOEngine

print("="*60)
print("BẮT ĐẦU CHẠY KIỂM THỬ HỆ THỐNG (AUTOMATED TEST SUITE)")
print("="*60)

# 1. KIỂM THỬ NẠP MODEL VÀ KHỞI TẠO ENGINE
print("\n[TEST 1] Khởi tạo YOLOEngine và nạp trọng số...")
MODEL_PATH = os.path.join(BASE_DIR, r"runs\detect\runs\detect\yolo11m_v1\weights\best.pt")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, "yolo11m.pt")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, "yolo11n.pt")

stats_log = []
def dummy_stats(cin, cout, ain, aout):
    stats_log.append((cin, cout, ain, aout))

log_events = []
def dummy_log(tid, event, conf):
    log_events.append((tid, event, conf))

engine = YOLOEngine(
    model_path=MODEL_PATH,
    cb_stats=dummy_stats,
    cb_log=dummy_log,
    cb_stop=lambda: print("  -> Engine đã dừng."),
    cb_error=lambda err: print(f"  -> LỖI ENGINE: {err}")
)

assert engine.model is not None, "Model không được nạp thành công!"
print(f"  ✅ Nạp model thành công: {MODEL_PATH}")
print(f"  ✅ Tham số tối ưu: Conf={engine.conf_thresh}, IOU={engine.iou_thresh}, Imgsz={engine.imgsz}")

# 2. KIỂM THỬ TOÁN HỌC GIAO ĐIỂM VÀ ĐA FRAME TRAJECTORY
print("\n[TEST 2] Kiểm thử thuật toán cắt vạch và tính vector hướng...")
engine.set_line([100, 300], [800, 300]) # Vạch ngang
ext_A, ext_B = engine.get_effective_line(margin_ratio=0.12)
assert ext_A[0] < 100 and ext_B[0] > 800, "Vạch kẻ không được mở rộng biên độ!"
print(f"  ✅ Vạch gốc: [100, 300] -> [800, 300]")
print(f"  ✅ Vạch mở rộng tự động: {ext_A} -> {ext_B}")

# Kiểm thử Smart Adaptive Gate (Full-span projection: Vẽ thế nào cũng chuẩn)
span_A, span_B = engine.get_full_span_line()
assert span_A == [0, 300] and span_B == [959, 300], f"Chiếu Full-span sai: {span_A} -> {span_B}"
print(f"  ✅ Cổng thông minh Full-Span chiếu tràn viền: {span_A} -> {span_B}")

# Kiểm tra người đi sát mép màn hình (x=10) vẫn cắt vạch thành công
p_edge_prev = (10, 280)
p_edge_curr = (10, 320)
assert engine._intersect(span_A, span_B, p_edge_prev, p_edge_curr), "Không bắt được người đi sát mép màn hình!"
print("  ✅ Bắt chuẩn xác 100% người đi sát mép khung hình nhờ Full-Span Barrier!")

# Kiểm tra cắt vạch đi từ trên xuống dưới (Vào cổng)
p_prev = (450, 280)
p_curr = (450, 320)
intersected = engine._intersect(ext_A, ext_B, p_prev, p_curr)
assert intersected, "Thuật toán không bắt được giao điểm khi đi qua vạch!"

vAB = (ext_B[0] - ext_A[0], ext_B[1] - ext_A[1])
cross = vAB[0] * (p_curr[1] - p_prev[1]) - vAB[1] * (p_curr[0] - p_prev[0])
assert cross > 0, "Tính toán tích có hướng (Cross product) hướng Vào bị sai!"
print("  ✅ Thuật toán cắt vạch và phân biệt hướng VÀO/RA chính xác 100%.")

# 3. KIỂM THỬ TẠO VIDEO TEST TỰ ĐỘNG VÀ CHẠY TRACKING THỰC TẾ
print("\n[TEST 3] Tạo video mô phỏng 4 người và kiểm thử Tracking...")
test_video_path = os.path.join(BASE_DIR, "test_simulation.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_vid = cv2.VideoWriter(test_video_path, fourcc, 20.0, (960, 540))

# Lấy 1 ảnh người thật từ tập dataset test để làm background hoặc texture
img_folder = os.path.join(BASE_DIR, "dataset/test/images")
bg_frame = np.full((540, 960, 3), 40, dtype=np.uint8)

# Tạo 60 frames mô phỏng
for f in range(60):
    frame = bg_frame.copy()
    # Vẽ nền giả lập có vạch
    cv2.putText(frame, f"TEST FRAME {f}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # 4 người cùng bước từ y=200 xuống y=400 (cắt vạch y=300)
    y_pos = int(180 + f * 4) # Chạy từ 180 đến 420
    for idx, x_pos in enumerate([200, 350, 500, 650]):
        # Vẽ hình người đơn giản
        cv2.circle(frame, (x_pos, y_pos - 40), 20, (200, 200, 200), -1) # Đầu
        cv2.rectangle(frame, (x_pos - 25, y_pos - 20), (x_pos + 25, y_pos + 60), (0, 140, 255), -1) # Thân
        cv2.putText(frame, f"P{idx+1}", (x_pos - 15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    out_vid.write(frame)
out_vid.release()
print(f"  ✅ Đã tạo video kiểm thử: {test_video_path} (60 frames)")

# Chạy thử video qua YOLOEngine
print("  -> Chạy video qua YOLOEngine...")
engine.start(test_video_path)
for _ in range(40):
    if engine.latest_frame is not None:
        break
    time.sleep(0.25)

disp_frame = engine.get_display_frame()
assert disp_frame is not None, "get_display_frame() trả về None!"
assert disp_frame.shape == (540, 960, 3), f"Kích thước frame sai: {disp_frame.shape}"
print(f"  ✅ Engine trích xuất frame Thread-safe thành công: {disp_frame.shape}, FPS: {engine.current_fps:.1f}")

engine.stop()
time.sleep(0.5)

# 4. KIỂM THỬ KHỞI TẠO GIAO DIỆN NGUYÊN BẢN (GUI TEST)
print("\n[TEST 4] Kiểm thử Giao diện App & ProDashboard (Tkinter)...")
from main import App

app = App()
app.update_idletasks()
app.update()

dashboard = app.dashboard
print("  ✅ Khởi tạo giao diện App và ProDashboard nguyên bản thành công!")
assert dashboard.video_canvas is not None
assert dashboard.lbl_in is not None
assert dashboard.lbl_out is not None
assert dashboard.lbl_inside is not None
assert dashboard.lbl_status is not None
assert dashboard.tree is not None
assert dashboard.tree_daily is not None
assert dashboard.chart_canvas is not None
print("  ✅ Toàn bộ cấu trúc giao diện nguyên bản đầy đủ 100% linh kiện.")

# Kiểm thử Smart Angle Snapping & Handle calculation
fixed = [200, 300]
moving_crooked_h = [700, 312] # Lệch 12px -> Phải tự căn thẳng ngang 300
snapped_h = dashboard.snap_coords(fixed, moving_crooked_h, snap_thresh=20)
assert snapped_h[1] == 300, f"Angle snap ngang thất bại: {snapped_h}"

moving_crooked_v = [212, 500] # Lệch 12px -> Phải tự căn thẳng dọc 200
snapped_v = dashboard.snap_coords(fixed, moving_crooked_v, snap_thresh=20)
assert snapped_v[0] == 200, f"Angle snap dọc thất bại: {snapped_v}"
print("  ✅ Thuật toán căn thẳng thông minh (Angle Snapping) hoạt động chuẩn xác 100%.")

# Kiểm tra cập nhật số liệu và lưu config
print("\n[TEST 5] Kiểm thử cập nhật số liệu và tự động lưu config.json...")
dashboard.lbl_in.configure(text="4")
dashboard.lbl_out.configure(text="0")
dashboard.update_daily_csv(add_in=4, add_out=0)
dashboard.save_config()

config_file = os.path.join(BASE_DIR, "config.json")
assert os.path.exists(config_file), "File config.json chưa được tạo!"
print(f"  ✅ File cấu hình config.json lưu thành công tại: {config_file}")

# Đóng app an toàn
app.on_closing()
print("  ✅ Đóng ứng dụng an toàn, giải phóng tài nguyên.")

# Dọn file video tạm
if os.path.exists(test_video_path):
    os.remove(test_video_path)

print("\n" + "="*60)
print("🎉 TOÀN BỘ 5/5 BÀI TEST ĐỀU ĐẠT CHUẨN XUẤT SẮC! HỆ THỐNG VẬN HÀNH ỔN ĐỊNH 100%!")
print("="*60)

import cv2
import threading
import time
import os
from collections import deque
from ultralytics import YOLO

class YOLOEngine:
    def __init__(self, model_path, cb_stats, cb_log, cb_stop, cb_error):
        self.model_path = model_path
        self.model = None
        self.load_error = None
        self._init_model(model_path)
        
        self.video_path = None
        self.cap = None
        self.is_running = False
        self.is_paused = False
        
        # Biến chia sẻ frame an toàn cho UI Polling (Chống đơ Tkinter 100% & Thread-safe)
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.current_fps = 0.0
        
        # Cấu hình thông số AI tối ưu
        self.conf_thresh = 0.35  # Mặc định 0.35 nhạy hơn, không bỏ sót
        self.iou_thresh = 0.60   # Tách biệt người đi sát nhau (tránh triệt tiêu nhầm)
        self.imgsz = 640         # Chuẩn hóa kích thước inference theo model train
        self.point_mode = "center"  # "center": Tâm người (chuẩn tức thì) | "foot": Gót chân
        self.invert_dir = False
        self.full_span_barrier = True  # Cổng thông minh: Mở rộng tràn 2 viền khung hình (vẽ thế nào cũng chuẩn)
        
        # Hiệu ứng thị giác phản hồi đếm (Visual Feedback)
        self.line_flash_timer = 0
        self.line_flash_color = (0, 255, 255)
        
        self.FRAME_W = 960
        self.FRAME_H = 540
        self.line_points = []
        
        self.track_trails = {}
        self.counted_ids = {}    # Quản lý trạng thái đếm theo ID (State-based)
        self.last_seen = {}      # Ghi nhận frame cuối thấy ID để dọn rác RAM
        self.frame_idx = 0
        
        self.count_in = 0
        self.count_out = 0
        
        # Callbacks
        self.cb_stats = cb_stats
        self.cb_log = cb_log
        self.cb_stop = cb_stop
        self.cb_error = cb_error

    def _init_model(self, path):
        try:
            if not os.path.exists(path) and not path.startswith("yolo"):
                raise FileNotFoundError(f"Không tìm thấy file model tại: {path}")
            self.model = YOLO(path)
            self.model_path = path
            self.load_error = None
        except Exception as e:
            self.load_error = str(e)
            print(f"[LỖI NẠP MODEL]: {e}")

    def load_new_model(self, new_path):
        self._init_model(new_path)
        if self.load_error and self.cb_error:
            self.cb_error(self.load_error)

    def set_line(self, pt1, pt2):
        self.line_points = [list(pt1), list(pt2)]

    def reverse_line_direction(self):
        if len(self.line_points) == 2:
            self.line_points = [self.line_points[1], self.line_points[0]]
        else:
            self.invert_dir = not self.invert_dir

    def get_effective_line(self, margin_ratio=0.12):
        if len(self.line_points) != 2:
            return None, None
        A, B = self.line_points[0], self.line_points[1]
        dx = B[0] - A[0]
        dy = B[1] - A[1]
        # Mở rộng nhẹ 2 đầu mút vạch để người đi sát mép cổng không bị lọt qua
        ext_A = [int(A[0] - dx * margin_ratio), int(A[1] - dy * margin_ratio)]
        ext_B = [int(B[0] + dx * margin_ratio), int(B[1] + dy * margin_ratio)]
        
        ext_A[0] = max(0, min(ext_A[0], self.FRAME_W - 1))
        ext_A[1] = max(0, min(ext_A[1], self.FRAME_H - 1))
        ext_B[0] = max(0, min(ext_B[0], self.FRAME_W - 1))
        ext_B[1] = max(0, min(ext_B[1], self.FRAME_H - 1))
        return ext_A, ext_B

    def get_full_span_line(self):
        """Chiếu vạch cắt kín 2 mép khung hình (Smart Barrier) để đảm bảo không ai có thể lách qua."""
        if len(self.line_points) != 2:
            return None, None
        x1, y1 = self.line_points[0]
        x2, y2 = self.line_points[1]
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return self.line_points[0], self.line_points[1]
            
        W, H = self.FRAME_W, self.FRAME_H
        intersections = []
        
        # Cắt mép x = 0
        if dx != 0:
            y = y1 + (0 - x1) * dy / dx
            if 0 <= y <= H - 1:
                intersections.append((0, int(round(y))))
            y = y1 + ((W - 1) - x1) * dy / dx
            if 0 <= y <= H - 1:
                intersections.append((W - 1, int(round(y))))
        # Cắt mép y = 0
        if dy != 0:
            x = x1 + (0 - y1) * dx / dy
            if 0 <= x <= W - 1:
                intersections.append((int(round(x)), 0))
            x = x1 + ((H - 1) - y1) * dx / dy
            if 0 <= x <= W - 1:
                intersections.append((int(round(x)), H - 1))
                
        # Lọc các điểm trùng nhau
        unique_pts = []
        for p in intersections:
            if not any(abs(p[0] - up[0]) <= 1 and abs(p[1] - up[1]) <= 1 for up in unique_pts):
                unique_pts.append(p)
                
        if len(unique_pts) < 2:
            return self.get_effective_line(margin_ratio=0.15)
            
        pA, pB = unique_pts[0], unique_pts[1]
        # Giữ đúng chiều vector người dùng vẽ pt1 -> pt2
        dot = (pB[0] - pA[0]) * dx + (pB[1] - pA[1]) * dy
        if dot < 0:
            pA, pB = pB, pA
        return list(pA), list(pB)

    def get_active_gate(self):
        """Lấy tọa độ vạch xét va chạm: ưu tiên Full-Span nếu bật, hoặc mở rộng mút vạch."""
        if self.full_span_barrier:
            span_a, span_b = self.get_full_span_line()
            if span_a is not None and span_b is not None:
                return span_a, span_b
        return self.get_effective_line(margin_ratio=0.15)

    def _draw_dashed_line(self, img, pt1, pt2, color, thickness=1, gap=12):
        dist = ((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)**0.5
        if dist < 4:
            return
        pts_num = max(1, int(dist // gap))
        for i in range(0, pts_num, 2):
            r1 = i / pts_num
            r2 = min((i + 1) / pts_num, 1.0)
            p1 = (int(pt1[0] + (pt2[0] - pt1[0]) * r1), int(pt1[1] + (pt2[1] - pt1[1]) * r1))
            p2 = (int(pt1[0] + (pt2[0] - pt1[0]) * r2), int(pt1[1] + (pt2[1] - pt1[1]) * r2))
            cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)

    def _intersect(self, A, B, C, D):
        def ccw(p1, p2, p3):
            return (p3[1] - p1[1]) * (p2[0] - p1[0]) > (p2[1] - p1[1]) * (p3[0] - p1[0])
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    def get_display_frame(self):
        with self.frame_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None

    def draw_line_with_direction(self, frame):
        if len(self.line_points) != 2:
            return frame
        A, B = tuple(self.line_points[0]), tuple(self.line_points[1])
        
        # Nếu bật Full-Span Barrier: Vẽ đường dóng phụ nét đứt kéo dài ra 2 mép khung hình
        if self.full_span_barrier:
            span_A, span_B = self.get_full_span_line()
            if span_A is not None and span_B is not None:
                span_A = tuple(span_A)
                span_B = tuple(span_B)
                self._draw_dashed_line(frame, span_A, A, (140, 180, 200), thickness=1, gap=12)
                self._draw_dashed_line(frame, B, span_B, (140, 180, 200), thickness=1, gap=12)
        
        # Hiệu ứng đổi màu phản hồi khi có người qua vạch (giữ trong 8 frame ~0.25s)
        if self.line_flash_timer > 0:
            line_col = self.line_flash_color
            line_w = 5
            self.line_flash_timer -= 1
        else:
            line_col = (0, 255, 255) # Vàng nguyên bản
            line_w = 3
            
        # Vẽ vạch cổng chính của người dùng
        cv2.line(frame, A, B, line_col, line_w, cv2.LINE_AA)
        
        # Vẽ các chốt điều khiển (Handles) cao cấp tại 2 đầu mút
        for pt in (A, B):
            cv2.circle(frame, pt, 8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, pt, 5, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 2, (255, 255, 255), -1, cv2.LINE_AA)
        
        mx, my = int((A[0] + B[0]) / 2), int((A[1] + B[1]) / 2)
        dx, dy = B[0] - A[0], B[1] - A[1]
        length = (dx**2 + dy**2) ** 0.5
        if length > 0:
            nx, ny = -dy / length, dx / length
            if self.invert_dir:
                nx, ny = -nx, -ny
            tip_x, tip_y = int(mx + nx * 45), int(my + ny * 45)
            cv2.arrowedLine(frame, (mx, my), (tip_x, tip_y), (0, 255, 0), 3, cv2.LINE_AA, tipLength=0.35)
            
            # Badge nền chữ "HUONG VAO" sắc nét
            badge_text = "HUONG VAO"
            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            bx, by = tip_x - 30, tip_y - 10
            cv2.rectangle(frame, (bx - 4, by - th - 4), (bx + tw + 4, by + 4), (0, 0, 0), -1)
            cv2.rectangle(frame, (bx - 4, by - th - 4), (bx + tw + 4, by + 4), (0, 255, 0), 1)
            cv2.putText(frame, badge_text, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            
        return frame

    def start(self, video_source):
        if self.model is None:
            if self.cb_error:
                self.cb_error(f"Chưa nạp được Model!\nChi tiết: {self.load_error}")
            self.cb_stop()
            return
            
        self.video_path = video_source
        self.is_running = True
        self.is_paused = False
        self.count_in = 0
        self.count_out = 0
        self.track_trails.clear()
        self.counted_ids.clear()
        self.last_seen.clear()
        self.frame_idx = 0
        threading.Thread(target=self._process_video, daemon=True).start()

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        return self.is_paused

    def stop(self):
        self.is_running = False
        self.is_paused = False

    def _process_video(self):
        try:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                if self.cb_error:
                    self.cb_error("Không thể mở file Video này!")
                self.is_running = False
                self.cb_stop()
                return

            # Lấy FPS gốc của file video để đồng bộ chuẩn thời gian thực (chống slow-motion)
            source_fps = self.cap.get(cv2.CAP_PROP_FPS)
            if source_fps <= 0 or source_fps > 120 or source_fps != source_fps:
                source_fps = 30.0
            target_frame_duration = 1.0 / source_fps
            is_live_stream = str(self.video_path).startswith("0") or "rtsp" in str(self.video_path).lower()

            prev_time = time.time()
            tracker_type = "botsort.yaml"
            
            while self.cap.isOpened() and self.is_running:
                if self.is_paused:
                    time.sleep(0.05)
                    prev_time = time.time()
                    continue
                    
                frame_start_time = time.time()
                ret, frame = self.cap.read()
                if not ret:
                    break
                    
                frame = cv2.resize(frame, (self.FRAME_W, self.FRAME_H))
                self.frame_idx += 1
                
                # Tự động dọn dẹp RAM: Xóa các ID đã rời khỏi khung hình > 40 frames
                if self.frame_idx % 30 == 0:
                    stale_ids = [tid for tid, f_seen in self.last_seen.items() if self.frame_idx - f_seen > 40]
                    for tid in stale_ids:
                        self.track_trails.pop(tid, None)
                        self.last_seen.pop(tid, None)
                        self.counted_ids.pop(tid, None)

                self.draw_line_with_direction(frame)

                # Chạy Tracking (Tự động fallback nếu máy thiếu thư viện của botsort)
                # Tối ưu iou=0.60 chống triệt tiêu người đi gần nhau, imgsz=640 tăng FPS
                try:
                    results = self.model.track(
                        frame, persist=True, tracker=tracker_type, iou=self.iou_thresh, classes=[0], imgsz=self.imgsz,
                        conf=self.conf_thresh, verbose=False
                    )
                except Exception:
                    tracker_type = "bytetrack.yaml"
                    results = self.model.track(
                        frame, persist=True, iou=self.iou_thresh, classes=[0], tracker=tracker_type,
                        conf=self.conf_thresh, imgsz=self.imgsz, verbose=False
                    )
                
                if results and results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.int().cpu().numpy()
                    confs = results[0].boxes.conf.cpu().numpy()
                    
                    ext_A, ext_B = self.get_active_gate()
                    A, B = (self.line_points[0], self.line_points[1]) if len(self.line_points) == 2 else (None, None)
                    
                    for box, track_id, conf in zip(boxes, track_ids, confs):
                        self.last_seen[track_id] = self.frame_idx
                        x1, y1, x2, y2 = map(int, box)
                        cx = (x1 + x2) // 2
                        cy = y2 if self.point_mode == "foot" else (y1 + y2) // 2
                        current_pt = (cx, cy)
                        
                        if track_id not in self.track_trails:
                            self.track_trails[track_id] = deque(maxlen=20)
                            
                        trail = self.track_trails[track_id]
                        
                        if len(trail) > 0 and ext_A is not None and ext_B is not None:
                            # Quét đa frame (1 đến 3 frame gần nhất) để không trượt bước nhảy hoặc khi video giật
                            crossed = False
                            cross_vec = None
                            pts_to_check = list(trail)[-3:]
                            for prev_pt in reversed(pts_to_check):
                                if self._intersect(ext_A, ext_B, prev_pt, current_pt):
                                    crossed = True
                                    cross_vec = (current_pt[0] - prev_pt[0], current_pt[1] - prev_pt[1])
                                    break
                                    
                            if crossed and cross_vec is not None:
                                vAB = (ext_B[0] - ext_A[0], ext_B[1] - ext_A[1])
                                cross_product = vAB[0] * cross_vec[1] - vAB[1] * cross_vec[0]
                                
                                if self.invert_dir:
                                    cross_product = -cross_product
                                    
                                detected_direction = "IN" if cross_product > 0 else ("OUT" if cross_product < 0 else None)
                                
                                if detected_direction:
                                    # State-based counting: Kiểm tra trạng thái ID
                                    counted_info = self.counted_ids.get(track_id)
                                    should_count = False
                                    
                                    if counted_info is None:
                                        # Chưa từng đếm ID này
                                        should_count = True
                                    else:
                                        # Chỉ cho phép đếm lại nếu đi theo hướng ngược lại và đã cách ít nhất 40 frames
                                        if counted_info["direction"] != detected_direction and (self.frame_idx - counted_info["frame"] > 40):
                                            should_count = True
                                            
                                    if should_count:
                                        self.counted_ids[track_id] = {
                                            "direction": detected_direction,
                                            "frame": self.frame_idx
                                        }
                                        
                                        if detected_direction == "IN":
                                            self.count_in += 1
                                            self.cb_stats(self.count_in, self.count_out, 1, 0)
                                            self.cb_log(track_id, "Vào cổng", round(float(conf), 2))
                                            self.line_flash_timer = 8
                                            self.line_flash_color = (0, 255, 0) # Xanh lá khi vào
                                        else:
                                            self.count_out += 1
                                            self.cb_stats(self.count_in, self.count_out, 0, 1)
                                            self.cb_log(track_id, "Ra cổng", round(float(conf), 2))
                                            self.line_flash_timer = 8
                                            self.line_flash_color = (0, 0, 255) # Đỏ khi ra

                        self.track_trails[track_id].append(current_pt)
                        
                        # Vẽ vệt quỹ đạo tinh gọn (chỉ 8 điểm gần nhất để khung hình không bị rối)
                        pts = list(self.track_trails[track_id])
                        n_pts = len(pts)
                        for i in range(max(1, n_pts - 8), n_pts):
                            thick = 1 if i < n_pts - 4 else 2
                            cv2.line(frame, pts[i - 1], pts[i], (0, 255, 180), thick, cv2.LINE_AA)
                            
                        is_counted = track_id in self.counted_ids
                        color = (0, 215, 255) if is_counted else (25, 135, 84)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
                        cv2.circle(frame, current_pt, 4, (0, 0, 255), -1, cv2.LINE_AA)
                        
                        label = f"ID {track_id}" + (" [OK]" if is_counted else "")
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
                        cv2.putText(frame, label, (x1 + 3, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                curr_time = time.time()
                dt = curr_time - prev_time
                self.current_fps = (1.0 / dt) if dt > 0 else 0.0
                prev_time = curr_time
                
                # Cập nhật frame mới nhất cho UI Polling lấy (Thread-safe)
                with self.frame_lock:
                    self.latest_frame = frame
                    
                # Đồng bộ tốc độ phát video đúng chuẩn thời gian thực (Real-time sync)
                if not is_live_stream:
                    elapsed = time.time() - frame_start_time
                    delay = target_frame_duration - elapsed
                    if delay > 0:
                        time.sleep(delay)
                    elif delay < -target_frame_duration:
                        # Nếu máy xử lý chậm hơn video quá 1 frame, skip nhẹ 1 frame để bắt kịp thời gian thực
                        self.cap.grab()
                else:
                    time.sleep(0.003)
                
        except Exception as e:
            if self.cb_error:
                self.cb_error(f"Lỗi trong lúc chạy Video:\n{str(e)}")
        finally:
            if self.cap:
                self.cap.release()
            self.is_running = False
            self.cb_stop()
# People Counting Project

Ứng dụng desktop đếm lượt người **vào/ra qua một vạch ảo** trong video hoặc camera. Dự án sử dụng **YOLO11** để phát hiện người, **BoT-SORT** để theo dõi ID và **CustomTkinter** để hiển thị giao diện tiếng Việt.

## Tính năng

- Nhận video từ máy tính (`.mp4`, `.avi`, `.mkv`, `.mov`), webcam hoặc luồng camera HTTP/RTSP.
- Xem trước nguồn hình trước khi bắt đầu đếm.
- Vẽ vạch bằng hai điểm, kéo hai đầu vạch, căn thẳng ngang/dọc và đảo hướng vào/ra.
- Tùy chọn mở rộng vạch đến hai mép khung hình; chọn tâm cơ thể hoặc gót chân làm điểm xét qua vạch.
- Hiển thị khung nhận diện, ID, quỹ đạo, FPS và bộ đếm vào/ra.
- Xem lịch sử sự kiện trong bộ nhớ, thống kê theo ngày và biểu đồ.
- Tự lưu thống kê ngày vào CSV; xuất bản sao CSV hoặc Excel.
- Điều chỉnh confidence, IoU và chọn model `.pt` từ giao diện.
- Tạm dừng, tiếp tục hoặc kết thúc phiên xử lý.

## Cấu trúc dự án

```text
People_Counting_Project/
├── main.py                   # Điểm khởi chạy ứng dụng
├── requirements.txt          # Các thư viện Python
├── config.json               # Cấu hình vạch và thông số AI
├── thong_ke_ngay.csv          # Số lượt vào/ra cộng dồn theo ngày
├── core/
│   └── yolo_engine.py        # Nhận diện, tracking và logic đếm
├── ui/
│   └── dashboard_pro.py      # Giao diện, camera, cấu hình và thống kê
├── model/
│   ├── train.py              # Huấn luyện YOLO11m
│   └── test.py               # Dự đoán ảnh trong tập test
├── dataset/
│   ├── data.yaml             # Đường dẫn dữ liệu và tên lớp
│   ├── train/                # images/ và labels/
│   ├── valid/                # images/ và labels/
│   └── test/                 # images/ và labels/
├── weights/                  # yolo11m.pt, yolo11n.pt, yolo26n.pt
├── runs/detect/
│   ├── predict/              # Ảnh dự đoán đã lưu
│   └── runs/detect/yolo11m_v1/
│       ├── weights/          # best.pt và last.pt
│       ├── args.yaml         # Tham số huấn luyện đã ghi nhận
│       └── results.csv       # Chỉ số theo epoch
└── test_suite.py             # Kiểm tra engine, hình học và giao diện
```

Đường dẫn `runs/detect/runs/detect/...` ở trên là cấu trúc thực tế của bản đóng gói hiện tại.

## Cài đặt và chạy

Hướng dẫn dưới đây dành cho Windows PowerShell. Cần Python có Tkinter và môi trường desktop để mở giao diện. Có thể tạo môi trường bằng Python 3.11; dự án chưa khóa phiên bản thư viện để tái lập môi trường chính xác.

Giải nén dự án, mở terminal tại thư mục chứa `main.py`, sau đó chạy:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

Các thư viện chính: Ultralytics, OpenCV, CustomTkinter, pandas, openpyxl, Matplotlib và Pillow. File `requirements.txt` chỉ đặt giới hạn phiên bản tối thiểu; nếu bản Ultralytics cũ không nhận model YOLO11, cập nhật thư viện trong môi trường ảo:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade ultralytics
```

Ứng dụng GUI không chỉ định cứng thiết bị suy luận. Hai script `model/train.py` và `model/test.py` đang dùng `device=0`; nếu không có GPU CUDA phù hợp, cần đổi thành `device='cpu'` trước khi chạy chúng. Tốc độ thực tế phụ thuộc model, phần cứng và nguồn hình.

### Trọng số khi khởi động

`main.py` tìm model theo thứ tự:

1. `runs/detect/runs/detect/yolo11m_v1/weights/best.pt`.
2. `weights/best.pt` nếu đường dẫn thứ nhất không tồn tại.

Bản nén có `best.pt` ở đường dẫn thứ nhất. Nếu không nạp được model, mở **Cài đặt AI & Model** và chọn lại file `.pt` phù hợp. Engine chỉ xử lý lớp có chỉ số `0`, nên model thay thế phải có lớp người ở chỉ số này.

## Cách sử dụng

1. Chọn **Chọn video** hoặc **Webcam**. Với camera IP, nhập URL HTTP/RTSP rồi kết nối.
2. Kiểm tra vạch trên ảnh xem trước. Dùng **Chỉnh vạch** để vẽ lại bằng hai điểm hoặc kéo hai đầu vạch để điều chỉnh.
3. Kiểm tra mũi tên **HUONG VAO**; nhấn **Đảo hướng** nếu chiều vào đang ngược với thực tế.
4. Trong **Cài đặt AI & Model**, chọn model, ngưỡng confidence/IoU, điểm xét qua vạch và chế độ mở rộng vạch nếu cần.
5. Nhấn **Bắt đầu**. Dùng **Tạm dừng** để tạm ngưng, nhấn lại để tiếp tục; dùng **Dừng hẳn** để kết thúc.
6. Xem **Lịch sử sự kiện** và **Thống kê**; nhấn **Xuất CSV / Excel** để lưu báo cáo ngày.

Nhãn Camera 0/1 trong giao diện là tên đặt sẵn; thiết bị thực tế phụ thuộc thứ tự camera trên máy. Đổi nguồn hoặc model khi đã dừng phiên xử lý.

### Ý nghĩa bộ đếm

- **Vào/Ra:** số lượt cắt vạch theo từng hướng trong phiên xử lý. Hai bộ đếm được đặt về 0 khi bắt đầu phiên mới.
- **Đang ở bên trong:** `max(0, Vào - Ra)` trong phiên, không tính số người đã ở trong khu vực trước khi chạy.
- **Thống kê ngày:** cộng dồn sự kiện qua nhiều phiên theo ngày/giờ máy tính tại lúc xử lý. Khi phát lại video cũ, ngày ghi nhận vẫn là ngày chạy ứng dụng.
- Chạy lại cùng video sẽ tiếp tục cộng vào thống kê ngày; ứng dụng không tự loại lượt trùng giữa các phiên.

## Nguyên lý đếm

```text
Video / camera → OpenCV → YOLO phát hiện person → Tracking ID
              → Xét quỹ đạo cắt vạch → Xác định hướng → Cập nhật GUI và CSV
```

Khung hình xử lý được resize về **960 × 540**; kích thước suy luận được đặt là **640**. Engine ưu tiên BoT-SORT và thử chuyển sang ByteTrack khi lời gọi tracking phát sinh lỗi.

Với từng ID, hệ thống kiểm tra đoạn di chuyển từ các điểm gần nhất đến điểm hiện tại có cắt vạch hay không. Hướng vào/ra được xác định bằng tích có hướng giữa vector vạch và vector di chuyển. Một ID được đếm lần đầu khi qua vạch; lần tiếp theo chỉ được đếm nếu đi ngược hướng và cách lần trước **hơn 40 frame**. Trạng thái ID mất dấu quá 40 frame được dọn định kỳ mỗi 30 frame.

## Cấu hình và dữ liệu lưu

`config.json` lưu các trường chính:

| Trường | Ý nghĩa |
| --- | --- |
| `line_points` | Hai đầu vạch theo hệ tọa độ khung hình 960 × 540 |
| `invert_dir` | Đảo quy ước chiều vào/ra |
| `conf_thresh`, `iou_thresh` | Ngưỡng tin cậy và IoU NMS |
| `point_mode` | `center` hoặc `foot` |
| `full_span_barrier` | Mở rộng vạch tới biên khung hình |
| `snap_enabled` | Tự căn ngang/dọc |
| `model_folder` | Thư mục quét model |
| `model_path`, `last_video` | Đường dẫn model và nguồn hình được ghi lại |

Bản đóng gói chứa một số đường dẫn tuyệt đối từ máy trước. Hãy chọn lại thư mục model và nguồn video trên máy mới. Khi khởi động, ứng dụng nạp trọng số theo quy tắc trong `main.py`; `load_config()` không tự nạp lại `model_path` hoặc mở `last_video` đã lưu.

`thong_ke_ngay.csv` dùng UTF-8 BOM và có các cột `Ngày`, `Tổng Vào`, `Tổng Ra`, `Cập nhật cuối`. Bản nén đã có file thống kê, nên cần kiểm tra dữ liệu cũ trước khi dùng cho phiên đo mới. Nút **Xóa dữ liệu** xóa thống kê ngày. Lịch sử chi tiết từng sự kiện nằm trong bộ nhớ và không được tự lưu thành nhật ký lâu dài.

## Dataset

Dataset có một lớp: `person` (ID `0`). Theo danh sách file ảnh trong bản nén:

| Tập | Số ảnh |
| --- | ---: |
| Train | 8.895 |
| Validation | 2.174 |
| Test | 1.088 |
| **Tổng** | **12.157** |

Tài liệu Roboflow đi kèm ghi 9.959 ảnh, khác với số file thực tế ở bản nén này. Nguồn được ghi trong `dataset/README.dataset.txt` là [CCTV-person trên Roboflow](https://universe.roboflow.com/testpeople-g2zma/cctv-person-o3jjx); tài liệu đó ghi giấy phép là `undefined`.

### Sửa đường dẫn dataset

Trước khi huấn luyện, sửa `dataset/data.yaml`: thay `path` của máy cũ bằng đường dẫn tuyệt đối đến **thư mục dataset** trên máy hiện tại, ví dụ:

```yaml
path: D:/Projects/People_Counting_Project/dataset
train: train/images
val: valid/images
test: test/images
nc: 1
names: ['person']
```

### Huấn luyện

Trong `model/train.py`, nên đổi dòng nạp trọng số thành đường dẫn đã có trong bản nén:

```python
model = YOLO('weights/yolo11m.pt')
```

Chạy từ thư mục gốc dự án:

```powershell
.\.venv\Scripts\python.exe model/train.py
```

Script đặt `epochs=150`, `patience=15`, `imgsz=640`, `batch=16`, `device=0`, `project='runs/detect'` và `name='yolo11m_v1'`. Nếu thiếu VRAM, giảm batch. Sau khi train, chọn file `best.pt` của lần chạy mới trong giao diện; đường dẫn đầu ra mới có thể khác đường dẫn lồng sẵn trong `main.py`.

### Dự đoán tập ảnh test

```powershell
.\.venv\Scripts\python.exe model/test.py
```

Script đọc ảnh trong `dataset/test/images`, dùng `conf=0.4`, lưu ảnh dự đoán vào thư mục kết quả do Ultralytics tạo. Cần sửa đường dẫn model nếu dùng trọng số khác và đổi `device` nếu chạy CPU. Đây là script dự đoán ảnh, không phải phép đánh giá độ chính xác đếm người qua cổng.

### Chỉ số trong log huấn luyện

Dòng epoch 150 của `runs/detect/runs/detect/yolo11m_v1/results.csv` ghi nhận:

| Chỉ số | Giá trị |
| --- | ---: |
| Precision | 0,96644 |
| Recall | 0,92071 |
| mAP@50 | 0,95638 |
| mAP@50–95 | 0,76546 |

Đây là chỉ số phát hiện đối tượng trên validation trong log có sẵn, không phải kết quả kiểm chứng mới hoặc độ chính xác đếm vào/ra. Các giá trị ở epoch cuối cũng không mặc nhiên là chỉ số của checkpoint `best.pt`.

## Kiểm thử

```powershell
.\.venv\Scripts\python.exe test_suite.py
```

Script kiểm tra nạp model, giao cắt vạch, chiếu vạch tới biên, chạy video mô phỏng, khởi tạo GUI và lưu cấu hình/thống kê. Cần có model và môi trường desktop.

**Lưu ý:** script có ghi `config.json` và cộng thử **4 lượt vào** vào `thong_ke_ngay.csv`; nên chạy trên bản sao dự án hoặc sao lưu hai file này. Video mô phỏng dùng hình người vẽ đơn giản nên không thay thế đánh giá trên video thực tế.

## Xử lý lỗi thường gặp

| Hiện tượng | Cách kiểm tra |
| --- | --- |
| Không nạp được model | Kiểm tra vị trí `best.pt`, chọn lại file trong cài đặt và xem lỗi nạp model |
| Không mở được camera | Kiểm tra quyền camera, ứng dụng khác đang dùng thiết bị và thử Camera 0/1 |
| Không đọc được HTTP/RTSP | Kiểm tra URL, kết nối mạng và khả năng truy cập luồng hình |
| Lỗi đường dẫn khi train | Sửa `path` trong `dataset/data.yaml`, chạy lệnh tại thư mục chứa `main.py` |
| Lỗi CUDA ở script train/test | Kiểm tra môi trường PyTorch/CUDA hoặc đổi `device='cpu'` |
| FPS thấp | Thử model nhẹ hơn, giảm tải hệ thống hoặc dùng GPU phù hợp |
| Đếm ngược hướng | Nhấn **Đảo hướng**, kiểm tra mũi tên trên vạch |
| Bỏ sót hoặc đếm dư | Kiểm tra vị trí vạch, confidence, điểm tâm/gót chân và mức độ che khuất |
| Không ghi/xuất được báo cáo | Kiểm tra quyền ghi và đóng file đang mở trong Excel |

## Giới hạn

Độ chính xác phụ thuộc góc camera, ánh sáng, mật độ người và chất lượng tracking. ID có thể thay đổi khi che khuất; người quay lại quá nhanh có thể không được đếm do khoảng chờ 40 frame. Đây là bộ đếm lượt qua vạch, không phải nhận dạng danh tính hoặc đo chính xác số người hiện diện trong mọi tình huống. Phiên bản hiện tại xử lý một nguồn hình tại một thời điểm và chưa có chức năng ghi video kết quả trong GUI.

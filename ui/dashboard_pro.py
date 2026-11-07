import os
from pathlib import Path
import json
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ================= GIAO DIỆN NGUYÊN BẢN (LIGHT MODE) =================
ctk.set_appearance_mode("Light")
BG_COLOR = "#F3F6FC"
SIDEBAR_COLOR = "#11283B"
CARD_COLOR = "#FFFFFF"
TEXT_MAIN = "#172B3A"
GREEN_ACCENT = "#198754"
BLUE_ACCENT = "#0D6EFD"
RED_ACCENT = "#DC3545"
ORANGE_ACCENT = "#FD7E14"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAILY_CSV_FILE = os.path.join(BASE_DIR, "thong_ke_ngay.csv")
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")

class ProDashboard(ctk.CTkFrame):
    def __init__(self, master, engine):
        super().__init__(master, fg_color=BG_COLOR)
        self.master = master
        self.engine = engine
        self.preview_frame = None
        self.current_display_frame = None
        self.click_points = []
        self.log_data = []
        
        self.img_scale = 1.0
        self.img_offset_x = 0
        self.img_offset_y = 0
        
        # Biến tối ưu vẽ Canvas và biểu đồ Matplotlib
        self.canvas_img_id = None
        self.last_chart_draw = 0
        
        # Biến tương tác chuột thông minh (Vẽ & Kéo thả chốt vạch)
        self.dragging_idx = None
        self.is_drawing_new = False
        self.drag_start_pt = None
        self.snap_enabled = True
        
        # Biến xem trước Webcam trực tiếp (Live Preview)
        self.live_cap = None
        self.is_live_previewing = False
        self.preview_thread = None
        
        self.init_daily_csv()
        self.load_config()
        self.setup_ui()
        self.refresh_daily_summary_ui()
        self.update_clock()

    # ================= QUẢN LÝ CẤU HÌNH TỰ ĐỘNG LƯU (TỐI ƯU TÍNH NĂNG) =================
    def save_config(self):
        cfg = {
            "line_points": self.engine.line_points,
            "invert_dir": self.engine.invert_dir,
            "conf_thresh": self.engine.conf_thresh,
            "iou_thresh": self.engine.iou_thresh,
            "point_mode": self.engine.point_mode,
            "full_span_barrier": self.engine.full_span_barrier,
            "model_path": self.engine.model_path,
            "model_folder": getattr(self, "model_folder", os.path.join(BASE_DIR, "weights")),
            "snap_enabled": self.snap_enabled,
            "last_video": str(self.engine.video_path) if self.engine.video_path is not None else ""
        }
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Lỗi lưu cấu hình: {e}")

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                self.model_folder = cfg.get("model_folder", os.path.join(BASE_DIR, "weights"))
                self.snap_enabled = bool(cfg.get("snap_enabled", True))
                if "line_points" in cfg and len(cfg["line_points"]) == 2:
                    self.engine.set_line(cfg["line_points"][0], cfg["line_points"][1])
                    self.click_points = list(cfg["line_points"])
                if "invert_dir" in cfg:
                    self.engine.invert_dir = cfg["invert_dir"]
                if "conf_thresh" in cfg:
                    self.engine.conf_thresh = cfg["conf_thresh"]
                if "iou_thresh" in cfg:
                    self.engine.iou_thresh = cfg["iou_thresh"]
                if "point_mode" in cfg:
                    self.engine.point_mode = cfg["point_mode"]
                if "full_span_barrier" in cfg:
                    self.engine.full_span_barrier = cfg["full_span_barrier"]
            except Exception as e:
                print(f"Lỗi nạp cấu hình: {e}")

    # ================= QUẢN LÝ FILE CSV THỐNG KÊ THEO NGÀY =================
    def init_daily_csv(self):
        if not os.path.exists(DAILY_CSV_FILE):
            df = pd.DataFrame(columns=["Ngày", "Tổng Vào", "Tổng Ra", "Cập nhật cuối"])
            df.to_csv(DAILY_CSV_FILE, index=False, encoding="utf-8-sig")

    def update_daily_csv(self, add_in=0, add_out=0):
        today_str = datetime.datetime.now().strftime("%d/%m/%Y")
        now_time = datetime.datetime.now().strftime("%H:%M:%S")
        
        try:
            df = pd.read_csv(DAILY_CSV_FILE, encoding="utf-8-sig")
        except Exception:
            df = pd.DataFrame(columns=["Ngày", "Tổng Vào", "Tổng Ra", "Cập nhật cuối"])

        if today_str in df["Ngày"].values:
            idx = df.index[df["Ngày"] == today_str][0]
            df.loc[idx, "Tổng Vào"] = int(df.loc[idx, "Tổng Vào"]) + add_in
            df.loc[idx, "Tổng Ra"] = int(df.loc[idx, "Tổng Ra"]) + add_out
            df.loc[idx, "Cập nhật cuối"] = now_time
        else:
            new_row = pd.DataFrame([{
                "Ngày": today_str,
                "Tổng Vào": add_in,
                "Tổng Ra": add_out,
                "Cập nhật cuối": now_time
            }])
            df = pd.concat([df, new_row], ignore_index=True)

        try:
            df.to_csv(DAILY_CSV_FILE, index=False, encoding="utf-8-sig")
        except Exception as e:
            print(f"Lỗi ghi CSV: {e}")
            
        # Vẽ lại biểu đồ có điều tiết (chỉ vẽ lại nếu đã cách >= 4 giây để chống đơ luồng giao diện khi đông người)
        now_ts = datetime.datetime.now().timestamp()
        if now_ts - self.last_chart_draw > 4.0:
            self.refresh_daily_summary_ui()
            self.last_chart_draw = now_ts

    def setup_ui(self):
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0, minsize=275)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=8, minsize=470)
        self.grid_rowconfigure(2, weight=2, minsize=210)

        self.setup_sidebar()
        self.setup_header()
        self.setup_video_section()
        self.setup_kpi_section()
        self.setup_daily_csv_section()
        self.setup_table_section()

    # ================= 1. SIDEBAR (NGUYÊN BẢN) =================
    def setup_sidebar(self):
        sidebar = ctk.CTkFrame(self, width=190, fg_color=SIDEBAR_COLOR, corner_radius=0)
        sidebar.grid(row=0, column=0, rowspan=3, sticky="nsew")
        sidebar.grid_propagate(False)
        
        ctk.CTkLabel(sidebar, text="TDV", font=("Segoe UI", 30, "bold"), text_color="white", justify="left").pack(pady=(25, 30), padx=20, anchor="w")
        
        ctk.CTkLabel(sidebar, text="PEOPLE FLOW", text_color="#94A9BB", font=("Segoe UI", 10)).pack(anchor="w", padx=22, pady=(0, 32))
        ctk.CTkLabel(sidebar, text="KHÔNG GIAN LÀM VIỆC", text_color="#7E98AE", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=22, pady=(0, 14))
        self.nav_btns = {}
        menu_items = [
            ("overview", "01   Tổng quan", self.show_overview),
            ("history", "02   Lịch sử sự kiện", self.open_full_history_modal),
            ("statistics", "03   Thống kê", self.show_statistics),
            ("settings", "04   Cài đặt AI & Model", self.open_settings_modal)
        ]
        
        for key, text, cmd in menu_items:
            btn = ctk.CTkButton(
                sidebar, text=text, fg_color="#078796" if key == "overview" else "transparent",
                text_color="white", hover_color="#1C4056", font=("Segoe UI", 13, "bold"),
                anchor="w", height=42, corner_radius=8,
                command=lambda k=key, c=cmd: self.activate_nav(k, c)
            )
            btn.pack(pady=4, padx=15, fill="x")
            self.nav_btns[key] = btn

        ctk.CTkLabel(
            sidebar, text="HỆ THỐNG ĐẾM NGƯỜI\n\nYOLO11 · BoT-SORT", 
            text_color="#A0ABC0", justify="left", font=("Segoe UI", 11)
        ).pack(side="bottom", pady=20, padx=20, anchor="w")

    def activate_nav(self, key, callback):
        for k, b in self.nav_btns.items():
            b.configure(fg_color="#078796" if k == key else "transparent")
        callback()

    # ================= 2. HEADER (NGUYÊN BẢN) =================
    def setup_header(self):
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=1, columnspan=2, sticky="ew", padx=24, pady=(20, 12))
        title = ctk.CTkFrame(header, fg_color="transparent")
        title.pack(side="left")
        ctk.CTkLabel(title, text="Tổng quan hệ thống", text_color=TEXT_MAIN, font=("Segoe UI", 26, "bold")).pack(anchor="w")
        ctk.CTkLabel(title, text="Theo dõi luồng người và hoạt động qua cổng", text_color="#7A899B", font=("Segoe UI", 12)).pack(anchor="w", pady=(2,0))
        self.lbl_clock = ctk.CTkLabel(header, text="", text_color="#64748B", font=("Segoe UI", 12), justify="right")
        self.lbl_clock.pack(side="right")

    # ================= 3. KHU VỰC VIDEO (NGUYÊN BẢN) =================
    def setup_video_section(self):
        card = ctk.CTkFrame(self, fg_color="white", corner_radius=18, border_width=1, border_color="#E4EBF2")
        card.grid(row=1, column=1, sticky="nsew", padx=(24,10), pady=(0,12))
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(1, weight=1)
        top = ctk.CTkFrame(card, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", padx=18, pady=(15,12))
        self.lbl_cam_title = ctk.CTkLabel(top, text="Màn hình giám sát", font=("Segoe UI",15,"bold"), text_color=TEXT_MAIN)
        self.lbl_cam_title.pack(side="left")
        self.lbl_badge = ctk.CTkLabel(top, text="  AI TRACKING  ", fg_color="#E8F5F4", text_color="#087F84", corner_radius=6, font=("Segoe UI",10,"bold"), height=26)
        self.lbl_badge.pack(side="right")
        self.video_canvas = tk.Canvas(card, bg="#101D2B", highlightthickness=0, cursor="crosshair", width=1, height=1)
        self.video_canvas.grid(row=1,column=0,sticky="nsew",padx=16)
        for event, callback in [("<Button-1>",self.on_canvas_mouse_down),("<B1-Motion>",self.on_canvas_mouse_drag),("<ButtonRelease-1>",self.on_canvas_mouse_up),("<Motion>",self.on_canvas_mouse_hover),("<Configure>",self._resize_video)]:
            self.video_canvas.bind(event, callback)
        self.canvas_text = self.video_canvas.create_text(0,0,text="Sẵn sàng giám sát\n\nChọn video hoặc webcam để xem hình ảnh.\nĐặt hai điểm trên hình để tạo vạch đếm.",fill="#9DB2C7",font=("Segoe UI",13),justify="center")
        controls = ctk.CTkFrame(card,fg_color="transparent")
        controls.grid(row=2,column=0,sticky="ew",padx=16,pady=(14,8))
        specs = [("btn_select","Chọn video",self.select_video,False),("btn_cam","Webcam",self.open_camera_selector_modal,False),("btn_play","Bắt đầu",self.toggle_tracking,True),("btn_pause","Tạm dừng",self.pause_tracking,False),("btn_redraw","Chỉnh vạch",self.reset_line,False),("btn_flip","Đảo hướng",self.flip_direction,False)]
        for col,(attr,text,command,primary) in enumerate(specs):
            controls.grid_columnconfigure(col,weight=1,uniform="control")
            button=ctk.CTkButton(controls,text=text,command=command,height=38,width=72,corner_radius=8,font=("Segoe UI",11,"bold"),fg_color="#087F8C" if primary else "#EDF2F7",hover_color="#066571" if primary else "#DDE7EF",text_color="white" if primary else "#33485E")
            button.grid(row=0,column=col,sticky="ew",padx=(0,6) if col<5 else 0)
            setattr(self,attr,button)
        self.btn_pause.configure(state="disabled")
        footer=ctk.CTkFrame(card,fg_color="transparent")
        footer.grid(row=3,column=0,sticky="ew",padx=18,pady=(0,12))
        ctk.CTkLabel(footer,text="Kéo hai đầu vạch để điều chỉnh vùng đi qua",font=("Segoe UI",10),text_color="#8293A5").pack(side="left")
        self.lbl_status=ctk.CTkLabel(footer,text=f"FPS: 0.0  |  Conf: {self.engine.conf_thresh:.2f}",font=("Segoe UI",10),text_color="#64748B")
        self.lbl_status.pack(side="right")

    # ================= 4. KHU VỰC KPI (NGUYÊN BẢN) =================
    def setup_kpi_section(self):
        rail=ctk.CTkFrame(self,fg_color="transparent",width=255)
        rail.grid_propagate(False)
        rail.grid(row=1,column=2,sticky="nsew",padx=(8,24),pady=(0,12))
        rail.grid_columnconfigure(0,weight=1)
        for i in range(3): rail.grid_rowconfigure(i,weight=1,uniform="kpi")
        def metric(row,title,caption,tint,color,attr):
            box=ctk.CTkFrame(rail,fg_color="white",corner_radius=18,border_width=1,border_color="#E4EBF2")
            box.grid(row=row,column=0,sticky="nsew",pady=(0,10))
            ctk.CTkLabel(box,text=title,font=("Segoe UI",12,"bold"),text_color="#52677C").pack(anchor="w",padx=20,pady=(16,0))
            value=ctk.CTkLabel(box,text="0",font=("Segoe UI",36,"bold"),text_color=color)
            value.pack(anchor="w",padx=20,pady=(0,0))
            setattr(self,attr,value)
            label=ctk.CTkLabel(box,text=caption,font=("Segoe UI",10),text_color=color,fg_color=tint,corner_radius=5,height=24)
            label.pack(anchor="w",padx=20,pady=(0,12))
            return label
        metric(0,"LƯỢT VÀO","  Ghi nhận trong phiên  ","#E8F7EF","#11956B","lbl_in")
        self.lbl_inside=metric(1,"LƯỢT RA","Đang ở bên trong: 0 người","#FFF0F1","#D95664","lbl_out")
        box=ctk.CTkFrame(rail,fg_color="#EAF4F6",corner_radius=18,border_width=1,border_color="#D9E9ED")
        box.grid(row=2,column=0,sticky="nsew")
        ctk.CTkLabel(box,text="TRẠNG THÁI NGUỒN",font=("Segoe UI",11,"bold"),text_color="#52677C").pack(anchor="w",padx=20,pady=(18,10))
        self.lbl_cam_state=ctk.CTkLabel(box,text="Chưa chọn nguồn",font=("Segoe UI",12,"bold"),text_color="#287984",fg_color="#DDEFF1",corner_radius=6,height=30,wraplength=210)
        self.lbl_cam_state.pack(anchor="w",padx=20,pady=(0,8))
        ctk.CTkLabel(box,text="Dữ liệu cập nhật trong phiên xử lý",font=("Segoe UI",10),text_color="#718B97",wraplength=210).pack(anchor="w",padx=20,pady=(0,12))

    # ================= 5. THỐNG KÊ TỔNG HỢP THEO NGÀY (NGUYÊN BẢN) =================
    def setup_daily_csv_section(self):
        daily_card = ctk.CTkFrame(self, fg_color=CARD_COLOR, corner_radius=16, border_width=1, border_color="#E4EBF2")
        daily_card.grid(row=2, column=1, sticky="nsew", padx=(20, 10), pady=(5, 15))
        daily_card.grid_columnconfigure(0, weight=3, uniform="daily")
        daily_card.grid_columnconfigure(1, weight=2, uniform="daily")
        daily_card.grid_rowconfigure(1, weight=1)
        
        top = ctk.CTkFrame(daily_card, fg_color="transparent")
        top.grid(row=0, column=0, columnspan=2, sticky="ew", padx=15, pady=(10, 2))
        ctk.CTkLabel(
            top, text="Thống kê lượt vào / ra theo ngày", 
            font=("Segoe UI", 14, "bold"), text_color=TEXT_MAIN
        ).pack(side="left")
        
        ctk.CTkButton(
            top, text="Xóa dữ liệu", fg_color="#F1F5F9", text_color=RED_ACCENT, hover_color="#FEE2E2",
            height=26, width=110, font=("Segoe UI", 11, "bold"), command=self.clear_daily_csv
        ).pack(side="right", padx=4)
        
        ctk.CTkButton(
            top, text="Xuất CSV / Excel", fg_color="#E8F5E9", text_color=GREEN_ACCENT, hover_color="#C8E6C9",
            height=26, width=135, font=("Segoe UI", 11, "bold"), command=self.export_daily_csv_copy
        ).pack(side="right", padx=4)

        # Trái: Biểu đồ Cột theo Ngày
        self.fig, self.ax = plt.subplots(figsize=(3.8, 1.5), dpi=100)
        self.fig.patch.set_facecolor('#FFFFFF')
        self.chart_canvas = FigureCanvasTkAgg(self.fig, master=daily_card)
        self.chart_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=(14, 10), pady=(5, 10))

        # Phải: Bảng CSV theo Ngày
        cols = ("Date", "In", "Out", "Updated")
        self.tree_daily = ttk.Treeview(daily_card, columns=cols, show="headings", height=3)
        self.tree_daily.heading("Date", text="Ngày"); self.tree_daily.column("Date", width=85, anchor="center")
        self.tree_daily.heading("In", text="Tổng Vào"); self.tree_daily.column("In", width=65, anchor="center")
        self.tree_daily.heading("Out", text="Tổng Ra"); self.tree_daily.column("Out", width=65, anchor="center")
        self.tree_daily.heading("Updated", text="Cập nhật"); self.tree_daily.column("Updated", width=75, anchor="center")
        self.tree_daily.grid(row=1, column=1, sticky="nsew", padx=(5, 15), pady=(5, 10))

    def refresh_daily_summary_ui(self):
        try:
            df = pd.read_csv(DAILY_CSV_FILE, encoding="utf-8-sig")
        except Exception:
            return

        for item in self.tree_daily.get_children():
            self.tree_daily.delete(item)
        for _, row in df.iloc[::-1].iterrows():
            self.tree_daily.insert("", "end", values=(row["Ngày"], int(row["Tổng Vào"]), int(row["Tổng Ra"]), row["Cập nhật cuối"]))

        df_recent = df.tail(7)
        dates = df_recent["Ngày"].tolist() if not df_recent.empty else [datetime.datetime.now().strftime("%d/%m/%Y")]
        in_vals = df_recent["Tổng Vào"].astype(int).tolist() if not df_recent.empty else [0]
        out_vals = df_recent["Tổng Ra"].astype(int).tolist() if not df_recent.empty else [0]

        self.ax.clear()
        self.ax.set_facecolor('#FFFFFF')
        x = range(len(dates))
        w = 0.24
        bars1 = self.ax.bar([i - w/2 for i in x], in_vals, width=w, color='#18AE87', label='Tổng Vào')
        bars2 = self.ax.bar([i + w/2 for i in x], out_vals, width=w, color='#EC7884', label='Tổng Ra')
        
        self.ax.set_xlim(-0.9, max(len(dates) - 0.1, 0.9))
        self.ax.set_xticks(list(x))
        self.ax.set_xticklabels(dates, fontsize=8, color='#333333', fontweight='bold')
        self.ax.tick_params(axis='y', labelsize=8, colors='#555555')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_color('#E2E8F0')
        self.ax.spines['bottom'].set_color('#E2E8F0')
        self.ax.set_ylim(0, max(max(in_vals + out_vals, default=5) + 4, 5))
        self.ax.set_axisbelow(True)
        self.ax.yaxis.grid(True, color="#EEF2F6", linewidth=0.7)
        from matplotlib.ticker import MaxNLocator
        self.ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
        self.ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=2)
        if not any(in_vals + out_vals):
            self.ax.text(0.5, 0.55, "Chưa có lượt qua cổng", transform=self.ax.transAxes, ha="center", va="center", color="#8A9CAE", fontsize=10)

        for bar in bars1:
            h = bar.get_height()
            if h > 0:
                self.ax.text(bar.get_x() + bar.get_width()/2, h + 0.1, int(h), ha='center', va='bottom', fontsize=8, color='#198754', fontweight='bold')
        for bar in bars2:
            h = bar.get_height()
            if h > 0:
                self.ax.text(bar.get_x() + bar.get_width()/2, h + 0.1, int(h), ha='center', va='bottom', fontsize=8, color='#DC3545', fontweight='bold')

        self.fig.tight_layout(pad=1.1)
        self.chart_canvas.draw_idle()

    def export_daily_csv_copy(self):
        save_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile=f"ThongKe_TheoNgay_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
            filetypes=[("CSV File", "*.csv"), ("Excel File", "*.xlsx")]
        )
        if save_path:
            df = pd.read_csv(DAILY_CSV_FILE, encoding="utf-8-sig")
            if save_path.endswith(".xlsx"):
                df.to_excel(save_path, index=False)
            else:
                df.to_csv(save_path, index=False, encoding="utf-8-sig")
            messagebox.showinfo("Thành công", f"Đã xuất bảng tổng hợp theo ngày tại:\n{save_path}")

    def clear_daily_csv(self):
        if messagebox.askyesno("Xác nhận", "Bạn có chắc muốn xóa làm mới lịch sử thống kê các ngày trong CSV?"):
            df = pd.DataFrame(columns=["Ngày", "Tổng Vào", "Tổng Ra", "Cập nhật cuối"])
            df.to_csv(DAILY_CSV_FILE, index=False, encoding="utf-8-sig")
            self.refresh_daily_summary_ui()

    # ================= 6. BẢNG LỊCH SỬ TỪNG LƯỢT (NGUYÊN BẢN) =================
    def setup_table_section(self):
        table_card = ctk.CTkFrame(self, fg_color=CARD_COLOR, corner_radius=16, border_width=1, border_color="#E4EBF2")
        table_card.configure(width=255)
        table_card.pack_propagate(False)
        table_card.grid(row=2, column=2, sticky="nsew", padx=(10, 20), pady=(5, 15))
        
        top_bar = ctk.CTkFrame(table_card, fg_color="transparent")
        top_bar.pack(fill="x", padx=15, pady=(10, 5))
        ctk.CTkLabel(top_bar, text="Lịch sử từng người", font=("Segoe UI", 12, "bold"), text_color=TEXT_MAIN).pack(side="left")
        
        ctk.CTkButton(
            top_bar, text="Xem tất cả", fg_color="transparent", text_color=BLUE_ACCENT,
            hover_color="#EFF6FF", font=("Segoe UI", 12, "bold"), width=65, height=24,
            command=self.open_full_history_modal
        ).pack(side="right")

        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview", background="white", foreground="#33485E", rowheight=32, fieldbackground="white", borderwidth=0, font=("Segoe UI", 10))
        style.configure("Treeview.Heading", background="#F8FAFC", foreground="#475569", font=('Segoe UI', 10, 'bold'), borderwidth=0)

        self.tree = ttk.Treeview(table_card, columns=("Time", "ID", "Event"), show="headings", height=5)
        self.tree.heading("Time", text="Thời gian"); self.tree.column("Time", width=75, anchor="center")
        self.tree.heading("ID", text="Mã ID"); self.tree.column("ID", width=70, anchor="center")
        self.tree.heading("Event", text="Sự kiện"); self.tree.column("Event", width=85, anchor="center")
        self.tree.pack(fill="both", expand=True, padx=12, pady=(0, 10))

    # ================= LOGIC HIỂN THỊ & VẼ VẠCH =================
    def update_clock(self):
        now = datetime.datetime.now()
        days = ["Chủ nhật", "Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7"]
        self.lbl_clock.configure(text=f"{days[now.isoweekday()%7]}, {now.strftime('%d/%m/%Y')}  |  {now.strftime('%H:%M:%S')}")
        self.after(1000, self.update_clock)

    def on_canvas_resize(self, event):
        if self.current_display_frame is not None and not self.engine.is_running:
            self.render_frame_on_canvas(self.current_display_frame)

    def render_frame_on_canvas(self, frame_bgr):
        self.current_display_frame = frame_bgr
        cw = self.video_canvas.winfo_width()
        ch = self.video_canvas.winfo_height()
        if cw < 50 or ch < 50:
            cw, ch = 800, 450
            
        fh, fw = frame_bgr.shape[:2]
        scale = min(cw / fw, ch / fh)
        new_w, new_h = int(fw * scale), int(fh * scale)
        
        self.img_scale = scale
        self.img_offset_x = (cw - new_w) // 2
        self.img_offset_y = (ch - new_h) // 2
        
        resized = cv2.resize(frame_bgr, (new_w, new_h))
        img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self.tk_img = ImageTk.PhotoImage(Image.fromarray(img_rgb))
        
        # Tối ưu vẽ Canvas: Chỉ tạo item 1 lần, các frame sau chỉ cập nhật ảnh (giảm 80% tải CPU)
        if self.canvas_img_id is None:
            self.video_canvas.delete("all")
            self.canvas_img_id = self.video_canvas.create_image(self.img_offset_x, self.img_offset_y, anchor="nw", image=self.tk_img)
        else:
            self.video_canvas.coords(self.canvas_img_id, self.img_offset_x, self.img_offset_y)
            self.video_canvas.itemconfig(self.canvas_img_id, image=self.tk_img)

    def canvas_to_frame_coords(self, cx, cy):
        if self.img_scale <= 0:
            return 0, 0
        fx = int((cx - self.img_offset_x) / self.img_scale)
        fy = int((cy - self.img_offset_y) / self.img_scale)
        fx = max(0, min(fx, self.engine.FRAME_W - 1))
        fy = max(0, min(fy, self.engine.FRAME_H - 1))
        return fx, fy

    def frame_to_canvas_coords(self, fx, fy):
        cx = int(fx * self.img_scale + self.img_offset_x)
        cy = int(fy * self.img_scale + self.img_offset_y)
        return cx, cy

    def snap_coords(self, fixed_pt, moving_pt, snap_thresh=20):
        if not self.snap_enabled or fixed_pt is None:
            return moving_pt
        fx, fy = fixed_pt
        mx, my = moving_pt
        # Tự động căn ngang (Horizontal Snap)
        if abs(my - fy) <= snap_thresh:
            my = fy
        # Tự động căn dọc (Vertical Snap)
        elif abs(mx - fx) <= snap_thresh:
            mx = fx
        return [mx, my]

    def on_canvas_mouse_hover(self, event):
        if self.engine.is_running or self.preview_frame is None or len(self.click_points) != 2:
            self.video_canvas.configure(cursor="crosshair")
            return
        # Đổi trỏ chuột khi lướt qua chốt mút (Handles)
        for pt in self.click_points:
            cpx, cpy = self.frame_to_canvas_coords(pt[0], pt[1])
            if ((event.x - cpx)**2 + (event.y - cpy)**2)**0.5 <= 18:
                self.video_canvas.configure(cursor="fleur")
                return
        self.video_canvas.configure(cursor="crosshair")

    def on_canvas_mouse_down(self, event):
        if self.engine.is_running:
            messagebox.showinfo("Đang chạy", "Hãy nhấn 'Dừng hẳn' trước khi kẻ hoặc chỉnh sửa vạch.")
            return
        if self.preview_frame is None:
            messagebox.showwarning("Chưa có video", "Vui lòng nhấn 'Chọn video' trước khi kẻ vạch!")
            return

        # 1. Kiểm tra nếu click trúng chốt mút (Handles) -> Bật chế độ kéo thả chốt
        if len(self.click_points) == 2:
            for idx, pt in enumerate(self.click_points):
                cpx, cpy = self.frame_to_canvas_coords(pt[0], pt[1])
                dist = ((event.x - cpx)**2 + (event.y - cpy)**2)**0.5
                if dist <= 22:
                    self.dragging_idx = idx
                    self.is_drawing_new = False
                    self.video_canvas.configure(cursor="fleur")
                    return

        # 2. Nếu click ra ngoài -> Bắt đầu kẻ vạch mới hoặc đặt điểm tiếp theo
        fx, fy = self.canvas_to_frame_coords(event.x, event.y)
        self.canvas_img_id = None
        self.dragging_idx = None
        self.drag_start_pt = [fx, fy]
        
        if len(self.click_points) != 1:
            # Bắt đầu Điểm 1 mới
            self.click_points = [[fx, fy]]
            self.is_drawing_new = True
            self.redraw_preview_line()
        else:
            # Đang có sẵn Điểm 1, đây là Điểm 2
            p2 = self.snap_coords(self.click_points[0], [fx, fy], snap_thresh=20)
            self.click_points.append(p2)
            self.is_drawing_new = False
            self.drag_start_pt = None
            self.engine.set_line(self.click_points[0], self.click_points[1])
            self.redraw_preview_line()
            self.save_config()

    # Alias tương thích ngược
    on_canvas_click = on_canvas_mouse_down

    def on_canvas_mouse_drag(self, event):
        if self.engine.is_running or self.preview_frame is None:
            return
            
        fx, fy = self.canvas_to_frame_coords(event.x, event.y)
        
        # Kéo thả chốt mút có sẵn
        if self.dragging_idx is not None and len(self.click_points) == 2:
            other_idx = 1 - self.dragging_idx
            fixed_pt = self.click_points[other_idx]
            snapped_pt = self.snap_coords(fixed_pt, [fx, fy], snap_thresh=20)
            self.click_points[self.dragging_idx] = snapped_pt
            self.engine.set_line(self.click_points[0], self.click_points[1])
            self.redraw_preview_line()
            return
            
        # Vừa nhấn vừa kéo để tạo Điểm 2 (Live preview kéo chuột)
        if self.is_drawing_new and len(self.click_points) >= 1:
            fixed_pt = self.click_points[0]
            snapped_pt = self.snap_coords(fixed_pt, [fx, fy], snap_thresh=20)
            self.redraw_preview_line(temp_second_pt=snapped_pt)

    def on_canvas_mouse_up(self, event):
        if self.engine.is_running or self.preview_frame is None:
            return
            
        # Kết thúc kéo thả chốt
        if self.dragging_idx is not None:
            self.dragging_idx = None
            self.video_canvas.configure(cursor="crosshair")
            self.save_config()
            self.redraw_preview_line()
            return
            
        # Kết thúc rê chuột tạo Điểm 2
        if self.is_drawing_new and len(self.click_points) == 1 and self.drag_start_pt is not None:
            fx, fy = self.canvas_to_frame_coords(event.x, event.y)
            dist = ((fx - self.drag_start_pt[0])**2 + (fy - self.drag_start_pt[1])**2)**0.5
            if dist >= 25:
                p2 = self.snap_coords(self.click_points[0], [fx, fy], snap_thresh=20)
                self.click_points.append(p2)
                self.is_drawing_new = False
                self.drag_start_pt = None
                self.engine.set_line(self.click_points[0], self.click_points[1])
                self.redraw_preview_line()
                self.save_config()

    def redraw_preview_line(self, temp_second_pt=None):
        if self.preview_frame is None:
            return
        temp = self.preview_frame.copy()
        
        if temp_second_pt is not None and len(self.click_points) >= 1:
            p1 = tuple(self.click_points[0])
            p2 = tuple(temp_second_pt)
            cv2.line(temp, p1, p2, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(temp, p1, 7, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(temp, p2, 7, (0, 255, 0), -1, cv2.LINE_AA)
            dx, dy = abs(p2[0] - p1[0]), abs(p2[1] - p1[1])
            snap_tag = " [Thang ngang]" if dy == 0 else (" [Thang doc]" if dx == 0 else "")
            cv2.putText(temp, f"Tha chuot de chot{snap_tag}", (p2[0] + 12, p2[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
        elif len(self.click_points) == 1:
            p1 = tuple(self.click_points[0])
            cv2.circle(temp, p1, 8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(temp, p1, 5, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(temp, "Diem 1 (Click hoac keo chuot de dat Diem 2...)", (p1[0] + 12, p1[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        elif len(self.click_points) == 2:
            self.engine.set_line(self.click_points[0], self.click_points[1])
            self.engine.draw_line_with_direction(temp)
            
        self.render_frame_on_canvas(temp)

    # ================= QUẢN LÝ WEBCAM & LIVE PREVIEW =================
    def start_live_preview(self, source):
        self.stop_live_preview()
        self.is_live_previewing = True
        self.preview_thread = threading.Thread(target=self._live_preview_worker, args=(source,), daemon=True)
        self.preview_thread.start()

    def stop_live_preview(self):
        self.is_live_previewing = False
        if self.live_cap is not None:
            try:
                self.live_cap.release()
            except Exception:
                pass
            self.live_cap = None
        if self.preview_thread is not None and self.preview_thread.is_alive():
            self.preview_thread.join(timeout=0.6)
            self.preview_thread = None

    def _live_preview_worker(self, source):
        try:
            cam_src = int(source) if isinstance(source, int) or (isinstance(source, str) and source.isdigit()) else source
            cap = cv2.VideoCapture(cam_src)
            if not cap.isOpened():
                self.after(0, lambda: messagebox.showerror("Lỗi Camera", f"Không thể mở nguồn Camera: {source}\nVui lòng kiểm tra lại thiết bị!"))
                return
            self.live_cap = cap
            while self.is_live_previewing and not self.engine.is_running:
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.08)
                    continue
                resized = cv2.resize(frame, (self.engine.FRAME_W, self.engine.FRAME_H))
                self.preview_frame = resized
                self.after_idle(self._refresh_live_canvas)
                time.sleep(0.035)  # ~28 FPS mượt mà
        except Exception as e:
            print(f"[LIVE PREVIEW ERROR]: {e}")
        finally:
            if self.live_cap is not None:
                try:
                    self.live_cap.release()
                except Exception:
                    pass
                self.live_cap = None

    def _refresh_live_canvas(self):
        if not self.engine.is_running and self.preview_frame is not None:
            self.redraw_preview_line()

    def open_camera_selector_modal(self):
        if self.engine.is_running:
            messagebox.showinfo("Đang chạy", "Vui lòng nhấn 'Dừng hẳn' trước khi đổi nguồn Camera.")
            return

        win = ctk.CTkToplevel(self)
        win.title("Chọn Nguồn Camera / Webcam")
        win.geometry("520x430")
        win.attributes("-topmost", True)
        
        ctk.CTkLabel(win, text="📷 CHỌN NGUỒN WEBCAM / CAMERA GIÁM SÁT", font=("Segoe UI", 16, "bold"), text_color=TEXT_MAIN).pack(pady=(16, 4))
        ctk.CTkLabel(win, text="Hệ thống hỗ trợ cả Webcam tích hợp, iVCam, DroidCam và Camera IP RTSP.", font=("Segoe UI", 11), text_color="#64748B").pack(pady=(0, 12))

        frame_cams = ctk.CTkFrame(win, fg_color="#F8FAFC", corner_radius=10)
        frame_cams.pack(fill="x", padx=25, pady=6)

        def pick_cam(idx, name):
            win.destroy()
            self.load_video_source(idx, f"📷 {name}", is_live=True)

        # Nút Camera 0 (iVCam)
        btn_c0 = ctk.CTkButton(
            frame_cams, text="📱 Camera 0: e2eSoft iVCam (Camera điện thoại / Ngoại vi)",
            fg_color="#FFFFFF", text_color=TEXT_MAIN, hover_color="#E2E8F0",
            border_width=1, border_color="#CBD5E1", height=42, anchor="w", font=("Segoe UI", 12, "bold"),
            command=lambda: pick_cam(0, "e2eSoft iVCam (Webcam 0)")
        )
        btn_c0.pack(fill="x", padx=12, pady=(10, 5))

        # Nút Camera 1 (Webcam Laptop)
        btn_c1 = ctk.CTkButton(
            frame_cams, text="💻 Camera 1: HD User Facing (Webcam tích hợp Laptop)",
            fg_color="#FFFFFF", text_color=TEXT_MAIN, hover_color="#E2E8F0",
            border_width=1, border_color="#CBD5E1", height=42, anchor="w", font=("Segoe UI", 12, "bold"),
            command=lambda: pick_cam(1, "Webcam Laptop (Webcam 1)")
        )
        btn_c1.pack(fill="x", padx=12, pady=5)

        # Mục nhập Camera IP / RTSP / DroidCam
        ctk.CTkLabel(win, text="Hoặc kết nối Camera IP qua WiFi (DroidCam, RTSP, HTTP):", font=("Segoe UI", 12, "bold"), text_color=TEXT_MAIN).pack(pady=(12, 4), padx=25, anchor="w")
        
        ip_frame = ctk.CTkFrame(win, fg_color="transparent")
        ip_frame.pack(fill="x", padx=25, pady=4)
        
        entry_ip = ctk.CTkEntry(ip_frame, placeholder_text="Ví dụ: http://192.168.1.15:4747/video hoặc rtsp://...", height=36, font=("Segoe UI", 12))
        entry_ip.pack(side="left", fill="x", expand=True, padx=(0, 8))
        
        def connect_ip():
            url = entry_ip.get().strip()
            if not url:
                messagebox.showwarning("Chưa nhập URL", "Vui lòng nhập đường link stream của Camera!")
                return
            win.destroy()
            self.load_video_source(url, f"🌐 IP Cam: {url}", is_live=True)
            
        btn_connect_ip = ctk.CTkButton(ip_frame, text="Kết nối", fg_color=BLUE_ACCENT, hover_color="#0B5ED7", width=90, height=36, font=("Segoe UI", 12, "bold"), command=connect_ip)
        btn_connect_ip.pack(side="right")
        
        ctk.CTkButton(win, text="Đóng", fg_color="#E2E8F0", text_color=TEXT_MAIN, hover_color="#CBD5E1", height=32, width=100, font=("Segoe UI", 12), command=win.destroy).pack(pady=(16, 10))

    # ================= CÁC HÀM SỰ KIỆN NÚT BẤM =================
    def select_video(self):
        self.stop_live_preview()
        if self.engine.is_running:
            self.engine.stop()
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov"), ("All files", "*.*")])
        if path:
            self.load_video_source(path, f"📹 Video: {os.path.basename(path)}", is_live=False)

    def open_webcam(self):
        self.open_camera_selector_modal()

    def load_video_source(self, source, title_text, is_live=None):
        self.stop_live_preview()
        if is_live is None:
            is_live = isinstance(source, int) or (isinstance(source, str) and (source.isdigit() or source.startswith("http") or source.startswith("rtsp")))
            
        if isinstance(source, str) and source.isdigit():
            source = int(source)
            
        self.engine.video_path = source
        self.lbl_cam_title.configure(text=title_text)
        
        if len(self.engine.line_points) != 2:
            p1 = [int(self.engine.FRAME_W * 0.2), int(self.engine.FRAME_H * 0.65)]
            p2 = [int(self.engine.FRAME_W * 0.8), int(self.engine.FRAME_H * 0.65)]
            self.click_points = [p1, p2]
            self.engine.set_line(p1, p2)
            self.save_config()
            
        if is_live:
            self.lbl_cam_state.configure(text="🟢 Đang xem trước Webcam Live", text_color=GREEN_ACCENT, fg_color="#E8F5E9")
            self.start_live_preview(source)
        else:
            cap = cv2.VideoCapture(source)
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                messagebox.showerror("Lỗi", "Không thể đọc nguồn Video này!")
                return
            self.preview_frame = cv2.resize(frame, (self.engine.FRAME_W, self.engine.FRAME_H))
            self.redraw_preview_line()
            self.lbl_cam_state.configure(text="🟢 Đã sẵn sàng (Nhấn Bắt đầu)", text_color=GREEN_ACCENT, fg_color="#E8F5E9")
            
        self.save_config()

    def reset_line(self):
        self.canvas_img_id = None
        self.click_points.clear()
        self.dragging_idx = None
        self.is_drawing_new = False
        self.drag_start_pt = None
        self.engine.line_points.clear()
        if self.preview_frame is not None:
            self.render_frame_on_canvas(self.preview_frame)

    def flip_direction(self):
        self.engine.reverse_line_direction()
        if len(self.engine.line_points) == 2:
            self.click_points = list(self.engine.line_points)
        if not self.engine.is_running and self.preview_frame is not None:
            self.redraw_preview_line()
        self.save_config()

    def toggle_tracking(self):
        if self.engine.video_path is None:
            messagebox.showwarning("Chưa chọn Video", "Vui lòng nhấn 'Chọn video' hoặc 'Webcam' trước!")
            return
        if not self.engine.is_running:
            if len(self.engine.line_points) != 2:
                messagebox.showwarning("Chưa kẻ vạch", "Vui lòng click chuột 2 điểm trên hình để kẻ vạch đếm!")
                return
                
            self.stop_live_preview()
            time.sleep(0.15)
            
            self.btn_play.configure(text="⏹ Dừng hẳn", fg_color=RED_ACCENT, hover_color="#BB2D3B")
            self.btn_pause.configure(state="normal", text="Tạm dừng", fg_color="#F1F5F9", text_color=TEXT_MAIN)
            self.btn_select.configure(state="disabled")
            if hasattr(self, 'btn_cam'):
                self.btn_cam.configure(state="disabled")
            self.lbl_in.configure(text="0")
            self.lbl_out.configure(text="0")
            self.lbl_inside.configure(text="Đang ở bên trong: 0 người")
            self.lbl_cam_state.configure(text="🟢 Đang chạy Tracking...", text_color=GREEN_ACCENT, fg_color="#E8F5E9")
            
            self.engine.start(self.engine.video_path)
            self.master.start_ui_polling()
        else:
            self.engine.stop()

    def pause_tracking(self):
        if not self.engine.is_running:
            return
        paused = self.engine.toggle_pause()
        if paused:
            self.btn_pause.configure(text="▶ Tiếp tục", fg_color=ORANGE_ACCENT, text_color="white")
            self.lbl_cam_state.configure(text="🟠 Đang tạm dừng", text_color=ORANGE_ACCENT, fg_color="#FFF3CD")
        else:
            self.btn_pause.configure(text="Tạm dừng", fg_color="#F1F5F9", text_color=TEXT_MAIN)
            self.lbl_cam_state.configure(text="🟢 Đang chạy Tracking...", text_color=GREEN_ACCENT, fg_color="#E8F5E9")

    # ================= CỬA SỔ POPUP =================
    def open_full_history_modal(self):
        win = self._open_page("history")
        
        ctk.CTkLabel(win, text="🕒 CHI TIẾT TỪNG LƯỢT QUA CỔNG TRONG PHIÊN", font=("Segoe UI", 16, "bold")).pack(pady=12)
        
        cols = ("STT", "Ngày", "Thời gian", "Mã ID", "Sự kiện", "Độ tin cậy")
        tree_all = ttk.Treeview(win, columns=cols, show="headings")
        for c in cols:
            tree_all.heading(c, text=c)
            tree_all.column(c, anchor="center", width=95)
        tree_all.pack(fill="both", expand=True, padx=15, pady=10)
        
        history_timer = [None]
        def cancel_history(event):
            if event.widget is tree_all and history_timer[0] is not None:
                try:
                    self.after_cancel(history_timer[0])
                except tk.TclError:
                    pass
        tree_all.bind("<Destroy>", cancel_history, add="+")
        def refresh_history():
            if not tree_all.winfo_exists():
                return
            tree_all.delete(*tree_all.get_children())
            for idx, item in enumerate(reversed(self.log_data), 1):
                tree_all.insert("", "end", values=(idx, item["Ngày"], item["Thời gian"], f"ID {item['Mã Định Danh']}", item["Hành Động"], item["Độ tin cậy"]))
            history_timer[0] = self.after(1000, refresh_history)
        refresh_history()

    def open_advanced_settings(self):
        win = self._open_page("settings", scroll=True)
        
        ctk.CTkLabel(win, text="⚙️ CÀI ĐẶT HỆ THỐNG AI", font=("Segoe UI", 16, "bold")).pack(pady=10)
        
        lbl_c = ctk.CTkLabel(win, text=f"Ngưỡng tin cậy (Confidence): {self.engine.conf_thresh:.2f}", font=("Segoe UI", 13, "bold"))
        lbl_c.pack(pady=(4, 2))
        
        def on_conf_change(val):
            self.engine.conf_thresh = round(float(val), 2)
            lbl_c.configure(text=f"Ngưỡng tin cậy (Confidence): {self.engine.conf_thresh:.2f}")
            self.save_config()
            
        slider = ctk.CTkSlider(win, from_=0.15, to=0.85, number_of_steps=70, command=on_conf_change)
        slider.set(self.engine.conf_thresh)
        slider.pack(pady=4, padx=30, fill="x")

        lbl_iou = ctk.CTkLabel(win, text=f"Ngưỡng gộp box (IOU NMS): {self.engine.iou_thresh:.2f}", font=("Segoe UI", 13, "bold"))
        lbl_iou.pack(pady=(6, 2))
        
        def on_iou_change(val):
            self.engine.iou_thresh = round(float(val), 2)
            lbl_iou.configure(text=f"Ngưỡng gộp box (IOU NMS): {self.engine.iou_thresh:.2f}")
            self.save_config()
            
        slider_iou = ctk.CTkSlider(win, from_=0.20, to=0.90, number_of_steps=70, command=on_iou_change)
        slider_iou.set(self.engine.iou_thresh)
        slider_iou.pack(pady=4, padx=30, fill="x")

        ctk.CTkLabel(win, text="🛡️ Cổng thông minh (Vẽ thế nào cũng chuẩn):", font=("Segoe UI", 13, "bold")).pack(pady=(8, 2))
        full_span_var = ctk.BooleanVar(value=self.engine.full_span_barrier)
        def toggle_full_span():
            self.engine.full_span_barrier = full_span_var.get()
            self.save_config()
            if not self.engine.is_running and self.preview_frame is not None:
                self.redraw_preview_line()
        ctk.CTkSwitch(win, text="Cổng thông minh tràn viền (Chống lách mép vạch)", variable=full_span_var, command=toggle_full_span, font=("Segoe UI", 12)).pack(pady=2)

        snap_var = ctk.BooleanVar(value=self.snap_enabled)
        def toggle_snap():
            self.snap_enabled = snap_var.get()
        ctk.CTkSwitch(win, text="Tự động căn thẳng Ngang / Dọc (Angle Snapping)", variable=snap_var, command=toggle_snap, font=("Segoe UI", 12)).pack(pady=2)

        ctk.CTkLabel(win, text="Điểm xét va chạm qua vạch:", font=("Segoe UI", 13, "bold")).pack(pady=(8, 2))
        pt_var = ctk.StringVar(value=self.engine.point_mode)
        
        def change_pt():
            self.engine.point_mode = pt_var.get()
            self.save_config()
            
        ctk.CTkRadioButton(win, text="Tâm cơ thể (Khuyên dùng: Chuẩn tức thì, không bị trễ)", variable=pt_var, value="center", command=change_pt).pack(pady=1)
        ctk.CTkRadioButton(win, text="Gót chân (Dành cho góc camera nhìn từ trên cao xuống)", variable=pt_var, value="foot", command=change_pt).pack(pady=1)

        ctk.CTkLabel(win, text="Chuyển đổi Model AI:", font=("Segoe UI", 13, "bold")).pack(pady=(10, 2))
        btn_box = ctk.CTkFrame(win, fg_color="transparent")
        btn_box.pack(pady=4)

        def use_best():
            p = os.path.join(BASE_DIR, r"runs\detect\runs\detect\yolo11m_v1\weights\best.pt")
            if os.path.exists(p):
                self.engine.load_new_model(p)
                self.save_config()
                messagebox.showinfo("Đã đổi Model", "Đang dùng model best.pt (Tập CCTV đã train)!")
            else:
                messagebox.showwarning("Không tìm thấy", f"Không tìm thấy: {p}")

        def use_coco():
            p = os.path.join(BASE_DIR, "yolo11m.pt")
            if not os.path.exists(p):
                p = os.path.join(BASE_DIR, "yolo11n.pt")
            if os.path.exists(p):
                self.engine.load_new_model(p)
                self.save_config()
                messagebox.showinfo("Đã đổi Model", f"Đã chuyển sang dùng {os.path.basename(p)} chuẩn COCO!\n(Bắt người ở góc ngang / quay điện thoại cực chuẩn!)")
            else:
                messagebox.showwarning("Không tìm thấy", "Không tìm thấy file yolo11m.pt hoặc yolo11n.pt!")

        ctk.CTkButton(btn_box, text="🎯 Dùng best.pt", width=120, height=32, font=("Segoe UI", 11, "bold"), command=use_best).pack(side="left", padx=5)
        ctk.CTkButton(btn_box, text="🚀 Dùng yolo11m (COCO)", fg_color=GREEN_ACCENT, hover_color="#157347", width=170, height=32, font=("Segoe UI", 11, "bold"), command=use_coco).pack(side="left", padx=5)

        def pick_model():
            p = filedialog.askopenfilename(filetypes=[("YOLO Weights", "*.pt")])
            if p:
                self.engine.load_new_model(p)
                if not self.engine.load_error:
                    self.save_config()
                    messagebox.showinfo("Đã đổi Model", f"Đã nạp thành công trọng số:\n{p}")
                
        ctk.CTkButton(win, text="📁 Chọn file .pt khác từ máy tính...", fg_color=BLUE_ACCENT, height=30, command=pick_model).pack(pady=(12, 10))

    def _resize_video(self, event):
        if self.current_display_frame is None:
            self.video_canvas.coords(self.canvas_text, event.width // 2, event.height // 2)
        else:
            self.on_canvas_resize(event)

    def show_statistics(self):
        self.refresh_daily_summary_ui()
        win = self._open_page("statistics")
        ctk.CTkLabel(win, text="THỐNG KÊ LƯỢT VÀO / RA THEO NGÀY", font=("Segoe UI", 19, "bold")).pack(pady=20)
        table = ttk.Treeview(win, columns=("date", "in", "out", "updated"), show="headings")
        for key, label in zip(("date", "in", "out", "updated"), ("Ngày", "Tổng vào", "Tổng ra", "Cập nhật")):
            table.heading(key, text=label)
            table.column(key, width=160, anchor="center")
        table.pack(fill="both", expand=True, padx=20, pady=10)
        for item in self.tree_daily.get_children():
            table.insert("", "end", values=self.tree_daily.item(item, "values"))
        ctk.CTkButton(win, text="Xuất CSV / Excel", command=self.export_daily_csv_copy).pack(pady=15)

    def open_settings_modal(self):
        if self.engine.is_running:
            messagebox.showinfo("Cài đặt", "Dừng phiên theo dõi trước khi đổi model.")
            return
        win = self._open_page("settings", scroll=True)
        ctk.CTkLabel(win, text="THƯ VIỆN MODEL", font=("Segoe UI", 21, "bold"), text_color=TEXT_MAIN).pack(anchor="w", padx=24, pady=(24, 6))
        ctk.CTkLabel(win, text="Chọn thư mục chứa các file .pt, sau đó chọn model để nạp.", text_color="#64748B").pack(anchor="w", padx=24)
        folder = tk.StringVar(value=getattr(self, "model_folder", os.path.join(BASE_DIR, "weights")))
        row = ctk.CTkFrame(win, fg_color="transparent")
        row.pack(fill="x", padx=24, pady=18)
        ctk.CTkEntry(row, textvariable=folder, height=38).pack(side="left", fill="x", expand=True, padx=(0, 8))
        files = {}
        selected = tk.StringVar()
        choice = ctk.CTkComboBox(win, variable=selected, values=["Chưa có model"], state="readonly", width=710, height=40)
        info = ctk.CTkLabel(win, text="", text_color="#64748B", wraplength=700, justify="left")
        def scan():
            files.clear()
            directory = Path(folder.get().strip())
            if directory.is_dir():
                for path in sorted(directory.rglob("*.pt")):
                    files[str(path.relative_to(directory))] = str(path)
            values = list(files) or ["Không tìm thấy file .pt"]
            choice.configure(values=values)
            current = next((key for key, value in files.items() if os.path.normcase(value) == os.path.normcase(self.engine.model_path)), values[0])
            selected.set(current)
            info.configure(text=f"{len(files)} model • Đang dùng: {self.engine.model_path}")
        def browse():
            path = filedialog.askdirectory(parent=self.master, title="Chọn thư mục chứa model .pt", initialdir=folder.get() if os.path.isdir(folder.get()) else BASE_DIR)
            if path:
                folder.set(path)
                self.model_folder = path
                self.save_config()
                scan()
        ctk.CTkButton(row, text="Chọn thư mục", width=130, height=38, fg_color="#078796", command=browse).pack(side="left")
        choice.pack(fill="x", padx=24)
        info.pack(anchor="w", padx=24, pady=15)
        def load():
            if self.engine.is_running:
                messagebox.showinfo("Model", "Dừng phiên trước khi đổi model.", parent=self.master)
                return
            path = files.get(selected.get())
            if not path:
                messagebox.showwarning("Model", "Chọn thư mục có file .pt trước.", parent=self.master)
                return
            previous = (self.engine.model, self.engine.model_path)
            self.engine.load_new_model(path)
            if self.engine.load_error:
                self.engine.model, self.engine.model_path = previous
                if previous[0] is not None:
                    self.engine.load_error = None
                return
            self.model_folder = folder.get().strip()
            self.save_config()
            info.configure(text=f"Đã nạp thành công: {path}", text_color=GREEN_ACCENT)
        ctk.CTkButton(win, text="Nạp model đã chọn", height=40, command=load).pack(anchor="w", padx=24, pady=8)
        ctk.CTkButton(win, text="Làm mới danh sách", fg_color="#E9EFF5", text_color=TEXT_MAIN, command=scan).pack(anchor="w", padx=24, pady=8)
        ctk.CTkLabel(win, text="Danh sách bao gồm cả thư mục con. Chỉ nạp file model bạn tin cậy.", text_color="#64748B").pack(anchor="w", padx=24, pady=10)
        self._build_ai_controls(win, info)
        scan()

    def _set_nav(self, key):
        for name, button in self.nav_btns.items():
            button.configure(fg_color="#078796" if name == key else "transparent")

    def _open_page(self, key, scroll=False):
        if not hasattr(self, "_overview_widgets"):
            self._overview_widgets = [widget for widget in self.grid_slaves()
                                      if int(widget.grid_info().get("column", 0)) >= 1
                                      and int(widget.grid_info().get("row", 0)) >= 1]
        for widget in self._overview_widgets:
            widget.grid_remove()
        previous = getattr(self, "_content_page", None)
        if previous is not None and previous.winfo_exists():
            previous.destroy()
        outer = ctk.CTkFrame(self, fg_color=CARD_COLOR, corner_radius=14)
        outer.grid(row=1, column=1, rowspan=2, columnspan=2, sticky="nsew", padx=20, pady=(10, 15))
        self._content_page = outer
        if scroll:
            content = ctk.CTkScrollableFrame(outer, fg_color="transparent")
            content.pack(fill="both", expand=True, padx=12, pady=12)
        else:
            content = outer
        self._set_nav(key)
        return content

    def show_overview(self):
        previous = getattr(self, "_content_page", None)
        if previous is not None and previous.winfo_exists():
            previous.destroy()
        self._content_page = None
        for widget in getattr(self, "_overview_widgets", []):
            widget.grid()
        self._set_nav("overview")
        self.refresh_daily_summary_ui()
        if self.current_display_frame is not None:
            self.after_idle(lambda: self.render_frame_on_canvas(self.current_display_frame))

    def _build_ai_controls(self, parent, model_info):
        panel = ctk.CTkFrame(parent, fg_color="#F4F7FB", corner_radius=12)
        panel.pack(fill="x", padx=24, pady=(16, 24))
        ctk.CTkLabel(panel, text="THÔNG SỐ AI & TRACKING", font=("Segoe UI", 17, "bold"), text_color=TEXT_MAIN).pack(anchor="w", padx=20, pady=(20,12))
        def redraw():
            if not self.engine.is_running and self.preview_frame is not None:
                self.redraw_preview_line()
        def add_slider(title, attr, low, high):
            label = ctk.CTkLabel(panel, text=f"{title}: {getattr(self.engine, attr):.2f}", font=("Segoe UI",12,"bold"))
            label.pack(anchor="w", padx=20, pady=(8,4))
            def change(value):
                setattr(self.engine, attr, round(float(value),2))
                label.configure(text=f"{title}: {getattr(self.engine, attr):.2f}")
                self.save_config()
            slider = ctk.CTkSlider(panel, from_=low, to=high, number_of_steps=round((high-low)*100), command=change, progress_color="#078796", button_color="#078796")
            slider.set(getattr(self.engine,attr))
            slider.pack(fill="x", padx=24, pady=(0,12))
        add_slider("Ngưỡng tin cậy (Confidence)", "conf_thresh", .15, .85)
        add_slider("Ngưỡng gộp box (IOU NMS)", "iou_thresh", .20, .90)
        ctk.CTkLabel(panel, text="Vạch đếm & tương tác",font=("Segoe UI",13,"bold")).pack(anchor="w",padx=20,pady=(10,8))
        full = tk.BooleanVar(value=self.engine.full_span_barrier)
        def toggle_full():
            self.engine.full_span_barrier=full.get()
            self.save_config()
            redraw()
        ctk.CTkSwitch(panel,text="Mở rộng vạch đến hai mép khung hình",variable=full,command=toggle_full,progress_color="#078796").pack(anchor="w",padx=20,pady=8)
        snap = tk.BooleanVar(value=self.snap_enabled)
        def toggle_snap():
            self.snap_enabled=snap.get()
            self.save_config()
        ctk.CTkSwitch(panel,text="Tự động căn thẳng ngang / dọc",variable=snap,command=toggle_snap,progress_color="#078796").pack(anchor="w",padx=20,pady=8)
        ctk.CTkLabel(panel,text="Điểm xét qua vạch",font=("Segoe UI",13,"bold")).pack(anchor="w",padx=20,pady=(15,8))
        point = tk.StringVar(value=self.engine.point_mode)
        def set_point():
            self.engine.point_mode=point.get()
            self.save_config()
        for text,value in [("Tâm cơ thể", "center"),("Gót chân — phù hợp camera nhìn từ trên cao", "foot")]:
            ctk.CTkRadioButton(panel,text=text,variable=point,value=value,command=set_point,fg_color="#078796",hover_color="#066D79").pack(anchor="w",padx=20,pady=7)
        ctk.CTkLabel(panel,text="Chuyển nhanh model",font=("Segoe UI",13,"bold")).pack(anchor="w",padx=20,pady=(18,8))
        buttons=ctk.CTkFrame(panel,fg_color="transparent")
        buttons.pack(fill="x",padx=20,pady=(0,12))
        def load_path(path):
            if self.engine.is_running:
                model_info.configure(text="Dừng phiên theo dõi trước khi chuyển model.",text_color=RED_ACCENT)
                return
            if not path or not os.path.isfile(path):
                model_info.configure(text="Không tìm thấy model. Chọn thư mục hoặc file .pt khác.",text_color=RED_ACCENT)
                return
            previous=(self.engine.model,self.engine.model_path,self.engine.load_error)
            self.engine.load_new_model(str(path))
            if self.engine.load_error:
                error=self.engine.load_error
                self.engine.model,self.engine.model_path,self.engine.load_error=previous
                model_info.configure(text=error,text_color=RED_ACCENT)
                return
            self.save_config()
            model_info.configure(text=f"Đang dùng: {path}",text_color=GREEN_ACCENT)
        def use_best():
            folder=Path(getattr(self,"model_folder",os.path.join(BASE_DIR,"weights")))
            candidates=list(folder.rglob("best.pt")) if folder.is_dir() else []
            if not candidates:
                candidates=list((Path(BASE_DIR)/"runs").rglob("best.pt"))
            load_path(max(candidates,key=lambda path:path.stat().st_mtime) if candidates else None)
        def use_coco():
            candidates=[Path(getattr(self,"model_folder",os.path.join(BASE_DIR,"weights")))/"yolo11m.pt",Path(BASE_DIR)/"weights/yolo11m.pt",Path(BASE_DIR)/"yolo11m.pt"]
            load_path(next((path for path in candidates if path.is_file()),None))
        def pick():
            path=filedialog.askopenfilename(parent=self.master,title="Chọn model .pt",filetypes=[("YOLO model","*.pt")])
            if path:load_path(path)
        for text,cmd,color in [("Dùng best.pt mới nhất",use_best,"#078796"),("Dùng yolo11m (COCO)",use_coco,"#198754"),("Chọn file .pt khác",pick,BLUE_ACCENT)]:
            ctk.CTkButton(buttons,text=text,command=cmd,fg_color=color,height=36).pack(side="left",padx=(0,8))
        ctk.CTkLabel(panel,text="Thay đổi thông số được lưu tự động. Chọn best.pt sẽ ưu tiên file mới nhất trong thư mục model.",text_color="#64748B",wraplength=700,justify="left").pack(anchor="w",padx=20,pady=(0,20))

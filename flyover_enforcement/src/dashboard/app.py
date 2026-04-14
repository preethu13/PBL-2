"""
Streamlit Dashboard for Flyover Enforcement System.

Pages:
  - Dashboard     : Stats + recent violation cards
  - Live Detection: Pick video from data/raw_footage/ → run pipeline → show alerts only
  - Violations    : Full log table with CSV export
  - Manual Review : Approve / reject flagged violations
  - DIP Debug     : Frame enhancement comparison
  - Settings      : YAML config + DB info

Launch: streamlit run src/dashboard/app.py
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

# ── Path setup ────────────────────────────────────────────────────────────────
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)

from src.violation.logger import ViolationLogger
from src.preprocessing.night_mode import NightModeProcessor
from src.preprocessing.enhancement import Enhancer
from src.preprocessing.noise_removal import NoiseRemover
from src.anpr.plate_dip import PlateDIP

FOOTAGE_DIR = os.path.join(project_root, "data", "raw_footage")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flyover Enforcement System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-header {
    font-size: 2rem; font-weight: 800; color: #1a237e;
    text-align: center; padding: 0.5rem 0;
    border-bottom: 3px solid #c62828; margin-bottom: 1.2rem;
}
.stat-card {
    background: linear-gradient(135deg, #1a237e, #283593);
    color: white; padding: 1.2rem; border-radius: 12px;
    text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
.stat-number { font-size: 2.4rem; font-weight: 800; line-height: 1; }
.stat-label  { font-size: 0.78rem; opacity: 0.9; margin-top: 0.3rem; }

/* Violation alert card */
.vio-card {
    background: linear-gradient(135deg,#fff5f5,#fff);
    border: 2px solid #e53935; border-radius: 10px;
    padding: 14px 18px; margin: 8px 0;
    box-shadow: 0 2px 8px rgba(229,57,53,0.15);
}
.vio-plate {
    font-size: 1.4rem; font-weight: 800; color: #b71c1c;
    letter-spacing: 2px; font-family: monospace;
}
.vio-meta { font-size: 0.82rem; color: #555; margin-top: 4px; }
.vio-fine  { font-size: 1rem; font-weight: 700; color: #c62828; }

/* Alert banner */
.alert-banner {
    background: #c62828; color: white;
    padding: 12px 20px; border-radius: 8px;
    font-size: 1.1rem; font-weight: 700;
    text-align: center; margin: 6px 0;
    animation: pulse 1s infinite;
}
@keyframes pulse {
  0%,100% { opacity:1; } 50% { opacity:0.75; }
}

/* Detection log rows */
.det-row {
    background:#f8f9fa; border-left:4px solid #1a237e;
    padding: 8px 12px; border-radius:6px; margin:4px 0;
    font-size: 0.84rem;
}
.det-restricted { border-left-color: #c62828; }

/* Sidebar */
section[data-testid="stSidebar"] { background-color: #0d1b2a; }
section[data-testid="stSidebar"] * { color: #dde6f0 !important; }

/* Status badges */
.badge {
    display:inline-block; padding:2px 10px; border-radius:20px;
    font-weight:700; font-size:0.76rem; margin:2px;
}
.badge-red    { background:#ffebee; color:#c62828; }
.badge-green  { background:#e8f5e9; color:#2e7d32; }
.badge-orange { background:#fff3e0; color:#e65100; }
.badge-blue   { background:#e3f2fd; color:#1565c0; }
</style>
""", unsafe_allow_html=True)


# ── Session‑state init ────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "db_path":   os.path.join(project_root, "data", "violations", "violations.db"),
        "logger":    None,
        "running":   False,
        "vio_feed":  [],   # live violation cards produced during current run
        "det_log":   [],   # text log of detected vehicles
        "proc_cnt":  0,
        "vio_cnt":   0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if st.session_state.logger is None:
        st.session_state.logger = ViolationLogger(st.session_state.db_path)


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar() -> str:
    st.sidebar.markdown("## 🚦 Flyover Enforcement")
    st.sidebar.markdown("**Kerala MVD · Automated System**")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigate",
        ["Dashboard", "Live Detection", "Violations",
         "Manual Review", "DIP Debug", "Settings"],
        key="nav_radio",
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(f"🕐 {datetime.now().strftime('%H:%M:%S')}")

    # Working refresh button
    if st.sidebar.button("🔄 Refresh", key="refresh_btn", width='stretch'):
        # Re-create logger so stats are fresh
        st.session_state.logger = ViolationLogger(st.session_state.db_path)
        st.rerun()

    return page


# ── Stat cards ────────────────────────────────────────────────────────────────
def render_stats():
    stats = st.session_state.logger.get_stats()
    cards = [
        ("#1a237e,#283593", stats["total"],         "Total"),
        ("#c62828,#e53935", stats["today"],         "Today"),
        ("#1565c0,#1976d2", stats["this_week"],     "This Week"),
        ("#e65100,#ef6c00", stats["pending"],       "Pending"),
        ("#2e7d32,#388e3c", stats["sent"],          "Sent"),
        ("#6a1b9a,#7b1fa2", stats["manual_review"], "Review"),
    ]
    cols = st.columns(6)
    for col, (g, v, lbl) in zip(cols, cards):
        col.markdown(
            f'<div class="stat-card" style="background:linear-gradient(135deg,{g});">'
            f'<div class="stat-number">{v}</div>'
            f'<div class="stat-label">{lbl}</div></div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Dashboard
# ══════════════════════════════════════════════════════════════════════════════
def render_dashboard():
    st.markdown('<div class="main-header">🚦 Flyover Enforcement Dashboard</div>',
                unsafe_allow_html=True)
    render_stats()
    st.markdown("---")

    col_snap, col_recent = st.columns([2, 1])

    with col_snap:
        st.subheader("📂 Footage Folder")
        os.makedirs(FOOTAGE_DIR, exist_ok=True)
        videos = [f for f in os.listdir(FOOTAGE_DIR)
                  if f.lower().endswith((".mp4",".avi",".mov",".mkv"))]
        if videos:
            st.success(f"**{len(videos)} video(s)** found in `data/raw_footage/`")
            for v in videos:
                sz = os.path.getsize(os.path.join(FOOTAGE_DIR, v)) / 1024 / 1024
                st.markdown(
                    f'<div class="det-row">🎬 <strong>{v}</strong> — {sz:.1f} MB</div>',
                    unsafe_allow_html=True,
                )
            st.info("👈 Go to **Live Detection** to process a video.")
        else:
            st.warning(
                "No videos found. Drop your footage files (.mp4/.avi/.mov) "
                "into `data/raw_footage/` and click Refresh."
            )

        # Latest snapshots
        snap_dir = os.path.join(project_root, "data", "violations")
        snaps = sorted([f for f in os.listdir(snap_dir) if f.endswith(".jpg")],
                       reverse=True)[:4]
        if snaps:
            st.markdown("**Latest Violation Snapshots**")
            img_cols = st.columns(min(len(snaps), 4))
            for ic, s in zip(img_cols, snaps):
                ic.image(os.path.join(snap_dir, s), caption=s,
                         width='stretch')

    with col_recent:
        st.subheader("🔴 Recent Violations")
        violations = st.session_state.logger.get_all_violations(limit=15)
        if violations:
            for v in violations:
                st.markdown(
                    f'<div class="vio-card">'
                    f'<div class="vio-plate">{v["plate"]}</div>'
                    f'<div class="vio-meta">'
                    f'{v["vehicle_class"]} · {v["timestamp"]}<br>'
                    f'{v["location"]}</div>'
                    f'<div class="vio-fine">₹{v["fine_amount"]} — '
                    f'<span class="badge badge-orange">{v["status"].upper()}</span>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No violations recorded yet.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Live Detection
# ══════════════════════════════════════════════════════════════════════════════
def render_live_detection():
    st.markdown('<div class="main-header">📹 Live Detection</div>',
                unsafe_allow_html=True)

    # ── Scan footage folder ───────────────────────────────────────────────────
    os.makedirs(FOOTAGE_DIR, exist_ok=True)
    videos = sorted([
        f for f in os.listdir(FOOTAGE_DIR)
        if f.lower().endswith((".mp4",".avi",".mov",".mkv"))
    ])

    if not videos:
        st.warning(
            "📂 No video files found in `data/raw_footage/`.\n\n"
            "Place your footage files there (e.g. `test.mp4`) then click 🔄 Refresh."
        )
        return

    st.info(f"📂 Found **{len(videos)}** video(s) in `data/raw_footage/`")

    # ── Video selector ────────────────────────────────────────────────────────
    selected_video = st.selectbox(
        "Select footage to analyse",
        videos,
        key="sel_video",
    )
    video_path = os.path.join(FOOTAGE_DIR, selected_video)

    # ── Settings ──────────────────────────────────────────────────────────────
    with st.expander("⚙ Detection Settings", expanded=False):
        c1, c2, c3 = st.columns(3)
        conf_thresh = c1.slider("YOLO confidence",      0.2, 0.9, 0.45, 0.05, key="d_conf")
        every_n     = c2.slider("Process every N frames", 1, 15, 5, 1,          key="d_n")
        max_frames  = c3.slider("Max frames to analyse", 50, 2000, 300, 50,     key="d_max")
        night_force = st.checkbox("Force night-mode DIP", False, key="d_night")

    # ── Control buttons ───────────────────────────────────────────────────────
    col_run, col_stop, _ = st.columns([1, 1, 4])
    run_clicked  = col_run.button("▶ Start Detection",  type="primary", key="btn_run")
    stop_clicked = col_stop.button("⏹ Stop",            key="btn_stop")

    if stop_clicked:
        st.session_state.running = False
        st.info("Detection stopped.")

    if not run_clicked and not st.session_state.running:
        st.markdown("---")
        st.markdown(
            "**How it works:**  \n"
            "1. Select a video above  \n"
            "2. Click **Start Detection**  \n"
            "3. Violations, detected two-wheelers, and number plates appear below in real-time  \n"
            "4. All violations are auto-saved to the database and a PDF fine report is generated"
        )
        return

    st.session_state.running = True
    # Reset per-run state
    st.session_state.vio_feed = []
    st.session_state.det_log  = []
    st.session_state.proc_cnt = 0
    st.session_state.vio_cnt  = 0

    # ── Import pipeline modules ───────────────────────────────────────────────
    try:
        from src.detection.vehicle_detector import VehicleDetector
        from src.detection.roi_manager import ROIManager
        from src.detection.motion_filter import MotionFilter
        from src.anpr.plate_detector import PlateDetector
        from src.anpr.plate_dip import PlateDIP
        from src.anpr.ocr_engine import OCREngine
        from src.violation.logic_engine import ViolationEngine
        from src.violation.deduplicator import ViolationDeduplicator
        from src.reporting.pdf_generator import FineReportGenerator
    except Exception as ex:
        st.error(f"Import error: {ex}")
        st.session_state.running = False
        return

    # ── Open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"❌ Cannot open `{video_path}`")
        st.session_state.running = False
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25
    w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    st.success(
        f"🎬 **{selected_video}** — {w}×{h} @ {fps:.0f} fps · {total_frames} frames"
    )
    st.markdown("---")

    # ── Init modules ──────────────────────────────────────────────────────────
    config_path = os.path.join(project_root, "config", "settings.yaml")
    night_proc  = NightModeProcessor()
    enhancer    = Enhancer()
    denoiser    = NoiseRemover()
    roi_mgr     = ROIManager(config_path)
    motion_flt  = MotionFilter()
    detector    = VehicleDetector(conf_threshold=conf_thresh)
    plate_det   = PlateDetector()
    plate_dip_  = PlateDIP()
    ocr         = OCREngine()
    vio_eng     = ViolationEngine(
        snapshot_dir=os.path.join(project_root, "data", "violations")
    )
    dedup       = ViolationDeduplicator()
    pdf_gen     = FineReportGenerator()

    # ── UI layout: stats + alert feed side by side ────────────────────────────
    col_stats, col_alerts = st.columns([1, 2])

    with col_stats:
        st.markdown("### 📊 Stats")
        stat_frame   = st.empty()
        stat_vio     = st.empty()
        stat_mode    = st.empty()
        st.markdown("---")
        st.markdown("### 🚗 Detection Log")
        det_ph       = st.empty()

    with col_alerts:
        st.markdown("### 🚨 Violation Alerts")
        alert_ph     = st.empty()

    progress_bar = st.progress(0)

    # ── Processing loop ───────────────────────────────────────────────────────
    frame_idx    = 0
    proc_cnt     = 0
    vio_cnt      = 0
    det_log      = []    # last N detection lines
    vio_cards    = []    # accumulated violation HTML cards

    try:
        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # Skip frames
            if frame_idx % every_n != 0:
                continue
            if proc_cnt >= max_frames:
                break
            proc_cnt += 1

            # ── DIP preprocessing ─────────────────────────────────────────
            is_night = night_force or night_proc.detect_night(frame)
            if is_night:
                enhanced = night_proc.process(denoiser.bilateral_filter(frame))
            else:
                enhanced = enhancer.auto_enhance(
                    denoiser.gaussian_blur(frame, ksize=3), is_night=False
                )

            # ── Motion + detection ────────────────────────────────────────
            motion_mask = motion_flt.get_moving_mask(enhanced)
            detections  = detector.detect(enhanced)
            det_dicts   = [
                {"bbox": d.bbox, "class_id": d.class_id,
                 "class_name": d.class_name, "confidence": d.confidence,
                 "is_restricted": d.is_restricted, "label": d.label}
                for d in detections
            ]
            moving = motion_flt.filter_detections(det_dicts, motion_mask)

            for det in moving:
                x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
                restricted = det["is_restricted"]
                vtype = det["label"]
                conf  = det["confidence"]

                badge = "badge-red" if restricted else "badge-green"
                flag  = "🚫 RESTRICTED" if restricted else "✅ ALLOWED"
                det_cls = "det-row det-restricted" if restricted else "det-row"
                det_log.insert(0, (
                    f'<div class="{det_cls}">'
                    f'<span class="badge {badge}">{flag}</span> '
                    f'<strong>{vtype}</strong> — conf: {conf:.0%} '
                    f'· frame {frame_idx}</div>'
                ))

                if not restricted:
                    continue

                # ROI gate
                cx, cy = (x1+x2)//2, (y1+y2)//2
                if not roi_mgr.is_in_roi((cx, cy)):
                    continue

                # ── ANPR ──────────────────────────────────────────────────
                crop = enhanced[max(y1,0):y2, max(x1,0):x2]
                plate_text = None

                if crop.size > 0:
                    pb = plate_det.detect_plate(crop)
                    if pb:
                        pc = plate_det.crop_plate(crop, pb)
                        if pc.size > 0:
                            cleaned = plate_dip_.preprocess(pc)
                            plate_text, is_valid, _ = ocr.process(cleaned)

                # ── Violation evaluation ───────────────────────────────────
                violation = vio_eng.evaluate(
                    det, plate_text, datetime.now(), enhanced, is_in_roi=True
                )
                if violation and dedup.check_and_record(violation.plate):
                    vio_cnt += 1
                    st.session_state.logger.log_violation(violation)
                    pdf_path = pdf_gen.generate_pdf(violation)
                    if pdf_path:
                        violation.pdf_path = pdf_path
                        st.session_state.logger.update_pdf_path(
                            violation.id, pdf_path
                        )

                    # Build violation card HTML
                    snap_html = ""
                    if violation.snapshot_path and os.path.exists(
                        violation.snapshot_path
                    ):
                        snap_html = (
                            f'<img src="file://{violation.snapshot_path}" '
                            f'style="width:100%;border-radius:6px;margin-top:6px;" '
                            f'alt="snapshot" />'
                        )

                    card_html = (
                        f'<div class="vio-card">'
                        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                        f'<div class="vio-plate">{violation.plate}</div>'
                        f'<div class="badge badge-red">#{vio_cnt} &nbsp;🔔</div>'
                        f'</div>'
                        f'<div class="vio-meta">'
                        f'🏍 <strong>{violation.vehicle_class}</strong> · '
                        f'{violation.timestamp}<br>'
                        f'📍 {violation.location}</div>'
                        f'<div class="vio-fine" style="margin-top:4px;">'
                        f'💰 Fine: ₹{violation.fine_amount} &nbsp;'
                        f'<span class="badge badge-orange">'
                        f'{violation.status.upper()}</span></div>'
                        f'{"<div>" + snap_html + "</div>" if snap_html else ""}'
                        f'</div>'
                    )
                    vio_cards.insert(0, card_html)

                    # Flashing alert banner
                    alert_ph.markdown(
                        f'<div class="alert-banner">'
                        f'🚨 VIOLATION DETECTED! &nbsp;&nbsp;'
                        f'{violation.plate} &nbsp;·&nbsp; {violation.vehicle_class}'
                        f'</div>'
                        + "".join(vio_cards[:10]),
                        unsafe_allow_html=True,
                    )

            # ── Update stats column ───────────────────────────────────────
            pct = min(frame_idx / max(total_frames, 1), 1.0)
            progress_bar.progress(pct)

            with col_stats:
                stat_frame.markdown(
                    f'<div class="det-row">'
                    f'📽 Frame <strong>{frame_idx}</strong> / {total_frames}<br>'
                    f'Analysed: <strong>{proc_cnt}</strong></div>',
                    unsafe_allow_html=True,
                )
                stat_vio.markdown(
                    f'<div class="det-row det-restricted">'
                    f'🚨 Violations: <strong style="font-size:1.4rem;">'
                    f'{vio_cnt}</strong></div>',
                    unsafe_allow_html=True,
                )
                night_txt = "🌙 Night Mode" if is_night else "☀ Day Mode"
                stat_mode.markdown(
                    f'<div class="det-row">{night_txt}</div>',
                    unsafe_allow_html=True,
                )
                det_ph.markdown(
                    "".join(det_log[:12]),
                    unsafe_allow_html=True,
                )

            # If no violations yet, show waiting message
            if vio_cnt == 0:
                alert_ph.markdown(
                    '<div class="det-row">⏳ Monitoring... no violations detected yet.</div>',
                    unsafe_allow_html=True,
                )

    finally:
        cap.release()
        st.session_state.running = False

    # ── Final summary ─────────────────────────────────────────────────────────
    progress_bar.progress(1.0)
    st.markdown("---")
    if vio_cnt > 0:
        st.success(
            f"✅ Processing complete! "
            f"Frames: {proc_cnt} | **{vio_cnt} violation(s) recorded** | "
            f"PDFs saved to `data/violations/reports/`"
        )
    else:
        st.info(
            f"✅ Processing complete — {proc_cnt} frames analysed. "
            f"No violations detected (check ROI settings or confidence threshold)."
        )
    # Refresh logger stats
    st.session_state.logger = ViolationLogger(st.session_state.db_path)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Violations
# ══════════════════════════════════════════════════════════════════════════════
def render_violations():
    st.markdown('<div class="main-header">📋 Violation Records</div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    plate_filter  = c1.text_input("🔍 Filter by Plate", "", key="vf_plate")
    status_filter = c2.selectbox(
        "Status", ["All","pending","sent","manual_review","rejected"],
        key="vf_status"
    )
    limit = c3.number_input("Max rows", 10, 2000, 200, key="vf_limit")

    violations = st.session_state.logger.get_all_violations(limit=int(limit))
    if not violations:
        st.info("No violations found."); return

    df = pd.DataFrame(violations)
    if plate_filter:
        df = df[df["plate"].str.contains(plate_filter.upper(), na=False)]
    if status_filter != "All":
        df = df[df["status"] == status_filter]

    st.metric("Matching records", len(df))
    st.dataframe(
        df[["id","plate","vehicle_class","timestamp","fine_amount",
            "status","confidence","location"]],
        width='stretch', hide_index=True,
    )

    csv_data = df.to_csv(index=False)
    st.download_button(
        "📥 Download CSV", csv_data,
        "violations_export.csv", "text/csv", key="dl_csv"
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Manual Review
# ══════════════════════════════════════════════════════════════════════════════
def render_manual_review():
    st.markdown('<div class="main-header">👁 Manual Review</div>',
                unsafe_allow_html=True)

    violations   = st.session_state.logger.get_all_violations(limit=500)
    review_items = [v for v in violations if v["status"] == "manual_review"]

    if not review_items:
        st.success("✅ No violations pending manual review!")
        return

    st.warning(f"⚠ {len(review_items)} violations need review")

    for v in review_items:
        with st.expander(f"🔍 {v['id']} — {v['plate']} ({v['timestamp']})"):
            ci, cd = st.columns([1,1])
            with ci:
                snap = v.get("snapshot_path","")
                if snap and os.path.exists(snap):
                    st.image(snap, caption="CCTV Snapshot", width='stretch')
                else:
                    st.info("No snapshot available")
            with cd:
                st.write(f"**Plate:** {v['plate']}")
                st.write(f"**Vehicle:** {v['vehicle_class']}")
                st.write(f"**Time:** {v['timestamp']}")
                st.write(f"**Location:** {v['location']}")
                st.write(f"**Fine:** ₹{v['fine_amount']}")
                st.write(f"**Confidence:** {v.get('confidence',0):.1%}")
                ca, cr = st.columns(2)
                if ca.button("✅ Approve", key=f"apv_{v['id']}"):
                    st.session_state.logger.update_status(v["id"],"pending")
                    st.rerun()
                if cr.button("❌ Reject",  key=f"rej_{v['id']}"):
                    st.session_state.logger.update_status(v["id"],"rejected")
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DIP Debug
# ══════════════════════════════════════════════════════════════════════════════
def render_dip_debug():
    st.markdown('<div class="main-header">🔬 DIP Debug Panel</div>',
                unsafe_allow_html=True)
    st.markdown("Upload a CCTV frame to visualise each DIP step.")

    uploaded = st.file_uploader(
        "📷 Upload frame (jpg/png)", type=["jpg","jpeg","png","bmp"],
        key="dip_upload"
    )
    if not uploaded:
        return

    frame = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        st.error("Failed to decode image."); return

    night_proc = NightModeProcessor()
    enhancer   = Enhancer()
    denoiser   = NoiseRemover()
    is_night   = night_proc.detect_night(frame)
    mean_b     = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))

    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
             caption="Original Frame", width='stretch')
    st.info(f"{'🌙 NIGHT' if is_night else '☀ DAY'} MODE  (mean brightness = {mean_b:.1f})")
    st.markdown("---")

    c1, c2 = st.columns(2)
    apply_night   = c1.checkbox("Night Mode pipeline", is_night,  key="dd_night")
    apply_clahe   = c1.checkbox("CLAHE",               True,      key="dd_clahe")
    clip_limit    = c1.slider("CLAHE clip limit", 1.0, 5.0, 3.0, 0.5, key="dd_clip")
    apply_denoise = c2.checkbox("Bilateral Denoise",   True,      key="dd_denoise")
    apply_sharpen = c2.checkbox("Unsharp Mask",        True,      key="dd_sharpen")
    gamma_val     = c2.slider("Gamma (< 1 = brighter)", 0.25, 3.0, 1.0, 0.05, key="dd_gamma")

    processed = frame.copy()
    if apply_denoise:  processed = denoiser.bilateral_filter(processed)
    if gamma_val != 1.0: processed = enhancer.gamma_correction(processed, gamma=gamma_val)
    if apply_clahe:    processed = enhancer.clahe(processed, clip_limit=clip_limit)
    if apply_sharpen:  processed = enhancer.unsharp_mask(processed)
    if apply_night:    processed = night_proc.process(frame)

    co, ce = st.columns(2)
    co.markdown("**Original**")
    co.image(cv2.cvtColor(frame,     cv2.COLOR_BGR2RGB), width='stretch')
    ce.markdown("**Enhanced**")
    ce.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), width='stretch')

    if apply_night:
        st.subheader("🌙 Night Mode: Step by Step")
        grid = night_proc.visualize_pipeline_steps(frame)
        st.image(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB),
                 caption="Steps: Original → Gamma → Bilateral → CLAHE → Unsharp → Result",
                 width='stretch')

    st.markdown("---")
    st.subheader("🔤 Plate DIP Pipeline")
    plate_up = st.file_uploader("Upload plate crop", type=["jpg","jpeg","png"],
                                key="dd_plate")
    if plate_up:
        pimg = cv2.imdecode(np.frombuffer(plate_up.read(), np.uint8), cv2.IMREAD_COLOR)
        if pimg is not None:
            pdip = PlateDIP()
            st.image(
                cv2.cvtColor(pdip.visualize_pipeline(pimg), cv2.COLOR_BGR2RGB),
                caption="Original → Resize → Gray → Denoise → Deskew → Binarize → Morph → Pad",
                width='stretch',
            )
            st.image(pdip.preprocess(pimg), caption="Final (fed to OCR)", width=300)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Settings
# ══════════════════════════════════════════════════════════════════════════════
def render_settings():
    st.markdown('<div class="main-header">⚙ Settings</div>', unsafe_allow_html=True)

    cfg = os.path.join(project_root, "config", "settings.yaml")
    if os.path.exists(cfg):
        with open(cfg) as f:
            st.code(f.read(), language="yaml")
    else:
        st.warning("Config file not found")

    st.markdown("---")
    st.subheader("Database")
    st.write(f"`{st.session_state.db_path}`")
    st.json(st.session_state.logger.get_stats())


# ══════════════════════════════════════════════════════════════════════════════
# Entry point — called ONCE by Streamlit
# ══════════════════════════════════════════════════════════════════════════════
def main():
    init_session()
    page = render_sidebar()

    if   page == "Dashboard":      render_dashboard()
    elif page == "Live Detection":  render_live_detection()
    elif page == "Violations":      render_violations()
    elif page == "Manual Review":   render_manual_review()
    elif page == "DIP Debug":       render_dip_debug()
    elif page == "Settings":        render_settings()


if __name__ == "__main__":
    main()

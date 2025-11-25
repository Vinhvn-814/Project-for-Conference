import os
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
# Removed geopandas and xarray imports for lightweight cloud deployment

import streamlit as st
import pydeck as pdk

# Lưu ý: Phiên bản này loại bỏ việc tải file .zarr
# DATA_DIR = r"C:\Users\Victus\Downloads\Conference"
# CURR_ZARR = os.path.join(DATA_DIR, "curr1.zarr")

# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide", page_title="PlasticSource", page_icon="🌊")
st.title("PlasticSource - Mobile Simulation")
st.caption("Backward Trajectory Simulation (Optimized for Cloud Deployment)")

# Physical constants and limits
METERS_PER_DEG_LAT   = 111320.0
MAX_PATHS_TO_DRAW    = 300
MAX_POINTS_PER_PATH  = 20000
LIGHT_CSV_MAX_STEPS  = 200
RANDOM_SEED_FOR_VIEW = 42
SPEED_CAP_MPS        = 3.0   # Speed cap to prevent numerical instability

# Start point optimization parameters
AUTO_OFFSHORE_START_KM = 100.0
AUTO_OFFSHORE_STEP_M   = 500.0
MIN_START_SPEED        = 0.05  # Minimum velocity required to start (avoid dead zones)

# ---- Start Beaches (Optimized Coordinates) ----
# Sử dụng các điểm trung tâm đại diện cho mục đích mô phỏng
TAIWAN_BEACHING_SITES = {
    "North - Keelung/Yehliu": (121.70, 25.20),
    "East - Hualien":         (121.65, 23.90),
    "East - Taitung":         (121.35, 22.80),
    "South - Kaohsiung":      (120.20, 22.60),
    "West - Hsinchu":         (120.80, 24.85),
}

# Define Local Domain (Taiwan vicinity) for source analysis
LOCAL_BOX = {
    "lon_min": 119.5, "lon_max": 122.5,
    "lat_min": 21.5,  "lat_max": 25.8
}

# ---------------- Math Tools ----------------
def meters_to_deg_lon(m, lat):
    return m / (METERS_PER_DEG_LAT * np.cos(np.deg2rad(lat)) + 1e-12)

def meters_to_deg_lat(m):
    return m / METERS_PER_DEG_LAT

def haversine(lon1, lat1, lon2, lat2):
    """Calculate distance between two points on Earth (meters)"""
    lon1 = np.asarray(lon1); lat1 = np.asarray(lat1)
    lon2 = np.asarray(lon2); lat2 = np.asarray(lat2)
    R = 6371000.0
    phi1 = np.deg2rad(lat1); phi2 = np.deg2rad(lat2)
    dphi = np.deg2rad(lat2-lat1); dl = np.deg2rad(lon2-lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dl/2)**2
    return R*2*np.arctan2(np.sqrt(a), np.sqrt(1-a))

# ---------------- Land (Beaching) - SIMPLIFIED/REMOVED ----------------
# Logic kiểm tra đất liền phức tạp đã bị loại bỏ cho phiên bản Cloud nhẹ
def is_land(lon, lat):
    return False

def check_beach(lon_arr, lat_arr):
    return np.zeros_like(np.asarray(lon_arr, dtype=bool))

# ---------------- MOCK VELOCITY PROVIDER ----------------
# Phần này thay thế logic tải Zarr bằng một mô hình toán học đơn giản
@st.cache_data(show_spinner=False)
def get_mock_velocity(lon, lat):
    """Mô phỏng trường vận tốc (U, V tính bằng m/s) dựa trên vị trí."""
    lon = np.asarray(lon, dtype="float64")
    lat = np.asarray(lat, dtype="float64")
    
    # Dòng chảy cơ bản (Giống Kuroshio hướng Bắc)
    v_base = 0.5 + 0.5 * (lon - 120.0) / 2.0
    u_base = 0.1 - 0.2 * (lon - 120.0) / 2.0 
    
    # Thêm biến thiên không gian nhỏ
    noise_u = np.sin(lat * 8) * 0.1 * np.cos(lon * 5)
    noise_v = np.cos(lon * 7) * 0.1 * np.sin(lat * 6)
    
    u = u_base + noise_u
    v = v_base + noise_v
    
    # Áp dụng giới hạn tốc độ
    spd = np.hypot(u, v)
    over = spd > SPEED_CAP_MPS
    if np.any(over):
        scale = SPEED_CAP_MPS / (spd + 1e-12)
        u = np.where(over, u*scale, u)
        v = np.where(over, v*scale, v)

    # Áp dụng kiểm tra tốc độ tối thiểu
    slow = spd < MIN_START_SPEED
    if np.any(slow):
        u[slow] = 0.0
        v[slow] = 0.0
        
    # Cấu trúc nhà cung cấp giả để tương thích với các hàm khác
    mock_provider = {
        "curr": {
            "u": np.array([u]), # 1 lát thời gian
            "v": np.array([v]), # 1 lát thời gian
            "bbox": (118.0, 125.0, 20.0, 27.0) # Khung giới hạn giả
        }
    }
    return u, v, mock_provider

# ---------------- Interpolation & Time Logic - SIMPLIFIED ----------------
def get_velocity(lon_arr, lat_arr, prov, hours_elapsed):
    """Truy xuất vận tốc từ nhà cung cấp giả (bỏ qua nội suy thời gian)."""
    # Vì dữ liệu giả là tĩnh, chúng ta gọi hàm tính toán giả trực tiếp.
    uc, vc, _ = get_mock_velocity(lon_arr, lat_arr)
    
    # Tính toán lại giới hạn tốc độ cho bước vận tốc cuối cùng
    spd = np.hypot(uc, vc)
    over = spd > SPEED_CAP_MPS
    if np.any(over):
        scale = SPEED_CAP_MPS / (spd + 1e-12)
        uc = np.where(over, uc*scale, uc)
        vc = np.where(over, vc*scale, vc)
        
    return uc, vc

def rk4_step(lon, lat, dt_seconds, prov, step_idx, total_steps):
    """Bước tích hợp Runge-Kutta bậc 4."""
    lon = np.asarray(lon, dtype="float64")
    lat = np.asarray(lat, dtype="float64")
    
    hours_elapsed = (step_idx * abs(dt_seconds)) / 3600.0
    
    def get_v(l, la): return get_velocity(l, la, prov, hours_elapsed)
    def to_dlon(u, lat_here): return u / (METERS_PER_DEG_LAT*np.cos(np.deg2rad(lat_here)) + 1e-12)
    def to_dlat(v): return v / METERS_PER_DEG_LAT

    u1, v1 = get_v(lon, lat)
    dlon1 = to_dlon(u1, lat); dlat1 = to_dlat(v1)

    lon2 = lon + 0.5*dt_seconds*dlon1; lat2 = lat + 0.5*dt_seconds*dlat1
    u2, v2 = get_v(lon2, lat2)
    dlon2 = to_dlon(u2, lat2); dlat2 = to_dlat(v2)

    lon3 = lon + 0.5*dt_seconds*dlon2; lat3 = lat + 0.5*dt_seconds*dlat2
    u3, v3 = get_v(lon3, lat3)
    dlon3 = to_dlon(u3, lat3); dlat3 = to_dlat(v3)

    lon4 = lon + dt_seconds*dlon3; lat4 = lat + dt_seconds*dlat3
    u4, v4 = get_v(lon4, lat4)
    dlon4 = to_dlon(u4, lat4); dlat4 = to_dlat(v4)

    new_lon = lon + (dt_seconds/6.0)*(dlon1 + 2*dlon2 + 2*dlon3 + dlon4)
    new_lat = lat + (dt_seconds/6.0)*(dlat1 + 2*dlat2 + 2*dlat3 + dlat4)
    return new_lon, new_lat

# ---------------- Offshore Utilities - SIMPLIFIED ----------------
def _inside_bbox(lon, lat, bbox):
    return (bbox[0] <= lon <= bbox[1]) and (bbox[2] <= lat <= bbox[3])

def _valid_water(lon, lat, prov):
    # Kiểm tra đơn giản cho dữ liệu giả
    u, v, _ = get_mock_velocity(lon, lat)
    return np.hypot(u, v) >= MIN_START_SPEED

def snap_to_valid_ocean(lon0, lat0, prov, max_km=AUTO_OFFSHORE_START_KM):
    # Trong mô hình giả này, chúng ta giả định điểm bắt đầu là hợp lệ nếu vận tốc đủ lớn.
    st.write(f"Using initial point {lon0:.2f}, {lat0:.2f} for simulation.")
    return float(lon0), float(lat0)

# ---------------- Main Loop ----------------
def run_backtracking(release_sites, n_particles_per_site, n_steps, dt_minutes, prov):
    dt_seconds = -abs(dt_minutes) * 60.0
    run_id = f"run-{uuid.uuid4().hex[:8]}"
    rng = np.random.default_rng(RANDOM_SEED_FOR_VIEW)

    nudged_sites = []
    for site_name, lon0, lat0 in release_sites:
        # snap_to_valid_ocean đã được đơn giản hóa
        lon1, lat1 = snap_to_valid_ocean(lon0, lat0, prov) 
        nudged_sites.append((site_name, lon1, lat1))

    parts = []
    for site_idx, (site_name, lon0, lat0) in enumerate(nudged_sites):
        # Phân tán các hạt nhẹ (bán kính 2 km)
        rs = np.sqrt(rng.random(n_particles_per_site))*2000.0 
        ang = rng.random(n_particles_per_site)*2*np.pi
        dx, dy = rs*np.cos(ang), rs*np.sin(ang)
        lons = lon0 + meters_to_deg_lon(dx, lat0)
        lats = lat0 + meters_to_deg_lat(dy)
        for pid in range(n_particles_per_site):
            parts.append((f"{site_idx}_{pid}", site_idx, site_name, lons[pid], lats[pid]))

    N = len(parts)
    lon = np.array([p[3] for p in parts], dtype="float64")
    lat = np.array([p[4] for p in parts], dtype="float64")
    pid_list = [p[0] for p in parts]
    # siteidx_l = [p[1] for p in parts] # Không được sử dụng sau này
    sitename_l = [p[2] for p in parts]

    active = np.ones(N, dtype=bool)
    status = np.full(N, "active", dtype=object)
    
    traj_lon = np.zeros((N, n_steps+1))
    traj_lat = np.zeros((N, n_steps+1))
    traj_lon[:,0] = lon
    traj_lat[:,0] = lat

    # Khung giới hạn giả
    bbox = prov["curr"]["bbox"] 

    progress_bar = st.progress(0)
    for s in range(1, n_steps+1):
        idx = np.where(active)[0]
        if idx.size > 0:
            new_lon, new_lat = rk4_step(lon[idx], lat[idx], dt_seconds, prov, s, n_steps)

            oob = (new_lon < bbox[0]) | (new_lon > bbox[1]) | (new_lat < bbox[2]) | (new_lat > bbox[3])
            if np.any(oob):
                # Dừng các hạt OOB, giữ vị trí hợp lệ cuối cùng
                new_lon[oob] = lon[idx][oob]; new_lat[oob] = lat[idx][oob]
                status[idx[oob]] = "oob_stop"; active[idx[oob]] = False

            # Kiểm tra đất liền hiện là một hàm giả (trả về False)
            land = check_beach(new_lon, new_lat)
            if np.any(land):
                new_lon[land] = lon[idx][land]; new_lat[land] = lat[idx][land]
                status[idx[land]] = "beached"; active[idx[land]] = False

            # Cập nhật chỉ các hạt đang hoạt động, không OOB/không đất liền
            upd_mask = ~oob & ~land
            lon[idx[upd_mask]] = new_lon[upd_mask]
            lat[idx[upd_mask]] = new_lat[upd_mask]
        
        traj_lon[:,s] = lon
        traj_lat[:,s] = lat
        
        if s % 10 == 0: progress_bar.progress(s / n_steps)
    
    progress_bar.empty()

    rows = []
    base_time = datetime.utcnow()
    # Sử dụng bước nhảy lớn hơn cho CSV/biểu đồ nhẹ
    step_stride = 5 
    
    for s in range(0, n_steps+1, step_stride):
        t_str = (base_time - timedelta(minutes=s*abs(dt_minutes))).isoformat()
        for i in range(N):
            rows.append({
                "particle_id": pid_list[i],
                "run_id": run_id,
                "site_name": sitename_l[i],
                "lon": traj_lon[i,s],
                "lat": traj_lat[i,s],
                "step_index": s,
                "time": t_str,
                "status": status[i]
            })
            
    return pd.DataFrame(rows), run_id, {"total_particles": N}

# ---------------- UI ----------------
with st.sidebar:
    st.header("Input Parameters")
    
    sample_sites = TAIWAN_BEACHING_SITES
    site_name = st.selectbox("Select Start Point", list(sample_sites.keys()))
    init_lon, init_lat = sample_sites[site_name]
    
    n_particles = st.slider("Number of Particles", 10, 500, 50)
    days = st.slider("Backtracking Days", 1, 30, 7)
    dt_mins = 60 # Kích thước bước cố định cho dữ liệu giả 
    total_steps = int(days * 24 * 60 / dt_mins)
    
    st.info(f"Total Steps: {total_steps} (Step Size: {dt_mins} min)")
    st.info("Mode: Mathematical Simulation (Optimized for Mobile/Cloud)")

# Load Mock Providers 
u_mock, v_mock, providers = get_mock_velocity(init_lon, init_lat)

if st.button("Start Backtracking", use_container_width=True):
    # Lưu tọa độ ban đầu để tính toán thống kê sau
    st.session_state["start_coords"] = (init_lon, init_lat)
    
    with st.spinner("Calculating..."):
        df, run_id, diag = run_backtracking(
            [(site_name, init_lon, init_lat)], 
            n_particles, total_steps, dt_mins, providers
        )
    st.success("Done!")
    st.session_state["df"] = df

if "df" in st.session_state:
    df = st.session_state["df"]
    start_lon, start_lat = st.session_state["start_coords"]
    
    # ★★★ 1. Distance Stats & Source Attribution ★★★
    stats_direct = []
    source_results = {} # pid -> color_list
    
    particles = df.groupby("particle_id")
    
    for pid, group in particles:
        g = group.sort_values("step_index")
        lons = g["lon"].values
        lats = g["lat"].values
        
        # A. Distance (Net Displacement)
        d_direct = haversine(start_lon, start_lat, lons[-1], lats[-1])
        stats_direct.append(d_direct)
        
        # B. Source Attribution (Check last position)
        origin_lon = lons[-1]
        origin_lat = lats[-1]
        is_local = (
            LOCAL_BOX["lon_min"] <= origin_lon <= LOCAL_BOX["lon_max"] and
            LOCAL_BOX["lat_min"] <= origin_lat <= LOCAL_BOX["lat_max"]
        )
        # Green=Local, Red=External/Far-field
        source_results[pid] = [50, 255, 50, 200] if is_local else [255, 50, 50, 200]
        
    avg_direct = np.mean(stats_direct)
    
    # Tóm tắt thống kê
    n_local = sum(1 for c in source_results.values() if c[1] == 255)
    n_total = len(source_results)
    ratio_local = (n_local / n_total) * 100 if n_total > 0 else 0
    ratio_ext = 100 - ratio_local

    st.markdown("### 📊 Drift Statistics & Origin Analysis")
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg. Net Displacement", f"{avg_direct/1000:.1f} km")
    c2.metric("Local Origin Estimate", f"{ratio_local:.1f}%")
    c3.metric("External Origin Estimate", f"{ratio_ext:.1f}%")
    st.divider()
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★

    st.subheader("Trajectory Map")
    pids = df["particle_id"].unique()
    # Chỉ hiển thị tối đa 100 hạt để hiển thị trên thiết bị di động mượt mà hơn
    show_pids = np.random.choice(pids, min(len(pids), 100), replace=False) 
    df_show = df[df["particle_id"].isin(show_pids)]
    
    paths = []
    for pid in show_pids:
        d = df_show[df_show["particle_id"]==pid].sort_values("step_index")
        path = d[["lon", "lat"]].values.tolist()
        # Sử dụng Màu nguồn
        p_color = source_results.get(pid, [200, 200, 200, 200])
        paths.append({"path": path, "pid": pid, "color": p_color})
        
    # Dữ liệu điểm bắt đầu (nơi tìm thấy rác thải nhựa)
    starts = pd.DataFrame([{"lon": start_lon, "lat": start_lat}])
    
    layer_path = pdk.Layer(
        "PathLayer", data=paths, get_path="path",
        get_color="color", # Dynamic Color
        width_min_pixels=2
    )
    layer_scatter = pdk.Layer(
        "ScatterplotLayer", data=starts, get_position=["lon", "lat"],
        get_color=[255, 100, 0, 255], get_radius=2000 # Bán kính lớn hơn để dễ nhìn
    )
    
    view_state = pdk.ViewState(
        latitude=start_lat, longitude=start_lon, zoom=7
    )
    
    st.pydeck_chart(pdk.Deck(
        layers=[layer_path, layer_scatter],
        initial_view_state=view_state,
        map_style="dark"
    ))
    
    st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), "traj.csv")


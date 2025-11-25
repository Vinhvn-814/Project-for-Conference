import os
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
# import geopandas as gpd # REMOVED for cloud deployment
# import xarray as xr # REMOVED for cloud deployment
# from shapely.geometry import Point # REMOVED for cloud deployment
# from shapely.prepared import prep # REMOVED for cloud deployment

import streamlit as st
import pydeck as pdk

# Cấu hình trang - GIỐNG CODE GỐC
st.set_page_config(layout="wide", page_title="PlasticSource")
st.title("PlasticSource")

# ★★★ KHÔNG LOAD ZARR (CHỈ GIỮ LẠI ĐƯỜNG DẪN MANG TÍNH BIỂU TƯỢNG) ★★★
# DATA_DIR = r"C:\Users\Victus\Downloads\Conference" 
# CURR_ZARR = os.path.join(DATA_DIR, "curr1.zarr") 

# Physical constants and limits - GIỐNG CODE GỐC
METERS_PER_DEG_LAT   = 111320.0
MAX_PATHS_TO_DRAW    = 300
MAX_POINTS_PER_PATH  = 20000
LIGHT_CSV_MAX_STEPS  = 200
RANDOM_SEED_FOR_VIEW = 42
SPEED_CAP_MPS        = 3.0   
AUTO_OFFSHORE_START_KM = 100.0
AUTO_OFFSHORE_STEP_M   = 500.0
MIN_START_SPEED        = 0.05  

# ---- Start Beaches (Optimized Coordinates) - GIỐNG CODE GỐC ----
TAIWAN_BEACHING_SITES = {
    "North - Keelung/Yehliu": ((121.60, 25.43), (121.90, 24.98)),
    "East - Hualien":         ((121.55, 24.17), (121.75, 23.67)),
    "East - Taitung":         ((121.28, 22.95), (121.45, 22.70)),
    "South - Kaohsiung":      ((120.08, 22.80), (120.30, 22.45)),
    "West - Hsinchu":         ((120.70, 25.00), (120.90, 24.70)),
}

# Define Local Domain (Taiwan vicinity) for source analysis - GIỐNG CODE GỐC
LOCAL_BOX = {
    "lon_min": 119.5, "lon_max": 122.5,
    "lat_min": 21.5,  "lat_max": 25.8
}

# ---------------- Math Tools - GIỐNG CODE GỐC ----------------
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

# ---------------- Land (Beaching) - SỬ DỤNG MOCK STUB ----------------
# Các hàm này trả về FALSE để tránh dependency GeoPandas
def is_land(lon, lat):
    return False

def check_beach(lon_arr, lat_arr):
    return np.zeros_like(np.asarray(lon_arr, dtype=bool))

# ---------------- MOCK VELOCITY FACTORY - THAY THẾ load_zarr ----------------
@st.cache_resource(show_spinner=False)
def build_velocity_providers(curr_path):
    """Tạo dữ liệu vận tốc giả (Mock Data Provider)"""
    
    # Định nghĩa lưới giả tĩnh (Static Mock Grid)
    mock_lat = np.linspace(20.0, 27.0, 71)
    mock_lon = np.linspace(118.0, 125.0, 71)
    
    LON, LAT = np.meshgrid(mock_lon, mock_lat)
    
    # Dòng chảy giả (U, V) - Giống Kuroshio hướng Bắc
    V = 0.5 + 0.5 * (LON - 120.0) / 2.0 + np.cos(LON * 7) * 0.1 * np.sin(LAT * 6)
    U = 0.1 - 0.2 * (LON - 120.0) / 2.0 + np.sin(LAT * 8) * 0.1 * np.cos(LON * 5)
    
    # Thêm chiều thời gian (1 lát duy nhất)
    u3d = np.array([U], dtype='float32') # (1, lat, lon)
    v3d = np.array([V], dtype='float32') # (1, lat, lon)
    
    curr = {
        "lat": mock_lat, "lon": mock_lon,
        "u": u3d, "v": v3d,
        "bbox": (float(mock_lon[0]), float(mock_lon[-1]), float(mock_lat[0]), float(mock_lat[-1])),
        "n_times": 1, # Quan trọng để hàm nội suy thời gian hoạt động
        "max_speed": float(np.nanmax(np.hypot(u3d, v3d)))
    }
    return {"curr": curr}

# ---------------- Interpolation & Time Logic - GIỮ LẠI LOGIC GỐC ----------------
# Các hàm này sử dụng dữ liệu giả tĩnh 1 lát thời gian (n_times=1)
# nhưng vẫn giữ nguyên cấu trúc tính toán phức tạp của Code.py gốc
def interpolate_2d_slice(lon_arr, lat_arr, u_grid, v_grid, lat_ax, lon_ax):
    """Sử dụng Bilinear Interpolation trên lưới giả tĩnh."""
    x = np.asarray(lon_arr, dtype="float64")
    y = np.asarray(lat_arr, dtype="float64")
    
    x_c = np.clip(x, lon_ax[0], lon_ax[-1])
    y_c = np.clip(y, lat_ax[0], lat_ax[-1])
    
    j = np.searchsorted(lon_ax, x_c) - 1
    i = np.searchsorted(lat_ax, y_c) - 1
    j = np.clip(j, 0, len(lon_ax)-2)
    i = np.clip(i, 0, len(lat_ax)-2)
    
    x0, x1 = lon_ax[j], lon_ax[j+1]
    y0, y1 = lat_ax[i], lat_ax[i+1]
    
    wx = (x_c - x0) / (x1 - x0 + 1e-12)
    wy = (y_c - y0) / (y1 - y0 + 1e-12)
    
    u00 = u_grid[i, j];   u01 = u_grid[i, j+1]
    u10 = u_grid[i+1, j]; u11 = u_grid[i+1, j+1]
    v00 = v_grid[i, j];   v01 = v_grid[i, j+1]
    v10 = v_grid[i+1, j]; v11 = v_grid[i+1, j+1]
    
    u_out = (u00*(1-wx)*(1-wy) + u01*wx*(1-wy) + u10*(1-wx)*wy + u11*wx*wy)
    v_out = (v00*(1-wx)*(1-wy) + v01*wx*(1-wy) + v10*(1-wx)*wy + v11*wx*wy)
    
    oob = (x < lon_ax[0]) | (x > lon_ax[-1]) | (y < lat_ax[0]) | (y > lat_ax[-1])
    u_out[oob] = np.nan
    v_out[oob] = np.nan
    return u_out, v_out

def get_velocity(lon_arr, lat_arr, prov, hours_elapsed):
    curr = prov["curr"]
    n_times = curr["n_times"]
    
    # 1. Frequency check (Simplfied as n_times=1)
    float_idx = hours_elapsed / 24.0
    
    # 2. Calculate Reverse Index (Time Interpolation)
    target_idx = (n_times - 1) - float_idx
    
    # Vì n_times=1, target_idx sẽ < 0, nhưng idx0 và idx1 sẽ được clip về 0
    idx0 = int(np.floor(target_idx))
    idx1 = int(np.ceil(target_idx))
    
    w1 = target_idx - idx0 
    w0 = 1.0 - w1
    
    idx0 = np.clip(idx0, 0, n_times - 1)
    idx1 = np.clip(idx1, 0, n_times - 1)
    
    # Khi idx0=idx1=0, w0=1, w1=0. Chỉ sử dụng lát cắt 0
    u_slice0 = curr["u"][idx0]; v_slice0 = curr["v"][idx0]
    u_slice1 = curr["u"][idx1]; v_slice1 = curr["v"][idx1]
    
    uc0, vc0 = interpolate_2d_slice(lon_arr, lat_arr, u_slice0, v_slice0, curr["lat"], curr["lon"])
    uc1, vc1 = interpolate_2d_slice(lon_arr, lat_arr, u_slice1, v_slice1, curr["lat"], curr["lon"])
    
    uc = uc0 * w0 + uc1 * w1
    vc = vc0 * w0 + vc1 * w1
    
    # Xử lý NaN
    nan_mask = np.isnan(uc) | np.isnan(vc)
    if np.any(nan_mask):
        uc = np.where(nan_mask, 0.0, uc)
        vc = np.where(nan_mask, 0.0, vc)

    # Giới hạn tốc độ
    spd = np.hypot(uc, vc)
    over = spd > SPEED_CAP_MPS
    if np.any(over):
        scale = SPEED_CAP_MPS / (spd + 1e-12)
        uc = np.where(over, uc*scale, uc)
        vc = np.where(over, vc*scale, vc)
        
    return uc, vc

def rk4_step(lon, lat, dt_seconds, prov, step_idx, total_steps):
    """Bước tích hợp Runge-Kutta bậc 4 - GIỐNG CODE GỐC."""
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

# ---------------- Offshore Utilities - MOCK STUB ----------------
# Các hàm này được đơn giản hóa để tránh dependency GeoPandas
def _inside_bbox(lon, lat, bbox):
    return (bbox[0] <= lon <= bbox[1]) and (bbox[2] <= lat <= bbox[3])

def _valid_water(lon, lat, prov):
    if not _inside_bbox(lon, lat, prov["curr"]["bbox"]): return False
    if is_land(lon, lat): return False
    # Check velocity at Time 0
    u0_slice = prov["curr"]["u"][0]
    v0_slice = prov["curr"]["v"][0]
    uc, vc = interpolate_2d_slice([lon], [lat], u0_slice, v0_slice, prov["curr"]["lat"], prov["curr"]["lon"])
    u_val, v_val = uc[0], vc[0]
    
    if np.isnan(u_val) or np.isnan(v_val): return False
    if np.hypot(u_val, v_val) < MIN_START_SPEED: return False
    return True

def snap_to_valid_ocean(lon0, lat0, prov, max_km=AUTO_OFFSHORE_START_KM):
    # Vì là mock, chúng ta chỉ kiểm tra đơn giản và trả về điểm ban đầu
    if _valid_water(lon0, lat0, prov): return float(lon0), float(lat0)
    
    # Quay trở lại điểm ban đầu hoặc điểm gần nhất của BBOX (nếu không hợp lệ)
    bb = prov["curr"]["bbox"]
    return float(np.clip(lon0, bb[0], bb[1])), float(np.clip(lat0, bb[2], bb[3]))

# ---------------- Main Loop - GIỐNG CODE GỐC ----------------
def run_backtracking(release_sites, n_particles_per_site, n_steps, dt_minutes, prov):
    dt_seconds = -abs(dt_minutes) * 60.0
    run_id = f"run-{uuid.uuid4().hex[:8]}"
    rng = np.random.default_rng(RANDOM_SEED_FOR_VIEW)

    # Logic tương tự code gốc
    nudged_sites = []
    for site_name, lon0, lat0 in release_sites:
        lon1, lat1 = snap_to_valid_ocean(lon0, lat0, prov)
        nudged_sites.append((site_name, lon1, lat1))

    # Khởi tạo hạt
    parts = []
    for site_idx, (site_name, lon0, lat0) in enumerate(nudged_sites):
        # Tạo hạt trong bán kính nhỏ 100m (theo code gốc)
        ang = rng.random(n_particles_per_site)*2*np.pi
        rs  = np.sqrt(rng.random(n_particles_per_site))*100.0 
        dx, dy = rs*np.cos(ang), rs*np.sin(ang)
        lons = lon0 + meters_to_deg_lon(dx, lat0)
        lats = lat0 + meters_to_deg_lat(dy)
        for pid in range(n_particles_per_site):
            parts.append((f"{site_idx}_{pid}", site_idx, site_name, lons[pid], lats[pid]))

    N = len(parts)
    lon = np.array([p[3] for p in parts], dtype="float64")
    lat = np.array([p[4] for p in parts], dtype="float64")
    pid_list = [p[0] for p in parts]
    sitename_l = [p[2] for p in parts]

    active = np.ones(N, dtype=bool)
    status = np.full(N, "active", dtype=object)
    
    traj_lon = np.zeros((N, n_steps+1))
    traj_lat = np.zeros((N, n_steps+1))
    traj_lon[:,0] = lon
    traj_lat[:,0] = lat

    bbox = prov["curr"]["bbox"]

    progress_bar = st.progress(0)
    for s in range(1, n_steps+1):
        idx = np.where(active)[0]
        if idx.size > 0:
            new_lon, new_lat = rk4_step(lon[idx], lat[idx], dt_seconds, prov, s, n_steps)

            # Kiểm tra Out of Bounds
            oob = (new_lon < bbox[0]) | (new_lon > bbox[1]) | (new_lat < bbox[2]) | (new_lat > bbox[3])
            if np.any(oob):
                new_lon[oob] = lon[idx][oob]; new_lat[oob] = lat[idx][oob]
                status[idx[oob]] = "oob_stop"; active[idx[oob]] = False

            # Kiểm tra đất liền (luôn FALSE trong mock)
            land = check_beach(new_lon, new_lat)
            if np.any(land):
                new_lon[land] = lon[idx][land]; new_lat[land] = lat[idx][land]
                status[idx[land]] = "beached"; active[idx[land]] = False

            # Cập nhật
            lon[idx] = new_lon
            lat[idx] = new_lat
        
        traj_lon[:,s] = lon
        traj_lat[:,s] = lat
        
        if s % 10 == 0: progress_bar.progress(s / n_steps)
    
    progress_bar.empty()

    rows = []
    base_time = datetime.utcnow()
    # Sử dụng step_stride tương tự code gốc
    step_stride = 1 if n_steps < 1000 else 5 
    
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

# ---------------- UI - KHÔI PHỤC TOÀN BỘ GIAO DIỆN GỐC ----------------
st.sidebar.header("Input Parameters")
sample_sites = {name: ((box[0][0]+box[1][0])/2.0, (box[0][1]+box[1][1])/2.0) for name, box in TAIWAN_BEACHING_SITES.items()}
site_name = st.sidebar.selectbox("Select Start Point", list(sample_sites.keys()))
init_lon, init_lat = sample_sites[site_name]

# Dùng st.sidebar.slider và st.sidebar.selectbox theo code gốc
n_particles = st.sidebar.slider("Number of Particles", 10, 1000, 10)
days = st.sidebar.selectbox("Backtracking Days", [7, 14, 30], index=0)
dt_mins = st.sidebar.slider("Step Size (min)", 10, 120, 10) # Dùng slider và default 10
total_steps = int(days * 24 * 60 / dt_mins)

st.sidebar.info(f"Total Steps: {total_steps}")

# Load Data (Mock)
providers = build_velocity_providers("MOCK_PATH") # Gọi mock factory

if st.button("Start Backtracking", use_container_width=True): # Dùng use_container_width
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
    
    # ★★★ 1. Distance Stats & Source Attribution - GIỐNG CODE GỐC ★★★
    stats_direct = []
    stats_total = [] # Cần tính toán Total Distance
    source_results = {}
    
    particles = df.groupby("particle_id")
    
    for pid, group in particles:
        g = group.sort_values("step_index")
        lons = g["lon"].values
        lats = g["lat"].values
        
        # A. Distance
        d_direct = haversine(lons[0], lats[0], lons[-1], lats[-1])
        stats_direct.append(d_direct)
        
        if len(lons) > 1:
            d_segs = haversine(lons[:-1], lats[:-1], lons[1:], lats[1:])
            d_total = np.sum(d_segs)
        else:
            d_total = 0.0
        stats_total.append(d_total)
        
        # B. Source Attribution (Check last position)
        origin_lon = lons[-1]
        origin_lat = lats[-1]
        is_local = (
            LOCAL_BOX["lon_min"] <= origin_lon <= LOCAL_BOX["lon_max"] and
            LOCAL_BOX["lat_min"] <= origin_lat <= LOCAL_BOX["lat_max"]
        )
        source_results[pid] = [50, 255, 50, 200] if is_local else [255, 50, 50, 200]
        
    avg_direct = np.mean(stats_direct)
    avg_total = np.mean(stats_total) # Tính toán Total Distance
    
    # Stats
    n_local = sum(1 for c in source_results.values() if c[1] == 255)
    n_total = len(source_results)
    ratio_local = (n_local / n_total) * 100 if n_total > 0 else 0
    ratio_ext = 100 - ratio_local

    st.markdown("### 📊 Drift Statistics & Source Analysis")
    c1, c2, c3, c4 = st.columns(4) # Khôi phục 4 cột
    c1.metric("Net Displacement", f"{avg_direct/1000:.1f} km")
    c2.metric("Total Distance", f"{avg_total/1000:.1f} km")
    c3.metric("Local Source", f"{ratio_local:.1f}%")
    c4.metric("External Source", f"{ratio_ext:.1f}%")
    st.divider()
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★

    st.subheader("Trajectory Map")
    pids = df["particle_id"].unique()
    show_pids = np.random.choice(pids, min(len(pids), 100), replace=False) # Giới hạn 100 hạt
    df_show = df[df["particle_id"].isin(show_pids)]
    
    paths = []
    for pid in show_pids:
        d = df_show[df_show["particle_id"]==pid].sort_values("step_index")
        path = d[["lon", "lat"]].values.tolist()
        p_color = source_results.get(pid, [200, 200, 200, 200])
        paths.append({"path": path, "pid": pid, "color": p_color})
        
    starts = df_show[df_show["step_index"]==0] # Lấy các điểm bắt đầu
    
    layer_path = pdk.Layer(
        "PathLayer", data=paths, get_path="path",
        get_color="color", 
        width_min_pixels=2
    )
    layer_scatter = pdk.Layer(
        "ScatterplotLayer", data=starts, get_position=["lon", "lat"],
        get_color=[255, 100, 0, 255], get_radius=150 # Bán kính nhỏ hơn (150)
    )
    
    view_state = pdk.ViewState(
        latitude=starts["lat"].mean(), longitude=starts["lon"].mean(), zoom=6
    )
    
    st.pydeck_chart(pdk.Deck(
        layers=[layer_path, layer_scatter],
        initial_view_state=view_state,
        map_style="dark"
    ))
    
    st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), "traj.csv")

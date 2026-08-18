"""
Hurricane Watch — global tropical-cyclone monitoring board.

Map centrepiece shows every active system worldwide (storms + invests),
with forecast track + cone of uncertainty on the live storms, and a grid
of summary cards for everything being monitored.

Run:
    uv run streamlit run app.py
"""

import math
import time
import threading

import numpy as np
import streamlit as st
import pydeck as pdk
from tropycal import realtime

st.set_page_config(page_title="Hurricane Watch", layout="wide")

st.markdown("""
<style>
  .block-container {padding-top: 1rem; max-width: 100%;}
  .hw-legend span {display:inline-flex; align-items:center; margin-right:16px;
      font-size:.76rem; color:#c4cad6;}
  .hw-dot {width:11px; height:11px; border-radius:50%; margin-right:6px;
      display:inline-block;}
  .hw-card {background:#12151c; border:1px solid #222836; border-radius:12px;
      padding:14px 16px 12px; margin-bottom:12px; border-left:5px solid #444;}
  .hw-card h4 {margin:0 0 2px; font-size:1.02rem; color:#e9ecf2;}
  .hw-card .klass {font-size:.8rem; color:#aab2c0; margin-bottom:10px;}
  .hw-grid {display:flex; gap:16px; font-size:.8rem;}
  .hw-grid div span {display:block; color:#7b8496; font-size:.66rem;
      text-transform:uppercase; letter-spacing:.4px;}
  .hw-grid div b {font-size:.98rem; color:#e9ecf2;}
</style>
""", unsafe_allow_html=True)

st.title("🌀 Hurricane Watch")
st.caption("Live global tropical-cyclone monitor · NHC + JTWC via Tropycal")


# ---------------------------------------------------------------------------
# Classification + colour (Saffir-Simpson, winds in knots)
# ---------------------------------------------------------------------------
def category(vmax):
    v = vmax or 0
    if v >= 137: return 5
    if v >= 113: return 4
    if v >= 96:  return 3
    if v >= 83:  return 2
    if v >= 64:  return 1
    return 0


def color(vmax, invest=False):
    if invest:
        return [150, 160, 175]
    return {
        5: [255, 45, 85], 4: [255, 96, 55], 3: [255, 149, 0],
        2: [255, 214, 10], 1: [255, 245, 120],
    }.get(category(vmax), [48, 209, 220] if (vmax or 0) >= 34 else [100, 160, 255])


def classify(vmax, stype, basin, invest=False):
    if invest:
        return "Invest / Area of Interest"
    v = vmax or 0
    cat = category(v)
    special = {"EX": "Post-Tropical Cyclone", "SS": "Subtropical Storm",
               "SD": "Subtropical Depression", "LO": "Remnant Low",
               "DB": "Disturbance", "WV": "Tropical Wave"}
    if stype in special:
        return special[stype]
    if cat >= 1:
        if basin == "west_pacific":
            return "Super Typhoon" if v >= 130 else "Typhoon"
        if basin in ("north_indian", "south_indian", "australia", "south_pacific"):
            return f"Cyclone (Cat {cat})"
        return f"Hurricane (Cat {cat})"
    if v >= 34:
        return "Tropical Storm"
    return "Tropical Depression"


_COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def movement(track, times):
    if len(track) < 2 or len(times) < 2:
        return "—"
    (lon1, lat1), (lon2, lat2) = track[-2], track[-1]
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlon)
    brg = (math.degrees(math.atan2(y, x)) + 360) % 360
    comp = _COMPASS[round(brg / 22.5) % 16]
    a = (math.sin((p2 - p1) / 2) ** 2 +
         math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2)
    dist = 3440.065 * 2 * math.asin(math.sqrt(a))
    try:
        hrs = (times[-1] - times[-2]).total_seconds() / 3600
        return f"{comp} at {dist / hrs:.0f} kt" if hrs else comp
    except Exception:
        return comp


# ---------------------------------------------------------------------------
# Cone of uncertainty -> polygon ring for a PolygonLayer
# ---------------------------------------------------------------------------
def cone_polygon(forecast, basin):
    try:
        from tropycal import utils
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cone = utils.generate_nhc_cone(forecast, basin, cone_days=5)
        grid = np.asarray(cone["cone"], dtype=float)
        cs = plt.contour(np.asarray(cone["lon"]), np.asarray(cone["lat"]),
                         grid, levels=[0.5])
        best = max(cs.allsegs[0], key=len) if cs.allsegs[0] else None
        plt.close("all")
        if best is None or len(best) < 3:
            return None
        return [[float(x), float(y)] for x, y in best]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Live data — NHC + JTWC, storms + invests, forecast + cone
# ---------------------------------------------------------------------------
def _read(rt, want_cone):
    systems = []
    for sid in rt.list_active_storms():
        try:
            s = rt.get_storm(sid)
            track = [[float(lo), float(la)] for lo, la in zip(s.lon, s.lat)]
            if not track:
                continue
            times = list(getattr(s, "date", []) or getattr(s, "time", []))
            vmax = float(s.vmax[-1]) if len(getattr(s, "vmax", [])) else 0.0
            stype = s.type[-1] if getattr(s, "type", None) is not None else None
            basin = getattr(s, "basin", None)
            invest = bool(getattr(s, "invest", False))
            mslp_raw = s.mslp[-1] if getattr(s, "mslp", None) is not None else None
            mslp = float(mslp_raw) if mslp_raw and not math.isnan(float(mslp_raw)) else None

            # forecast track + cone (live storms only, not invests)
            fc_track, cone = [], None
            if not invest:
                try:
                    fc = s.get_forecast_realtime()
                    fc_track = [[float(lo), float(la)]
                                for lo, la in zip(fc["lon"], fc["lat"])]
                    if want_cone and fc_track:
                        cone = cone_polygon(fc, basin or "north_atlantic")
                except Exception:
                    pass

            systems.append({
                "id": sid, "name": str(s.name).title(),
                "basin": basin, "invest": invest,
                "track": track, "pos": track[-1],
                "fc_track": ([track[-1]] + fc_track) if fc_track else [],
                "cone": cone,
                "vmax": vmax, "mslp": mslp,
                "klass": classify(vmax, stype, basin, invest),
                "cat": category(vmax),
                "move": movement(track, times),
                "time": times[-1].strftime("%d %b %H:%MZ")
                        if times and hasattr(times[-1], "strftime") else "—",
            })
        except Exception:
            continue
    return systems


# NHC (Atlantic / E-Pac / C-Pac) — fast, shown immediately.
@st.cache_data(ttl=600, show_spinner="Loading storms (NHC)…")
def get_nhc():
    try:
        return _read(realtime.Realtime(), want_cone=True)
    except Exception:
        return []


# JTWC (rest of world) — slow (~2 min), loaded in a background thread so it
# never blocks the NHC map. Result lives in this shared object across reruns.
@st.cache_resource(ttl=600)
def jtwc_state():
    return {"data": None, "started": False}


def get_systems():
    nhc = get_nhc()
    state = jtwc_state()
    if not state["started"]:
        state["started"] = True

        def bg():
            try:
                state["data"] = _read(
                    realtime.Realtime(jtwc=True, jtwc_source="ucar"), want_cone=True)
            except Exception:
                state["data"] = []
        threading.Thread(target=bg, daemon=True).start()

    out = list(nhc)
    if state["data"]:
        have = {s["id"] for s in out}
        out += [s for s in state["data"] if s["id"] not in have]
    loading = state["data"] is None
    return out, loading


systems, jtwc_loading = get_systems()
if jtwc_loading:
    st.caption("⏳ Loading global (JTWC) systems in the background…")

if not systems:
    st.info("No active tropical systems anywhere right now.")
    st.stop()

# strongest first
systems.sort(key=lambda s: (s["invest"], -s["vmax"]))


# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------
_LEG = [("Cat 5", [255, 45, 85]), ("Cat 4", [255, 96, 55]),
        ("Cat 3", [255, 149, 0]), ("Cat 2", [255, 214, 10]),
        ("Cat 1", [255, 245, 120]), ("Trop. Storm", [48, 209, 220]),
        ("Depression", [100, 160, 255]), ("Invest", [150, 160, 175])]
st.markdown('<div class="hw-legend">' + "".join(
    f'<span><span class="hw-dot" style="background:rgb({r},{g},{b})"></span>{l}</span>'
    for l, (r, g, b) in _LEG) + "</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# MAP — all systems at once, satellite basemap, forecast + cone
# ---------------------------------------------------------------------------
obs_paths, fc_paths, cones, dots = [], [], [], []
for s in systems:
    col = color(s["vmax"], s["invest"])
    if len(s["track"]) >= 2:
        obs_paths.append({"path": s["track"], "color": col})
    if len(s["fc_track"]) >= 2:
        fc_paths.append({"path": s["fc_track"], "color": [255, 255, 255]})
    if s["cone"]:
        cones.append({"polygon": s["cone"]})
    dots.append({
        "position": s["pos"], "color": col,
        "name": s["name"], "klass": s["klass"],
        "wind": f'{s["vmax"]:.0f} kt ({s["vmax"] * 1.15078:.0f} mph)'
                if s["vmax"] else "—",
        "pressure": f'{s["mslp"]:.0f} mb' if s["mslp"] else "N/A",
        "move": s["move"],
    })

st.pydeck_chart(pdk.Deck(
    map_provider="carto", map_style="dark",
    initial_view_state=pdk.ViewState(latitude=15, longitude=-40, zoom=1.2),
    layers=[
        pdk.Layer("PolygonLayer", cones, get_polygon="polygon",
                  get_fill_color=[255, 255, 255, 30],
                  get_line_color=[255, 255, 255, 110],
                  stroked=True, filled=True, line_width_min_pixels=1),
        pdk.Layer("PathLayer", obs_paths, get_path="path", get_color="color",
                  width_min_pixels=7, opacity=0.25),
        pdk.Layer("PathLayer", obs_paths, get_path="path", get_color="color",
                  width_min_pixels=2.5),
        pdk.Layer("PathLayer", fc_paths, get_path="path", get_color="color",
                  width_min_pixels=1.5, opacity=0.9),
        pdk.Layer("ScatterplotLayer", dots, get_position="position",
                  get_fill_color="color", get_radius=55000, radius_min_pixels=6,
                  stroked=True, get_line_color=[10, 12, 20],
                  line_width_min_pixels=2, pickable=True),
    ],
    tooltip={
        "html": "<b>{name}</b><br/>{klass}<br/>💨 {wind}<br/>"
                "🔽 {pressure}<br/>➡️ {move}",
        "style": {"backgroundColor": "#12151c", "color": "#e6e9ef",
                  "fontSize": "12px", "padding": "8px 10px",
                  "borderRadius": "6px", "border": "1px solid #2a2f3a"},
    },
), height=520)


# ---------------------------------------------------------------------------
# SUMMARY CARDS — everything being monitored
# ---------------------------------------------------------------------------
st.subheader(f"Currently monitoring — {len(systems)} system"
             f"{'s' if len(systems) != 1 else ''}")

cols = st.columns(3)
for i, s in enumerate(systems):
    r, g, b = color(s["vmax"], s["invest"])
    with cols[i % 3]:
        st.markdown(f"""
<div class="hw-card" style="border-left-color:rgb({r},{g},{b})">
  <h4>{s['name']}</h4>
  <div class="klass">{s['klass']}</div>
  <div class="hw-grid">
    <div><span>Winds</span><b>{f'{s["vmax"]:.0f} kt' if s['vmax'] else '—'}</b></div>
    <div><span>Pressure</span><b>{f'{s["mslp"]:.0f} mb' if s['mslp'] else 'N/A'}</b></div>
    <div><span>Moving</span><b>{s['move']}</b></div>
  </div>
  <div style="color:#697083;font-size:.68rem;margin-top:8px">
    {s['pos'][1]:.1f}°, {s['pos'][0]:.1f}° · {s['time']}
  </div>
</div>""", unsafe_allow_html=True)


# Background JTWC not ready yet -> wait briefly, then rerun to pick it up.
if jtwc_loading:
    time.sleep(4)
    st.rerun()

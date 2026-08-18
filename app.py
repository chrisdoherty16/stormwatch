"""
Hurricane Watch — live tropical-cyclone dashboard.

Run:
    uv run streamlit run app.py
"""

import math
from datetime import datetime

import streamlit as st
import pydeck as pdk
from tropycal import realtime

st.set_page_config(page_title="Hurricane Watch", layout="wide")

st.markdown("""
<style>
  .block-container {padding-top: 1rem;}
  div[data-testid="stMetric"] {background:#12151c; border:1px solid #222836;
      border-radius:10px; padding:10px 14px;}
</style>
""", unsafe_allow_html=True)

st.title("🌀 Hurricane Watch")


# ---------------------------------------------------------------------------
# Classification + colour (Saffir-Simpson, winds in knots)
# ---------------------------------------------------------------------------
def category(vmax):
    """Return Saffir-Simpson category number (0 = below hurricane strength)."""
    v = vmax or 0
    if v >= 137: return 5
    if v >= 113: return 4
    if v >= 96:  return 3
    if v >= 83:  return 2
    if v >= 64:  return 1
    return 0


def color(vmax):
    return {
        5: [255, 45, 85], 4: [255, 96, 55], 3: [255, 149, 0],
        2: [255, 214, 10], 1: [255, 245, 120],
    }.get(category(vmax), [48, 209, 220] if (vmax or 0) >= 34 else [100, 160, 255])


def classify(vmax, stype, basin):
    """Human-readable classification, e.g. 'Hurricane (Cat 2)' or 'Typhoon'."""
    v = vmax or 0
    west_pac = basin == "west_pacific"
    cat = category(v)
    # special / non-tropical types first
    special = {
        "EX": "Post-Tropical Cyclone", "SS": "Subtropical Storm",
        "SD": "Subtropical Depression", "LO": "Remnant Low",
        "DB": "Disturbance", "WV": "Tropical Wave",
    }
    if stype in special:
        return special[stype]
    if cat >= 1:
        if west_pac:
            return "Super Typhoon" if v >= 130 else "Typhoon"
        return f"Hurricane (Cat {cat})"
    if v >= 34:
        return "Tropical Storm"
    return "Tropical Depression"


# ---------------------------------------------------------------------------
# Movement (bearing + speed) from the last two track fixes
# ---------------------------------------------------------------------------
_COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def movement(track, times):
    if len(track) < 2 or len(times) < 2:
        return None
    (lon1, lat1), (lon2, lat2) = track[-2], track[-1]
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    # bearing
    y = math.sin(dlon) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlon)
    brg = (math.degrees(math.atan2(y, x)) + 360) % 360
    compass = _COMPASS[round(brg / 22.5) % 16]
    # distance (nm) via haversine
    a = (math.sin((p2 - p1) / 2) ** 2 +
         math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2)
    dist_nm = 3440.065 * 2 * math.asin(math.sqrt(a))
    try:
        hrs = (times[-1] - times[-2]).total_seconds() / 3600
        spd = dist_nm / hrs if hrs else 0
        return f"{compass} at {spd:.0f} kt"
    except Exception:
        return compass


# ---------------------------------------------------------------------------
# Live data
# ---------------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner="Loading active storms…")
def get_storms():
    rt = realtime.Realtime()
    out = []
    for sid in rt.list_active_storms():
        try:
            s = rt.get_storm(sid)
            track = [[float(lo), float(la)] for lo, la in zip(s.lon, s.lat)]
            times = list(getattr(s, "date", []) or getattr(s, "time", []))
            vmax = float(s.vmax[-1])
            stype = s.type[-1] if getattr(s, "type", None) is not None else None
            basin = getattr(s, "basin", None)
            mslp = s.mslp[-1] if getattr(s, "mslp", None) is not None else None
            out.append({
                "id": sid,
                "name": str(s.name).title(),
                "basin": basin,
                "track": track,
                "pos": track[-1],
                "vmax": vmax,
                "mslp": float(mslp) if mslp and not math.isnan(float(mslp)) else None,
                "klass": classify(vmax, stype, basin),
                "cat": category(vmax),
                "move": movement(track, times),
                "time": times[-1].strftime("%d %b %H:%MZ")
                        if times and hasattr(times[-1], "strftime") else "—",
            })
        except Exception:
            continue
    return out


storms = get_storms()
if not storms:
    st.info("No active storms right now.")
    st.stop()


# ---------------------------------------------------------------------------
# Storm selector
# ---------------------------------------------------------------------------
labels = [f'{s["name"]} — {s["klass"]}' for s in storms]
choice = st.selectbox("Storm", labels)
storm = storms[labels.index(choice)]


# ---------------------------------------------------------------------------
# Map — satellite imagery + glowing track + hover tooltip
# ---------------------------------------------------------------------------
paths = [{"path": s["track"], "color": color(s["vmax"])} for s in storms]
dots = [{
    "position": s["pos"], "color": color(s["vmax"]),
    "name": s["name"], "klass": s["klass"],
    "wind": f'{s["vmax"]:.0f} kt ({s["vmax"] * 1.15078:.0f} mph)',
    "pressure": f'{s["mslp"]:.0f} mb' if s["mslp"] else "N/A",
    "move": s["move"] or "—",
} for s in storms]

satellite = pdk.Layer(
    "TileLayer",
    data="https://server.arcgisonline.com/ArcGIS/rest/services/"
         "World_Imagery/MapServer/tile/{z}/{y}/{x}",
    min_zoom=0, max_zoom=19, tile_size=256,
)

st.pydeck_chart(pdk.Deck(
    map_style=None,  # satellite tiles provide the basemap
    initial_view_state=pdk.ViewState(
        latitude=storm["pos"][1], longitude=storm["pos"][0], zoom=4),
    layers=[
        satellite,
        # glow underlay
        pdk.Layer("PathLayer", paths, get_path="path", get_color="color",
                  width_min_pixels=7, opacity=0.25),
        # core track
        pdk.Layer("PathLayer", paths, get_path="path", get_color="color",
                  width_min_pixels=2.5),
        # current-position dots
        pdk.Layer("ScatterplotLayer", dots, get_position="position",
                  get_fill_color="color", get_radius=55000,
                  radius_min_pixels=7, stroked=True,
                  get_line_color=[10, 12, 20], line_width_min_pixels=2,
                  pickable=True),
    ],
    tooltip={
        "html": "<b>{name}</b><br/>{klass}<br/>💨 {wind}<br/>"
                "🔽 {pressure}<br/>➡️ {move}",
        "style": {"backgroundColor": "#12151c", "color": "#e6e9ef",
                  "fontSize": "12px", "padding": "8px 10px",
                  "borderRadius": "6px", "border": "1px solid #2a2f3a"},
    },
), height=460)


# ---------------------------------------------------------------------------
# Status bulletin — core metrics, clean row
# ---------------------------------------------------------------------------
st.subheader(f'{storm["name"]} — {storm["klass"]}')
c1, c2, c3, c4 = st.columns(4)
c1.metric("Sustained winds", f'{storm["vmax"]:.0f} kt',
          f'{storm["vmax"] * 1.15078:.0f} mph')
c2.metric("Min pressure", f'{storm["mslp"]:.0f} mb' if storm["mslp"] else "N/A")
c3.metric("Movement", storm["move"] or "—")
c4.metric("Last update", storm["time"])
st.caption(f'Position: {storm["pos"][1]:.1f}°, {storm["pos"][0]:.1f}°  ·  '
           f'Source: NHC/JTWC via Tropycal')


# ---------------------------------------------------------------------------
# Latest NHC / CPHC bulletins (RSS)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def get_bulletins():
    import urllib.request, xml.etree.ElementTree as ET
    feeds = {
        "Atlantic": "https://www.nhc.noaa.gov/index-at.xml",
        "E. Pacific": "https://www.nhc.noaa.gov/index-ep.xml",
        "C. Pacific": "https://www.nhc.noaa.gov/index-cp.xml",
    }
    items = []
    for basin, url in feeds.items():
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "stormwatch/1.0"})
            with urllib.request.urlopen(req, timeout=8) as r:
                root = ET.fromstring(r.read())
            for it in root.iter("item"):
                title = (it.findtext("title") or "").strip()
                if not title:
                    continue
                desc = (it.findtext("description") or "")
                desc = " ".join(desc.replace("<", " <").split())
                # crude tag strip
                clean, keep = [], True
                for ch in desc:
                    if ch == "<": keep = False
                    elif ch == ">": keep = True
                    elif keep: clean.append(ch)
                items.append({
                    "basin": basin, "title": title,
                    "desc": " ".join("".join(clean).split()),
                    "link": (it.findtext("link") or "").strip(),
                    "pub": (it.findtext("pubDate") or "").strip(),
                })
        except Exception:
            continue
    # de-dupe on title
    seen, uniq = set(), []
    for i in items:
        if i["title"] not in seen:
            seen.add(i["title"]); uniq.append(i)
    return uniq[:8]


st.subheader("📰 Latest NHC bulletins")
bulletins = get_bulletins()
if not bulletins:
    st.caption("No bulletins retrieved right now.")
for b in bulletins:
    with st.expander(f'[{b["basin"]}] {b["title"]}'):
        st.caption(b["pub"])
        st.write(b["desc"] or "—")
        if b["link"]:
            st.markdown(f'[Open full product on nhc.noaa.gov]({b["link"]})')

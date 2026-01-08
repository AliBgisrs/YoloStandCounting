import os
import cv2
import csv
import json
import math
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from werkzeug.utils import secure_filename

# ---------- Optional deps for georeference auto-detection ----------
try:
    import rasterio
    _HAS_RIO = True
except Exception:
    _HAS_RIO = False

try:
    from pyproj import CRS  # also used for resolving EPSG from WKT (worldfile)
    _HAS_PYPROJ = True
except Exception:
    _HAS_PYPROJ = False

# ---------- Optional Shapefile export (if available) ----------
try:
    import fiona
    from fiona.crs import from_epsg
    _HAS_FIONA = True
except Exception:
    _HAS_FIONA = False
# -----------------------------
# Config
# -----------------------------
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

# Use the SAME weights as your notebook; change if your path differs.
MODEL_PATH = r"C:/webapps/plant_counter_app/models/best.pt"

# Default confidence (UI can override)
CONF_MIN = 0.20

# Each class encodes "plants per box"
PLANTS_PER_CLASS = {0: 1, 1: 2, 2: 3}

# BGR colors for drawing
CLASS_COLORS = {
    0: (0, 255, 255),   # yellow
    1: (128, 0, 0),     # navy/dark-blue
    2: (255, 255, 255), # white
    3: (0, 255, 0),     # green (fallback)
}

# Tiling
TILE_SIZE          = 896
TILE_OVERLAP       = 224
CORE_MARGIN        = 32
IOU_NMS            = 0.65
TILE_THRESHOLD_PX  = 1200

# =============================
# App
# =============================
app = Flask(__name__)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

model = YOLO(MODEL_PATH)
MODEL_NAMES = getattr(model, "names", {})

# =============================
# Helpers
# =============================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def unique_name(base: str, suffix: str, ext: str = ".jpg") -> str:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")
    return os.path.join(RESULT_FOLDER, f"{base}_{suffix}_{ts}{ext}")

def parse_roi(roi_obj, img_w: int, img_h: int):
    """Leaflet CRS.Simple → image pixel ROI (x1,y1,x2,y2)."""
    try:
        x1_raw = int(roi_obj["_southWest"]["lng"])
        y_top_raw = int(roi_obj["_northEast"]["lat"])
        x2_raw = int(roi_obj["_northEast"]["lng"])
        y_bottom_raw = int(roi_obj["_southWest"]["lat"])
        y1 = int(img_h - y_top_raw)
        y2 = int(img_h - y_bottom_raw)
        x1 = int(x1_raw)
        x2 = int(x2_raw)
    except Exception:
        x1 = int(roi_obj.get("x1", 0));           y1 = int(roi_obj.get("y1", 0))
        x2 = int(roi_obj.get("x2", img_w));       y2 = int(roi_obj.get("y2", img_h))
    x1 = clamp(x1, 0, img_w); x2 = clamp(x2, 0, img_w)
    y1 = clamp(y1, 0, img_h); y2 = clamp(y2, 0, img_h)
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return x1, y1, x2, y2

def best_text_color(bgr):
    b, g, r = bgr
    lum = 0.114*b + 0.587*g + 0.299*r
    return (0, 0, 0) if lum > 180 else (255, 255, 255)

def iter_tiles_whole(w, h, tile=TILE_SIZE, overlap=TILE_OVERLAP):
    stride = max(1, tile - overlap)
    xs = list(range(0, max(1, w - tile + 1), stride))
    ys = list(range(0, max(1, h - tile + 1), stride))
    if xs and xs[-1] + tile < w: xs.append(w - tile)
    if ys and ys[-1] + tile < h: ys.append(h - tile)
    for y in ys:
        for x in xs:
            x1, y1 = int(max(0, x)), int(max(0, y))
            x2, y2 = int(min(w, x1 + tile)), int(min(h, y1 + tile))
            yield x1, y1, x2, y2

def box_iou_xyxy(a, b):
    xx1 = max(a[0], b[0]); yy1 = max(a[1], b[1])
    xx2 = min(a[2], b[2]); yy2 = min(a[3], b[3])
    iw = max(0.0, xx2 - xx1); ih = max(0.0, yy2 - yy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0

def nms_classwise(dets, iou_thr=IOU_NMS):
    out, by_cls = [], {}
    for d in dets:
        by_cls.setdefault(d["cls"], []).append(d)
    for _, items in by_cls.items():
        items = sorted(items, key=lambda d: d["conf"], reverse=True)
        keep = []
        while items:
            best = items.pop(0)
            keep.append(best)
            items = [d for d in items if box_iou_xyxy(best["xyxy"], d["xyxy"]) < iou_thr]
        out.extend(keep)
    return out

def in_core(cx, cy, x1t, y1t, x2t, y2t, margin=CORE_MARGIN):
    return (x1t + margin) <= cx <= (x2t - margin) and (y1t + margin) <= cy <= (y2t - margin)

def save_geojson(features, path):
    fc = {"type": "FeatureCollection", "features": features}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False, indent=2)

def save_shapefile(polygons, shp_path, epsg=None, crs_wkt=None):
    """Write polygons to ESRI Shapefile if Fiona is available."""
    if not _HAS_FIONA:
        return None
    schema = {
        'geometry': 'Polygon',
        'properties': {
            'cls_id': 'int', 'cls_name': 'str', 'conf': 'float',
            'plants_pb': 'int', 'plants': 'int', 'src_image': 'str'
        }
    }
    crs = from_epsg(int(epsg)) if epsg else (crs_wkt or None)
    with fiona.open(shp_path, 'w', driver='ESRI Shapefile', schema=schema, crs=crs) as dst:
        for poly in polygons:
            dst.write({
                'geometry': {'type': 'Polygon', 'coordinates': poly['coords']},
                'properties': {
                    'cls_id': int(poly['props']['cls_id']),
                    'cls_name': str(poly['props']['cls_name']),
                    'conf': float(poly['props']['conf']),
                    'plants_pb': int(poly['props']['plants_per_box']),
                    'plants': int(poly['props']['plants']),
                    'src_image': str(poly['props']['source_image'])
                }
            })
    return shp_path

# ---------- Auto-georef (GeoTIFF / worldfile + optional .prj) ----------
WORLD_FILE_EXTS = {
    ".tif": ".tfw", ".tiff": ".tfw",
    ".jpg": ".jgw", ".jpeg": ".jgw",
    ".png": ".pgw",
    ".bmp": ".bpw"
}
PRJ_EXT = ".prj"

def read_worldfile(img_path):
    base, ext = os.path.splitext(img_path)
    wf_ext = WORLD_FILE_EXTS.get(ext.lower())
    if not wf_ext:
        return None, None, None
    wf_path = base + wf_ext
    if not os.path.exists(wf_path):
        return None, None, None
    try:
        with open(wf_path, "r", encoding="utf-8") as f:
            vals = [float(l.strip()) for l in f.readlines()[:6]]
        if len(vals) != 6:
            return None, None, None
        A, D, B, E, C, F = vals
        # Worldfile is top-left pixel center; convert to GDAL GT (UL corner)
        GT0 = C - 0.5 * A - 0.5 * B
        GT1 = A
        GT2 = B
        GT3 = F - 0.5 * D - 0.5 * E
        GT4 = D
        GT5 = E
        gt = [GT0, GT1, GT2, GT3, GT4, GT5]

        epsg = None
        crs_wkt = None
        prj_path = base + PRJ_EXT
        if os.path.exists(prj_path):
            with open(prj_path, "r", encoding="utf-8") as pf:
                crs_wkt = pf.read()
            if _HAS_PYPROJ:
                try:
                    epsg = CRS.from_wkt(crs_wkt).to_epsg()
                except Exception:
                    epsg = None
        return gt, epsg, crs_wkt
    except Exception:
        return None, None, None

def read_rasterio_gt(img_path):
    if not _HAS_RIO:
        return None, None, None
    try:
        with rasterio.open(img_path) as ds:  # type: ignore
            tf = ds.transform  # Affine(a,b,c,d,e,f)
            gt = [tf.c, tf.a, tf.b, tf.f, tf.d, tf.e]  # rasterio → GDAL GT
            epsg = None
            crs_wkt = None
            if ds.crs:
                try:
                    epsg = ds.crs.to_epsg()
                except Exception:
                    epsg = None
                try:
                    crs_wkt = ds.crs.to_wkt()
                except Exception:
                    crs_wkt = None
            return gt, epsg, crs_wkt
    except Exception:
        return None, None, None

def auto_georef(img_path):
    """Return (gt, epsg, crs_wkt)."""
    gt, epsg, wkt = read_rasterio_gt(img_path)
    if gt is not None:
        return gt, epsg, wkt
    gt, epsg, wkt = read_worldfile(img_path)
    return gt, epsg, wkt

def to_map(px, py, gt):
    GT0, GT1, GT2, GT3, GT4, GT5 = gt
    x = GT0 + px * GT1 + py * GT2
    y = GT3 + px * GT4 + py * GT5
    return float(x), float(y)

def utm_epsg_from_lonlat(lon, lat):
    zone = int(math.floor((lon + 180.0) / 6.0) + 1)
    if lat >= 0:
        return 32600 + zone  # UTM north
    else:
        return 32700 + zone  # UTM south

# =============================
# Routes
# =============================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No file part 'image' in request."}), 400
    file = request.files["image"]
    if not file or file.filename.strip() == "":
        return jsonify({"error": "Empty filename."}), 400

    filename = secure_filename(file.filename)
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(image_path)

    img = cv2.imread(image_path)
    if img is None:
        return jsonify({"error": "Could not read uploaded image."}), 400

    h, w = img.shape[:2]
    return jsonify({"image_path": image_path, "width": w, "height": h})

@app.route("/analyze", methods=["POST"])
def analyze_roi():
    try:
        data = request.get_json(force=True)
    except Exception:
        data = json.loads(request.data or "{}")

    image_path = data.get("image_path")
    roi = data.get("roi")
    img_w = int(data.get("width", 0))
    img_h = int(data.get("height", 0))
    conf_min = float(data.get("conf_min", CONF_MIN))

    if not image_path or roi is None or img_w <= 0 or img_h <= 0:
        return jsonify({"error": "Missing or invalid payload fields."}), 400

    img = cv2.imread(image_path)
    if img is None:
        return jsonify({"error": "Could not read image from path."}), 400

    # Auto-georef (if available)
    gt, epsg, crs_wkt = auto_georef(image_path)

    # ROI crop
    x1, y1, x2, y2 = parse_roi(roi, img_w, img_h)
    if x2 <= x1 or y2 <= y1:
        return jsonify({
            "total_boxes_kept": 0,
            "class_box_counts": {},
            "plants_per_class": PLANTS_PER_CLASS,
            "per_class_plants": {},
            "total_plants": 0,
            "conf_threshold": conf_min,
            "class_ids": sorted(PLANTS_PER_CLASS.keys()),
            "class_names": {int(k): str(MODEL_NAMES.get(int(k), f'class_{k}')) for k in PLANTS_PER_CLASS},
            "result_image": image_path,
            "geojson_path": None,
            "georef_used": False,
            "epsg": None,
            "has_crs_wkt": False,
            "crs_wkt_path": None,
            "centroids_csv": None,
            "utm_epsg": None,
            "utm_geojson_path": None,
            "utm_centroids_csv": None,
            "utm_shapefile_path": None
        })

    cropped = img[y1:y2, x1:x2].copy()
    H, W = cropped.shape[:2]
    use_tiling = max(H, W) > TILE_THRESHOLD_PX

    collected = []   # [{'cls','conf','xyxy'(ROI coords)}]

    if use_tiling:
        for x1t, y1t, x2t, y2t in iter_tiles_whole(W, H, TILE_SIZE, TILE_OVERLAP):
            tile = cropped[y1t:y2t, x1t:x2t]
            res = model.predict(tile, imgsz=TILE_SIZE, conf=0.0, verbose=False)
            if not res:
                continue
            r = res[0]
            if getattr(r, "boxes", None) is None:
                continue
            for b in r.boxes:
                cls_id = int(b.cls[0]) if b.cls is not None else -1
                conf   = float(b.conf[0]) if b.conf is not None else 0.0
                x1b, y1b, x2b, y2b = [float(v) for v in b.xyxy[0].tolist()]

                # Keep only detections whose center lies in tile core
                cx, cy = 0.5*(x1b+x2b), 0.5*(y1b+y2b)
                if not in_core(cx, cy, 0, 0, (x2t-x1t), (y2t-y1t), margin=CORE_MARGIN):
                    continue

                # FIXED: offset both corners by tile origin
                collected.append({
                    "cls": cls_id, "conf": conf,
                    "xyxy": [x1b + x1t, y1b + y1t, x2b + x1t, y2b + y1t]
                })
    else:
        imgsz = max(640, min(1280, max(H, W)))
        res = model.predict(cropped, imgsz=imgsz, conf=0.0, verbose=False)
        if res:
            r = res[0]
            if getattr(r, "boxes", None) is not None:
                for b in r.boxes:
                    cls_id = int(b.cls[0]) if b.cls is not None else -1
                    conf   = float(b.conf[0]) if b.conf is not None else 0.0
                    x1b, y1b, x2b, y2b = [float(v) for v in b.xyxy[0].tolist()]
                    collected.append({"cls": cls_id, "conf": conf, "xyxy": [x1b, y1b, x2b, y2b]})

    # Stitch (class-wise NMS)
    collected = nms_classwise(collected, iou_thr=IOU_NMS)

    # Draw + count + build export features (first in source map coords if available)
    class_box_counts = {int(k): 0 for k in PLANTS_PER_CLASS}
    kept = 0
    export_features_src = []  # polygons in source map coords (or pixels)

    for d in collected:
        cls_id, conf = d["cls"], d["conf"]
        x1r, y1r, x2r, y2r = d["xyxy"]  # ROI coords
        if conf < conf_min or cls_id not in PLANTS_PER_CLASS:
            continue

        # Draw
        color = CLASS_COLORS.get(cls_id, (0, 255, 0))
        cv2.rectangle(cropped, (int(x1r), int(y1r)), (int(x2r), int(y2r)), color, 2)
        cls_name = MODEL_NAMES.get(cls_id, f"class_{cls_id}")
        cv2.putText(cropped, str(cls_name), (int(x1r), max(0, int(y1r) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, best_text_color(color), 1, cv2.LINE_AA)

        class_box_counts[cls_id] += 1
        kept += 1

        # Absolute image pixel coords
        abs_x1, abs_y1 = float(x1 + x1r), float(y1 + y1r)
        abs_x2, abs_y2 = float(x1 + x2r), float(y1 + y2r)

        # Build polygon ring in source coordinates if georef exists; else pixels
        if gt is not None:
            xA, yA = to_map(abs_x1, abs_y1, gt)
            xB, yB = to_map(abs_x2, abs_y1, gt)
            xC, yC = to_map(abs_x2, abs_y2, gt)
            xD, yD = to_map(abs_x1, abs_y2, gt)
            ring_src = [[xA,yA], [xB,yB], [xC,yC], [xD,yD], [xA,yA]]
            coord_space = "map"
        else:
            ring_src = [[abs_x1,abs_y1], [abs_x2,abs_y1], [abs_x2,abs_y2], [abs_x1,abs_y2], [abs_x1,abs_y1]]
            coord_space = "image_pixels"

        export_features_src.append({
            "type": "Feature",
            "geometry": { "type": "Polygon", "coordinates": [ring_src] },
            "properties": {
                "cls_id": int(cls_id),
                "cls_name": str(cls_name),
                "conf": float(conf),
                "plants_per_box": int(PLANTS_PER_CLASS[cls_id]),
                "plants": int(PLANTS_PER_CLASS[cls_id]),
                "source_image": os.path.basename(image_path),
                "roi_x1": int(x1), "roi_y1": int(y1), "roi_x2": int(x2), "roi_y2": int(y2),
                "conf_threshold_used": float(conf_min),
                "tiling": bool(use_tiling),
                "coord_space": coord_space,
                "epsg": int(epsg) if epsg is not None else None
            }
        })

    # Plants per class & total
    per_class_plants = {int(c): int(class_box_counts[int(c)] * PLANTS_PER_CLASS[int(c)])
                        for c in PLANTS_PER_CLASS}
    total_plants = int(sum(per_class_plants.values()))

    # Save annotated ROI image
    base = os.path.splitext(os.path.basename(image_path))[0]
    roi_tag = f"roi_{x1}_{y1}_{x2}_{y2}"
    result_img_path = unique_name(base, roi_tag, ext=".jpg")
    cv2.imwrite(result_img_path, cropped)

    # Save original-CRS GeoJSON (map or pixels)
    result_geojson_path = unique_name(base, roi_tag, ext=".geojson")
    save_geojson(export_features_src, result_geojson_path)

    # Save CRS WKT if CRS exists but EPSG is unknown
    crs_wkt_path = None
    if gt is not None and epsg is None and crs_wkt:
        crs_wkt_path = result_geojson_path.replace(".geojson", ".wkt")
        try:
            with open(crs_wkt_path, "w", encoding="utf-8") as wf:
                wf.write(crs_wkt)
        except Exception:
            crs_wkt_path = None

    # ---------- UTM export (preferred) ----------
    utm_epsg = None
    utm_geojson_path = None
    utm_centroids_csv = None
    utm_shapefile_path = None

    if gt is not None and _HAS_PYPROJ and (epsg is not None or crs_wkt):
        # Build source CRS
        src_crs = CRS.from_epsg(int(epsg)) if epsg is not None else CRS.from_wkt(crs_wkt)

        # Compute ROI center (map coords) then lon/lat to pick UTM zone
        cx_px = (x1 + x2) / 2.0
        cy_px = (y1 + y2) / 2.0
        cx_map, cy_map = to_map(cx_px, cy_px, gt)
        to_wgs = Transformer.from_crs(src_crs, CRS.from_epsg(4326), always_xy=True)
        lon, lat = to_wgs.transform(cx_map, cy_map)
        utm_epsg = utm_epsg_from_lonlat(lon, lat)
        utm_crs = CRS.from_epsg(utm_epsg)
        to_utm = Transformer.from_crs(src_crs, utm_crs, always_xy=True)

        # Reproject polygons to UTM
        utm_features = []
        for feat in export_features_src:
            ring = feat["geometry"]["coordinates"][0]
            ring_utm = [list(to_utm.transform(x, y)) for (x, y) in ring]
            utm_features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring_utm]},
                "properties": feat["properties"] | {"coord_space": f"UTM_{utm_epsg}"}
            })

        # Write UTM GeoJSON
        utm_geojson_path = unique_name(base, roi_tag + "_utm", ext=".geojson")
        save_geojson(utm_features, utm_geojson_path)

        # UTM centroids CSV
        utm_centroids_csv = utm_geojson_path.replace(".geojson", "_centroids.csv")
        try:
            with open(utm_centroids_csv, "w", newline="", encoding="utf-8") as cf:
                writer = csv.writer(cf)
                writer.writerow(["Easting", "Northing", "cls_id", "cls_name", "conf", "plants_per_box", "plants"])
                for f in utm_features:
                    ring = f["geometry"]["coordinates"][0]
                    xs = [p[0] for p in ring[:-1]]
                    ys = [p[1] for p in ring[:-1]]
                    cx = sum(xs)/4.0; cy = sum(ys)/4.0
                    p = f["properties"]
                    writer.writerow([cx, cy, p["cls_id"], p["cls_name"], p["conf"],
                                     p["plants_per_box"], p["plants"]])
        except Exception:
            utm_centroids_csv = None

        # Optional UTM Shapefile
        if _HAS_FIONA:
            try:
                shp_path = utm_geojson_path.replace(".geojson", ".shp")
                shp_polys = []
                for f in utm_features:
                    shp_polys.append({
                        'coords': [tuple(map(tuple, f['geometry']['coordinates'][0]))],
                        'props': f['properties']
                    })
                utm_shapefile_path = save_shapefile(shp_polys, shp_path, epsg=utm_epsg, crs_wkt=None)
            except Exception:
                utm_shapefile_path = None

    # Centroids CSV in source coords (optional, unchanged)
    cent_csv_path = result_geojson_path.replace(".geojson", "_centroids.csv")
    try:
        with open(cent_csv_path, "w", newline="", encoding="utf-8") as cf:
            writer = csv.writer(cf)
            writer.writerow(["x", "y", "cls_id", "cls_name", "conf", "plants_per_box", "plants",
                             "coord_space"])
            for feat in export_features_src:
                ring = feat["geometry"]["coordinates"][0]
                xs = [p[0] for p in ring[:-1]]
                ys = [p[1] for p in ring[:-1]]
                cx = sum(xs)/4.0; cy = sum(ys)/4.0
                p = feat["properties"]
                writer.writerow([cx, cy, p["cls_id"], p["cls_name"], p["conf"],
                                 p["plants_per_box"], p["plants"], p.get("coord_space")])
    except Exception:
        cent_csv_path = None

    # Response
    return jsonify({
        "total_boxes_kept": int(kept),
        "class_box_counts": {str(k): int(v) for k, v in class_box_counts.items()},
        "plants_per_class": PLANTS_PER_CLASS,
        "per_class_plants": {str(k): int(v) for k, v in per_class_plants.items()},
        "total_plants": total_plants,
        "conf_threshold": conf_min,
        "class_ids": sorted([int(k) for k in PLANTS_PER_CLASS.keys()]),
        "class_names": {int(k): str(MODEL_NAMES.get(int(k), f'class_{k}')) for k in PLANTS_PER_CLASS},
        "result_image": result_img_path,

        # Original map/pixel outputs
        "geojson_path": result_geojson_path,
        "centroids_csv": cent_csv_path,
        "georef_used": bool(gt is not None),
        "epsg": int(epsg) if epsg is not None else None,
        "has_crs_wkt": bool(crs_wkt is not None),
        "crs_wkt_path": crs_wkt_path,
        "georef_source": ("rasterio" if (_HAS_RIO and gt is not None) else
                          ("worldfile" if gt is not None else None)),

        # UTM outputs (preferred for GIS)
        "utm_epsg": int(utm_epsg) if utm_epsg else None,
        "utm_geojson_path": utm_geojson_path,
        "utm_centroids_csv": utm_centroids_csv,
        "utm_shapefile_path": utm_shapefile_path
    })

# =============================
# Entrypoint
# =============================
if __name__ == "__main__":
    app.run(debug=True)
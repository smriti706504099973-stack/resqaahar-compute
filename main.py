from __future__ import annotations

import io
import json
import os
import re
from datetime import datetime, timedelta, timezone
import google.generativeai as genai
import requests
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from PIL.ExifTags import TAGS

app = FastAPI(title="ResQaahar Compute")

AMRITSAR_LAT = 31.6340
AMRITSAR_LON = 74.8723

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "").strip()

GEMINI_PROMPT = """You are an AI for a food rescue app in India. Analyze this food image. You must return ONLY a valid JSON object matching this schema: {"items": [{"name": "string", "qty": "number or descriptive string"}], "requires_refrigeration": boolean}. Do not include markdown formatting or backticks in the response."""


@app.middleware("http")
async def require_internal_secret(request: Request, call_next):
    path = request.url.path
    if path == "/health" or path == "/":
        return await call_next(request)
    if path.startswith("/docs") or path == "/openapi.json" or path == "/redoc":
        return await call_next(request)
    secret = (os.getenv("INTERNAL_SERVICE_SECRET") or "").strip()
    if not secret:
        return JSONResponse(
            status_code=503,
            content={
                "detail": (
                    "Compute INTERNAL_SERVICE_SECRET is missing. "
                    "Set it in Compute/.env to the same value as Backend INTERNAL_SERVICE_SECRET, "
                    "then restart uvicorn."
                ),
            },
        )
    auth = request.headers.get("authorization") or ""
    if auth != f"Bearer {secret}":
        return JSONResponse(status_code=403, content={"detail": "Forbidden"})
    return await call_next(request)


def _parse_exif_datetime(s: str) -> datetime | None:
    s = str(s).strip()
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def exif_check(image_bytes: bytes) -> None:
    if os.getenv("LENIENT_EXIF") == "1":
        return
    img = Image.open(io.BytesIO(image_bytes))
    exif = img.getexif()
    if not exif:
        raise HTTPException(
            status_code=400,
            detail="Missing EXIF metadata (liveness check). Use device camera capture.",
        )
    tags = {TAGS.get(k, k): v for k, v in exif.items()}
    dt_raw = tags.get("DateTimeOriginal") or tags.get("DateTime")
    if not dt_raw:
        raise HTTPException(
            status_code=400,
            detail="Missing capture timestamp in EXIF (possible non-camera image).",
        )
    captured = _parse_exif_datetime(str(dt_raw))
    if captured is None:
        raise HTTPException(
            status_code=400,
            detail="Could not parse image capture time from EXIF.",
        )
    if captured < datetime.now(timezone.utc) - timedelta(hours=2):
        raise HTTPException(
            status_code=400,
            detail="Image is too old. Please take a live photo.",
        )
    software = (tags.get("Software") or "").lower()
    bad = ("photoshop", "gimp", "canva", "pixelmator", "affinity")
    if any(b in software for b in bad):
        raise HTTPException(
            status_code=400,
            detail="Image editing software detected in EXIF.",
        )


def fetch_amritsar_temp_c() -> float:
    key = (os.getenv("OPENWEATHER_API_KEY") or "").strip()
    if not key:
        raise HTTPException(
            status_code=503,
            detail="OPENWEATHER_API_KEY is required for refrigerated donations",
        )
    try:
        r = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={
                "lat": AMRITSAR_LAT,
                "lon": AMRITSAR_LON,
                "appid": key,
                "units": "metric",
            },
            timeout=20,
        )
        r.raise_for_status()
        return float(r.json()["main"]["temp"])
    except HTTPException:
        raise
    except requests.RequestException as e:
        raise HTTPException(
            status_code=502,
            detail=f"Weather service unavailable: {e!s}",
        ) from e
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(
            status_code=502,
            detail=f"Invalid weather API response: {e!s}",
        ) from e


def compute_expiry_time(requires_refrigeration: bool) -> datetime:
    now = datetime.now(timezone.utc)
    if not requires_refrigeration:
        return now + timedelta(hours=12)
    temp_c = fetch_amritsar_temp_c()
    if temp_c > 35:
        return now + timedelta(minutes=45)
    return now + timedelta(hours=4)


def _strip_json_fences(text: str) -> str:
    t = text.strip()
    if "```" in t:
        parts = t.split("```")
        if len(parts) >= 2:
            inner = parts[1]
            inner = re.sub(r"^json\s*", "", inner.strip(), flags=re.IGNORECASE)
            return inner.strip()
    return t


def _normalize_gemini_payload(parsed: dict) -> dict:
    """Map Gemini schema to fields Node.js expects."""
    items_out = []
    total_weight_hint = 0.0
    for it in parsed.get("items") or []:
        if not isinstance(it, dict):
            continue
        name = str(it.get("name") or "Food").strip() or "Food"
        qty = it.get("qty", "")
        qty_s = str(qty).strip() if qty is not None else ""
        items_out.append({"name": name, "quantity_estimate": qty_s})
        if isinstance(qty, (int, float)):
            total_weight_hint += float(qty)
    if not items_out:
        items_out = [{"name": "Food", "quantity_estimate": ""}]
    weight_kg = max(0.5, min(500.0, total_weight_hint if total_weight_hint > 0 else 1.0))
    return {
        "items": items_out,
        "requires_refrigeration": bool(parsed.get("requires_refrigeration")),
        "weight_kg": weight_kg,
    }


def gemini_analyze(image_bytes: bytes, mime_type: str) -> dict:
    key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not key:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY is not configured",
        )
    def gemini_analyze(image_bytes: bytes, mime_type: str) -> dict:
    key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not key:
        raise HTTPException(status_code=503, detail="Missing GEMINI_API_KEY")

    genai.configure(api_key=key)
    model = genai.GenerativeModel(GEMINI_MODEL)
    try:
        resp = model.generate_content(
            [
                GEMINI_PROMPT,
                {"mime_type": mime_type or "image/jpeg", "data": image_bytes},
            ]
        )
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini request failed or timed out: {e!s}",
        ) from e

    text = (getattr(resp, "text", None) or "").strip()
    if not text and resp.candidates:
        parts = []
        for p in resp.candidates[0].content.parts:
            if hasattr(p, "text") and p.text:
                parts.append(p.text)
        text = "".join(parts).strip()
    if not text:
        raise HTTPException(
            status_code=502,
            detail="Gemini returned an empty response",
        )
    try:
        raw = json.loads(_strip_json_fences(text))
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini did not return valid JSON: {e!s}",
        ) from e
    if not isinstance(raw, dict):
        raise HTTPException(status_code=502, detail="Gemini JSON must be an object")
    return _normalize_gemini_payload(raw)


@app.post("/analyze-food-image")
async def analyze_food_image(file: UploadFile = File(...)):
    raw = await file.read()
    if len(raw) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large")
    mime = (file.content_type or "").split(";")[0].strip().lower() or "image/jpeg"
    if not mime.startswith("image/"):
        mime = "image/jpeg"

    try:
        exif_check(raw)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e!s}") from e

    parsed = gemini_analyze(raw, mime)
    requires = parsed["requires_refrigeration"]
    try:
        expiry = compute_expiry_time(requires)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not compute expiry: {e!s}",
        ) from e

    parsed["expiry_time"] = expiry.isoformat()
    parsed["requires_refrigeration"] = requires
    return parsed


def _route_distance(pts: list[tuple[float, float]], order: list[int]) -> float:
    total = 0.0
    for a, b in zip(order, order[1:]):
        dy = (pts[a][0] - pts[b][0]) * 111_000
        dx = (pts[a][1] - pts[b][1]) * 82_000
        total += (dx * dx + dy * dy) ** 0.5
    return total


@app.post("/compute-route")
async def compute_route(body: dict):
    import itertools

    start = body.get("start") or {}
    stops = body.get("stops") or []
    end = body.get("end")

    pts: list[tuple[float, float]] = []
    if start.get("lat") is None:
        return {"legs": [], "polyline": []}
    pts.append((float(start["lat"]), float(start["lng"])))
    for s in stops:
        if s.get("lat") is not None:
            pts.append((float(s["lat"]), float(s["lng"])))
    has_end = bool(end and end.get("lat") is not None)
    if has_end:
        pts.append((float(end["lat"]), float(end["lng"])))

    if len(pts) < 2:
        return {"legs": list(range(len(pts))), "polyline": [[p[0], p[1]] for p in pts]}

    if has_end and len(pts) >= 3:
        n = len(pts)
        middle = list(range(1, n - 1))
        best_order: list[int] | None = None
        best_d = float("inf")
        for perm in itertools.permutations(middle):
            order = [0] + list(perm) + [n - 1]
            d = _route_distance(pts, order)
            if d < best_d:
                best_d = d
                best_order = order
        assert best_order is not None
        ordered_pts = [pts[i] for i in best_order]
        return {"legs": best_order, "polyline": [[p[0], p[1]] for p in ordered_pts]}

    start_idx = 0
    others = [i for i in range(len(pts)) if i != start_idx]
    best_order: list[int] | None = None
    best_d = float("inf")
    for perm in itertools.permutations(others):
        order = [start_idx] + list(perm)
        d = _route_distance(pts, order)
        if d < best_d:
            best_d = d
            best_order = order
    assert best_order is not None
    ordered_pts = [pts[i] for i in best_order]
    return {"legs": best_order, "polyline": [[p[0], p[1]] for p in ordered_pts]}


@app.get("/health")
def health():
    return {"ok": True, "service": "resqaahar-compute"}

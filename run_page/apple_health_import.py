#!/usr/bin/env python3
"""Convert an Apple Health export into Running Page activities.

The importer intentionally reads only workout-related information from the
export. Other HealthKit records are ignored, except heart-rate samples used to
fill averages for older workouts that do not contain WorkoutStatistics.

Example:
    python run_page/apple_health_import.py \
      /path/to/apple_health_export/export.xml \
      --routes /path/to/apple_health_export/workout-routes \
      --output src/static/activities.json
"""

from __future__ import annotations

import argparse
import bisect
import contextlib
import datetime as dt
import hashlib
import io
import json
import math
import re
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path


APPLE_DATE_FORMAT = "%Y-%m-%d %H:%M:%S %z"
GPX_NS = {"gpx": "http://www.topografix.com/GPX/1/1"}
HEART_RATE_TYPE = "HKQuantityTypeIdentifierHeartRate"
DEFAULT_EXPORT_MEMBER = "apple_health_export/导出.xml"


TYPE_MAP = {
    "Walking": ("Walk", "步行"),
    "Running": ("Run", "跑步"),
    "Cycling": ("Ride", "骑行"),
    "TableTennis": ("Training", "乒乓球"),
    "JumpRope": ("Training", "跳绳"),
    "FunctionalStrengthTraining": ("Training", "功能性力量训练"),
    "Badminton": ("Training", "羽毛球"),
    "Elliptical": ("Training", "椭圆机"),
    "Other": ("Training", "其他训练"),
}


def parse_apple_date(value: str) -> dt.datetime:
    return dt.datetime.strptime(value, APPLE_DATE_FORMAT)


@contextlib.contextmanager
def open_export(export_path: Path, export_member: str):
    """Open either a standalone XML export or an XML member inside a ZIP."""

    if export_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(export_path) as archive:
            member = export_member
            if member not in archive.namelist():
                candidates = [
                    name
                    for name in archive.namelist()
                    if name.startswith("apple_health_export/")
                    and name.endswith(".xml")
                    and not name.endswith("export_cda.xml")
                ]
                if len(candidates) != 1:
                    raise FileNotFoundError(
                        f"Could not identify Health export XML in {export_path}"
                    )
                member = candidates[0]
            with archive.open(member) as raw:
                with io.TextIOWrapper(raw, encoding="utf-8") as text:
                    yield text
    else:
        with export_path.open("r", encoding="utf-8") as text:
            yield text


def format_duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def stable_id(start: dt.datetime, activity_type: str, source: str) -> int:
    payload = f"{start.isoformat()}|{activity_type}|{source}".encode()
    # Keep the ID positive and within SQLite's signed 63-bit integer range.
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def encode_polyline(points: list[tuple[float, float]], precision: int = 5) -> str:
    """Encode latitude/longitude pairs using Google's polyline algorithm."""

    factor = 10**precision
    output: list[str] = []
    previous_lat = 0
    previous_lon = 0

    def encode_value(value: int) -> None:
        value = ~(value << 1) if value < 0 else value << 1
        while value >= 0x20:
            output.append(chr((0x20 | (value & 0x1F)) + 63))
            value >>= 5
        output.append(chr(value + 63))

    for lat, lon in points:
        current_lat = round(lat * factor)
        current_lon = round(lon * factor)
        encode_value(current_lat - previous_lat)
        encode_value(current_lon - previous_lon)
        previous_lat = current_lat
        previous_lon = current_lon
    return "".join(output)


def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    radius = 6_371_000.0
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    value = math.sin(dlat / 2) ** 2 + (
        math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return radius * 2 * math.atan2(math.sqrt(value), math.sqrt(1 - value))


def simplify_points(
    points: list[tuple[float, float, float | None]], min_distance_m: float = 5.0
) -> list[tuple[float, float, float | None]]:
    """Reduce payload size without materially changing the displayed route."""

    if len(points) <= 2:
        return points
    kept = [points[0]]
    for point in points[1:-1]:
        if haversine_m(kept[-1][:2], point[:2]) >= min_distance_m:
            kept.append(point)
    kept.append(points[-1])
    return kept


def read_route(route_path: Path) -> dict[str, object]:
    root = ET.parse(route_path).getroot()
    points: list[tuple[float, float, float | None]] = []
    for element in root.findall(".//gpx:trkpt", GPX_NS):
        lat = float(element.attrib["lat"])
        lon = float(element.attrib["lon"])
        elevation_element = element.find("gpx:ele", GPX_NS)
        elevation = (
            float(elevation_element.text)
            if elevation_element is not None and elevation_element.text
            else None
        )
        points.append((lat, lon, elevation))

    simplified = simplify_points(points)
    elevation_gain = 0.0
    previous_elevation: float | None = None
    for _, _, elevation in points:
        if elevation is not None and previous_elevation is not None:
            elevation_gain += max(0.0, elevation - previous_elevation)
        if elevation is not None:
            previous_elevation = elevation

    return {
        "summary_polyline": encode_polyline([(p[0], p[1]) for p in simplified]),
        "elevation_gain": round(elevation_gain, 1),
        "raw_points": len(points),
        "encoded_points": len(simplified),
    }


def statistic_value(workout: ET.Element, suffix: str, attribute: str) -> float | None:
    for statistic in workout.findall("WorkoutStatistics"):
        if statistic.attrib.get("type", "").endswith(suffix):
            value = statistic.attrib.get(attribute)
            if value is not None:
                return float(value)
    return None


def distance_m(workout: ET.Element) -> float:
    for suffix in (
        "DistanceWalkingRunning",
        "DistanceCycling",
        "DistanceSwimming",
        "DistanceDownhillSnowSports",
    ):
        for statistic in workout.findall("WorkoutStatistics"):
            if statistic.attrib.get("type", "").endswith(suffix):
                value = float(statistic.attrib.get("sum", "0"))
                unit = statistic.attrib.get("unit", "m")
                if unit == "km":
                    return value * 1000
                if unit == "mi":
                    return value * 1609.344
                return value
    return 0.0


def parse_workouts(
    export_path: Path, export_member: str, routes_dir: Path
) -> list[dict[str, object]]:
    workouts: list[dict[str, object]] = []
    collecting = False
    block: list[str] = []

    with open_export(export_path, export_member) as source:
        for line in source:
            stripped = line.lstrip()
            if not collecting and stripped.startswith("<Workout "):
                collecting = True
                block = [line]
                continue
            if not collecting:
                continue
            block.append(line)
            if stripped.startswith("</Workout>"):
                workout = ET.fromstring("".join(block))
                attr = workout.attrib
                raw_type = attr["workoutActivityType"].removeprefix(
                    "HKWorkoutActivityType"
                )
                activity_type, title = TYPE_MAP.get(
                    raw_type, ("Training", raw_type or "训练")
                )
                start_local = parse_apple_date(attr["startDate"])
                end_local = parse_apple_date(attr["endDate"])
                duration_seconds = float(attr.get("duration", "0"))
                if attr.get("durationUnit") == "min":
                    duration_seconds *= 60
                elif attr.get("durationUnit") == "hr":
                    duration_seconds *= 3600

                route_file = None
                file_reference = workout.find("./WorkoutRoute/FileReference")
                if file_reference is not None:
                    route_file = Path(file_reference.attrib["path"]).name
                route_data: dict[str, object] = {
                    "summary_polyline": "",
                    "elevation_gain": 0.0,
                }
                if route_file and (routes_dir / route_file).is_file():
                    route_data = read_route(routes_dir / route_file)

                distance = distance_m(workout)
                average_hr = statistic_value(workout, "HeartRate", "average")
                source_name = attr.get("sourceName", "Apple Health")
                start_utc = start_local.astimezone(dt.timezone.utc)
                workouts.append(
                    {
                        "run_id": stable_id(start_local, raw_type, source_name),
                        "name": title,
                        "distance": round(distance, 3),
                        "moving_time": format_duration(duration_seconds),
                        "type": activity_type,
                        "subtype": raw_type,
                        "start_date": start_utc.isoformat(sep=" "),
                        "start_date_local": start_local.replace(tzinfo=None).isoformat(
                            sep=" "
                        ),
                        "location_country": None,
                        "summary_polyline": route_data["summary_polyline"],
                        "average_heartrate": average_hr,
                        "average_speed": (
                            round(distance / duration_seconds, 4)
                            if duration_seconds > 0
                            else 0.0
                        ),
                        "elevation_gain": route_data["elevation_gain"],
                        # Do not publish the personal device name from HealthKit.
                        "source": "Apple Health",
                        "streak": 0,
                        "_start": start_local,
                        "_end": end_local,
                        "_route_file": route_file,
                    }
                )
                collecting = False
                block = []

    workouts.sort(key=lambda item: item["_start"])
    return workouts


ATTRIBUTE_RE = re.compile(r'(\w+)="([^"]*)"')


def fill_missing_heart_rates(
    export_path: Path, export_member: str, workouts: list[dict[str, object]]
) -> int:
    """Compute missing workout averages from raw Apple Watch HR samples."""

    missing = [item for item in workouts if item["average_heartrate"] is None]
    if not missing:
        return 0
    missing.sort(key=lambda item: item["_start"])
    sums = [0.0] * len(missing)
    counts = [0] * len(missing)
    starts = [item["_start"] for item in missing]

    needle = f'type="{HEART_RATE_TYPE}"'
    with open_export(export_path, export_member) as source:
        for line in source:
            if needle not in line or not line.lstrip().startswith("<Record "):
                continue
            attributes = dict(ATTRIBUTE_RE.findall(line))
            source_name = " ".join(attributes.get("sourceName", "").split())
            if "Apple Watch" not in source_name:
                continue
            timestamp = parse_apple_date(attributes["startDate"])
            index = bisect.bisect_right(starts, timestamp) - 1
            if index >= 0 and timestamp <= missing[index]["_end"]:
                sums[index] += float(attributes["value"])
                counts[index] += 1

    filled = 0
    for item, total, count in zip(missing, sums, counts, strict=True):
        if count:
            item["average_heartrate"] = round(total / count, 2)
            filled += 1
    return filled


def add_streaks(workouts: list[dict[str, object]]) -> None:
    streak = 0
    last_date: dt.date | None = None
    for workout in workouts:
        date = workout["_start"].date()
        if last_date is None or date > last_date + dt.timedelta(days=1):
            streak = 1
        elif date == last_date + dt.timedelta(days=1):
            streak += 1
        workout["streak"] = streak
        last_date = date


def public_activity(workout: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in workout.items() if not key.startswith("_")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path, help="Apple Health export ZIP or export.xml path")
    parser.add_argument(
        "--export-member",
        default=DEFAULT_EXPORT_MEMBER,
        help="XML member path when the input is a ZIP",
    )
    parser.add_argument("--routes", type=Path, required=True, help="workout-routes directory")
    parser.add_argument("--output", type=Path, required=True, help="activities.json output")
    parser.add_argument(
        "--skip-heart-rate-fallback",
        action="store_true",
        help="Do not scan raw heart-rate records for older workouts",
    )
    args = parser.parse_args()

    workouts = parse_workouts(args.export, args.export_member, args.routes)
    filled = 0
    if not args.skip_heart_rate_fallback:
        filled = fill_missing_heart_rates(args.export, args.export_member, workouts)
    add_streaks(workouts)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        json.dump(
            [public_activity(item) for item in workouts],
            destination,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    types = Counter(item["type"] for item in workouts)
    routed = sum(bool(item["summary_polyline"]) for item in workouts)
    with_hr = sum(item["average_heartrate"] is not None for item in workouts)
    print(f"Imported {len(workouts)} workouts: {dict(types)}")
    print(f"Routes: {routed}; heart-rate averages: {with_hr} ({filled} reconstructed)")
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1024 / 1024:.1f} MiB)")


if __name__ == "__main__":
    main()

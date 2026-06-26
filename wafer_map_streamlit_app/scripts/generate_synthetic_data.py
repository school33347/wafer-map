from __future__ import annotations

import csv
import math
from pathlib import Path
from random import Random


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "synthetic_wafer_map.csv"


def gaussian(distance: float, sigma: float) -> float:
    return math.exp(-(distance**2) / (2 * sigma**2))


def build_rows() -> list[dict[str, object]]:
    rng = Random(20260623)
    rows: list[dict[str, object]] = []
    wafer_radius = 12.0
    lot_id = "LOT-SYN-0623"
    wafer_id = "WAFER-DEMO-001"

    for die_x in range(-12, 13):
        for die_y in range(-12, 13):
            radius = math.sqrt(die_x**2 + die_y**2)
            if radius > wafer_radius:
                continue

            x_norm = die_x / wafer_radius
            y_norm = die_y / wafer_radius
            radius_norm = radius / wafer_radius
            theta_deg = math.degrees(math.atan2(die_y, die_x))

            edge_ring = gaussian(radius_norm - 0.88, 0.055)
            center_bump = gaussian(radius_norm, 0.22)
            gradient = 2.8 * x_norm - 1.6 * y_norm
            local_distance = math.sqrt((die_x - 4.0) ** 2 + (die_y + 5.0) ** 2)
            local_defect = gaussian(local_distance, 0.85)
            noise = rng.gauss(0, 0.42)

            thickness_nm = 100.0 + gradient + 5.4 * center_bump - 4.7 * edge_ring - 11.5 * local_defect + noise
            sheet_resistance = (
                55.0
                - 0.18 * (thickness_nm - 100.0)
                + 2.8 * edge_ring
                - 1.7 * center_bump
                + 8.2 * local_defect
                + rng.gauss(0, 0.18)
            )

            if radius_norm >= 0.82:
                zone = "edge"
            elif radius_norm <= 0.25:
                zone = "center"
            else:
                zone = "middle"

            rows.append(
                {
                    "lot_id": lot_id,
                    "wafer_id": wafer_id,
                    "site_id": f"X{die_x:+03d}_Y{die_y:+03d}",
                    "die_x": die_x,
                    "die_y": die_y,
                    "radius_norm": round(radius_norm, 5),
                    "theta_deg": round(theta_deg, 3),
                    "zone": zone,
                    "thickness_nm": round(thickness_nm, 4),
                    "sheet_resistance_ohm_sq": round(sheet_resistance, 4),
                    "synthetic_pattern_note": "edge_ring+center_anomaly+gradient+local_defect",
                }
            )
    return rows


def main() -> None:
    rows = build_rows()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()

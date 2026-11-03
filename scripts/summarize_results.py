#!/usr/bin/env python3
"""Export one row per completed mode/seed; do not pool lane observations."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--include-truncated", action="store_true")
    args = parser.parse_args()
    rows = []
    for manifest in sorted(args.results.glob("*/seed_*/manifest.json")):
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        if meta["status"] != "completed" and not (
            args.include_truncated and meta["status"] == "truncated"
        ):
            continue
        row = {
            key: meta[key]
            for key in (
                "mode",
                "seed",
                "status",
                "steps",
                "simulation_time",
                "remaining_vehicles",
            )
        }
        row.update(json.loads((manifest.parent / "traffic_metrics.json").read_text()))
        rows.append(row)
    if not rows:
        parser.exit(1, "No hay corridas completadas para resumir.\n")
    target = args.results / "summary.csv"
    if target.exists():
        parser.exit(
            1,
            "summary.csv ya existe; consérvalo o cambia de carpeta antes de resumir.\n",
        )
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} corridas resumidas en {target}")


if __name__ == "__main__":
    main()

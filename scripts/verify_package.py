#!/usr/bin/env python3
"""Verify original bytes, explicit seeds and all scenario dependencies offline."""
from pathlib import Path
import hashlib
import json
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_experiment import read_seeds


def main():
    manifest = json.loads((ROOT / "original/MANIFEST.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = ROOT / item["path"]
        assert path.is_file(), f"Falta {path}"
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        ), f"Cambio en {path}"
    seeds = read_seeds(ROOT / "config/seeds.txt")
    assert len(seeds) == 100 and seeds == sorted(
        seeds
    ), "Deben conservarse las 100 semillas ordenadas."
    csv_lines = (ROOT / "config/seeds.csv").read_text().splitlines()[1:]
    assert csv_lines == [
        f"{i},{s}" for i, s in enumerate(seeds, 1)
    ], "TXT/CSV inconsistentes."
    scenario = ROOT / "scenarios/bologna"
    config = ET.parse(scenario / "run.sumocfg")
    for tag in ("net-file", "route-files", "additional-files"):
        for name in config.find("input/" + tag).get("value").split(","):
            assert (scenario / name).is_file(), f"Dependencia ausente: {name}"
            assert (scenario / name).read_bytes() == (
                ROOT / "original" / name
            ).read_bytes(), name
            ET.parse(scenario / name)
    for folder in (ROOT / "src", ROOT / "scripts"):
        for path in folder.glob("*.py"):
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
    print(
        f'OK: {len(manifest["files"])} originales íntegros, 100 semillas y escenario XML completo.'
    )


if __name__ == "__main__":
    main()

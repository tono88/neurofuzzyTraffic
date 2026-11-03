#!/usr/bin/env python3
"""Execute a paired design with explicit seeds and isolated child processes.

Standard-library-only planning and Estandar runs. Fuzzy/NeuroFuzzy additionally
load the pinned scientific environment. See README.md for executable examples.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
MODES = ("Estandar", "Fuzzy", "NeuroFuzzy")


def read_seeds(path: Path) -> list[int]:
    """Read one integer per line, rejecting duplicate or invalid replicates."""
    lines = [s.strip() for s in path.read_text(encoding="utf-8-sig").splitlines()]
    values = [int(s) for s in lines if s and not s.startswith("#")]
    if (
        not values
        or len(values) != len(set(values))
        or any(not 1 <= s <= 2147483647 for s in values)
    ):
        raise ValueError(
            "La lista debe contener semillas positivas únicas (enteros de 32 bits)."
        )
    return values


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_hashes() -> dict[str, str]:
    """Hash all executable inputs, independently of generated output paths."""
    files = [
        Path(__file__).resolve(),
        *sorted((ROOT / "src").glob("*.py")),
        *sorted((ROOT / "scenarios/bologna").glob("*")),
        ROOT / "requirements.txt",
        ROOT / "config/seeds.txt",
        ROOT / "config/seeds.csv",
    ]
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in files if p.is_file()}


def save_json(path: Path, value: object) -> None:
    """Replace metadata atomically to avoid accepting a partial manifest."""
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def find_sumo(home: str | None = None) -> tuple[Path, Path]:
    """Locate SUMO and its matching TraCI tools; never install silently."""
    home = home or os.environ.get("SUMO_HOME")
    executable = shutil.which("sumo")
    if home:
        base = Path(home).resolve()
        binary = base / "bin" / ("sumo.exe" if os.name == "nt" else "sumo")
    elif executable:
        binary = Path(executable).resolve()
        base = binary.parent.parent
    else:
        raise RuntimeError(
            "Instala SUMO 1.16.0 y declara SUMO_HOME; consulta README.md."
        )
    if not binary.is_file() or not (base / "tools/traci").is_dir():
        raise RuntimeError(f"No se encuentran el ejecutable y tools/traci en {base}.")
    return binary, base


def environment(binary: Path) -> dict:
    result = subprocess.run(
        [str(binary), "--version"], capture_output=True, text=True, check=True
    )
    packages = {}
    for name in (
        "numpy",
        "tensorflow",
        "tensorflow-intel",
        "scikit-fuzzy",
        "scipy",
        "networkx",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "XlsxWriter",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "sumo": result.stdout.splitlines()[0],
        "packages": packages,
    }


def make_configuration(out: Path) -> Path:
    """Use unchanged network/demand and redirect every SUMO output per run.

    Only detector file destinations and configuration paths change. Detector
    geometry, sampling periods, demand, signal plans and vehicle types persist.
    """
    scenario = ROOT / "scenarios/bologna"
    detectors = ET.parse(scenario / "joined_detectors.add.xml")
    for element in detectors.iter():
        if "file" in element.attrib:
            element.set("file", (out / "e1_output.xml").as_posix())
    detectors.write(out / "detectors.add.xml", encoding="utf-8", xml_declaration=True)
    tree = ET.parse(scenario / "run.sumocfg")
    for tag in ("net-file", "route-files", "additional-files"):
        element = tree.find("input/" + tag)
        paths = []
        for name in element.get("value").split(","):
            path = (
                out / "detectors.add.xml"
                if name == "joined_detectors.add.xml"
                else scenario / name
            )
            paths.append(path.as_posix())
        element.set("value", ",".join(paths))
    for tag, name in [
        ("output/tripinfo-output", "tripinfo.xml"),
        ("output/save-state.prefix", "state"),
        ("report/log", "sumo.log"),
    ]:
        tree.find(tag).set("value", (out / name).as_posix())
    config = out / "run.sumocfg"
    tree.write(config, encoding="utf-8", xml_declaration=True)
    return config


class PredictionMetrics:
    """Online finite-pair metrics and threshold confusion, without storing rows.

    MAPE follows the source convention: replace a zero denominator with one.
    These are descriptive comparisons; see methodological limits in the docs.
    """

    def __init__(self):
        self.n = 0
        self.mean = self.m2 = self.sse = self.sae = self.sape = 0.0
        self.confusion = [[0, 0, 0] for _ in range(3)]

    def add(self, row):
        _, _, truth, prediction, low, high = row
        if prediction is None or not all(
            math.isfinite(float(x)) for x in (truth, prediction, low, high)
        ):
            return
        self.n += 1
        delta = truth - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (truth - self.mean)
        err = truth - prediction
        self.sse += err * err
        self.sae += abs(err)
        self.sape += abs(err / (truth if truth != 0 else 1.0))

        def category(value):
            return 0 if value < low else (1 if value < high else 2)

        self.confusion[category(truth)][category(prediction)] += 1

    def result(self):
        return {
            "n_valid_pairs": self.n,
            "RMSE": math.sqrt(self.sse / self.n) if self.n else None,
            "MAE": self.sae / self.n if self.n else None,
            "MAPE_percent_source_convention": (
                100 * self.sape / self.n if self.n else None
            ),
            "R2": 1 - self.sse / self.m2 if self.n >= 2 and self.m2 > 0 else None,
            "classes": ["Bajo", "Medio", "Alto"],
            "confusion_rows_true_columns_predicted": self.confusion,
        }


def traffic_metrics(path: Path) -> dict:
    """Summarize arrived trips only; interrupted runs are explicitly marked."""
    fields = ("duration", "waitingTime", "timeLoss", "routeLength")
    totals = dict.fromkeys(fields, 0.0)
    count = 0
    for _, node in ET.iterparse(path, events=("end",)):
        if node.tag == "tripinfo" and float(node.get("arrival", "-1")) >= 0:
            count += 1
            for field in fields:
                totals[field] += float(node.get(field, "0"))
        node.clear()
    return {
        "arrived_trips": count,
        **{f"mean_{k}": v / count if count else None for k, v in totals.items()},
    }


def worker(job_path: Path) -> int:
    """Run exactly one mode/seed. Process isolation fixes hash/RNG startup state."""
    job = json.loads(job_path.read_text(encoding="utf-8"))
    out = job_path.parent
    meta = {
        **job,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
    }
    save_json(out / "manifest.json", meta)
    started = False
    try:
        home = Path(job["sumo_home"])
        os.environ["SUMO_HOME"] = str(home)
        sys.path.insert(0, str(home / "tools"))
        import traci

        config = make_configuration(out)
        command = [
            job["sumo_binary"],
            "-c",
            str(config),
            "--seed",
            str(job["seed"]),
            "--queue-output",
            str(out / "queue.xml"),
        ]
        meta["sumo_command"] = command
        mode = job["mode"]
        if mode != "Estandar":
            # Imports follow child environment configuration, before any model.
            import tensorflow as tf

            tf.config.threading.set_inter_op_parallelism_threads(1)
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.keras.utils.set_random_seed(job["seed"])
            tf.config.experimental.enable_op_determinism()
            from src import controllers
        traci.start(command)
        started = True
        steps = rows = 0
        metrics = PredictionMetrics()
        if mode == "Estandar":
            while traci.simulation.getMinExpectedNumber() > 0 and (
                job["max_steps"] is None or steps < job["max_steps"]
            ):
                traci.simulationStep()
                steps += 1
        else:
            function = (
                controllers.run_fuzzy if mode == "Fuzzy" else controllers.run_neurofuzzy
            )
            with (out / "details.csv").open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    ["step", "lane_id", "y_true", "y_pred", "med_lb", "med_ub"]
                )
                for row in function(max_steps=job["max_steps"]):
                    writer.writerow(row)
                    metrics.add(row)
                    steps = row[0]
                    rows += 1
            prediction = metrics.result()
            prediction["interpretation"] = (
                "identity_not_a_forecast"
                if mode == "Fuzzy"
                else "held_intersection_mean_prediction_vs_current_lane_density"
            )
            save_json(out / "prediction_metrics.json", prediction)
        remaining = traci.simulation.getMinExpectedNumber()
        sim_time = traci.simulation.getTime()
        traci.close()
        started = False
        save_json(out / "traffic_metrics.json", traffic_metrics(out / "tripinfo.xml"))
        meta.update(
            status="completed" if remaining == 0 else "truncated",
            steps=steps,
            simulation_time=sim_time,
            remaining_vehicles=remaining,
            detail_rows=rows,
        )
        meta["output_sha256"] = {
            p.name: digest(p)
            for p in out.iterdir()
            if p.suffix in (".xml", ".json", ".csv")
            and p.name not in ("manifest.json", "job.json")
        }
    except Exception as error:
        meta.update(status="failed", error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
    finally:
        if started:
            try:
                traci.close()
            except Exception:
                pass
        meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save_json(out / "manifest.json", meta)
    return 0 if meta["status"] in ("completed", "truncated") else 1


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds-file", type=Path, default=ROOT / "config/seeds.txt")
    p.add_argument(
        "--seed", type=int, help="Una semilla de prueba; reemplaza seeds-file."
    )
    p.add_argument(
        "--limit", type=int, help="Primeras N semillas del listado, sin sorteo."
    )
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    p.add_argument(
        "--max-steps",
        type=int,
        help="Prueba recortada; omitir para el experimento completo.",
    )
    p.add_argument("--output", type=Path, default=ROOT / "results")
    p.add_argument("--sumo-home", help="Instalación de SUMO con bin/ y tools/.")
    p.add_argument(
        "--plan", action="store_true", help="Mostrar el diseño sin ejecutar SUMO."
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Omitir solamente corridas completadas e idénticas.",
    )
    p.add_argument("--_worker", type=Path, help=argparse.SUPPRESS)
    return p


def main(argv=None) -> int:
    p = parser()
    opts = p.parse_args(argv)
    if opts._worker:
        return worker(opts._worker.resolve())
    try:
        seeds = (
            [opts.seed]
            if opts.seed is not None
            else read_seeds(opts.seeds_file.resolve())
        )
        if opts.limit is not None:
            if opts.limit < 1:
                raise ValueError("--limit debe ser positivo.")
            seeds = seeds[: opts.limit]
        if any(s < 1 or s > 2147483647 for s in seeds):
            raise ValueError("Semilla fuera del rango admitido.")
        if opts.max_steps is not None and opts.max_steps < 1:
            raise ValueError("--max-steps debe ser positivo.")
        if len(opts.modes) != len(set(opts.modes)):
            raise ValueError("No repitas modos.")
        plan = {
            "seeds": seeds,
            "modes": opts.modes,
            "runs": len(seeds) * len(opts.modes),
            "max_steps": opts.max_steps,
        }
        if opts.plan:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0
        binary, home = find_sumo(opts.sumo_home)
        env_info = environment(binary)
        hashes = input_hashes()
        out_root = opts.output.resolve()
        out_root.mkdir(parents=True, exist_ok=True)
        # Preflight every destination before starting any simulation.
        jobs = []
        for seed in seeds:
            for mode in opts.modes:
                out = out_root / mode / f"seed_{seed}"
                job = {
                    "seed": seed,
                    "mode": mode,
                    "max_steps": opts.max_steps,
                    "sumo_binary": str(binary),
                    "sumo_home": str(home),
                    "environment": env_info,
                    "input_sha256": hashes,
                    "rng_policy": "same_seed_SUMO_Python_NumPy_TensorFlow_CPU_single_thread",
                }
                job["signature"] = hashlib.sha256(
                    json.dumps(job, sort_keys=True).encode()
                ).hexdigest()
                if out.exists():
                    manifest = out / "manifest.json"
                    previous = (
                        json.loads(manifest.read_text(encoding="utf-8"))
                        if manifest.exists()
                        else {}
                    )
                    outputs = previous.get("output_sha256", {})
                    valid_outputs = bool(outputs) and all(
                        (out / name).is_file() and digest(out / name) == sha
                        for name, sha in outputs.items()
                    )
                    if (
                        opts.resume
                        and previous.get("status") == "completed"
                        and previous.get("signature") == job["signature"]
                        and valid_outputs
                    ):
                        print(f"Conservada: {mode}, semilla {seed}", flush=True)
                        continue
                    raise RuntimeError(
                        f"El destino ya existe: {out}. Usa otra carpeta; no se sobrescribe."
                    )
                jobs.append((out, job))
        save_json(
            out_root / "plan.json",
            {**plan, "environment": env_info, "input_sha256": hashes},
        )
        for index, (out, job) in enumerate(jobs, 1):
            out.mkdir(parents=True, exist_ok=False)
            save_json(out / "job.json", job)
            child_env = {
                **os.environ,
                "PYTHONHASHSEED": str(job["seed"]),
                "TF_DETERMINISTIC_OPS": "1",
                "TF_ENABLE_ONEDNN_OPTS": "0",
                "CUDA_VISIBLE_DEVICES": "-1",
                "TF_CPP_MIN_LOG_LEVEL": "2",
                "OMP_NUM_THREADS": "1",
                "MPLBACKEND": "Agg",
                "PYTHONUTF8": "1",
            }
            print(
                f"[{index}/{len(jobs)}] {job['mode']}, semilla {job['seed']}",
                flush=True,
            )
            with (out / "console.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--_worker",
                        str(out / "job.json"),
                    ],
                    cwd=out,
                    env=child_env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            if result.returncode:
                raise RuntimeError(f'Falló la corrida; consulta {out / "console.log"}.')
        print(f"Ejecuciones terminadas. Resultados: {out_root}")
        return 0
    except (ValueError, OSError, RuntimeError) as error:
        p.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())

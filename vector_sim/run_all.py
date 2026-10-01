"""Run physical tests and all corrected studies into a separate output directory."""

import argparse, subprocess, sys, time
from pathlib import Path
from simulation_utils import ROOT, provenance, save_json

SCRIPTS = [
    "study1_sweeps.py",
    "study2_snr.py",
    "study3_heatmap.py",
    "study4_bearing.py",
    "study5_sensitivity.py",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "simulation_results" / "v2")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if (
        args.quick
        and args.out.resolve() == (ROOT / "simulation_results" / "v2").resolve()
    ):
        args.out = ROOT / "simulation_results" / "smoke"
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "unittest", "test_simulation", "-v"],
        cwd=ROOT,
        check=True,
    )
    timings = {}
    for script in SCRIPTS:
        print(f"Running {script}", flush=True)
        start = time.perf_counter()
        command = [sys.executable, str(ROOT / script), "--out", str(out)]
        if args.quick:
            command.append("--quick")
        with (out / (Path(script).stem + ".log")).open("w", encoding="utf-8") as log:
            subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True
            )
        timings[script] = time.perf_counter() - start
        print(f"Completed in {timings[script]:.1f} s", flush=True)
    save_json(out / "run_timings.json", timings)
    provenance(
        out, "run_all", {"complete": True, "quick": args.quick, "scripts": SCRIPTS}
    )
    print(f"Results: {out}")


if __name__ == "__main__":
    main()

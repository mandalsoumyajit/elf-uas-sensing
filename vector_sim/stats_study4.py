"""Report corrected localization metrics without selecting only favorable cells."""

from simulation_utils import arguments

if __name__ == "__main__":
    args = arguments(__doc__)
    print((args.out / "study4_summary.json").read_text(encoding="utf-8"))

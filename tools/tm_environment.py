"""Report detailed cation compositions of tetrahedral n-TM environments."""
import argparse
import json
from collections import Counter
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distortion_percolation.structure_analysis import DEFAULT_CATIONS, DEFAULT_OFFSET, atomic_write_text, comma_strings, indexed_paths, nearest_cations, offset_centers, positive_float, positive_int, vector3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perfect", default="perfect.vasp")
    parser.add_argument("--pattern", default="structure_{i}.vasp")
    parser.add_argument("--start", type=int)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--step", type=positive_int, default=1)
    parser.add_argument("--inclusive-stop", action="store_true")
    parser.add_argument("--offset", type=vector3, default=DEFAULT_OFFSET)
    parser.add_argument("--offset-mode", choices=("cartesian", "fractional"), default="cartesian")
    parser.add_argument("--cations", type=comma_strings, default=list(DEFAULT_CATIONS))
    parser.add_argument("--li", default="Li")
    parser.add_argument("--classes", type=comma_strings, default=["0", "1", "2", "3", "4"])
    parser.add_argument("--radius", type=positive_float, default=5.0)
    parser.add_argument("--output", type=Path, default=Path("tm_environments.json"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    from pymatgen.core import Structure

    try:
        classes = {int(item) for item in args.classes}
    except ValueError:
        parser.error("--classes must contain integers from 0 through 4")
    if not classes or not classes <= {0, 1, 2, 3, 4}:
        parser.error("--classes must contain integers from 0 through 4")
    perfect = Structure.from_file(args.perfect)
    centers = offset_centers(perfect, args.offset, args.offset_mode)
    result = []
    try:
        paths = indexed_paths(args.pattern, args.start, args.stop, args.step, inclusive_stop=args.inclusive_stop)
    except ValueError as exc:
        parser.error(str(exc))
    for path in paths:
        if not path.exists():
            continue
        structure = Structure.from_file(path)
        environments = Counter()
        for center in centers:
            nearby = nearest_cations(structure, center, args.cations, args.radius, 4)
            if len(nearby) < 4:
                continue
            symbols = sorted(site.specie.symbol for site in nearby)
            number = 4 - symbols.count(args.li)
            if number in classes:
                environments[f"{number}-TM:" + "-".join(symbols)] += 1
        result.append({"file": str(path), "environments": dict(environments)})
    atomic_write_text(args.output, json.dumps(result, indent=2), overwrite=args.overwrite)
    print(f"analyzed {len(result)} structures")


if __name__ == "__main__":
    main()

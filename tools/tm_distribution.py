"""Calculate 0-TM through 4-TM fractions using nearest-four or radius-count profiles."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distortion_percolation.structure_analysis import (
    DEFAULT_CATIONS, DEFAULT_OFFSET, TM_PROFILES, atomic_write_text, comma_strings, ensure_output_paths,
    indexed_paths, nearest_cations, offset_centers, positive_float, positive_int,
    resolve_tm_elements, vector3,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perfect", default="perfect.vasp"); parser.add_argument("--pattern", default="structure_{i}.vasp")
    parser.add_argument("--start", type=int); parser.add_argument("--stop", type=int); parser.add_argument("--step", type=positive_int, default=1)
    parser.add_argument("--inclusive-stop", action="store_true", help="Treat --stop as inclusive (default: exclusive)")
    parser.add_argument("--offset", type=vector3, default=DEFAULT_OFFSET); parser.add_argument("--offset-mode", choices=("cartesian", "fractional"), default="cartesian")
    parser.add_argument("--mode", choices=("nearest-four", "radius-count"), default="nearest-four")
    parser.add_argument("--cations", type=comma_strings, default=list(DEFAULT_CATIONS)); parser.add_argument("--li", default="Li")
    parser.add_argument("--tm-profile", choices=tuple(TM_PROFILES), default="standard"); parser.add_argument("--tm-elements", type=comma_strings)
    parser.add_argument("--radius", type=positive_float, default=5.0); parser.add_argument("--neighbors", type=positive_int, default=4)
    parser.add_argument("--exclude-element", default=None, help="Skip centers near this element, e.g. P")
    parser.add_argument("--exclude-radius", type=positive_float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("distribution.json")); parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.mode == "nearest-four" and args.neighbors != 4:
        parser.error("n-TM classification currently requires --neighbors 4")
    try: paths = indexed_paths(args.pattern, args.start, args.stop, args.step, inclusive_stop=args.inclusive_stop)
    except ValueError as exc: parser.error(str(exc))

    from pymatgen.core import Structure
    perfect = Structure.from_file(args.perfect); centers = offset_centers(perfect, args.offset, args.offset_mode)
    tm_elements = resolve_tm_elements(args.tm_elements, args.tm_profile); report = []
    for path in paths:
        if not path.exists(): print(f"warning: skipping missing {path}"); continue
        structure = Structure.from_file(path); counts = [0] * 5; skipped = 0
        for center in centers:
            if args.exclude_element and any(site.specie.symbol == args.exclude_element for site in structure.get_sites_in_sphere(center, args.exclude_radius)):
                skipped += 1; continue
            if args.mode == "nearest-four":
                nearby = nearest_cations(structure, center, args.cations, args.radius, 4)
                if len(nearby) != 4: skipped += 1; continue
                tm_count = 4 - sum(site.specie.symbol == args.li for site in nearby)
            else:
                tm_count = sum(site.specie.symbol in tm_elements for site in structure.get_sites_in_sphere(center, args.radius))
                tm_count = min(tm_count, 4)
            counts[tm_count] += 1
        total = sum(counts); report.append({"file": str(path), "counts": counts,
            "fractions": [value / total if total else 0 for value in counts], "skipped_centers": skipped})
    atomic_write_text(args.output, json.dumps(report, indent=2), overwrite=args.overwrite); print(f"analyzed {len(report)} structures")


if __name__ == "__main__": main()

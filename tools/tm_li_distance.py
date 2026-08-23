"""Find connected Li sites and transition-metal distances from Li triangle planes."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from distortion_percolation.structure_analysis import DEFAULT_OFFSET, DEFAULT_THRESHOLDS, TM_PROFILES, atomic_write_text, comma_strings, indexed_paths, nearest_cations, offset_centers, positive_float, positive_int, resolve_tm_elements, threshold_map, vector3


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
    parser.add_argument("--tm-profile", choices=tuple(TM_PROFILES), default="standard")
    parser.add_argument("--tm-elements", type=comma_strings)
    parser.add_argument("--li", default="Li")
    parser.add_argument("--radius", type=positive_float, default=5.0)
    parser.add_argument(
        "--thresholds", type=threshold_map, default=DEFAULT_THRESHOLDS,
        help="Minimum TM-to-three-Li-plane heights in A (default: Mn=2.507,Ti=2.524)",
    )
    parser.add_argument("--one-tm-policy", choices=("exclude", "threshold", "all"), default="threshold")
    parser.add_argument("--output", type=Path, default=Path("tm_li_distance.json"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.tm_elements = resolve_tm_elements(args.tm_elements, args.tm_profile)
    missing = sorted(set(args.tm_elements) - set(args.thresholds))
    if args.one_tm_policy == "threshold" and missing:
        parser.error("missing thresholds for TM elements: " + ", ".join(missing))

    import numpy as np
    from pymatgen.core import Structure

    perfect = Structure.from_file(args.perfect)
    centers = offset_centers(perfect, args.offset, args.offset_mode)
    cations = [args.li, *args.tm_elements]
    report = []
    try:
        paths = indexed_paths(args.pattern, args.start, args.stop, args.step, inclusive_stop=args.inclusive_stop)
    except ValueError as exc:
        parser.error(str(exc))
    for path in paths:
        if not path.exists():
            continue
        structure = Structure.from_file(path)
        connected = set(); records = []
        for center in centers:
            nearby = nearest_cations(structure, center, cations, args.radius, 4)
            lithium = [site for site in nearby if site.specie.symbol == args.li]
            metals = [site for site in nearby if site.specie.symbol in args.tm_elements]
            if len(lithium) == 4:
                connected.update(site.index for site in lithium)
            elif len(lithium) == 3 and len(metals) == 1:
                vectors = np.asarray([site.coords for site in lithium])
                normal = np.cross(vectors[1] - vectors[0], vectors[2] - vectors[0])
                norm = np.linalg.norm(normal)
                distance = None if norm == 0 else abs(float(np.dot(metals[0].coords - vectors[0], normal / norm)))
                symbol = metals[0].specie.symbol
                accepted = args.one_tm_policy == "all" or (args.one_tm_policy == "threshold" and distance is not None and distance > args.thresholds[symbol])
                records.append({"element": symbol, "distance_to_li_plane": distance, "connected": accepted})
                if accepted:
                    connected.update(int(site.index) for site in lithium)
        report.append({"file": str(path), "connected_li_indices": sorted(int(index) for index in connected), "one_tm": records})
    atomic_write_text(args.output, json.dumps(report, indent=2, allow_nan=False), overwrite=args.overwrite)
    print(f"analyzed {len(report)} structures")


if __name__ == "__main__":
    main()

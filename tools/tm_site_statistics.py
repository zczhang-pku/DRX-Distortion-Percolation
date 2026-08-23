"""Count transition-metal species in selected n-TM tetrahedral classes."""
import argparse
import json
from collections import Counter
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distortion_percolation.structure_analysis import DEFAULT_OFFSET, TM_PROFILES, atomic_write_text, comma_strings, indexed_paths, nearest_cations, offset_centers, positive_float, positive_int, resolve_tm_elements, vector3


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
    parser.add_argument("--classes", type=comma_strings, default=["1", "2", "3", "4"])
    parser.add_argument("--radius", type=positive_float, default=5.0)
    parser.add_argument("--exclude-element", help="Skip centers near this element, e.g. P")
    parser.add_argument("--exclude-radius", type=positive_float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("tm_site_statistics.json"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    from pymatgen.core import Structure

    args.tm_elements = resolve_tm_elements(args.tm_elements, args.tm_profile)
    classes = {int(value) for value in args.classes}
    if not classes <= {0, 1, 2, 3, 4}:
        parser.error("--classes must contain values from 0 to 4")
    perfect = Structure.from_file(args.perfect)
    centers = offset_centers(perfect, args.offset, args.offset_mode)
    total_counters = {number: Counter() for number in classes}
    site_counts = Counter()
    details = []
    cations = [args.li, *args.tm_elements]
    try:
        paths = indexed_paths(args.pattern, args.start, args.stop, args.step, inclusive_stop=args.inclusive_stop)
    except ValueError as exc:
        parser.error(str(exc))
    for path in paths:
        if not path.exists():
            continue
        structure = Structure.from_file(path)
        local = {number: Counter() for number in classes}
        local_sites = Counter()
        for center in centers:
            if args.exclude_element and any(site.specie.symbol == args.exclude_element for site in structure.get_sites_in_sphere(center, args.exclude_radius)):
                continue
            nearby = nearest_cations(structure, center, cations, args.radius, 4)
            if len(nearby) < 4:
                continue
            metals = [site.specie.symbol for site in nearby if site.specie.symbol in args.tm_elements]
            number = len(metals)
            if number in classes:
                local[number].update(metals); local_sites[number] += 1
        for number in classes:
            total_counters[number].update(local[number]); site_counts[number] += local_sites[number]
        details.append({"file": str(path), "sites": dict(local_sites),
                        "species_counts": {str(key): dict(value) for key, value in local.items()}})
    summary = {}
    for number in sorted(classes):
        denominator = site_counts[number] * number
        summary[str(number)] = {"site_count": site_counts[number], "species_counts": dict(total_counters[number]),
                                "probabilities": {symbol: total_counters[number][symbol] / denominator if denominator else 0
                                                  for symbol in args.tm_elements}}
    atomic_write_text(args.output, json.dumps({"summary": summary, "structures": details}, indent=2), overwrite=args.overwrite)
    print(f"analyzed {len(details)} structures")


if __name__ == "__main__":
    main()

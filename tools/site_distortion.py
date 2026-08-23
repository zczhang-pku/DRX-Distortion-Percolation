"""Measure PBC-shortest atomic displacements, including batch time series."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distortion_percolation.structure_analysis import (
    comma_strings, indexed_paths, load_structure_pair, OutputTransaction,
    species_group_labels, validate_structure_pair,
)


def matching_groups(value: str) -> list[list[str]]:
    groups = [comma_strings(group) for group in value.split(";") if group.strip()]
    if not groups:
        raise argparse.ArgumentTypeError("expected semicolon-separated species groups")
    return groups


def calculate(reference, observed, species_constraint: bool = True,
              lattice_tolerance: float = 1e-5, species_groups=None):
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    validate_structure_pair(reference, observed, lattice_tolerance)
    distances = reference.lattice.get_all_distances(observed.frac_coords, reference.frac_coords)
    cost = np.asarray(distances)
    observed_species = [site.species_string for site in observed]
    reference_species = [site.species_string for site in reference]
    if species_groups is not None:
        observed_groups = np.asarray(species_group_labels(observed_species, species_groups))
        reference_groups = np.asarray(species_group_labels(reference_species, species_groups))
        cost[observed_groups[:, None] != reference_groups[None, :]] = np.inf
    elif species_constraint:
        observed_species_array = np.asarray(observed_species)
        reference_species_array = np.asarray(reference_species)
        cost[observed_species_array[:, None] != reference_species_array[None, :]] = np.inf
    finite = cost[np.isfinite(cost)]
    penalty = (float(finite.max()) + 1) * (len(reference) + 1) if finite.size else 1e12
    rows, columns = linear_sum_assignment(np.where(np.isfinite(cost), cost, penalty))
    if not np.all(np.isfinite(cost[rows, columns])):
        raise ValueError("no complete one-to-one assignment exists")
    # Exact shortest Cartesian PBC vectors for arbitrary triclinic lattices.
    displacement = []
    for row, column in zip(rows, columns):
        vector = reference.lattice.get_distance_and_image(
            observed.frac_coords[row], reference.frac_coords[column]
        )[1]
        # get_distance_and_image returns jimage; obtain the corresponding Cartesian vector.
        image = vector
        frac_vector = reference.frac_coords[column] + image - observed.frac_coords[row]
        displacement.append(-reference.lattice.get_cartesian_coords(frac_vector))
    displacement = np.asarray(displacement)
    lengths = np.linalg.norm(displacement, axis=1)
    grouped = defaultdict(list)
    for row, length, vector in zip(rows, lengths, displacement):
        grouped[observed[row].species_string].append((float(length), vector.tolist()))
    result = {}
    for symbol, values in grouped.items():
        ds = np.asarray([item[0] for item in values]); vectors_array = np.asarray([item[1] for item in values])
        result[symbol] = {"count": len(ds), "mean_distance": float(ds.mean()), "std_distance": float(ds.std()),
                          "min_distance": float(ds.min()), "max_distance": float(ds.max()),
                          "mean_displacement": vectors_array.mean(axis=0).tolist(),
                          "all_distances": ds.tolist(), "all_displacements": vectors_array.tolist()}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("perfect")
    parser.add_argument("distorted", nargs="*")
    parser.add_argument("--pattern")
    parser.add_argument("--start", type=int); parser.add_argument("--stop", type=int)
    parser.add_argument("--step", type=int, default=1); parser.add_argument("--inclusive-stop", action="store_true")
    parser.add_argument("--allow-species-mismatch", action="store_true")
    parser.add_argument("--matching-groups", type=matching_groups,
                        help="Semicolon-separated interchangeable groups, e.g. Li,Mn,Ti;O, for a generic cation reference")
    parser.add_argument("--lattice-tolerance", type=float, default=1e-5)
    parser.add_argument("--elements", type=comma_strings,
                        help="Elements included in the batch mean (old displacement scripts used non-Li/non-O)")
    parser.add_argument("--exclude-elements", type=comma_strings, default=["Li", "O"])
    parser.add_argument("--output", type=Path, default=Path("site_distortion.json"))
    parser.add_argument("--timeseries", type=Path, help="Optional CSV with per-frame mean displacement")
    parser.add_argument("--summary-only", action="store_true"); parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.lattice_tolerance < 0: parser.error("--lattice-tolerance cannot be negative")
    try:
        paths = [Path(name) for name in args.distorted]
        if args.pattern:
            paths.extend(indexed_paths(args.pattern, args.start, args.stop, args.step,
                                       inclusive_stop=args.inclusive_stop))
    except ValueError as exc:
        parser.error(str(exc))
    if not paths: parser.error("provide distorted files or --pattern")
    targets = [args.output] + ([args.timeseries] if args.timeseries else [])

    reports = []; series = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        perfect, observed = load_structure_pair(args.perfect, path, args.lattice_tolerance)
        result = calculate(perfect, observed, not args.allow_species_mismatch,
                           args.lattice_tolerance, args.matching_groups)
        reports.append({"file": str(path), "elements": result})
        selected = args.elements or [symbol for symbol in result if symbol not in args.exclude_elements]
        values = [distance for symbol in selected for distance in result.get(symbol, {}).get("all_distances", [])]
        series.append((str(path), sum(values) / len(values) if values else 0.0))
    payload = reports
    if args.summary_only:
        payload = [{"file": row["file"], "elements": {symbol: {key: value for key, value in stats.items()
                   if not key.startswith("all_")} for symbol, stats in row["elements"].items()}} for row in reports]
    with OutputTransaction(targets, args.overwrite) as transaction:
        transaction.stage_path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if args.timeseries:
            with transaction.stage_path(args.timeseries).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle); writer.writerow(["filename", "mean_displacement_A"]); writer.writerows(series)
        transaction.commit()


if __name__ == "__main__": main()

"""Map distorted atoms one-to-one onto a compatible perfect periodic lattice."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distortion_percolation.structure_analysis import (
    comma_strings, indexed_paths, load_structure_pair, OutputTransaction, positive_int,
    species_group_labels, validate_structure_pair,
)


def matching_groups(value: str) -> list[list[str]]:
    groups = [comma_strings(group) for group in value.split(";") if group.strip()]
    if not groups:
        raise argparse.ArgumentTypeError("expected semicolon-separated species groups")
    return groups


def match_structure(perfect, distorted, species_policy: str, tolerance: float | None,
                    lattice_tolerance: float = 1e-5, assignment: str = "global",
                    species_groups=None):
    import numpy as np
    from pymatgen.core import Structure
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial import cKDTree

    validate_structure_pair(perfect, distorted, lattice_tolerance)
    if assignment == "legacy-independent":
        from itertools import product
        shifts = np.asarray(list(product((-1, 0, 1), repeat=3)))
        expanded = (np.asarray(perfect.frac_coords)[:, None, :] + shifts).reshape(-1, 3)
        distances, image_indices = cKDTree(expanded).query(np.asarray(distorted.frac_coords))
        coordinates = np.mod(expanded[image_indices], 1.0)
        mapped = Structure(perfect.lattice, [site.species for site in distorted], coordinates,
                           coords_are_cartesian=False)
        return mapped, {
            "algorithm": assignment,
            "maximum_fractional_distance": float(np.max(distances, initial=0)),
            "assignments": [{"distorted_index": index, "expanded_index": int(image),
                             "fractional_distance": float(distance)}
                            for index, (image, distance) in enumerate(zip(image_indices, distances))],
        }
    if assignment == "legacy-greedy-species":
        from collections import Counter, defaultdict
        species = [site.species_string for site in distorted]
        required = Counter(species); coordinates_by_species = defaultdict(list)
        for site in perfect:
            coordinates_by_species[site.species_string].append(site.frac_coords)
        for symbol, count in required.items():
            if len(coordinates_by_species[symbol]) < count:
                raise ValueError(f"reference contains too few {symbol} sites")
        trees = {symbol: cKDTree(coordinates) for symbol, coordinates in coordinates_by_species.items()}
        used = defaultdict(list); matched = []
        source_by_species = defaultdict(list)
        for site in distorted: source_by_species[site.species_string].append(site.frac_coords)
        for symbol in species:
            coordinate = source_by_species[symbol].pop(0); tree = trees[symbol]
            _, index = tree.query(coordinate)
            while int(index) in used[symbol]:
                _, indices = tree.query(coordinate, k=len(used[symbol]) + 1)
                index = np.atleast_1d(indices)[-1]
            used[symbol].append(int(index)); matched.append(np.mod(coordinates_by_species[symbol][int(index)], 1.0))
        mapped = Structure(perfect.lattice, [site.species for site in distorted], matched,
                           coords_are_cartesian=False)
        return mapped, {"algorithm": assignment, "matched_by_species": dict(used)}
    if assignment != "global":
        raise ValueError(f"unknown assignment algorithm {assignment}")
    # Pymatgen computes exact shortest Cartesian vectors under triclinic PBC.
    distances = perfect.lattice.get_all_distances(distorted.frac_coords, perfect.frac_coords)
    cost = np.asarray(distances, dtype=float)
    if species_groups is not None:
        observed_species = [site.species_string for site in distorted]
        reference_species = [site.species_string for site in perfect]
        observed_groups = np.asarray(species_group_labels(observed_species, species_groups))
        reference_groups = np.asarray(species_group_labels(reference_species, species_groups))
        mismatch = observed_groups[:, None] != reference_groups[None, :]
        cost[mismatch] = np.inf
    elif species_policy == "same-species":
        observed_species = np.asarray([site.species_string for site in distorted])
        reference_species = np.asarray([site.species_string for site in perfect])
        mismatch = observed_species[:, None] != reference_species[None, :]
        cost[mismatch] = np.inf
        if np.any(np.all(mismatch, axis=1)):
            missing = sorted(set(observed_species) - set(reference_species))
            raise ValueError(f"no compatible reference sites for species: {missing}")
    elif species_policy not in ("distorted", "perfect"):
        raise ValueError(f"unknown species policy {species_policy}")
    finite = cost[np.isfinite(cost)]
    penalty = (float(finite.max()) + 1.0) * (len(perfect) + 1) if finite.size else 1e12
    rows, columns = linear_sum_assignment(np.where(np.isfinite(cost), cost, penalty))
    assigned = cost[rows, columns]
    if not np.all(np.isfinite(assigned)):
        raise ValueError("a complete one-to-one species-compatible assignment does not exist")
    maximum = float(assigned.max(initial=0))
    if tolerance is not None and maximum > tolerance:
        raise ValueError(f"mapping distance {maximum:.6g} A exceeds tolerance {tolerance} A")
    # rows are distorted-site indices; columns are perfect-site indices. Do not confuse
    # expanded-image indices with site indices as the legacy KDTree scripts did.
    order = np.argsort(columns)
    observed_indices = rows[order]
    reference_indices = columns[order]
    if species_policy == "perfect":
        species = [perfect[index].species for index in reference_indices]
    else:
        species = [distorted[index].species for index in observed_indices]
    mapped = Structure(perfect.lattice, species,
                       [perfect[index].frac_coords for index in reference_indices],
                       coords_are_cartesian=False)
    report = {
        "maximum_distance_A": maximum,
        "mean_distance_A": float(assigned.mean()),
        "duplicate_reference_sites": len(reference_indices) - len(set(reference_indices)),
        "missing_reference_sites": sorted(set(range(len(perfect))) - set(reference_indices)),
        "assignments": [{"distorted_index": int(row), "perfect_index": int(column),
                         "distance_A": float(cost[row, column])} for row, column in zip(rows, columns)],
    }
    return mapped, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("perfect")
    parser.add_argument("distorted", nargs="*")
    parser.add_argument("--pattern", help="Indexed input pattern containing {i}")
    parser.add_argument("--start", type=int); parser.add_argument("--stop", type=int)
    parser.add_argument("--step", type=positive_int, default=1); parser.add_argument("--inclusive-stop", action="store_true")
    parser.add_argument("--output", default="match_{stem}.vasp", help="Output format containing {stem} in batch mode")
    parser.add_argument("--species-policy", choices=("same-species", "distorted", "perfect"), default="same-species")
    parser.add_argument("--matching-groups", type=matching_groups,
                        help="Semicolon-separated interchangeable groups, e.g. Li,Mn,Ti;O, for a generic cation reference")
    parser.add_argument("--assignment", choices=("global", "legacy-independent", "legacy-greedy-species"), default="global",
                        help="Use global Cartesian matching or an exact legacy compatibility algorithm")
    parser.add_argument("--tolerance", type=float)
    parser.add_argument("--lattice-tolerance", type=float, default=1e-5)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--compare-command", nargs="+", help="Optional external comparison command; {perfect}, {input}, and {output} are substituted")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.tolerance is not None and args.tolerance <= 0:
        parser.error("--tolerance must be positive")
    if args.lattice_tolerance < 0:
        parser.error("--lattice-tolerance cannot be negative")
    try:
        patterned = indexed_paths(args.pattern, args.start, args.stop, args.step,
                                  inclusive_stop=args.inclusive_stop) if args.pattern else []
    except ValueError as exc:
        parser.error(str(exc))
    inputs = [Path(name) for name in args.distorted] + patterned
    if not inputs:
        parser.error("provide distorted inputs or --pattern with a numeric range")
    if len(inputs) > 1 and "{stem}" not in args.output:
        parser.error("batch mode requires {stem} in --output")

    outputs = [Path(args.output.format(stem=path.stem)) for path in inputs]
    targets = outputs + ([args.report] if args.report else [])
    reports = []
    mapped_items = []
    for source, output in zip(inputs, outputs):
        perfect, distorted = load_structure_pair(args.perfect, source, args.lattice_tolerance)
        mapped, report = match_structure(perfect, distorted, args.species_policy, args.tolerance,
                                         args.lattice_tolerance, args.assignment, args.matching_groups)
        mapped_items.append((mapped, output)); reports.append({"input": str(source), "output": str(output), **report})
    with OutputTransaction(targets, args.overwrite) as transaction:
        for mapped, output in mapped_items:
            mapped.to(fmt="poscar", filename=transaction.stage_path(output))
        if args.report:
            transaction.stage_path(args.report).write_text(__import__("json").dumps(reports, indent=2), encoding="utf-8")
        transaction.commit()
    if args.compare_command:
        for source, output in zip(inputs, outputs):
            command = [part.format(perfect=args.perfect, input=str(source), output=str(output))
                       for part in args.compare_command]
            result = subprocess.run(command, check=False)
            if result.returncode:
                raise SystemExit(f"comparison command failed for {source}: exit {result.returncode}")


if __name__ == "__main__":
    main()

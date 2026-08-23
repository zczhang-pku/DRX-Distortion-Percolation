#!/usr/bin/env python
"""Analyze periodic connected components or dribble percolating sites."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distortion_percolation.percolation_core import (
    find_clusters,
    periodic_connectivity,
    periodic_span_heuristic,
    run_dribble,
    structure_from_indices,
)


def csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def triple(value: str) -> list[int]:
    result = [int(part) for part in value.split(",")]
    if len(result) != 3:
        raise argparse.ArgumentTypeError("expected x,y,z")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("structure", type=Path)
    parser.add_argument("--method", choices=("distance", "dribble"), default="distance")
    parser.add_argument("--species", type=csv, default=csv("Li"), help="included species")
    parser.add_argument("--cutoff", type=float, default=3.0)
    parser.add_argument("--periodic-span-threshold", type=float, default=0.5)
    parser.add_argument("--cation-species", type=csv, default=csv("Li,Ti,Mn"))
    parser.add_argument("--bond-rule", choices=("NearestNeighborBR", "MinCommonNeighborsBR", "MinCommonNNNeighborsBR"), default="MinCommonNeighborsBR")
    parser.add_argument("--num-neighbors", type=int, default=2)
    parser.add_argument("--supercell", type=triple, default=[1, 1, 1])
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-structure", type=Path, help="save largest/percolating component")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def ensure_new(path: Path | None, force: bool) -> None:
    if path and path.exists() and not force:
        raise SystemExit(f"refusing to overwrite {path}; pass --force")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    from pymatgen.core import Structure
    from pymatgen.io.vasp import Poscar

    structure = Structure.from_file(args.structure)
    output_json = args.output_json.resolve()
    output_structure = args.output_structure.resolve() if args.output_structure else None
    ensure_new(output_json, args.force)
    ensure_new(output_structure, args.force)

    if args.method == "distance":
        clusters = find_clusters(structure, cutoff=args.cutoff, species=args.species)
        selected = clusters[0] if clusters else []
        result = {
            "method": "distance",
            "clusters": clusters,
            "largest_size": len(selected),
            "periodic_connectivity": periodic_connectivity(structure, selected, cutoff=args.cutoff),
            "fractional_span_heuristic": periodic_span_heuristic(
                structure, selected, args.periodic_span_threshold
            ),
        }
    else:
        dribble_result = run_dribble(
            args.structure,
            cation_species=args.cation_species,
            percolating_species=args.species,
            bond_rule=args.bond_rule,
            num_neighbors=args.num_neighbors,
            supercell=args.supercell,
            verbose=args.verbose,
        )
        selected = sorted(dribble_result.percolating_sites)
        result = {
            "method": "dribble",
            "percolating_sites": selected,
            "percolating_count": len(selected),
            "percolating_clusters": [list(cluster) for cluster in dribble_result.percolating_clusters],
        }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if output_structure and selected:
        output_structure.parent.mkdir(parents=True, exist_ok=True)
        indexed_structure = structure if args.method == "distance" else dribble_result.structure
        Poscar(structure_from_indices(indexed_structure, selected)).write_file(output_structure)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

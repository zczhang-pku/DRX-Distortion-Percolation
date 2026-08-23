#!/usr/bin/env python
"""Create structures containing selected or connected sites."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distortion_percolation.percolation_core import find_clusters, parse_indices, structure_from_indices


def csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("structure", type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--indices", type=Path, help="JSON/Python literal integer list")
    source.add_argument("--largest-cluster", action="store_true")
    parser.add_argument("--cutoff", type=float, default=3.0)
    parser.add_argument("--cluster-species", type=csv, default=csv("Li"))
    parser.add_argument("--keep-non-target", action="store_true", help="retain all atoms other than --target")
    parser.add_argument("--target", default="Li")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    from pymatgen.core import Structure
    from pymatgen.io.vasp import Poscar

    output = args.output.resolve()
    if output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {output}; pass --force")
    structure = Structure.from_file(args.structure)
    if args.indices:
        indices = parse_indices(args.indices)
    else:
        clusters = find_clusters(structure, cutoff=args.cutoff, species=args.cluster_species)
        if not clusters:
            raise SystemExit("no sites matched the cluster criteria")
        indices = clusters[0]
    selected = structure_from_indices(
        structure,
        indices,
        keep_non_target=args.keep_non_target,
        target=args.target,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    Poscar(selected).write_file(output)
    print(f"wrote {len(selected)} sites to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

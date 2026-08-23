#!/usr/bin/env python
"""Run a single or batch dribble percolation count."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distortion_percolation.percolation_core import run_dribble, select_paths, structure_from_indices


def csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def triple(value: str) -> list[int]:
    result = [int(item) for item in value.split(",")]
    if len(result) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated integers")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--structure", type=Path, help="one structure file relative to --workdir")
    source.add_argument("--glob", dest="pattern", help="structure glob relative to --workdir")
    source.add_argument("--file-list", type=Path, help="one input path per line, relative to --workdir")
    source.add_argument("--template", help="input template containing {i}, relative to --workdir")
    parser.add_argument("--start", type=int)
    parser.add_argument("--stop", type=int, help="exclusive stop for --template")
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--index-multiplier", type=int, default=1)
    parser.add_argument("--index-offset", type=int, default=0)
    parser.add_argument(
        "--order",
        choices=("natural", "generated"),
        default="natural",
        help="path order; generated preserves template/file-list order",
    )
    parser.add_argument("--workdir", type=Path, default=Path.cwd())
    parser.add_argument("--cation-species", type=csv, default=csv("Li,Ti,Mn"))
    parser.add_argument("--percolating-species", type=csv, default=csv("Li"))
    parser.add_argument("--oxygen-species", type=csv, default=csv("O"))
    parser.add_argument(
        "--bond-rule",
        choices=("NearestNeighborBR", "MinCommonNeighborsBR", "MinCommonNNNeighborsBR"),
        default="MinCommonNNNeighborsBR",
    )
    parser.add_argument("--num-neighbors", type=int, default=2)
    parser.add_argument("--supercell", type=triple, default=[1, 1, 1])
    parser.add_argument("--formula-units", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True, help="new JSON or TSV summary")
    parser.add_argument(
        "--format",
        choices=("json", "tsv", "counts"),
        default="tsv",
        help="summary format; counts writes one percolating-site count per line",
    )
    parser.add_argument("--show-clusters", action="store_true")
    parser.add_argument("--save-structures-dir", type=Path, help="save sites using dribble lattice indices")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    workdir = args.workdir.resolve()
    try:
        structures = select_paths(
            workdir,
            single=args.structure,
            pattern=args.pattern,
            file_list=args.file_list,
            template=args.template,
            start=args.start,
            stop=args.stop,
            step=args.step,
            index_multiplier=args.index_multiplier,
            index_offset=args.index_offset,
            order=args.order,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc
    if not structures:
        raise SystemExit("no structures matched")
    output = args.output.resolve()
    targets = [output]
    if args.save_structures_dir:
        target_dir = args.save_structures_dir.resolve()
        targets.extend(target_dir / f"{path.stem}_percolating.vasp" for path in structures)
    if len({str(path).casefold() for path in targets}) != len(targets):
        raise SystemExit("multiple inputs resolve to the same output path")
    collisions = [str(path) for path in targets if path.exists()]
    if collisions and not args.force:
        raise SystemExit(f"refusing to overwrite existing outputs: {collisions}; pass --force")

    rows = []
    saved = []
    for structure in structures:
        result = run_dribble(
            structure,
            cation_species=args.cation_species,
            percolating_species=args.percolating_species,
            oxygen_species=args.oxygen_species,
            bond_rule=args.bond_rule,
            num_neighbors=args.num_neighbors,
            supercell=args.supercell,
            formula_units=args.formula_units,
            verbose=args.verbose,
        )
        row = {"structure": str(structure), "percolating_sites": len(result.percolating_sites)}
        if args.show_clusters:
            row["percolating_clusters"] = [list(cluster) for cluster in result.percolating_clusters]
        rows.append(row)
        if args.save_structures_dir and result.percolating_sites:
            from pymatgen.io.vasp import Poscar
            target = args.save_structures_dir.resolve() / f"{structure.stem}_percolating.vasp"
            target.parent.mkdir(parents=True, exist_ok=True)
            Poscar(structure_from_indices(result.structure, result.percolating_sites)).write_file(target)
            saved.append(target)
        print(f"{structure.name}: {row['percolating_sites']}")

    output.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "json":
        output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    elif args.format == "counts":
        output.write_text(
            "".join(f"{row['percolating_sites']}\n" for row in rows),
            encoding="utf-8",
        )
    else:
        output.write_text(
            "structure\tpercolating_sites\n"
            + "".join(f"{row['structure']}\t{row['percolating_sites']}\n" for row in rows),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

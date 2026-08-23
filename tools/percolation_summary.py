#!/usr/bin/env python
"""Filter mobile-ion channels and summarize their dribble percolation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distortion_percolation.percolation_core import (
    connected_mobile_indices,
    mapped_structure,
    percolate_filtered_structure,
    reference_centers,
    select_paths,
    structure_from_indices,
)


def csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def floats3(value: str) -> list[float]:
    result = [float(part) for part in value.split(",")]
    if len(result) != 3:
        raise argparse.ArgumentTypeError("expected x,y,z")
    return result


def ints3(value: str) -> list[int]:
    return [int(v) for v in floats3(value)]


def thresholds(value: str) -> dict[str, float]:
    try:
        data = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("thresholds must be a JSON object") from exc
    if not isinstance(data, dict):
        raise argparse.ArgumentTypeError("thresholds must be a JSON object")
    return {str(key): float(number) for key, number in data.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perfect", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--structure", type=Path)
    source.add_argument("--glob", dest="pattern", help="distorted structures relative to workdir")
    source.add_argument("--file-list", type=Path)
    source.add_argument("--template", help="input template containing {i}")
    parser.add_argument("--start", type=int)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--workdir", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offset", type=floats3, required=True, help="offset x,y,z")
    parser.add_argument("--offset-coordinates", choices=("cartesian", "fractional"), default="cartesian")
    parser.add_argument("--cation-species", type=csv, default=csv("Li,Mn,Ti,Zr,V,Mo"))
    parser.add_argument("--mobile-species", default="Li")
    parser.add_argument("--tm-thresholds", type=thresholds, default={})
    parser.add_argument("--threshold-direction", choices=("above", "below"), default="above")
    parser.add_argument("--radius", type=float, default=5.0)
    parser.add_argument(
        "--channel-mode", choices=("zero-tm", "one-tm", "zero-and-one-tm"), default="zero-and-one-tm"
    )
    parser.add_argument("--no-map", action="store_true", help="filter the distorted structure instead of mapping to perfect sites")
    parser.add_argument("--allow-duplicate-mapping", action="store_true")
    parser.add_argument("--max-mapping-distance", type=float)
    parser.add_argument("--bond-rule", choices=("NearestNeighborBR", "MinCommonNeighborsBR", "MinCommonNNNeighborsBR"), default="NearestNeighborBR")
    parser.add_argument("--num-neighbors", type=int, default=2)
    parser.add_argument("--supercell", type=ints3, default=[1, 1, 1])
    parser.add_argument("--save-filtered-dir", type=Path)
    parser.add_argument("--save-percolating-dir", type=Path)
    parser.add_argument("--force", action="store_true", help="allow replacing requested output files")
    parser.add_argument("--verbose", action="store_true")
    return parser


def guarded_write(path: Path, text: str, force: bool) -> None:
    if path.exists() and not force:
        raise SystemExit(f"refusing to overwrite {path}; pass --force")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        files = select_paths(
            args.workdir,
            single=args.structure,
            pattern=args.pattern,
            file_list=args.file_list,
            template=args.template,
            start=args.start,
            stop=args.stop,
            step=args.step,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc
    if not files:
        raise SystemExit("no distorted structures matched")
    include_zero = args.channel_mode != "one-tm"
    include_one = args.channel_mode != "zero-tm"
    if include_one and not args.tm_thresholds:
        raise SystemExit("one-TM mode requires nonempty --tm-thresholds; use --channel-mode zero-tm")
    output = args.output.resolve()
    targets = [output]
    if args.save_filtered_dir:
        targets.extend(args.save_filtered_dir.resolve() / f"{path.stem}_filtered.vasp" for path in files)
    if args.save_percolating_dir:
        targets.extend(args.save_percolating_dir.resolve() / f"{path.stem}_percolating.vasp" for path in files)
    if len({str(path).casefold() for path in targets}) != len(targets):
        raise SystemExit("multiple inputs resolve to the same output path")
    collisions = [str(path) for path in targets if path.exists()]
    if collisions and not args.force:
        raise SystemExit(f"refusing to overwrite existing outputs: {collisions}; pass --force")

    centers = reference_centers(
        args.perfect, args.offset, offset_coordinates=args.offset_coordinates
    )
    rows = []
    for path in files:
        selected, observations = connected_mobile_indices(
            path,
            centers,
            mobile_species=args.mobile_species,
            cation_species=args.cation_species,
            tm_thresholds=args.tm_thresholds,
            radius=args.radius,
            include_zero_tm=include_zero,
            include_one_tm=include_one,
            threshold_direction=args.threshold_direction,
        )
        base = (
            mapped_structure(
                args.perfect,
                path,
                strict_unique=not args.allow_duplicate_mapping,
                max_distance=args.max_mapping_distance,
            )
            if not args.no_map
            else __import__("pymatgen.core", fromlist=["Structure"]).Structure.from_file(path)
        )
        filtered = structure_from_indices(base, selected, keep_non_target=True, target=args.mobile_species)
        percolator = percolate_filtered_structure(
            filtered,
            cation_species=args.cation_species,
            percolating_species=[args.mobile_species],
            oxygen_species=["O"],
            bond_rule=args.bond_rule,
            num_neighbors=args.num_neighbors,
            supercell=args.supercell,
            verbose=args.verbose,
        )
        sites = sorted(percolator.percolating_sites)
        indexed_structure = percolator.structure
        row = {
            "structure": str(path),
            "connected_mobile_sites": len(selected),
            "percolating_sites": len(sites),
            "one_tm_observations": observations,
        }
        rows.append(row)
        if args.save_filtered_dir:
            from pymatgen.io.vasp import Poscar
            target = args.save_filtered_dir.resolve() / f"{path.stem}_filtered.vasp"
            if target.exists() and not args.force:
                raise SystemExit(f"refusing to overwrite {target}; pass --force")
            target.parent.mkdir(parents=True, exist_ok=True)
            Poscar(filtered).write_file(target)
        if args.save_percolating_dir and sites:
            from pymatgen.io.vasp import Poscar
            target = args.save_percolating_dir.resolve() / f"{path.stem}_percolating.vasp"
            if target.exists() and not args.force:
                raise SystemExit(f"refusing to overwrite {target}; pass --force")
            target.parent.mkdir(parents=True, exist_ok=True)
            Poscar(structure_from_indices(indexed_structure, sites)).write_file(target)
        print(f"{path.name}: connected={len(selected)}, percolating={len(sites)}")

    guarded_write(output, json.dumps(rows, indent=2), args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

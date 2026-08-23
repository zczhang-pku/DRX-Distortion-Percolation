#!/usr/bin/env python
"""Join VASP E0 energies to structure slots without silently compressing gaps."""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distortion_percolation.files import atomic_write_text, natural_key, preflight_outputs, resolve_in
ENERGY_PATTERN = re.compile(r"E0:\s*([-+0-9.eE]+)")


def read_energies(path: Path) -> list[float]:
    values = [float(m.group(1)) for line in path.read_text(encoding="utf-8").splitlines() if (m := ENERGY_PATTERN.search(line))]
    if not values: raise ValueError(f"No E0 energies found in {path}")
    return values


def sort_key(mode: str):
    if mode == "lexical": return lambda path: path.name
    if mode == "numeric":
        def numeric(path):
            if not path.name.isdigit(): raise ValueError(f"Numeric sorting requires numeric names: {path.name}")
            return int(path.name)
        return numeric
    return natural_key


def slots(workdir: Path, mode: str, pattern: str, structure_name: str, ordering: str = "natural") -> list[tuple[str, Path | None]]:
    candidates = sorted(workdir.glob(pattern), key=sort_key(ordering))
    if mode == "files": return [(path.name, path.resolve()) for path in candidates if path.is_file()]
    directories = [path.resolve() for path in candidates if path.is_dir()]
    return [(directory.name, directory / structure_name if (directory / structure_name).is_file() else None) for directory in directories]


def remove_spin(entry: dict) -> None:
    for site in entry["structure"]["sites"]:
        for species in site["species"]:
            species.pop("spin", None)
            if isinstance(species.get("properties"), dict): species["properties"].pop("spin", None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=Path.cwd())
    parser.add_argument("--energy-file", type=Path, default=Path("check.txt"))
    parser.add_argument("--mode", choices=("files", "directories"), default="files")
    parser.add_argument("--pattern")
    parser.add_argument("--sort", choices=("lexical", "natural", "numeric"), default="natural")
    parser.add_argument("--structure-name", choices=("POSCAR", "CONTCAR"), default="CONTCAR")
    parser.add_argument("--output", type=Path, default=Path("entries.json"))
    parser.add_argument("--auto-oxidation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(); workdir = args.workdir.resolve(); output = resolve_in(workdir, args.output)
    pattern = args.pattern or ("POSCAR_order_*.vasp" if args.mode == "files" else "[0-9]*")
    energies = read_energies(resolve_in(workdir, args.energy_file)); joined = slots(workdir, args.mode, pattern, args.structure_name, args.sort)
    if len(joined) != len(energies): raise ValueError(f"Slot count {len(joined)} differs from energy count {len(energies)}")
    missing = [name for name, path in joined if path is None]
    if missing and not args.allow_missing: raise FileNotFoundError("Missing structure in slot(s): " + ", ".join(missing))
    preflight_outputs([output], args.force)
    try:
        from pymatgen.core import Structure
        from pymatgen.entries.computed_entries import ComputedStructureEntry
        from pymatgen.transformations.standard_transformations import AutoOxiStateDecorationTransformation
    except ImportError as exc: raise SystemExit("pymatgen is required") from exc
    ox = AutoOxiStateDecorationTransformation(symm_tol=0, max_radius=4, max_permutations=100000, distance_scale_factor=1) if args.auto_oxidation else None
    entries = []
    for slot_index, ((slot_name, path), energy) in enumerate(zip(joined, energies)):
        if path is None: continue
        structure = Structure.from_file(path)
        if ox: structure = ox.apply_transformation(structure)
        entry = ComputedStructureEntry(structure, energy).as_dict(); remove_spin(entry)
        entry["data"] = {**entry.get("data", {}), "source_slot": slot_index, "source_name": slot_name}
        entries.append(entry)
    atomic_write_text(output, json.dumps(entries, indent=2) + "\n", args.force, preflighted=True)
    print(f"Wrote {len(entries)} entries from {len(joined)} aligned slots to {output}")
if __name__ == "__main__": main()

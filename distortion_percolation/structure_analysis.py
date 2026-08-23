"""Shared parsing, validation, file iteration, and local-environment helpers."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_CATIONS = ("Li", "Mn", "Ti", "Zr", "V", "Mo")
DEFAULT_TM = ("Mn", "Ti", "Zr", "V", "Mo")
TM_PROFILES = {
    "standard": DEFAULT_TM,
    "nb": (*DEFAULT_TM, "Nb"),
    "lmtvmo": ("Mn", "Ti", "V", "Mo"),
}
DEFAULT_OFFSET = (2.10 / 2, 2.81745 / 2, 0.93915 / 2)
# Minimum TM-to-three-Li-plane heights (angstrom) used to admit 1-TM
# tetrahedra into the LMTO example's Li-channel network. These inherited,
# element-specific analysis cutoffs are not fitted from LMTO.vasp; validate
# them against the convention used in a production study.
DEFAULT_THRESHOLDS = {"Mn": 2.507, "Ti": 2.524}


def comma_strings(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return result


def comma_floats(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc
    if not result:
        raise argparse.ArgumentTypeError("expected at least one number")
    return result


def vector3(value: str) -> tuple[float, float, float]:
    result = comma_floats(value)
    if len(result) != 3:
        raise argparse.ArgumentTypeError("expected exactly three comma-separated numbers")
    return result  # type: ignore[return-value]


def matrix3(value: str) -> list[list[int]]:
    try:
        rows = [[int(item) for item in row.split(",")] for row in value.split(";")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a;b;c rows with comma-separated integers") from exc
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise argparse.ArgumentTypeError("supercell must be 3x3, e.g. 2,1,0;-1,2,0;0,0,2")
    return rows


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return result


def positive_float(value: str) -> float:
    result = float(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("expected a positive number")
    return result


def threshold_map(value: str) -> dict[str, float]:
    result: dict[str, float] = {}
    try:
        for item in value.split(","):
            symbol, number = item.split("=", 1)
            symbol = symbol.strip()
            if not symbol or symbol in result:
                raise ValueError
            result[symbol] = float(number)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected unique SYMBOL=value pairs") from exc
    return result


def natural_key(value: str | Path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(value))]


def indexed_paths(pattern: str, start: int | None, stop: int | None, step: int,
                  *, inclusive_stop: bool = False) -> list[Path]:
    if step <= 0:
        raise ValueError("--step must be positive")
    if start is None and stop is None:
        return [Path(name) for name in sorted(glob.glob(pattern), key=natural_key)]
    if start is None or stop is None:
        raise ValueError("--start and --stop must be used together")
    if stop < start:
        raise ValueError("--stop must be greater than or equal to --start")
    if "{i}" not in pattern:
        raise ValueError("an indexed pattern must contain {i}")
    end = stop + 1 if inclusive_stop else stop
    return [Path(pattern.format(i=i)) for i in range(start, end, step)]


def resolve_tm_elements(explicit: list[str] | None, profile: str) -> list[str]:
    return explicit if explicit is not None else list(TM_PROFILES[profile])


def validate_structure_pair(reference, observed, lattice_tolerance: float = 1e-5,
                            require_same_count: bool = True) -> None:
    import numpy as np
    if require_same_count and len(reference) != len(observed):
        raise ValueError(f"atom-count mismatch: reference={len(reference)}, observed={len(observed)}")
    if not np.allclose(reference.lattice.matrix, observed.lattice.matrix, atol=lattice_tolerance, rtol=0):
        difference = float(np.max(np.abs(reference.lattice.matrix - observed.lattice.matrix)))
        raise ValueError(f"lattice mismatch: maximum component difference {difference:.6g} exceeds {lattice_tolerance}")


def load_structure_pair(reference_file: str | Path, observed_file: str | Path,
                        lattice_tolerance: float = 1e-5, *, require_same_count: bool = True):
    """Load a coordinate reference and observed structure and validate compatibility."""
    from pymatgen.core import Structure

    reference = Structure.from_file(reference_file)
    observed = Structure.from_file(observed_file)
    validate_structure_pair(reference, observed, lattice_tolerance, require_same_count)
    return reference, observed


def species_group_labels(symbols: Sequence[str], groups: Sequence[Sequence[str]]) -> list[int]:
    """Map species to disjoint matching groups used for sublattice-constrained assignment."""
    lookup: dict[str, int] = {}
    for group_index, group in enumerate(groups):
        for symbol in group:
            if symbol in lookup:
                raise ValueError(f"species {symbol!r} occurs in more than one matching group")
            lookup[symbol] = group_index
    missing = sorted(set(symbols).difference(lookup))
    if missing:
        raise ValueError("species missing from matching groups: " + ", ".join(missing))
    return [lookup[symbol] for symbol in symbols]


def offset_centers(structure, offset: Sequence[float], coordinate_mode: str = "cartesian"):
    import numpy as np
    offset_array = np.asarray(offset, dtype=float)
    if offset_array.shape != (3,):
        raise ValueError("offset must contain exactly three values")
    if coordinate_mode == "cartesian":
        offset_frac = structure.lattice.get_fractional_coords(offset_array)
    elif coordinate_mode == "fractional":
        offset_frac = offset_array
    else:
        raise ValueError("coordinate_mode must be cartesian or fractional")
    centers_frac = np.mod(structure.frac_coords + offset_frac, 1.0)
    return structure.lattice.get_cartesian_coords(centers_frac)


def nearest_cations(structure, center, cations: Iterable[str], radius: float, count: int):
    if radius <= 0 or count <= 0:
        raise ValueError("radius and count must be positive")
    allowed = set(cations)
    sites = [site for site in structure.get_sites_in_sphere(center, radius) if site.specie.symbol in allowed]
    sites.sort(key=lambda site: (float(site.nn_distance), int(site.index)))
    return sites[:count]


def strict_fractional_direction(delta: Sequence[float]):
    """Return an old-script direction bin; ties are deliberately unclassified."""
    absolute = [abs(float(value)) for value in delta]
    for axis in range(3):
        others = [absolute[index] for index in range(3) if index != axis]
        if absolute[axis] > others[0] and absolute[axis] > others[1]:
            sign = 1 if float(delta[axis]) > 0 else -1 if float(delta[axis]) < 0 else 0
            return (axis, sign) if sign else None
    return None


def legacy_direction_neighbors(frac_coords, lattice_matrix, center_index: int):
    """Select nearest all-atom neighbor per strict direction over the full PBC structure."""
    from math import sqrt
    coordinates = [tuple(float(value) for value in coordinate) for coordinate in frac_coords]
    lattice = [tuple(float(value) for value in row) for row in lattice_matrix]
    center = coordinates[center_index]
    selected: dict[tuple[int, int], tuple[int, float]] = {}
    for index, coordinate in enumerate(coordinates):
        if index == center_index:
            continue
        delta = tuple(value - round(value) for value in
                      (coordinate[axis] - center[axis] for axis in range(3)))
        direction = strict_fractional_direction(delta)
        if direction is None:
            continue
        cartesian = [sum(delta[row] * lattice[row][column] for row in range(3)) for column in range(3)]
        distance = sqrt(sum(value * value for value in cartesian))
        previous = selected.get(direction)
        if previous is None or distance < previous[1]:
            selected[direction] = (index, distance)
    return selected


def ensure_output_paths(paths: Iterable[Path], overwrite: bool = False) -> list[Path]:
    result = list(paths)
    duplicates = [path for path in set(result) if result.count(path) > 1]
    if duplicates:
        raise FileExistsError(f"duplicate output targets: {', '.join(map(str, duplicates))}")
    existing = [path for path in result if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("refusing to overwrite: " + ", ".join(map(str, existing)))
    for path in result:
        path.parent.mkdir(parents=True, exist_ok=True)
    return result


class OutputTransaction:
    """Stage outputs beside their targets and install them as one rollback-safe group."""

    def __init__(self, paths: Iterable[Path], overwrite: bool = False):
        self.paths = ensure_output_paths(paths, overwrite)
        self.overwrite = overwrite
        self.staged: dict[Path, Path] = {}
        self.backups: dict[Path, Path] = {}

    def stage_path(self, target: Path) -> Path:
        if target not in self.paths:
            raise ValueError(f"undeclared transaction target: {target}")
        if target in self.staged:
            return self.staged[target]
        fd, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".stage", dir=str(target.parent))
        os.close(fd)
        staged = Path(name)
        self.staged[target] = staged
        return staged

    def commit(self) -> None:
        missing = [path for path in self.paths if path not in self.staged or not self.staged[path].exists()]
        if missing:
            self.cleanup()
            raise RuntimeError("transaction outputs were not all staged: " + ", ".join(map(str, missing)))
        installed: list[Path] = []
        try:
            for target in self.paths:
                if target.exists():
                    fd, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".backup", dir=str(target.parent))
                    os.close(fd); backup = Path(name); backup.unlink(); os.replace(str(target), str(backup)); self.backups[target] = backup
                os.replace(str(self.staged[target]), str(target)); installed.append(target)
        except BaseException:
            for target in reversed(installed):
                if target.exists(): target.unlink()
            for target, backup in self.backups.items():
                if backup.exists(): os.replace(str(backup), str(target))
            self.cleanup()
            raise
        for backup in self.backups.values():
            if backup.exists(): backup.unlink()
        self.staged.clear(); self.backups.clear()

    def cleanup(self) -> None:
        for staged in self.staged.values():
            if staged.exists(): staged.unlink()
        for target, backup in self.backups.items():
            if backup.exists() and not target.exists(): os.replace(str(backup), str(target))
            elif backup.exists(): backup.unlink()
        self.staged.clear(); self.backups.clear()

    def __enter__(self): return self
    def __exit__(self, exc_type, exc, traceback):
        if exc_type is not None: self.cleanup()
        return False


def atomic_write_text(path: Path, text: str, *, overwrite: bool = False, encoding: str = "utf-8") -> None:
    with OutputTransaction([path], overwrite) as transaction:
        transaction.stage_path(path).write_text(text, encoding=encoding)
        transaction.commit()


def write_json(path: Path, value, *, overwrite: bool = False, allow_nan: bool = False) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=allow_nan), overwrite=overwrite)

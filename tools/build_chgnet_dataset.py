"""Build a validated CHGNet JSON dataset from VASP calculation directories."""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


def validate_dataset(dataset: dict[str, Any]) -> None:
    """Validate the aligned CHGNet dataset columns before writing."""
    required = {"structure", "uncorrected_total_energy", "energy_per_atom"}
    missing = required.difference(dataset)
    if missing:
        raise ValueError(f"dataset missing keys: {sorted(missing)}")
    size = len(dataset["structure"])
    for key, values in dataset.items():
        if key == "provenance" or values is None:
            continue
        if not isinstance(values, list) or len(values) != size:
            raise ValueError(f"dataset field {key!r} is not aligned with structure")
    for index, structure in enumerate(dataset["structure"]):
        if not isinstance(structure, dict):
            raise ValueError(f"structure {index} is not a mapping")


def atomic_write_json(path: Path, value: dict[str, Any], *, overwrite: bool) -> None:
    """Write JSON atomically without silently replacing an existing dataset."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _outcar_magmom_steps(path: Path, axis: str) -> list[list[float]]:
    """Parse each complete OUTCAR magnetization table for one axis."""
    heading = f"magnetization ({axis})"
    results: list[list[float]] = []
    reading = False
    current: list[float] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if heading in line:
            reading, current = True, []
            continue
        if not reading:
            continue
        match = re.match(r"^(\d+)\s+(?:[-+\d.Ee]+\s+){3}([-+\d.Ee]+)$", line)
        if match:
            current.append(float(match.group(2)))
        elif line.startswith("tot"):
            if current:
                results.append(current)
            reading = False
    return results


def _parse_one(directory: Path, check_convergence: bool, magmom_source: str,
               step_policy: str) -> dict[str, Any]:
    try:
        from pymatgen.io.vasp.outputs import Vasprun
    except ImportError as exc:
        raise RuntimeError("build_chgnet_dataset requires pymatgen") from exc
    run = Vasprun(str(directory / "vasprun.xml"), parse_dos=False, parse_eigen=False,
                  parse_projected_eigen=False, parse_potcar_file=False,
                  exception_on_bad_xml=False)
    outcar = directory / "OUTCAR"
    magnetic: list[list[float]] = []
    if magmom_source != "none":
        if not outcar.is_file():
            raise ValueError("magmom requested but OUTCAR is missing")
        magnetic = _outcar_magmom_steps(outcar, magmom_source[-1])
        if not magnetic:
            raise ValueError(f"no complete {magmom_source} tables found")
    ionic = run.ionic_steps
    if magmom_source != "none" and len(magnetic) not in {len(ionic), len(ionic) + 1}:
        raise ValueError(f"OUTCAR has {len(magnetic)} magmom steps, vasprun has {len(ionic)} ionic steps")
    if len(magnetic) == len(ionic) + 1:
        magnetic = magnetic[:len(ionic)]
    selected: list[int] = []
    nelm = run.parameters.get("NELM", 999999)
    for index, step in enumerate(ionic):
        if check_convergence and len(step.get("electronic_steps", [])) >= nelm:
            continue
        selected.append(index)
    if step_policy == "last" and selected:
        selected = selected[-1:]
    if not selected:
        raise RuntimeError("no selected ionic steps")
    structures = [ionic[i]["structure"] for i in selected]
    atom_counts = [len(value) for value in structures]
    if len(set(atom_counts)) != 1:
        raise ValueError("atom count changed between ionic steps")
    def serial(value: Any) -> Any:
        return value.tolist() if hasattr(value, "tolist") else value
    result: dict[str, Any] = {
        "structure": [value.as_dict() for value in structures],
        "uncorrected_total_energy": [float(ionic[i]["e_wo_entrp"]) for i in selected],
        "energy_per_atom": [float(ionic[i]["e_wo_entrp"]) / atom_counts[n] for n, i in enumerate(selected)],
        "force": [serial(ionic[i].get("forces")) for i in selected],
        "stress": [serial(ionic[i].get("stress")) for i in selected],
        "magmom": [magnetic[i] for i in selected] if magnetic else None,
    }
    for key in ("force", "stress"):
        values = result[key]
        if all(value is None for value in values):
            result[key] = None
        elif any(value is None for value in values):
            raise ValueError(f"partially missing {key} labels")
    if result["magmom"] is not None:
        for number, values in enumerate(result["magmom"]):
            if len(values) != atom_counts[number]:
                raise ValueError(f"magmom atom mismatch at selected step {number}")
    return result


def build_dataset(base_dir: Path, output: Path, *, check_convergence: bool,
                  required_files: set[str], magmom_source: str, step_policy: str,
                  strict: bool, overwrite: bool) -> tuple[int, int]:
    aggregate: dict[str, Any] = {key: [] for key in
        ("structure", "uncorrected_total_energy", "energy_per_atom", "force", "magmom", "stress")}
    parsed, skipped, missing_count = 0, 0, 0
    directories: list[dict[str, Any]] = []
    for root, _, files in os.walk(base_dir):
        found = set(files)
        missing = sorted(required_files - found)
        record: dict[str, Any] = {"directory": str(Path(root).resolve()),
            "discovered_files": sorted(found & (required_files | {"vasprun.xml", "OUTCAR", "OSZICAR"})),
            "missing_required": missing}
        if missing:
            missing_count += 1
            record.update(status="missing", reason=f"missing: {', '.join(missing)}")
            directories.append(record)
            continue
        try:
            value = _parse_one(Path(root), check_convergence, magmom_source, step_policy)
            validate_dataset(value)
            present = {key: value[key] is not None for key in ("force", "magmom", "stress")}
            existing = len(aggregate["structure"]) > 0
            for key in present:
                prior_present = bool(aggregate[key])
                if existing and prior_present != present[key]:
                    raise ValueError(f"inconsistent {key} availability across directories")
            for key in aggregate:
                if value[key] is not None:
                    aggregate[key].extend(value[key])
            record.update(status="parsed", reason=None, samples=len(value["structure"]))
            directories.append(record)
            parsed += 1
        except (OSError, RuntimeError, ValueError) as exc:
            skipped += 1
            record.update(status="skipped", reason=str(exc))
            directories.append(record)
            if strict:
                raise RuntimeError(f"failed parsing {root}: {exc}") from exc
            print(f"skip {root}: {exc}")
    for key in ("force", "magmom", "stress"):
        if not aggregate[key]:
            aggregate[key] = None
    validate_dataset(aggregate)
    document = dict(aggregate)
    document["provenance"] = {
        "base_dir": str(base_dir.resolve()), "directories": directories,
        "required_files": sorted(required_files), "magmom_source": magmom_source,
        "step_policy": step_policy, "check_electronic_convergence": check_convergence,
        "parsed_directories": parsed, "skipped_directories": skipped,
        "missing_directories": missing_count,
        "unparsed_directories": skipped + missing_count,
        "discovered_directories": len(directories),
    }
    atomic_write_json(output, document, overwrite=overwrite)
    return parsed, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_dir", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("dataset.json"))
    parser.add_argument("--required-file", action="append", default=None,
                        help="required filename; repeat (default: vasprun.xml, OSZICAR, OUTCAR)")
    parser.add_argument("--magmom-source", choices=("none", "outcar-x", "outcar-y", "outcar-z"), default="outcar-x")
    parser.add_argument("--steps", choices=("all", "last"), default="all")
    parser.add_argument("--allow-unconverged", action="store_true")
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if not args.base_dir.is_dir():
        parser.error("base_dir must be a directory")
    defaults = ["vasprun.xml", "OSZICAR"] + ([] if args.magmom_source == "none" else ["OUTCAR"])
    required = set(args.required_file or defaults)
    parsed, skipped = build_dataset(
        args.base_dir, args.output, check_convergence=not args.allow_unconverged,
        required_files=required, magmom_source=args.magmom_source,
        step_policy=args.steps, strict=args.strict, overwrite=args.overwrite)
    print(f"wrote {args.output}: parsed={parsed}, skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Analyze cation-anion bond lengths with robust or legacy neighbor selection."""
import argparse
import json
from collections import defaultdict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distortion_percolation.structure_analysis import (
    OutputTransaction, comma_strings, legacy_direction_neighbors, positive_float, positive_int,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("structure")
    parser.add_argument("--anion", default="O")
    parser.add_argument("--cations", type=comma_strings)
    parser.add_argument("--algorithm", choices=("nearest", "legacy-directions", "cation-directions"), default="nearest")
    parser.add_argument("--coordination", type=positive_int, default=6)
    parser.add_argument("--cutoff", type=positive_float, default=4.0)
    parser.add_argument("--output", type=Path, default=Path("bond_distances.json"))
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    targets = [args.output] + ([args.plot] if args.plot else [])

    import numpy as np
    from pymatgen.core import Structure

    structure = Structure.from_file(args.structure)
    allowed = set(args.cations or [element.symbol for element in structure.composition.elements if element.symbol != args.anion])
    bonds = defaultdict(list)
    for anion_index, anion in enumerate(structure):
        if anion.specie.symbol != args.anion:
            continue
        if args.algorithm == "legacy-directions":
            # Exactly preserve the old algorithm: full structure, strict dominant
            # fractional component (ties omitted), PBC shortest image, then cation filter.
            bins = legacy_direction_neighbors(structure.frac_coords, structure.lattice.matrix, anion_index)
            for index, distance in bins.values():
                symbol = structure[index].specie.symbol
                if symbol in allowed:
                    bonds[symbol].append(distance)
            continue
        all_neighbors = list(structure.get_neighbors(anion, args.cutoff))
        cation_neighbors = [neighbor for neighbor in all_neighbors if neighbor.specie.symbol in allowed]
        if args.algorithm == "nearest":
            selected = sorted(cation_neighbors, key=lambda site: site.nn_distance)[:args.coordination]
        else:
            # Improved cutoff-limited variant: cations first, then argmax direction bins.
            bins = {}
            for neighbor in cation_neighbors:
                delta = neighbor.frac_coords - anion.frac_coords
                delta -= np.round(delta)
                axis = int(np.argmax(np.abs(delta)))
                key = (axis, int(np.sign(delta[axis])))
                if key not in bins or neighbor.nn_distance < bins[key].nn_distance:
                    bins[key] = neighbor
            selected = list(bins.values())
        for neighbor in selected:
            bonds[neighbor.specie.symbol].append(float(neighbor.nn_distance))
    report = {symbol: {"count": len(values), "mean": float(np.mean(values)), "variance": float(np.var(values)),
                       "distances": values} for symbol, values in sorted(bonds.items()) if values}
    with OutputTransaction(targets, args.overwrite) as transaction:
        transaction.stage_path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        if args.plot:
            import matplotlib.pyplot as plt
            for symbol, values in bonds.items():
                plt.hist(values, bins=30, alpha=.6, label=symbol)
            plt.xlabel("Bond distance (A)"); plt.ylabel("Frequency"); plt.legend(); plt.tight_layout()
            plt.savefig(transaction.stage_path(args.plot), dpi=300, format=args.plot.suffix.lstrip(".") or "png"); plt.close()
        transaction.commit()
    for symbol, row in report.items():
        print(f"{symbol}-{args.anion}: mean={row['mean']:.5f} variance={row['variance']:.5f} count={row['count']}")


if __name__ == "__main__":
    main()

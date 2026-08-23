"""Calculate Warren-Cowley short-range-order parameters with OVITO."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distortion_percolation.structure_analysis import atomic_write_text, comma_strings


def calculate_native(structure, species=None, neighbors: int = 12):
    """Calculate directed Warren-Cowley parameters from nearest PBC neighbors."""
    import numpy as np

    allowed = set(species or [element.symbol for element in structure.composition.elements])
    indices = [index for index, site in enumerate(structure) if site.specie.symbol in allowed]
    if neighbors <= 0:
        raise ValueError("neighbors must be positive")
    if len(indices) <= neighbors:
        raise ValueError(f"need more than {neighbors} selected sites")
    labels = sorted({structure[index].specie.symbol for index in indices})
    concentrations = {
        symbol: sum(structure[index].specie.symbol == symbol for index in indices) / len(indices)
        for symbol in labels
    }
    distances = np.asarray(structure.lattice.get_all_distances(
        structure.frac_coords[indices], structure.frac_coords[indices]
    ))
    np.fill_diagonal(distances, np.inf)
    nearest = np.argpartition(distances, neighbors - 1, axis=1)[:, :neighbors]
    counts = {center: {neighbor: 0 for neighbor in labels} for center in labels}
    totals = {center: 0 for center in labels}
    for row, center_index in enumerate(indices):
        center = structure[center_index].specie.symbol
        for column in nearest[row]:
            neighbor = structure[indices[int(column)]].specie.symbol
            counts[center][neighbor] += 1
            totals[center] += 1
    parameters = {}
    conditional = {}
    for center in labels:
        conditional[center] = {}
        parameters[center] = {}
        for neighbor in labels:
            probability = counts[center][neighbor] / totals[center] if totals[center] else 0.0
            conditional[center][neighbor] = probability
            parameters[center][neighbor] = 1.0 - probability / concentrations[neighbor]
    return {
        "backend": "native-nearest-neighbor",
        "selected_species": labels,
        "selected_site_count": len(indices),
        "neighbors_per_site": neighbors,
        "concentrations": concentrations,
        "conditional_probabilities": conditional,
        "parameters": parameters,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Warren-Cowley parameters describe chemical short-range order in the analyzed structure; a perfect coordinate reference is not required.",
    )
    parser.add_argument("structure")
    parser.add_argument("--backend", choices=("native", "ovito"), default="native")
    parser.add_argument("--species", type=comma_strings,
                        help="Species included by the native backend, e.g. Li,Ti,Mn")
    parser.add_argument("--neighbors", type=int, default=12,
                        help="Nearest selected sites per center for the native backend")
    parser.add_argument("--neighbor-shells", type=comma_strings, default=["0", "12"],
                        help="Cumulative shell boundaries used only by the OVITO backend")
    parser.add_argument("--only-selected", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("warren_cowley.json"))
    parser.add_argument("--include-particles", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        shells = [int(value) for value in args.neighbor_shells]
    except ValueError:
        parser.error("--neighbor-shells must contain integers")
    if len(shells) < 2 or any(value < 0 for value in shells) or shells != sorted(shells):
        parser.error("--neighbor-shells must be nonnegative and ascending")

    if args.backend == "native":
        if args.neighbors <= 0:
            parser.error("--neighbors must be positive")
        from pymatgen.core import Structure
        try:
            report = calculate_native(Structure.from_file(args.structure), args.species, args.neighbors)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        try:
            from ovito.io import import_file
            import WarrenCowleyParameters as wc
        except ImportError as exc:
            parser.error(f"OVITO WarrenCowleyParameters plugin is required: {exc}")
        pipeline = import_file(args.structure)
        modifier = wc.WarrenCowleyParameters(nneigh=shells,
                                             only_selected=args.only_selected)
        pipeline.modifiers.append(modifier)
        try:
            data = pipeline.compute()
        except Exception as exc:
            parser.error("WarrenCowleyParameters failed; verify an OVITO-compatible plugin version and nneigh configuration: " + str(exc))
        required_attributes = ["Warren-Cowley parameters", "Warren-Cowley parameters by particle name"]
        missing_attributes = [name for name in required_attributes if name not in data.attributes]
        if missing_attributes:
            parser.error("incompatible WarrenCowleyParameters plugin: missing output attributes " + ", ".join(missing_attributes))
        report = {
            "backend": "ovito-plugin",
            "parameters": data.attributes[required_attributes[0]].tolist(),
            "by_particle_name": data.attributes[required_attributes[1]],
        }
        if args.include_particles:
            property_name = "Warren-Cowley parameter (shell=1)"
            if data.particles is None or property_name not in data.particles:
                parser.error("incompatible WarrenCowleyParameters plugin/configuration: missing particle property " + property_name)
            report["per_particle_first_shell"] = data.particles[property_name][...].tolist()
    atomic_write_text(args.output, json.dumps(report, indent=2, default=str), overwrite=args.overwrite)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

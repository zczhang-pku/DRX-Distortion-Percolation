"""Generate symmetry-distinct ordered structures from a disordered input."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distortion_percolation.structure_analysis import OutputTransaction, matrix3, positive_int


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("--supercell", type=matrix3, default=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        help="Any integer 3x3 matrix, rows separated by ';', e.g. 2,1,0;-1,2,0;0,0,2")
    parser.add_argument("--algorithm", choices=("energy", "enumerate"), default="energy",
                        help="Compatibility alias for --transform")
    parser.add_argument("--transform", choices=("order-disordered", "enumerate"),
                        help="Select the pymatgen transformation explicitly")
    parser.add_argument("--ranked", type=positive_int, default=2000)
    parser.add_argument("--max-cell-size", type=positive_int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--prefix", default="POSCAR_order_")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.transform is not None:
        args.algorithm = "enumerate" if args.transform == "enumerate" else "energy"

    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure
    from pymatgen.transformations.advanced_transformations import EnumerateStructureTransformation
    from pymatgen.transformations.standard_transformations import OrderDisorderedStructureTransformation

    structure = Structure.from_file(args.input)
    structure.make_supercell(args.supercell)
    transform = (OrderDisorderedStructureTransformation() if args.algorithm == "energy"
                 else EnumerateStructureTransformation(max_cell_size=args.max_cell_size))
    ranked = transform.apply_transformation(structure, return_ranked_list=args.ranked)
    candidates = [item["structure"] if isinstance(item, dict) else item for item in ranked]
    groups = StructureMatcher().group_structures(candidates)
    outputs = [args.output_dir / f"{args.prefix}{index}.vasp" for index in range(len(groups))]
    with OutputTransaction(outputs, args.overwrite) as transaction:
        for output, group in zip(outputs, groups):
            group[0].to(fmt="poscar", filename=transaction.stage_path(output))
        transaction.commit()
    print(f"wrote {len(groups)} unique structures")


if __name__ == "__main__":
    main()

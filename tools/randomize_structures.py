"""Create reproducible unique random decorations of selected sites."""
import argparse
import math
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distortion_percolation.structure_analysis import OutputTransaction, comma_strings, positive_int


def counts(value: str) -> dict[str, int]:
    try: result = {symbol.strip(): int(number) for symbol, number in (item.split("=", 1) for item in value.split(","))}
    except ValueError as exc: raise argparse.ArgumentTypeError("expected counts such as Li=12,Ti=4,Mn=4") from exc
    if not result or any(not symbol or count < 0 for symbol, count in result.items()):
        raise argparse.ArgumentTypeError("species must be nonempty and counts nonnegative")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input"); parser.add_argument("--site-species", type=comma_strings, required=True)
    parser.add_argument("--counts", type=counts, required=True); parser.add_argument("--number", type=positive_int, default=150)
    parser.add_argument("--seed", type=int, help="Optional seed; omit to preserve nondeterministic legacy behavior")
    parser.add_argument("--output-dir", type=Path, default=Path(".")); parser.add_argument("--prefix", default="POSCAR_")
    parser.add_argument("--overwrite", action="store_true"); args = parser.parse_args()
    from pymatgen.core import Structure
    source = Structure.from_file(args.input); pool = [index for index, site in enumerate(source) if site.specie.symbol in set(args.site_species)]
    occupation = [symbol for symbol, count in args.counts.items() for _ in range(count)]
    if len(occupation) != len(pool): parser.error(f"occupation count {len(occupation)} does not match candidate-site count {len(pool)}")
    possible = math.factorial(len(pool))
    for count in Counter(occupation).values(): possible //= math.factorial(count)
    if args.number > possible: parser.error(f"requested {args.number} unique structures but only {possible} occupations exist")
    outputs = [args.output_dir / f"{args.prefix}{run}.vasp" for run in range(args.number)]
    rng = random.Random(args.seed) if args.seed is not None else random.Random(); seen = set(); arrangements = []
    while len(arrangements) < args.number:
        shuffled = occupation.copy(); rng.shuffle(shuffled); key = tuple(shuffled)
        if key not in seen: seen.add(key); arrangements.append(shuffled)
    with OutputTransaction(outputs, args.overwrite) as transaction:
        for output, shuffled in zip(outputs, arrangements):
            result = source.copy()
            for index, symbol in zip(pool, shuffled): result.replace(index, symbol)
            result.to(fmt="poscar", filename=transaction.stage_path(output))
        transaction.commit()
    print(f"wrote {args.number} unique structures (seed={args.seed})")


if __name__ == "__main__": main()

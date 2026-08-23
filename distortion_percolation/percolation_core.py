"""Shared percolation and local-environment routines.

Heavy optional dependencies are imported inside functions so every command can show
its help even when the scientific environment is not active.
"""
from __future__ import annotations

import ast
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def parse_indices(path: str | Path) -> list[int]:
    """Read a JSON/Python list of site indices without executing the file."""
    text = Path(path).read_text(encoding="utf-8").strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = ast.literal_eval(text)
    if not isinstance(value, (list, tuple, set)) or not all(isinstance(i, int) for i in value):
        raise ValueError(f"{path} must contain a list of integer site indices")
    return sorted(set(value))


def build_dribble_input(
    structure_file: str | Path,
    cation_species: Sequence[str],
    percolating_species: Sequence[str] = ("Li",),
    oxygen_species: Sequence[str] = ("O",),
    bond_rule: str = "MinCommonNNNeighborsBR",
    num_neighbors: int | None = 2,
    formula_units: int = 1,
) -> str:
    rule: list[object] = [bond_rule]
    if num_neighbors is not None and bond_rule != "NearestNeighborBR":
        rule.append({"num_neighbors": num_neighbors})
    data = {
        "structure": str(Path(structure_file).resolve()),
        "formula_units": formula_units,
        "sublattices": {
            "cation": {
                "description": "cation sites",
                "sites": {"species": list(cation_species)},
            },
            "oxygen": {
                "description": "ignored anion sites",
                "sites": {"species": list(oxygen_species)},
                "ignore": True,
            },
        },
        "bonds": [{"sublattices": ["cation", "cation"], "bond_rules": [rule]}],
        "percolating_species": list(percolating_species),
    }
    return json.dumps(data)


@dataclass(frozen=True)
class DribbleResult:
    """Keep the lattice beside its percolator so site indices remain unambiguous."""

    percolator: object
    lattice: object

    @property
    def structure(self):
        return self.lattice.structure

    @property
    def percolating_sites(self):
        return self.percolator.percolating_sites

    @property
    def percolating_clusters(self):
        return self.percolator.percolating_clusters


def run_dribble(
    structure_file: str | Path,
    *,
    cation_species: Sequence[str],
    percolating_species: Sequence[str] = ("Li",),
    oxygen_species: Sequence[str] = ("O",),
    bond_rule: str = "MinCommonNNNeighborsBR",
    num_neighbors: int | None = 2,
    supercell: Sequence[int] = (1, 1, 1),
    formula_units: int = 1,
    verbose: bool = False,
) -> DribbleResult:
    """Return the dribble percolator and the exact structure indexed by it."""
    from dribble import Input, Lattice, Percolator

    payload = build_dribble_input(
        structure_file,
        cation_species,
        percolating_species,
        oxygen_species,
        bond_rule,
        num_neighbors,
        formula_units,
    )
    inp = Input.from_string(payload)
    lattice = Lattice.from_input_object(inp, supercell=list(supercell))
    percolator = Percolator.from_input_object(inp, lattice, verbose=verbose)
    return DribbleResult(percolator=percolator, lattice=lattice)


def structure_from_indices(structure, indices: Iterable[int], *, keep_non_target: bool = False, target="Li"):
    """Create a structure containing selected sites, optionally retaining non-target atoms."""
    from pymatgen.core import Structure

    selected = set(indices)
    invalid = selected.difference(range(len(structure)))
    if invalid:
        raise IndexError(f"site indices outside structure: {sorted(invalid)}")
    if keep_non_target:
        sites = [site for i, site in enumerate(structure) if site.species_string != target or i in selected]
    else:
        sites = [structure[i] for i in sorted(selected)]
    if not sites:
        raise ValueError("selection produced an empty structure")
    return Structure.from_sites(sites)


def mapped_structure(
    perfect_file: str | Path,
    distorted_file: str | Path,
    *,
    strict_unique: bool = True,
    max_distance: float | None = None,
):
    """Map distorted species to nearest periodic sites, rejecting ambiguous reuse."""
    import numpy as np
    from itertools import product
    from pymatgen.core import Structure
    from scipy.spatial import cKDTree

    perfect = Structure.from_file(perfect_file)
    distorted = Structure.from_file(distorted_file)
    shifts = np.array(list(product((-1, 0, 1), repeat=3)))
    expanded = (np.asarray(perfect.frac_coords)[:, None, :] + shifts).reshape(-1, 3)
    distances, indices = cKDTree(expanded).query(np.asarray(distorted.frac_coords))
    base_indices = indices // len(shifts)
    if strict_unique and len(set(base_indices.tolist())) != len(base_indices):
        duplicates = sorted({int(i) for i in base_indices if list(base_indices).count(i) > 1})
        raise ValueError(f"multiple distorted atoms map to perfect sites {duplicates}")
    if max_distance is not None:
        cart_distances = np.linalg.norm(
            np.dot(np.asarray(distorted.frac_coords) - expanded[indices], perfect.lattice.matrix), axis=1
        )
        bad = np.flatnonzero(cart_distances > max_distance)
        if len(bad):
            raise ValueError(f"mapped distances exceed {max_distance} A at distorted indices {bad.tolist()}")
    return Structure(
        perfect.lattice,
        [site.species_string for site in distorted],
        np.mod(expanded[indices], 1.0),
        coords_are_cartesian=False,
    )


def reference_centers(
    perfect_file: str | Path,
    offset: Sequence[float],
    *,
    offset_coordinates: str = "cartesian",
):
    """Generate wrapped Cartesian centers from Cartesian or fractional offsets."""
    import numpy as np
    from pymatgen.core import Structure

    perfect = Structure.from_file(perfect_file)
    offset_array = np.asarray(offset, dtype=float)
    if offset_coordinates == "cartesian":
        offset_fractional = perfect.lattice.get_fractional_coords(offset_array)
    elif offset_coordinates == "fractional":
        offset_fractional = offset_array
    else:
        raise ValueError("offset_coordinates must be 'cartesian' or 'fractional'")
    centers_fractional = np.mod(np.asarray(perfect.frac_coords) + offset_fractional, 1.0)
    return perfect.lattice.get_cartesian_coords(centers_fractional)


def connected_mobile_indices(
    structure_file: str | Path,
    centers,
    *,
    mobile_species: str = "Li",
    cation_species: Sequence[str],
    tm_thresholds: Mapping[str, float],
    radius: float = 5.0,
    include_zero_tm: bool = True,
    include_one_tm: bool = True,
    threshold_direction: str = "above",
):
    """Select mobile ions in 0-TM and threshold-qualified 1-TM tetrahedra."""
    import numpy as np
    from pymatgen.core import Structure

    structure = Structure.from_file(structure_file)
    cations = set(cation_species)
    selected: set[int] = set()
    observations: list[dict[str, float | str]] = []
    for center in centers:
        neighbors = [site for site in structure.get_sites_in_sphere(center, radius) if site.specie.symbol in cations]
        neighbors.sort(key=lambda site: np.linalg.norm(center - site.coords))
        nearest = neighbors[:4]
        mobile = [site for site in nearest if site.specie.symbol == mobile_species]
        tm = [site for site in nearest if site.specie.symbol != mobile_species]
        if include_zero_tm and len(mobile) == 4:
            selected.update(site.index for site in mobile)
        elif include_one_tm and len(mobile) == 3 and len(tm) == 1:
            coords = np.asarray([site.coords for site in mobile])
            normal = np.cross(coords[1] - coords[0], coords[2] - coords[0])
            norm = np.linalg.norm(normal)
            if norm == 0:
                continue
            distance = abs(float(np.dot(tm[0].coords - coords[0], normal / norm)))
            symbol = tm[0].specie.symbol
            observations.append({"species": symbol, "height": distance})
            threshold = tm_thresholds.get(symbol)
            qualifies = threshold is not None and (
                distance > threshold if threshold_direction == "above" else distance < threshold
            )
            if qualifies:
                selected.update(site.index for site in mobile)
    return selected, observations


def percolate_filtered_structure(
    structure,
    *,
    cation_species: Sequence[str],
    percolating_species: Sequence[str],
    oxygen_species: Sequence[str],
    bond_rule: str,
    num_neighbors: int | None,
    supercell: Sequence[int],
    verbose: bool = False,
):
    """Run dribble on an in-memory structure using an isolated temporary POSCAR."""
    from pymatgen.io.vasp import Poscar

    with tempfile.TemporaryDirectory(prefix="percolation-") as temp_dir:
        temp_file = Path(temp_dir) / "structure.vasp"
        Poscar(structure).write_file(temp_file)
        percolator = run_dribble(
            temp_file,
            cation_species=cation_species,
            percolating_species=percolating_species,
            oxygen_species=oxygen_species,
            bond_rule=bond_rule,
            num_neighbors=num_neighbors,
            supercell=supercell,
            verbose=verbose,
        )
    return percolator


def find_clusters(structure, *, cutoff: float = 3.0, species: Sequence[str] | None = None):
    """Find connected components using pymatgen's periodic distances."""
    allowed = set(species or [])
    nodes = [i for i, site in enumerate(structure) if not allowed or site.specie.symbol in allowed]
    remaining = set(nodes)
    clusters: list[list[int]] = []
    while remaining:
        stack = [remaining.pop()]
        cluster: list[int] = []
        while stack:
            current = stack.pop()
            cluster.append(current)
            neighbors = [i for i in tuple(remaining) if structure.get_distance(current, i) < cutoff]
            remaining.difference_update(neighbors)
            stack.extend(neighbors)
        clusters.append(sorted(cluster))
    return sorted(clusters, key=len, reverse=True)


def periodic_span_heuristic(structure, indices: Sequence[int], threshold: float = 0.5) -> list[bool]:
    """Legacy fractional-span heuristic; it is not a proof of percolation."""
    import numpy as np

    if not indices:
        return [False, False, False]
    coords = np.asarray(structure.frac_coords)[list(indices)]
    return [bool(np.ptp(coords[:, axis]) > threshold) for axis in range(3)]


def periodic_connectivity(structure, indices: Sequence[int], *, cutoff: float) -> list[bool]:
    """Detect non-contractible graph cycles and return their lattice directions.

    Each periodic neighbor edge carries a jimage translation. During graph traversal,
    inconsistent image labels identify a winding cycle; its nonzero components are
    strict evidence of periodic connectivity in those lattice directions.
    """
    import numpy as np

    selected = set(indices)
    if not selected:
        return [False, False, False]
    adjacency: dict[int, list[tuple[int, tuple[int, int, int]]]] = {i: [] for i in selected}
    for i in selected:
        for neighbor in structure.get_neighbors(structure[i], cutoff):
            j = int(neighbor.index)
            if j not in selected or j == i:
                continue
            image = tuple(int(value) for value in neighbor.image)
            adjacency[i].append((j, image))
    image_labels: dict[int, np.ndarray] = {}
    winding = np.zeros(3, dtype=bool)
    for root in selected:
        if root in image_labels:
            continue
        image_labels[root] = np.zeros(3, dtype=int)
        stack = [root]
        while stack:
            current = stack.pop()
            for neighbor, edge_image in adjacency[current]:
                candidate = image_labels[current] + np.asarray(edge_image, dtype=int)
                if neighbor not in image_labels:
                    image_labels[neighbor] = candidate
                    stack.append(neighbor)
                else:
                    winding |= candidate != image_labels[neighbor]
    return winding.tolist()


def natural_key(path: str | Path):
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", str(path))]


def select_paths(
    workdir: str | Path,
    *,
    single: str | Path | None = None,
    pattern: str | None = None,
    file_list: str | Path | None = None,
    template: str | None = None,
    start: int | None = None,
    stop: int | None = None,
    step: int = 1,
    index_multiplier: int = 1,
    index_offset: int = 0,
    order: str = "natural",
) -> list[Path]:
    """Resolve batch sources, preserving generated order or sorting naturally."""
    root = Path(workdir).resolve()
    modes = sum(value is not None for value in (single, pattern, file_list, template))
    if modes != 1:
        raise ValueError("choose exactly one of single, pattern, file_list, or template")
    if single is not None:
        path = Path(single)
        paths = [path if path.is_absolute() else root / path]
    elif pattern is not None:
        paths = list(root.glob(pattern))
    elif file_list is not None:
        list_path = Path(file_list)
        if not list_path.is_absolute():
            list_path = root / list_path
        paths = []
        for line in list_path.read_text(encoding="utf-8").splitlines():
            name = line.strip()
            if name and not name.startswith("#"):
                path = Path(name)
                paths.append(path if path.is_absolute() else root / path)
    else:
        if start is None or stop is None:
            raise ValueError("template mode requires start and stop")
        if step == 0:
            raise ValueError("step cannot be zero")
        if index_multiplier == 0:
            raise ValueError("index_multiplier cannot be zero")
        paths = [
            root / template.format(i=index_offset + index_multiplier * i)
            for i in range(start, stop, step)
        ]
    if order == "generated":
        paths = [path.resolve() for path in paths]
    elif order == "natural":
        paths = sorted((path.resolve() for path in paths), key=natural_key)
    else:
        raise ValueError("order must be 'natural' or 'generated'")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing input files: {missing}")
    return paths

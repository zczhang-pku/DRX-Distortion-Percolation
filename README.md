# DRX–Distortion–Percolation

A compact, parameterized analysis toolkit for coupling lattice distortion, local cation environments, short-range order, and Li-ion percolation in cation-disordered rocksalt (DRX) oxides.

The code is material-agnostic: species, thresholds, paths, frame ranges, and other system-specific values are supplied as arguments instead of being encoded in separate scripts.

## Layout

- `distortion_percolation/`: reusable scientific core library.
- `tools/`: command-line analysis, dataset, and training tools.
- `LMTO_analysis_example.ipynb`: worked reference-based analysis of the supplied structures.
- `requirements.txt`: core dependencies required to execute the example notebook.
- `CITATION.cff`: GitHub-compatible citation metadata for this repository and the associated work.
- `perfect.vasp`: ideal coordinate lattice used as the reference structure.
- `LMTO.vasp`: chemically decorated and distorted example structure.

## Requirements

Install only the dependencies needed by your workflow:

- Python 3.9+
- `pymatgen`, `numpy`, and `scipy` for structure analysis
- `matplotlib` and Jupyter for the example notebook
- `dribble` only for optional command-line Dribble percolation; the example notebook does not require it
- OVITO and a compatible `WarrenCowleyParameters` plugin only for the optional OVITO WCP backend; the native WCP backend uses pymatgen
- `chgnet` and `torch` for CHGNet training

Optional scientific dependencies are imported lazily where possible. To install the core notebook environment:

```powershell
py -3 -m pip install -r requirements.txt
```

Dribble, OVITO, CHGNet, and PyTorch are intentionally not included in the core requirements because they support optional command-line workflows and may require platform-specific installation.

## Reference structures

`perfect.vasp` supplies the ideal lattice and crystallographic site coordinates. `LMTO.vasp` supplies the observed species occupations and distorted atomic coordinates. They must have compatible lattice matrices and the same number of sites for reference-based mapping and displacement analysis.

The supplied `perfect.vasp` labels every cation site as Mn, whereas `LMTO.vasp` contains Li, Ti, and Mn on that cation sublattice. Therefore, reference mapping must treat `Li,Ti,Mn` as an interchangeable cation group while keeping oxygen separate:

```powershell
py -3 tools/match_structures.py perfect.vasp LMTO.vasp `
  --species-policy distorted `
  --matching-groups "Li,Ti,Mn;O" `
  --output LMTO_mapped.vasp `
  --report mapping.json

py -3 tools/site_distortion.py perfect.vasp LMTO.vasp `
  --matching-groups "Li,Ti,Mn;O" `
  --output displacement.json
```

This grouping constrains assignment to the correct sublattice without requiring the ideal reference to carry the observed chemical labels. It does not make different cations chemically equivalent in reported results: displacement statistics are still grouped by the observed Li, Ti, Mn, and O species.

The ideal coordinates are also used to generate fixed tetrahedral centers for n-TM and channel analysis. The distorted structure supplies the four neighboring cations around each center.

Warren–Cowley parameters are different: they quantify chemical short-range order in a single decorated structure and do not require ideal reference coordinates.

## Example notebook

Open `LMTO_analysis_example.ipynb` from the project root. This worked **DRX–Distortion–Percolation** example analyzes `LMTO.vasp` relative to `perfect.vasp` and covers:

1. sublattice-constrained global mapping to the perfect lattice;
2. 0-TM through 4-TM tetrahedral fractions;
3. per-element displacement and the atom-weighted mean displacement of all Ti and Mn sites;
4. detailed compositions and Ti/Mn fractions for 3-TM and 4-TM tetrahedra;
5. Li–O, Ti–O, and Mn–O bond-distance distributions;
6. 0-TM and threshold-qualified 0+1-TM channel-site counts, fractions of all Li, and side-by-side maps;
7. native Warren–Cowley short-range-order analysis and heat map.

Notebook outputs are written to `notebook_outputs/`. The committed notebook has been executed locally with the Anaconda Python environment and embeds the results directly beneath every step: Markdown tables, distributions, structural projections, channel-coverage maps, and the WCP heat map. The 0-TM and 0-TM + 1-TM Li networks are displayed side by side, and their unique-site counts are divided by all Li sites to report channel-site coverage. This fraction is explicitly presented as a coverage metric, not as a separate fixed-cutoff graph-percolation proof. The redundant graph-connectivity notebook section was removed. The notebook requires neither Dribble nor OVITO; those remain optional CLI backends.

The LMTO example has only Mn and Ti transition metals. Its default 1-TM admission cutoffs are therefore deliberately limited to:

```python
DEFAULT_THRESHOLDS = {"Mn": 2.507, "Ti": 2.524}
```

These values are minimum TM-to-three-Li-plane heights in angstrom. A 1-TM tetrahedron contributes its three Li sites when its TM height is greater than the corresponding cutoff. They are inherited analysis parameters rather than values fitted from `LMTO.vasp`; validate them against the channel definition used in a production study.

## Command-line usage

Run tools from the project root:

```powershell
py -3 tools/<tool>.py --help
```

Examples using the supplied structures:

```powershell
py -3 tools/tm_distribution.py --perfect perfect.vasp --pattern LMTO.vasp `
  --cations Li,Ti,Mn --tm-elements Ti,Mn --output tm_distribution.json

py -3 tools/tm_environment.py --perfect perfect.vasp --pattern LMTO.vasp `
  --cations Li,Ti,Mn --output tm_environments.json

py -3 tools/tm_site_statistics.py --perfect perfect.vasp --pattern LMTO.vasp `
  --tm-elements Ti,Mn --classes 0,1,2,3,4 --output tm_site_statistics.json

py -3 tools/tm_li_distance.py --perfect perfect.vasp --pattern LMTO.vasp `
  --tm-elements Ti,Mn --output tm_li_distance.json

py -3 tools/bond_distance.py LMTO.vasp --cations Li,Ti,Mn `
  --output bond_distances.json --plot bond_distances.png

py -3 tools/warren_cowley.py LMTO.vasp --backend native `
  --species Li,Ti,Mn --neighbors 12 --output warren_cowley.json
```

## Core library

| Module | Purpose |
|---|---|
| `distortion_percolation/__init__.py` | Package definition. |
| `distortion_percolation/files.py` | Path handling, file enumeration, preflight checks, and atomic writes. |
| `distortion_percolation/structure_analysis.py` | Shared structure-pair validation, sublattice matching groups, periodic-coordinate operations, tetrahedral centers, and transactional output. |
| `distortion_percolation/percolation_core.py` | Dribble input generation, percolation execution, reference-site mapping, cluster handling, and strict periodic connectivity. |

## Tools

### Structure and local-environment analysis

| Tool | Purpose |
|---|---|
| `match_structures.py` | Map observed sites onto a compatible perfect periodic lattice using global assignment, species constraints, or sublattice groups. |
| `site_distortion.py` | Compute single-frame, batch, and time-series PBC-shortest displacements from a reference structure. |
| `bond_distance.py` | Compute nearest-neighbor or directional cation–anion bond-distance statistics and plots. |
| `tm_distribution.py` | Calculate 0-TM through 4-TM tetrahedral fractions from reference-centered environments. |
| `tm_environment.py` | Count detailed cation compositions of selected n-TM tetrahedra. |
| `tm_site_statistics.py` | Calculate TM-species counts and probabilities within selected n-TM classes. |
| `tm_li_distance.py` | Measure TM distance from the three-Li plane and select 0-TM/1-TM connected Li sites. |
| `warren_cowley.py` | Calculate nearest-neighbor chemical short-range order natively, with an optional OVITO-plugin backend. |
| `draw_connected.py` | Write structures containing selected sites or a largest connected cluster. |
| `order_structures.py` | Enumerate/order disordered structures with supercell, ranking, and output-size controls. |
| `randomize_structures.py` | Generate unique randomized structures from species/count constraints. |

### Percolation

| Tool | Purpose |
|---|---|
| `run_dribble.py` | Run Dribble over a single structure or indexed structure series. |
| `percolation.py` | Analyze general distance/Dribble clusters and periodic connectivity. |
| `percolation_summary.py` | Map/filter mobile-ion 0-TM and 1-TM channels and summarize Dribble percolation. |

`percolation.py` and `percolation_summary.py` are intentionally separate. The former is a general cluster/connectivity analyzer; the latter is the reference-aware DRX channel workflow combining mapping, TM-height filtering, and Dribble percolation.

### Dataset and training

| Tool | Purpose |
|---|---|
| `collect_vasp_energies.py` | Collect energies from calculation slots while preserving missing entries. |
| `build_chgnet_dataset.py` | Parse VASP outputs into a validated CHGNet-compatible JSON dataset. |
| `train_chgnet.py` | Fine-tune a pretrained CHGNet model with configurable training parameters. |

These are the only active CHGNet capabilities. CHGNet Monte Carlo and graph-generation utilities are not included.

## Reproducibility

- Keep the exact `perfect.vasp` used for coordinate references with each analysis.
- Record matching groups, offsets, cation species, TM thresholds, radius, bond rule, and supercell.
- Treat the built-in 1-TM thresholds as explicit analysis parameters and validate them for the scientific convention being used.
- Keep inputs and generated reports in separate directories.
- Record frame ranges and random seeds for batch, ordering, randomization, and training workflows.

## Citation

If you use the DRX–Distortion–Percolation analysis, data, or methodology in this repository, please cite:

> Zichang Zhang, Lihua Feng, Jiewei Cheng, Peng-Hu Du, Chu-Liang Fu, Xintao Long, Anchun Tang, Longlong Fan, Kang Dong, Haoyu Wu, Weihan Li, Jian Peng, Shuo Wang, Dingguo Xia, Xueliang Sun, and Qiang Sun. **Interplay of Lattice Distortion and Cation Disorder Governs Li-Ion Transport in Cation-Disordered Rocksalt Cathodes.** *Journal of the American Chemical Society* (2026). https://doi.org/10.1021/jacs.6c05378

```bibtex
@article{zhang2026interplay,
  title   = {Interplay of Lattice Distortion and Cation Disorder Governs {Li-Ion} Transport in Cation-Disordered Rocksalt Cathodes},
  author  = {Zhang, Zichang and Feng, Lihua and Cheng, Jiewei and Du, Peng-Hu and Fu, Chu-Liang and Long, Xintao and Tang, Anchun and Fan, Longlong and Dong, Kang and Wu, Haoyu and Li, Weihan and Peng, Jian and Wang, Shuo and Xia, Dingguo and Sun, Xueliang and Sun, Qiang},
  journal = {Journal of the American Chemical Society},
  year    = {2026},
  doi     = {10.1021/jacs.6c05378},
  url     = {https://doi.org/10.1021/jacs.6c05378}
}
```

The work establishes lattice distortion as an active variable that couples with cation disorder and short-range order, reshapes nominal 0-TM/1-TM Li-ion pathways, and extends the static ideal-lattice percolation picture in DRX oxides.

## Acknowledgments

This work was partially supported by grants from the National Key Research and Development Program of China (2022YFB2502100), the National Natural Science Foundation of China (92372112, 22373005, 52573249), and the Commanding Heights of Science and Technology of Chinese Academy of Sciences (LDES15 0000). The calculations were supported by the High-performance Computing Platform of Peking University and Eastern Institute of Technology, Ningbo.

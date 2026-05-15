# SlotDeconv

SlotDeconv is a reference-based spatial transcriptomics deconvolution method for estimating spot-level cell-type proportions from spot-based spatial transcriptomics data. It learns discriminative cell-type signatures from annotated scRNA-seq data using slot-based prototype vectors and a diversity constraint, then estimates proportions with weighted NNLS initialization followed by spatial refinement.

This repository contains the model implementation, a command-line runner, example analysis notebooks, and the file format used for the ECCB 2026 manuscript:

**SlotDeconv: Spatial Transcriptomics Deconvolution via Diversity-Constrained Prototype Learning and Spatial Refinement**

## Repository Layout

- `model/slot_model.py`: SlotDeconv model, reference learning, NNLS initialization, and spatial refinement.
- `model/run_slot.py`: command-line runner for one dataset directory.
- `model/slot_utility.py`: data loading, spot alignment, and evaluation metrics.
- `model/plot.py`, `model/visz.py`: plotting and validation utilities.
- `experiments/tutorial.ipynb`: tutorial notebook.
- `experiments/mouse_brain_eccb.ipynb`: mouse brain benchmark analysis used for the ECCB manuscript.
- `experiments/PDAC_run.ipynb`: PDAC analysis notebook.
- `experiments/MOB_eccb.ipynb`: mouse olfactory bulb analysis notebook.
- `experiments/bench_bootstrap.py`: bootstrap and paired-test helper script for benchmark uncertainty.
- `experiments/generate_supplemental_review_analysis.py`: supplemental negative-PCC and Xenium shared-gene concordance analyses.
- `data/spotifydata_v2/`: example mouse brain benchmark directory in the expected CSV format, when included in the local checkout or downloaded from the data link below.

## Installation

The ECCB experiments were run with Python 3.8.20. A minimal environment can be created with:

```bash
conda create -n slotdeconv python=3.8 -y
conda activate slotdeconv
pip install -r requirements.txt
```

The pinned package versions in `requirements.txt` are:

```text
numpy==1.23.5
pandas==2.0.3
scipy==1.10.1
scikit-learn==1.3.2
torch==2.0.1
matplotlib==3.7.5
seaborn==0.13.2
scanpy==1.9.8
anndata==0.9.2
```

The core command-line runner uses `numpy`, `pandas`, `scipy`, `scikit-learn`, `torch`, and `matplotlib`. The notebooks additionally use `seaborn`, `scanpy`, and `anndata`.

## Input Format

`model/run_slot.py` expects a dataset directory containing:

```text
sc_count.csv          genes x cells scRNA-seq count matrix
st_count.csv          genes x spots spatial transcriptomics count matrix
sc_meta.csv           cell metadata indexed by cell id, with a cellType column
spatial_location.csv  spot coordinates indexed by spot id
eval_types.txt        one evaluated cell type per line
true_props.csv        optional, spots x cell types ground truth for benchmarking
```

Spot IDs must match between `st_count.csv` columns and `spatial_location.csv` rows. If `true_props.csv` is present, the same spot IDs are used for evaluation. Gene names are aligned between `sc_count.csv` and `st_count.csv` before fitting.

## Quick Start

From the repository root, run SlotDeconv on the example mouse brain directory:

```bash
PYTHONPATH=model python model/run_slot.py \
  --data_dir data/spotifydata_v2 \
  --output_dir results/example_mousebrain
```

Expected outputs:

```text
results/example_mousebrain/predicted_props.csv
results/example_mousebrain/predicted_props_nnls.csv
results/example_mousebrain/metrics_all.csv
results/example_mousebrain/metrics_valid.csv
results/example_mousebrain/celltype_metrics.csv
```

`predicted_props.csv` contains the spatially refined cell-type proportions. `predicted_props_nnls.csv` contains the NNLS-only initialization. Metric files are written when `true_props.csv` is available.

To run the non-spatial NNLS-only ablation:

```bash
PYTHONPATH=model python model/run_slot.py \
  --data_dir data/spotifydata_v2 \
  --output_dir results/example_mousebrain_nnls \
  --no_spatial
```

## Reproducing Manuscript Analyses

The main manuscript analyses are organized as notebooks and helper scripts:

- Mouse brain benchmark: `experiments/mouse_brain_eccb.ipynb`
- PDAC validation: `experiments/PDAC_run.ipynb`
- Mouse olfactory bulb validation: `experiments/MOB_eccb.ipynb`
- Benchmark uncertainty and paired tests: `experiments/bench_bootstrap.py`
- Supplemental negative-PCC and Xenium shared-gene concordance analyses: `experiments/generate_supplemental_review_analysis.py`

The notebooks preserve the manuscript analysis workflow and assume the processed
CSV inputs follow the format described above. If paths differ on your system,
update the path and output-directory variables in the relevant notebook cells.
The helper scripts use repository-relative defaults and also accept command-line
paths. For example, pass baseline prediction files explicitly when reproducing
the mouse brain benchmark table:

```bash
python experiments/bench_bootstrap.py \
  --data-dir data/spotifydata_v2 \
  --slot-dir results/mousebrain \
  --spotiphy results/mousebrain/baselines/spotiphy_props.csv \
  --cell2location results/mousebrain/baselines/spotiphy_cell2location_props.csv \
  --rctd results/mousebrain/baselines/spotiphy_RCTD.weights.norm.csv \
  --card results/mousebrain/baselines/spotiphy_CARD_proportions.csv
```
Baseline prediction CSVs for Spotiphy, Cell2location, RCTD, and CARD are not
generated by SlotDeconv; generate them with the corresponding method
implementations or place the processed files from the project data folder under
`results/mousebrain/baselines/`.

## Public Data and Accession Links

Processed example data and manuscript-ready CSV inputs are provided through the project data folder:

- Google Drive data folder: <https://drive.google.com/drive/folders/1-d2rTbGwN3zK3B5PGZM9LvhCXZEKdYSC?usp=drive_link>

Public source datasets used in the manuscript:

- Mouse brain benchmark and Xenium-aligned ground-truth proportions: Spotiphy study, Zenodo record 10520022, <https://zenodo.org/records/10520022>
- Human pancreatic ductal adenocarcinoma spatial transcriptomics: Moncada et al., GEO GSE111672, <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE111672>
- Mouse olfactory bulb scRNA-seq reference: GEO GSE121891, <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE121891>

Where redistribution is permitted, processed CSV files follow the `Input Format` section above. Otherwise, the notebooks document the conversion from public sources to the required CSV files.

## Citation

If you use SlotDeconv, please cite:

```text
Fang H, Qi C, Zou Y, Chen Y, Wei Z. SlotDeconv: Spatial Transcriptomics
Deconvolution via Diversity-Constrained Prototype Learning and Spatial
Refinement. ECCB 2026 Proceedings Track / Bioinformatics, 2026.
```

## Contact

For questions about the manuscript or code, please contact the corresponding author listed in the paper.

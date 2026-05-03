# Regional Downscaling of Summer Daily Maximum Temperature with Deep Learning

CSCI 1470 final project by Xi Chen.

This project studies statistical downscaling of summer daily maximum 2-meter temperature over the northeastern United States. The goal is to map coarse-resolution weather fields to higher-resolution temperature structure using deep learning, with comparisons across U-Net/CNN-style models and transformer-based models. The experiments evaluate overall prediction skill, spatial error structure, and weather-regime-dependent model behavior.

## Final Deliverables

- Poster: [`poster_xi_chen.jpg`](poster_xi_chen.jpg)
- Final report: [`paper/final_report_xi_chen.pdf`](paper/final_report_xi_chen.pdf)
- Code: [`src/`](src), [`data_preprocess/`](data_preprocess), and [`plots/`](plots)

For a quick review, start with the poster and final report, then use this README
as a map of the code and experiment artifacts.

The original project proposal is preserved separately in [`PROPOSAL.md`](PROPOSAL.md).

## Repository Structure

```text
.
├── src/                  # Dataset classes, model definitions, train/evaluate scripts
├── data_preprocess/      # Data download, regridding, split, and normalization notebooks
├── plots/                # Analysis notebooks and poster-figure generation scripts
├── figures/              # Final figures and summary tables used in the poster/report
├── outputs/              # Small final metric CSVs and best-model checkpoints
├── paper/                # Final written report PDF
├── poster_xi_chen.jpg    # Required high-resolution horizontal 4:3 poster
├── PROPOSAL.md           # Original project proposal
├── environment.yml       # Full conda environment export
├── environment-from-history.yml
└── req.txt
```

Large raw data files and generated NetCDF prediction arrays are intentionally not committed. They are ignored under `.data/` and `outputs/*.nc`.

## Guide To Files

### Core Code

- [`src/model_cnn.py`](src/model_cnn.py): U-Net/CNN-style downscaling models.
- [`src/model_transformer.py`](src/model_transformer.py): transformer-based downscaling model.
- [`src/dataset.py`](src/dataset.py): base dataset class for low-resolution temperature plus topography.
- [`src/dataset_x_only.py`](src/dataset_x_only.py): dataset for the low-resolution-temperature-only ablation.
- [`src/dataset_x_d2m_elev.py`](src/dataset_x_d2m_elev.py): dataset for low-resolution temperature, dew point, and elevation.
- [`src/dataset_t_elev_urban.py`](src/dataset_t_elev_urban.py): dataset for low-resolution temperature, elevation, and urban fraction.
- [`src/utils.py`](src/utils.py): shared helpers used by training/evaluation code.

### Training Scripts

- [`src/train_cnn.py`](src/train_cnn.py): main U-Net/CNN baseline using low-resolution temperature and topography.
- [`src/train_x_only.py`](src/train_x_only.py): U-Net ablation with only low-resolution temperature input.
- [`src/train_x_d2m_elev.py`](src/train_x_d2m_elev.py): U-Net variant with dew point and elevation features.
- [`src/train_unet_t_elev_urban.py`](src/train_unet_t_elev_urban.py): U-Net variant with elevation and urban fraction.
- [`src/train_transformer_x_only.py`](src/train_transformer_x_only.py): transformer ablation with only low-resolution temperature input.
- [`src/train_transformer_x_d2m_elev.py`](src/train_transformer_x_d2m_elev.py): transformer variant with dew point and elevation features.
- [`src/train_transformer_t_elev_urban.py`](src/train_transformer_t_elev_urban.py): transformer variant with elevation and urban fraction.

### Evaluation And Analysis

- `src/evaluate*.py`: generate test-set predictions for each trained model. These scripts write large `.nc` files locally, which are intentionally excluded from GitHub.
- [`src/feature_importance_rich.py`](src/feature_importance_rich.py): feature/error analysis for the richer predictor set.
- [`plots/`](plots): notebooks used to compare models, analyze weather-condition performance, and generate final figures.
- [`plots/make_poster_result_figures.py`](plots/make_poster_result_figures.py): script that generates the poster-ready result panels.

### Data Preparation

- [`data_preprocess/gridmet_4km.ipynb`](data_preprocess/gridmet_4km.ipynb): prepares the high-resolution GRIDMET target.
- [`data_preprocess/ERA5_Tmax.ipynb`](data_preprocess/ERA5_Tmax.ipynb): prepares ERA5 daily maximum-temperature predictors.
- [`data_preprocess/ERA5_extra_features.ipynb`](data_preprocess/ERA5_extra_features.ipynb): prepares additional ERA5 weather variables used in weather-regime analysis.
- [`data_preprocess/ETOPO.ipynb`](data_preprocess/ETOPO.ipynb): prepares elevation/topographic features.
- [`data_preprocess/Zenodo.ipynb`](data_preprocess/Zenodo.ipynb): prepares urban fraction data.
- [`data_preprocess/split.ipynb`](data_preprocess/split.ipynb) and [`data_preprocess/split_extra_features.ipynb`](data_preprocess/split_extra_features.ipynb): create chronological train/validation/test splits.
- [`data_preprocess/normalize.ipynb`](data_preprocess/normalize.ipynb) and [`data_preprocess/normalize_extra_features.ipynb`](data_preprocess/normalize_extra_features.ipynb): normalize model inputs and targets.

### Final Artifacts

- [`figures/model_comparison/`](figures/model_comparison): model-comparison plots and metric tables.
- [`figures/weather_condition_compare/`](figures/weather_condition_compare): weather-condition analysis using project-derived weather variables.
- [`figures/weather_condition_compare_era5/`](figures/weather_condition_compare_era5): weather-condition analysis using ERA5 weather variables.
- [`figures/spatial_bias_local_features/`](figures/spatial_bias_local_features): spatial error and local-feature diagnostics.
- [`figures/poster_results/`](figures/poster_results): final panels used in the poster.
- [`outputs/checkpoints/`](outputs/checkpoints): compact best-model checkpoints.
- [`outputs/*.csv`](outputs): training histories and compact summary tables.

### Files Intentionally Excluded

- `.data/`: raw and processed NetCDF/TIF data files.
- `outputs/*.nc`: full model prediction arrays.
- cache files such as `.DS_Store`, `__pycache__/`, and `.ipynb_checkpoints/`.

## Data

The project uses:

- GRIDMET 4 km daily maximum temperature as the high-resolution target
- ERA5-derived daily predictors as coarse-resolution inputs
- ETOPO elevation/topographic features
- Urban fraction and other static auxiliary features for some ablations

Raw and processed data are expected under `.data/`, which is excluded from Git because the files are large. The preprocessing notebooks in [`data_preprocess/`](data_preprocess) document the data construction workflow.

## Environment

Create the conda environment with:

```bash
conda env create -f environment.yml
conda activate csci1470-final-project
```

If the full export is too platform-specific, use:

```bash
conda env create -f environment-from-history.yml
```

## Training And Evaluation

The main training scripts are in [`src/`](src). Example commands:

```bash
python src/train_cnn.py
python src/train_x_only.py
python src/train_x_d2m_elev.py
python src/train_unet_t_elev_urban.py
python src/train_transformer_x_only.py
python src/train_transformer_x_d2m_elev.py
python src/train_transformer_t_elev_urban.py
```

Evaluation scripts write prediction NetCDFs to `outputs/`; these files are useful locally but are not tracked on GitHub because they are large:

```bash
python src/evaluate.py
python src/evaluate_x_only.py
python src/evaluate_x_d2m_elev.py
python src/evaluate_unet_t_elev_urban.py
python src/evaluate_transformer_x_only.py
python src/evaluate_transformer_t_elev_d2m.py
python src/evaluate_transformer_t_elev_urban.py
```

Final figures and summary tables are saved in [`figures/`](figures). The poster-specific figure pipeline is in [`plots/make_poster_result_figures.py`](plots/make_poster_result_figures.py).

## Included Outputs

The repository includes compact artifacts that help reproduce the final analysis:

- best-model checkpoints in `outputs/checkpoints/`
- training loss histories in `outputs/*.csv`
- final figure PNGs and summary CSVs in `figures/`

The large `.nc` prediction files can be regenerated by running the evaluation scripts after preparing `.data/`.

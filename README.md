# Regional Downscaling of Summer Daily Maximum 2-Meter Temperature over the Northeastern United States Using Deep Learning
## Motivation
This project aims to use deep learning to downscale summer daily maximum near-surface temperature over the northeastern United States, including Rhode Island and surrounding states. High-resolution weather and climate data are important for understanding regional heat exposure, especially because extreme summer heat can strongly affect human health, outdoor workers, agriculture, and ecosystems. However, generating high-resolution climate information using physical climate models is computationally expensive. This makes statistical and machine learning downscaling a useful alternative.
I will compare two deep learning approaches introduced in class, convolutional neural networks (CNNs) and transformers, for spatial downscaling. CNNs are effective at learning local spatial patterns, while transformers may better capture broader spatial dependencies. In addition to comparing their overall prediction skill, I am interested in identifying the conditions under which transformers provide an advantage, such as during large-scale heat events or spatially coherent temperature anomalies.
For training and evaluation, I will use summer (June, July, and August) daily maximum temperature data from 2000 to 2025 from both a high-resolution dataset (4 km) and a lower-resolution dataset (0.25°, about 25 km). I will also include static surface features such as elevation, since near-surface temperature is strongly influenced by topography. Because temperature patterns have strong temporal structure, I will not randomly shuffle the samples. Instead, I will use a chronological split to better reflect a realistic prediction setting: 2000–2018 for training, 2019–2021 for validation, and 2022–2025 for testing.
Model performance will be evaluated using quantitative metrics such as RMSE, MAE, and spatial correlation. As a non-deep-learning baseline, I will compare the models against linear interpolation (regridding).
## Key Limitations Anticipated
A key limitation of this project is the complexity of data preprocessing and alignment. The high-resolution and low-resolution datasets are defined on different spatial grids, so I will need to carefully regrid or interpolate them onto a consistent framework before training. Temporal alignment is also important, since I need to ensure that the daily maximum temperature fields from both datasets correspond exactly in time. In addition, missing values, land masks, and differences in spatial coverage may introduce extra preprocessing challenges. Defining clean input and output tensor structures for both the dynamic temperature fields and static features such as elevation may also take more time than expected.
Another limitation is project scope. Comparing multiple deep learning architectures, including CNNs and transformers, while also testing additional static features, could become ambitious within the available time. To keep the project manageable, I will first build a simple CNN baseline with a minimal feature set and only add transformers or extra predictors after the basic pipeline is working well.
A further challenge is that the dataset may still be limited for training more complex models, especially transformers, which can require more data and computation. This raises the risk of overfitting and may make it difficult to clearly separate true model improvements from noise. As a result, careful baseline comparison and scope control will be important throughout the project.

## Project Data Ideas
1. High-resolution target dataset:4 km daily high-resolution temperature dataset https://water.usgs.gov/catalog/datasets/ef98187e-8703-4ec6-afc1-4dbc72c9d6d8/
2. Low-resolution predictor dataset:ERA5 daily dataset https://cds.climate.copernicus.eu/datasets/derived-era5-single-levels-daily-statistics?tab=overview
3. Surface elevation dataset: https://www.ncei.noaa.gov/products/etopo-global-relief-model

## Conda Env
Run:
`conda env create -f environment.yml`
for all packages, specific versions, and build-specific information.

Run:
`conda env create -f environment-from-history.yml`
for no transitive dependencies or OS-specific builds.

Run:
`conda create -n csci1470-final-project --file req.txt`
for package specific builds.
# Oculoplastics Disgnosis

## Project Description
This project dedicates to automatically diagnose selected oculoplastics diseases.


## Projcet Orgnization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed 	   <- Processed images.
    │   ├── meta_data 	   <- Meta data about the dataset (i.e. prediction results, image_path, etc.)
    │   └── raw            <- The original, immutable data dump.
    │        └── some_data_identifier <- put images for each group into corresponding folder
    │             ├── normal
    │             ├── ptosis
    │             ├── thyroid_eye_disease
    │             └── others
    │
    ├── notebooks          <- Jupyter notebooks.
    ├── docker             <- Docker related files
    ├── reference          <- Related files from other projects, including related papers, models, etc.
    └── src                <- Source code

## Running the code
----------
### 1. Getting the data into the right place
- **Step 1**: To begin with, run the script `python src/first_time_setup.py` to build necessary folders
- **Step 2**: make a directory for current data (in case we have more data coming in the later course), for example, we can do `03032021`.
- **Step 3**: Put all the raw image data (i.e. full face images to be classified) into `data/raw/[some_data_identifier]/` in the following way:
    - Images with the same condition go into the same sub-folders
    - sub-folders are named with corresponding condition with lower case, connected by underscore
    - e.g. `data/raw/[some_data_identifier]/thyroid_eye_disease/`

### 2. Running different targets

#### 2.0 Overview
All the functionalities, from data cleaning to model testing, are implemented via `run.py`. Executing `run.py` with flags and arguments would accomplish corresponding targets

#### 2.1 Specifying model
To specify a model, use `-m [your_config_file_name]`. Make sure there is a corresponding `[your_config_file_name].json` in the folder `config/`

#### 2.2 Specifying data source
To specify a data source, edit the `data_sources` field of the json file. This will be the identifier for both most of the process related to data (e.g. loading raw images, saving processed images, saving metadata, etc.)

#### 2.3 Data Preparations:
- **Step 1**: Data cleaning: `python run.py -c -m [your_config_file_name]`; this will **rename the images**, **crop them with only eye regions**, and **save them into `processed/[your_data_sources]/`**
- **Step 2**: Manual inspection **IMPORTANT**: after running data cleaning, remember to check the resulting images in the `processed/[your_data_sources]/` and put all the in-correctly cropped images into `data/meta_data/[some_data_identifier]/error_img_df.csv`. In this way, although these images are processed, they will not appear in the final training steps
- **Step 3**: Dataloader preparation: `python run.py -p -m [your_config_file_name]`; this will prepare the needed pickle files for dataloaders. After this step, you should find a pickle file in `data/processed/[some_data_identifier]/img_info_df_no_error.pickle`

#### 2.4 Training
- `python run.py -t -m [your_config_file_name]`
- Make sure you finished all the step before doing training!

#### 2.5 Training results
- All the checkpoints and training loggings are all inside `data/checkpoints/[model_name]`, depending on the json file, [model_name] can be `three_classes_v1`, `ted_over_other_v2`, etc.
- Inside the directory:
    - `model_[model_name].pt`: the torch model file for this particular model version (selected based on the val acc);
    - `model_[model_name].json`: the training configuration;
    - `training_log.csv`: the training&validation acc&loss for each epoch.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


-------

# Motional Prediction Development Kit
This devkit contains a complete pipeline of training a motion forecasting model using Motional Prediction dataset. This includes a self-contained map API and an example implementation of the Dataset class to use our dataset.

Quick Start: [[Dataset]](#download-dataset) [[Run Examples]](#run) [[Devkit Structure]](#devkit-structure) [[Challenge Submission]](#submit-to-the-challenge)

## Change Log
 - 2022/12/15 Devkit initial release

## Download Dataset

- **Trajectories**: link to be posted in Janurary.
- **Map**: download the map from [nuPlan](https://www.nuscenes.org/nuplan) under nuPlan v1.1.

Note: we share the same dataset source as [nuPlan](https://www.nuscenes.org/nuplan). Here we provide `.parquet` files as a better representation for prediction task. We also provide a script in `_db_converter` if you want a highly customized data, for example longer inputs/targets beyond 10s segments we provide. You are able to process thoes `.db` files provided by nuPlan with the script. This could be useful in your own task formulation. However, as for challenge submission, we will stick to the `.parquet` format as in training and validation splits.

## Devkit Structure

Please use the information below to understand the devkit better if you want to start from our devkit. This devkit uses `PyTorch-Lightning` for training and `Hydra` for configurations.

*Want to create your own pipeline?* Directories/files with star(*) are what you want to primarily look into if you want to set up your own pipeline from scratch.

```
prediction_devkit
├── _db_converter           - Script for convert nuPlan .db to .parquet. Only used if a highly customized dataset wanted.
├── _doc                    - Resources for readme
├── callback                - Callback functions used during training
├── config                  - Hydra configs only
├── dataloader*             - Implementation of Dataset, DataModule
├── map*                    - Map API. This dir is self-contained and can be migrated to your own repo
│   ├── nuplan_map
│   │       ├── nuplan_map.py       - Main map API class
│   │       ├── vector_map_utils.py - Main utils to get vector map features from the map
├── metric                  - Modularized evaluation metrics
├── model                   - Define motion forecasting models
├── training_engine         - Main pipeline of model training
├── tutorials               - Some notebooks for getting familiar with data and map
├── main.py                 - The entry point of training
├── prediction_env.yaml*    - conda environment file
```

## Run

### Explore with data and map
See notebooks in `tutorials`.

### Create Cache (Feature Pre-processing)
Usually it is recommended to create cache for the features before actual training.
To create cache, simply run the training command. Here are examples of creating cache:

```
# if the cache has not been generated, or is not there, or to continue the incomplete generation
python main.py +training=training_example_minimum.yaml

# if you want to generate a new one and cover any existing part/entire cache
python main.py +training=training_example_minimum.yaml datamodule.force_cache=true

# use single thread [not recommended except for sanity check]
python main.py +training=training_example_minimum.yaml datamodule.force_cache=true datamodule.multiprocessing_cache=false
```
Note: you can freely ignore the error `RuntimeError: Timed out initializing process group in store based barrier on rank: 0` or `TimeoutError: The client socket` from other processes when waiting this cache to finish. You can use `CUDA_VISIBLE_DEVICES=0` or `trainer.devices=1` to avoid this issue.

Several configs related to data files:
- `datamodule.root`: the root directory of the dataset (trajectories)
- `datamodule.cache_dataset_root`: the directory contains your cached features
- `datamodule.map_root`: the directory contains map

Expected structures of data and map directories to use the provided DataModule:
```
root
├── metadata     - metadata of the dataset samples (mainly scenario labels)
├── train        - train trajectories
│   ├── 2021.10.08.18.57.48_veh-28_01057_01171_00001.parquet   - 10s segments of one scenario with trajectories
│   ├── ...
├── val          - val trajectories
│   ├── xxxxx.parquet
│   ├── ...
├── (test)       - test trajectories -- not available for public. The format is the same as train/val. 
│   ├── xxxxx.parquet
│   ├── ...

map
├── nuplan-maps-vx.x.json        - map meta information
├── sg-one-north                 - map dir for SG
├── us-ma-boston                 - map dir for Boston
├── us-nv-las-vegas-strip        - map dir for Las Vegas
├── us-pa-pittsburgh-hazelwood   - map dir for Pittsburgh

cache_root
├── train        - train features
│   ├── 2021.10.08.18.57.48_veh-28_01057_01171_00001.pt       - preprocessed features. must match the .parquet file name
│   ├── ...
├── val          - val features
│   ├── xxxxx.pt
│   ├── ...
```

Configs related to generate cache:
- `datamodule.generate_cache`: `true` should be true always
- `datamodule.force_cache`: `true` will cover pre-generated cache in the `cache_dataset_root` dir. `false` will skip any pre-generated ones.
- `datamodule.multiprocessing_cache`: `true` to cache in parallel processes to speed up. Recommended.
- `datamodule.num_cache_wokers`: number of CPU threads needed for caching depending on your machine.

### Training
We can use the generated cache, or re-use any pre-generated cache:
```
python main.py +training=training_example_minimum.yaml
```

### Evaluation
```
# validation set
python main.py +training=training_example_minimum.yaml py_func=validate checkpoint_to_validate="/path/to/your/model.ckpt"
# test set (optinal, and if you use this devkit for challenge submission)
python main.py +training=training_example_minimum.yaml py_func=test checkpoint_to_validate="/path/to/your/model.ckpt"
```
Note that the test function will generate prediction result files. We will compare prediction result files to give you flexibility using any training framework you have, as long as it generates result files with required format. Details will be listed in [Challenge Submission](#submit-to-the-challenge).

### Model Examples
We provide two examplar models to help understand the data and framework.
- Minimum model: for getting familiar with the data
```
python main.py +training=training_example_minimum.yaml
```

- [HiVT](https://github.com/ZikangZhou/HiVT): for understanding how an existing model can be intergrated in this framework. Note: because of the high number of lanes from map api, you might need to downsample the map and adjust the batch size (`datamodule.train_batch_size`) to fit this model in a regular GPU. 
```
python main.py +training=training_example_hivt.yaml
```
Directories/files that are mainly related to modeling:
- `model`: model definition
- `config/model`: model-specific configurations
- `config/training`: main configuration file about how to train this model

Note: the current provided example is making predictions on vehicles only. To make predictions on pedestrains and other types of objects, please change/create your own DataModule.

## Submit to the Challenge
### Task Definition
In Motional's prediction challenge, a motion forecating model is required to take 2s history and to make predictions of 8s future.
Though the dataset is provided in 20Hz, we will only evaluate the results in 10Hz.

### Submission Format
*Tentative*

We will require participants to submit their code (docker) to the challenge. The submission should include their trained model with the commond to run test. We will require the participants to generate files with prediction results. We will use the content in thoes files to compute the testing metric.

**Tentative** result file format is below:
 - One result file per sample (*not per batch*)
 - Filename: \[sample_name\].pt (e.g. 2021.10.08.18.57.48_veh-28_01057_01171_00001.pt) You will have this \[sample_name\] from the dataset metadata, as we show in `datamodule/dataset.py`. 
 - Content: a Python `dict` object:
 ```
 {
    "4c06fc2a347596b": torch.Tensor([K, T, 2]),
    [agent_id: str]: [prediction: torch.Tensor of shape [K, T, 2]] 
    # K is multi modalities (sorted by confidence descendingly if you have)
    # T is time steps (80) and 2 stands for x and y coordinates
    # remember to convert the result Tensor to cpu (.detach().cpu())
    
 }
 ```
We provide an exampl to save this kind of file at `LightningModuleWrapper.test_step()` in `training_engine/lightning_module_wrapper.py`.

### Important Dates
TBD


## Citation
```
TBD
```

## Contact
Welcome comments and issues under this repo. Your voice is valuable for our development.

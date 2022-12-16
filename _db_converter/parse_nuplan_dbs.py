import os

os.environ["PYTHONUNBUFFERED"] = "1"

# !!change following if your data is stored locally
# for s3 remote store
os.environ["NUPLAN_DATA_STORE"] = "s3"
os.environ["NUPLAN_CACHE_FROM_S3"] = "false" # always false to avoid duplicate download
os.environ["NUPLAN_DATA_ROOT_S3_URL"] = "s3://some-remote-path-for-data" # change this
os.environ["NUPLAN_MAPS_ROOT_S3_URL"] = "s3://some-remote-path-for-map"  # change this

# for local
# os.environ["NUPLAN_DATA_STORE"] = "local"
# NUPLAN_DATA_ROOT = "/where/you/put/nuplan/data"
# os.environ["NUPLAN_CACHE_FROM_S3"] = "false" # always false to avoid duplicate download
# os.environ["NUPLAN_DATA_ROOT_S3_URL"] = "s3://some-remote-path-for-data" # will be ignored, can be omitted
# os.environ["NUPLAN_MAPS_ROOT_S3_URL"] = "s3://some-remote-path-for-map" # will be ignored, can be omitted

from io import BytesIO
import pandas as pd
import yaml
import time
from multiprocessing import Pool, get_context
import numpy as np

from _db_converter.utils import (
    absolute_path_to_log_name,
    download_file_if_necessary,
    put_file_to_s3,
    get_log_metadata_from_db,
    get_all_lidar_pc_from_db,
    get_all_ego_from_db,
    get_all_lidar_box_with_attributes_from_db
)

from _db_converter.s3_utils import expand_s3_dir

# where you will put converted files, s3 or local
SAVE_DIR = "s3://where/to/save/converted/data" 

def get_splits():
    """
    return: a dictionary contains train, and val splits in 
    {
        "train": List[str]
        "val": List[str]
    } format. Each str is the db file name (without suffix .db and store path)
    """

    # train and val
    with open("splits.yaml", "r") as stream:
        try:
            splits = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    splits["train"] = splits["log_splits"]["train"][:]
    splits["val"] = splits["log_splits"]["val"][:]
    del splits["log_splits"]

    return splits
    

def save_dataframe(df, save_dir: str, fn: str):
    if save_dir.startswith("s3://"):
        buf = BytesIO()
        df.to_parquet(buf)
        buf.seek(0)
        content = buf.read()
        put_file_to_s3(save_dir, fn, content)
    else:
        os.makedirs(save_dir, exist_ok=True)
        df.to_parquet(os.path.join(save_dir, fn))



# print (splits)
# print (len(splits["train"]))
# print (len(splits["val"]))

def convert_one_log(db_file, split):
    EXTRACT_SCENARIO_LENGTH = 10.0
    INPUT_LENGTH = 2.0
    DATABASE_INTERVAL = 0.05 # highest

    EGO_VEHICLE_SIZE = {"length": 4.635, "width": 1.89, "height": 1.605} # length, width, height

    CONSIDERED_COLUMNS = ["timestamp", "track_token", "x", "y", "z", "vx", "vy", "vz", 
                            "length", "width", "height", "category_name", "city"]

    start = time.time()

    

    if os.getenv("NUPLAN_DATA_STORE") == "s3":
        download_path = "/data/cache/" # where you put downloaded db files
        if split == "train" or split == "val":
            remote_path = os.path.join(os.getenv('NUPLAN_DATA_ROOT_S3_URL'), "splits", "public_set", db_file+".db")
        
        log_file, download_time = download_file_if_necessary(download_path, remote_path)
    else: 
        log_file = os.path.join(download_path, split, db_file+".db")
    
    metadata :pd.DataFrame = get_log_metadata_from_db(log_file)
    ego      :pd.DataFrame = get_all_ego_from_db(log_file)
    # lidar_pc :pd.DataFrame = get_all_lidar_pc_from_db(log_file)
    lidar_box:pd.DataFrame = get_all_lidar_box_with_attributes_from_db(log_file)
    # scenario_tags: pd.DataFrame = get_all_scenario_tag_from_db(log_file)
    
    city_name = metadata["location"][0]
    assert len(city_name) > 1
    # print (city_name)

    # print (ego["timestamp"].unique())
    sample_step = int(EXTRACT_SCENARIO_LENGTH / DATABASE_INTERVAL)
    timestamps = ego["timestamp"].unique().tolist() # note: timestamp is already ordered, this is done in sql layer
    file_counter = 0

    metadata_record = {"filename": [], "scenario_tag": []}

    for index in range (0, len(timestamps), sample_step):

        if index + sample_step > len(timestamps):
            break
        
        # start = time.time()
        scenario_ts_ranges = timestamps[index:index+sample_step]
        input_steps = int(INPUT_LENGTH / DATABASE_INTERVAL)

        ego_trajectory = ego[ego["timestamp"].isin(scenario_ts_ranges)] # this is order-preserving
        agents_trajectory = lidar_box[lidar_box["timestamp"].isin(scenario_ts_ranges)] # this is order-preserving

        ego_tokens = ego_trajectory["token"].apply(lambda x: bytes(x).hex())
        ego_trajectory = ego_trajectory.assign(track_token=ego_tokens,
                                                length=EGO_VEHICLE_SIZE["length"],
                                                width=EGO_VEHICLE_SIZE["width"],
                                                height=EGO_VEHICLE_SIZE["height"],
                                                category_name="vehicle",
                                                city=city_name).reindex()

        agents_tokens = agents_trajectory["track_token"].apply(lambda x: bytes(x).hex())
        agents_trajectory = agents_trajectory.assign(track_token=agents_tokens,
                                                city=city_name).reindex()

        filename_to_save = f"{db_file}_{file_counter:05d}.parquet"
        metadata_record["filename"].append(filename_to_save)
        metadata_record["scenario_tag"].append(this_tag)
        
        tags = ego_trajectory[["type", "timestamp"]][:input_steps]
        non_unknown_tags = tags[~tags["type"].isnull()]
        if len(non_unknown_tags) == 0:
            this_tag = "unknown"
        else:
            this_tag = non_unknown_tags.groupby(["type"], as_index=False).count() \
                             .sort_values(["timestamp"], ascending=[False]) \
                             .reset_index(drop=True) \
                             .iloc[0]["type"]

        # print (this_tag)

        converted_df = pd.DataFrame()
        converted_df = pd.concat([ego_trajectory[CONSIDERED_COLUMNS],
                                    agents_trajectory[CONSIDERED_COLUMNS]],
                                    ignore_index=True)
        converted_df = converted_df.sort_values(by = ["timestamp", "track_token"], ascending = [True, True])
        converted_df = converted_df.reset_index(drop=True)
        assert len(converted_df) == (len(ego_trajectory) + len(agents_trajectory))

        save_dataframe(converted_df, os.path.join(SAVE_DIR, split), filename_to_save)
        file_counter += 1

        del converted_df

        # break
    del metadata, ego, lidar_box

    # if downloaded from remote, delete the db file because we don't need it
    if os.getenv("NUPLAN_DATA_STORE") == "s3":
        os.remove(log_file)

    metadata_record_df = pd.DataFrame(data=metadata_record)
    metadata_filename_to_save = f"{db_file}_meta.parquet"
    save_dataframe(metadata_record_df, os.path.join(SAVE_DIR, "metadata", split), metadata_filename_to_save)

    took = time.time() - start
    print (f"[{db_file}] took {took:.1f} ({download_time:.1f})s, extracted: {file_counter}, avg: {(took-download_time)/file_counter:.2f}s")

    return (file_counter, took, download_time, split)

def merge_meta(local_dataset_dir, splits):
    """
    merge per-db metadata to one to save time for DataModule creation when training
    """
    for split in splits:
        metadata_split_dir = os.path.join(local_dataset_dir, "metadata", split)
        metadata_paths = [os.path.join(metadata_split_dir, fn) for fn in os.listdir(metadata_split_dir) if fn.endswith(".parquet") and fn != f"{split}.parquet"]
        dataset_meta_list = []
        for p in metadata_paths:
            log_meta = pd.read_parquet(p)
            dataset_meta_list.append(log_meta)

        dataset_meta = pd.concat(dataset_meta_list,
                                    ignore_index=True)
        dataset_meta = dataset_meta.sort_values(by = ['filename'], ascending = [True])
        dataset_meta = dataset_meta.reset_index(drop=True)
        save_dataframe(dataset_meta, metadata_split_dir, f"{split}_meta.parquet")
        # optionally delete original per-db meta files
        # for p in metadata_paths:
        #     os.remove(p)
    

def generate_parse_stats(results):
    stat_dict = {"num_scenario": [],
                    "download_time": [],
                    "process_time": [], 
                    "total_time": [],
                    "split": []}
    
    for num_scenario, total_time, download_time, split in results:
        stat_dict["num_scenario"].append(num_scenario)
        stat_dict["total_time"].append(total_time)
        stat_dict["download_time"].append(download_time)
        stat_dict["process_time"].append(total_time-download_time)
        stat_dict["split"].append(split)
    
    stat_df = pd.DataFrame(data=stat_dict)
    stat_df.to_csv("data_convertion_stat.csv", index=False)

    total_num_scenario = stat_df["num_scenario"].sum()
    train_num_scenario = stat_df[stat_df["split"]=="train"]["num_scenario"].sum()
    val_num_scenario = stat_df[stat_df["split"]=="val"]["num_scenario"].sum()

    total_time = stat_df["total_time"].sum()
    download_time = stat_df["download_time"].sum()
    download_time_ratio = download_time/total_time

    avg_total_time = total_time / total_num_scenario
    avg_process_time = stat_df["process_time"].sum() / total_num_scenario


    print ("--------------------")
    print (f"scenarios: {total_num_scenario} ({train_num_scenario}/{val_num_scenario})")
    print (f"convertion time (thread time): {total_time:.1f}s (download: {download_time:.1f}s, {download_time_ratio*100:.1f}%)")
    print (f"avg (thread time): {avg_total_time:.3f}s")
    print ("--------------------")
    

def create_dataset():
    splits = get_splits()
    db_file_list = []
    split_list = []
    for split in ["train", "val"]:
        for db_file in splits[split]:
            db_file_list.append(db_file)
            split_list.append(split)
            # break
        # break
    
    # db_file_list = db_file_list[:10]
    # split_list = split_list[:10]

    workers = 30
    with get_context("spawn").Pool(processes=workers) as pool:
        results = pool.starmap(convert_one_log, zip(db_file_list, split_list))
    return results

if __name__ == "__main__":
    
    results = create_dataset()
    generate_parse_stats(results)

    # note: this function only works on local storage. 
    # It has not supported reading from remote `SAVE_DIR` yet.
    merge_meta(SAVE_DIR, ["train", "val"])
    
    

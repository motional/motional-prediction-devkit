import logging
import os
import warnings
# don't know why getting this warning for this dataloader example
warnings.filterwarnings("ignore", ".*one or more elements.*")
from itertools import permutations, product
from typing import Any, Callable, List, Optional, Tuple, Union, Dict

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm

# use dataset class from torch_geometric for example
from torch_geometric.data import Data
from torch_geometric.data import Dataset

# map api
from map.geometry.geometry import Point2D
from map.utils import get_all_maps_api, parquet_city_name_to_map_api_name
from map.nuplan_map.vector_map_utils import (
    get_neighbor_vector_map,
    get_traffic_light_encoding,
    merge_vector_map,
)

from dataloader.utils import TemporalData

logger = logging.getLogger(__name__)

"""
Following is an example dataset class to use our dataset. 
You are able to modify or copy everything in it to fulfill your own usage.
"""
class MotionalDataset(Dataset):

    ALL_CITIES = ['sg-one-north', 'us-ma-boston', 'us-nv-las-vegas-strip', 'us-pa-pittsburgh-hazelwood']

    def __init__(self,
                 raw_dataset_root: str,
                 split: str,
                 cache_dataset_root: str = None,
                 map_root: str = None,
                 map_radius: float = 50,
                 scenario_type_filter: List[str] = None,
                 generate_cache = False,
                 force_cache = False,
                 multiprocessing_cache = False,
                 num_cache_wokers = 16,
                 **kwargs) -> None:
        """
        raw_dataset_root: the directory holds the dataset. 
        split: which split is to use
        cache_dataset_root: the directory holds the cashed features. Please make sure
            you have the write permission of the given directory. If not provided, 
            by default it creates a subdirectory `cached` under the `raw_dataset_root`.
            This cache directory will have the same format as `raw_dataset_root`
        map_root: the directory holds map files. If not provided, by default it uses `map`
            under the `raw_dataset_root`.
        map_radius: the radius on map considered
        scenario_type_filter: a list of strings of scenarios considered. If None, use
            all scenario types.
        generate_cache: if True, generate cache for the given split
        force_cache: if True, overwrite the existing cache
        multiprocessing_cache: if True, cache data by multiprocessing
        num_cache_workers: number of threads will used if multiprocessing_cache=True
        """
        self._raw_dataset_root = raw_dataset_root
        self._split = split
        self._generate_cache = generate_cache
        self._force_cache = force_cache
        self._num_cache_workers = num_cache_wokers
        self._multiprocessing_cache = multiprocessing_cache

        # scenario type configs
        self._scenairo_type_filter = scenario_type_filter

        if self._split not in ["mini", "train", "val", "test"]:
            raise ValueError(f"Split '{split}' is not valid")

        # load metadata first
        # this step will determine (filter) the needed files
        # a DataFrame with columns "filename" and "scenario_tag"
        self._dataset_meta = self._load_meta(self._raw_dataset_root)
        
        # check raw dir
        self._split_dir = os.path.join(self._raw_dataset_root, self._split)
        if not os.path.exists(self._split_dir):
            raise FileNotFoundError(f"{self._split_dir} not found")

        # set up cache dir
        if cache_dataset_root is None:
            self._cached_dataset_root = os.path.join(self._raw_dataset_root, "cache")
        else:
            self._cached_dataset_root = cache_dataset_root
            if not os.path.exists(self._cached_dataset_root):
                os.makedirs(self._cached_dataset_root, exist_ok=False)
        self._cached_split_dir = os.path.join(self._cached_dataset_root, self._split)

        # if needed (usually in pre-processing), will load map api
        self._map_api = None
        self._map_root = map_root
        self._map_radius = map_radius

        # save other configs
        self._kwargs = kwargs
        super(MotionalDataset, self).__init__(raw_dataset_root)

    

    def _load_meta(self, raw_dataset_root: str) -> pd.DataFrame:
        """
            raw_dataset_root: place to hold raw dir
            return: a pd.Dataframe with columns `filename` and `scenario_tag`
                    that satisfy the `self._scenario_type_filter`. As the column
                    names suggest, this object stores all filenames of samples and
                    their corresponding scenario tag
        """
        # by default use the metadata dir under raw_dataset_root
        metadata_split_dir = os.path.join(raw_dataset_root, "metadata", self._split)
        if not os.path.exists(metadata_split_dir):
            raise FileNotFoundError(f"Metadata dir {metadata_split_dir} does not exsist")

        # metadata_paths = [os.path.join(metadata_split_dir, fn) for fn in os.listdir(metadata_split_dir) if fn.endswith(".parquet")]
        metadata_paths = [os.path.join(metadata_split_dir, f"{self._split}_meta.parquet")]
        dataset_meta_list = []
        for p in metadata_paths:
            log_meta = pd.read_parquet(p)
            dataset_meta_list.append(log_meta)

        dataset_meta = pd.concat(dataset_meta_list,
                                    ignore_index=True)
        dataset_meta = dataset_meta.sort_values(by = ['filename'], ascending = [True])
        dataset_meta.reset_index(drop=True)

        if self._scenairo_type_filter is not None and len(self._scenairo_type_filter) > 0:
            dataset_meta = dataset_meta[dataset_meta["scenario_tag"].isin(self._scenairo_type_filter)]

        del dataset_meta_list
        return dataset_meta

    
    def _get_raw_fn(self) -> List[str]:
        """
            get the filenames of the raw dataset
            Note: this does not contain the path
                this includes the file extension
                eg.: xxxxxx.parquet
        """
        return self._dataset_meta["filename"].tolist()

    def _get_cached_fn(self) -> List[str]:
        """
            get the filenames of currently cached
            Note: this does not contain the path
                this includes the file extension
                eg.: xxxxx.pt
        """
        if not os.path.exists(self._cached_split_dir):
            return []
        return os.listdir(self._cached_split_dir)


    def process_one(self, fn: str) -> int:
        """
        preprocessing (caching) logic of a single sample
        This is an example about preprocessing samples used by HiVT
        and has snippets borrowed from there
        """
        fn_woext = os.path.splitext(fn)[0]
        save_path = os.path.join(self._cached_split_dir, f"{fn_woext}.pt")
        trajectories = pd.read_parquet(os.path.join(self._split_dir, fn))
        # -----below is the preprocessing logic
        city_name = trajectories["city"].unique().tolist()[0]
        # get all vehicles for example
        agents = trajectories[trajectories["category_name"] == "vehicle"]
        # choose to use their timestamp, track_token, x, y, z, category_name
        agents = agents[["timestamp", "track_token", "x", "y", "category_name"]]
        timestamps = agents["timestamp"].unique()
        # do a downsample to frequency 10Hz
        timestamps = timestamps[1::2]

        # input/target split
        # target is the last 8s (80 timestamps after downsample to 10Hz)
        # input is the remaining (beginning 2s, 20 steps)
        input_timestamps = timestamps[:20].tolist()
        target_timestamps = timestamps[20:].tolist()
        input_agents = agents[agents["timestamp"].isin(input_timestamps)]
        target_agents = agents[agents["timestamp"].isin(target_timestamps)]

        # define valid agents by selecting agent in the present frame (the last frame of input)
        present_agents = input_agents[input_agents["timestamp"] == input_timestamps[-1]]
        present_agents_id = present_agents["track_token"].tolist() # will used to generate prediction result file. One should be careful about the ordering match between this and the features
        # filter target agents, needs to be valid, i.e. at least appear in present
        if len(target_agents) > 0: # in case dealing with test set
            target_agents = target_agents[target_agents["track_token"].isin(present_agents_id)]

        # just randomly assign an agent as origin
        # used for coordinate transform in training
        # needs to have at least two frames to determine the direction
        # here we always select the agent having most frames
        origin_agent_id = input_agents.groupby("track_token").size().idxmax()
        origin_agent = input_agents[input_agents["track_token"] == origin_agent_id]
        origin_coord = torch.tensor([origin_agent["x"].iloc[-1], origin_agent["y"].iloc[-1]], dtype=torch.float)
        if len(origin_agent["timestamp"].unique()) < 2:
            origin_heading = origin_coord - origin_coord
        else:
            origin_heading = origin_coord - torch.tensor([origin_agent["x"].iloc[-2], origin_agent["y"].iloc[-2]], dtype=torch.float)
        theta = torch.atan2(origin_heading[1], origin_heading[0])
        rotation = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                                [torch.sin(theta), torch.cos(theta)]])
        
        # preprocess input features
        num_agents = len(present_agents)
        input_agents_feature = torch.zeros((num_agents, len(input_timestamps), 2), dtype=torch.float) # input agent features (x, y)
        padding_mask = torch.ones((num_agents, len(input_timestamps)+len(target_timestamps)), dtype=torch.bool) # 1 means masked
        bos_mask = torch.zeros(num_agents, 20, dtype=torch.bool)
        rotate_angles = torch.zeros(num_agents, dtype=torch.float)
        target_agents_coords = torch.full((num_agents, len(target_timestamps), 2), torch.nan, dtype=torch.float) # gt trajectories
        # prefill nan to pad samples without gt
        
        for id, agent in present_agents.groupby("track_token"):
            index = present_agents_id.index(id)
            agent_appear = [input_timestamps.index(ts) for ts in agent['timestamp']]
            padding_mask[index, agent_appear] = False
            
            trajectory = torch.from_numpy(np.stack([agent['x'].values, agent['y'].values], axis=-1)).float()
            transformed_trajectory = torch.matmul(trajectory - origin_coord, rotation)
            input_agents_feature[index, agent_appear] = transformed_trajectory
            min_agent_appear = min(agent_appear)
            input_agents_feature[index, :min_agent_appear] = input_agents_feature[index, min_agent_appear] # padding with oldest step

            if min_agent_appear < len(input_timestamps) - 1:  # calculate the heading of the actor (approximately)
                heading_vector = input_agents_feature[index, agent_appear[-1]] - \
                                    input_agents_feature[index, agent_appear[-2]]
                rotate_angles[index] = torch.atan2(heading_vector[1], heading_vector[0])
            else:  # make no predictions for the actor if the number of valid time steps is less than 2
                padding_mask[index, len(input_agents_feature):] = True

        if len(target_agents) > 0:
            for id, agent in target_agents.groupby("track_token"):
                index = present_agents_id.index(id) # align input and target -- the same index means the same agent
                target_agent_appear = [target_timestamps.index(ts) for ts in agent['timestamp']]
                trajectory = torch.from_numpy(np.stack([agent['x'].values, agent['y'].values], axis=-1)).float()
                target_agents_coords[index, target_agent_appear] = trajectory

        # get map features using map api
        # this is an example and can be changed to any map feature you like
        # do this convertion to bridge the gap between two sets of names
        # see content of parquet_city_name_to_map_api_name() for more details
        map_api_city_name = parquet_city_name_to_map_api_name(city_name)
        present_agent_coords_before_transform = [Point2D(agent['x'], agent['y']) for _, agent in present_agents.iterrows()]
        vector_map_features = self.get_map_features_multiple_centers(map_api_city_name, 
                                present_agent_coords_before_transform, self._map_radius)
        num_lanes = len(vector_map_features["lane_coords"])
        vector_map_features["lane_coords"] = torch.matmul(vector_map_features["lane_coords"].reshape(num_lanes*2, 2) - origin_coord, 
                                                                rotation).reshape(num_lanes, 2, 2)
        lane_vectors = vector_map_features["lane_coords"][:, 1] - \
                        vector_map_features["lane_coords"][:, 0]

        agent_edge_index =  torch.LongTensor(list(permutations(range(num_agents), 2))).t().contiguous()
        positions = torch.cat([input_agents_feature, target_agents_coords], dim=1)
        
        is_intersection = torch.zeros(num_lanes, dtype=torch.uint8)
        turn_directions = torch.zeros(num_lanes, dtype=torch.uint8)
        traffic_controls = torch.zeros(num_lanes, dtype=torch.uint8)
        agent_indices = torch.arange(0, num_agents, dtype=torch.long)
        lane_actor_index = torch.LongTensor(list(product(
                                torch.arange(lane_vectors.size(0)), agent_indices
                            ))).t().contiguous()
        _lane_positions = (vector_map_features["lane_coords"][:, 1] + vector_map_features["lane_coords"][:, 0]) / 2.
        _node_positions = input_agents_feature[:, -1] # [N, 2]
        lane_actor_vectors = _lane_positions.repeat_interleave(num_agents, dim=0) - \
                                _node_positions.repeat(lane_vectors.size(0), 1)
        LOCAL_RADIUS = 50 # can be converted into parameter and change on your own
        mask = torch.norm(lane_actor_vectors, p=2, dim=-1) < LOCAL_RADIUS
        lane_actor_index = lane_actor_index[:, mask]
        lane_actor_vectors = lane_actor_vectors[mask]
        data = {
            "x": input_agents_feature,          # Tensor [N, input_T, 2], dtype=torch.float
            "positions": positions,             # Tensor [N, input_T+target_T, 2], dtype=torch.float (don't know why needed for HiVT)
            "edge_index": agent_edge_index,     # Tensor [2, N*(N-1)], dtype=torch.long
            "y": target_agents_coords,          # Tensor [N, target_T], dtype=torch.float
            "num_nodes": num_agents,            # int, N = num_agents
            "padding_mask": padding_mask,       # Tensor [N, input_T + target_T], dtype=torch.bool
            "bos_mask": bos_mask,               # Tensor [N, 20], dtype=torch.bool
            "rotate_angles": rotate_angles,     # Tensor [N], dtype=torch.float
            "lane_vectors": lane_vectors,       # Tensor [L, 2], dtype=torch.float
            "is_intersections": is_intersection,
            "turn_directions": turn_directions, 
            "traffic_controls": traffic_controls,
            "lane_actor_index": lane_actor_index,
            "lane_actor_vectors": lane_actor_vectors,
            "city": map_api_city_name,
            "origin": origin_coord.unsqueeze(0),
            "theta": theta,
            "sample_name": [fn_woext],       # needed in testing to generate result file
            "agents_count": [num_agents],    # needed in testing to generate result file
            "agents_id": present_agents_id   # List[str], used for note agents id and generate prediction result file for testing
        }
        data = TemporalData(**data)
        torch.save(data, save_path)
        # -----preprocessing logic end
        return 0

    def get_map_features_multiple_centers(self, city_name: str, coords: List[Point2D], radius: float) -> Dict:
        """
        get vector map from multiple local regions and merge
        city_name: city name to use map api
        coords: center of queried range
        radius: radius of queried map range
        """
        merged_lane_seg_coords = []
        merged_lane_seg_conns = []
        merged_lane_seg_ids = []

        for coord in coords:
            lane_seg_coords, lane_seg_conns, _, lane_seg_lane_ids, _ = get_neighbor_vector_map(self._map_api[city_name], coord, radius)
            merged_lane_seg_coords, merged_lane_seg_conns, merged_lane_seg_ids = merge_vector_map(
                merged_lane_seg_coords, merged_lane_seg_conns, merged_lane_seg_ids,
                lane_seg_coords, lane_seg_conns, lane_seg_lane_ids
            )

        return {
            "lane_coords": torch.from_numpy(np.stack([np.stack([np.array(lane[0]), np.array(lane[1])]) \
                                for lane in merged_lane_seg_coords])).float(), # a Tensor [N, 2, 2], where N is the number of lanes
            "lane_conns": torch.from_numpy(np.stack([np.array(conn) \
                                for conn in merged_lane_seg_conns])).float(),   # a Tensor [E, 2], where E is the number of connections
        }

    def process(self, 
                force_cache: bool=False, 
                multi_processing: bool=False, 
                num_cache_workers: int=16):
        """
        preprocess (cache) the features
        force_cache: if True, forced to overwrite any existing cached sample
        """
        raw_fn = self._get_raw_fn()
        cached_fn = self._get_cached_fn()

        if force_cache:
            file_names_to_cache = raw_fn
            logger.info(f"force caching all {len(file_names_to_cache)} samples in {self._split} set")
        else:
            if len(cached_fn) > 0:
                raw_fn_set = set([os.path.splitext(f)[0] for f in raw_fn])
                cached_fn_set = set([os.path.splitext(f)[0] for f in cached_fn])
                cached = raw_fn_set.intersection(cached_fn_set)
                if len(cached) == len(raw_fn_set):
                    logger.info(f"all samples are already cached in {self._split} set")
                    return
                else:
                    file_names_to_cache = [f"{fn}.parquet" for fn in raw_fn_set.difference(cached)] # note: add extension back
                    logger.info(f"continue caching {len(file_names_to_cache)} samples in {self._split} set")
            else:
                file_names_to_cache = raw_fn # note: with file extention, i.e. ".parquet"
                logger.info(f"caching all {len(file_names_to_cache)} samples in {self._split} set")

        if not os.path.exists(self._cached_split_dir):
            os.makedirs(self._cached_split_dir, exist_ok=False)
            logger.info(f"{self._cached_split_dir} does not exists. Created")

        # load map apis
        # to use one api of a city: self._map_api[name]
        logger.info("loading map apis.")
        self._map_api = get_all_maps_api(self._map_root, self.ALL_CITIES)
        logger.info("map api loading done.")
        logger.info("caching...... please hold tight (no progress bar when using multiprocessing)")
        if multi_processing:
            from multiprocessing import get_context
            with get_context("spawn").Pool(processes=num_cache_workers) as pool:
                results = pool.map(self.process_one, file_names_to_cache)
        else:
            for fn in tqdm(file_names_to_cache):
                self.process_one(fn)

        
    def len(self) -> int:
        return len(self._dataset_meta)

    def get(self, idx) -> Data:
        d = torch.load(os.path.join(self._cached_split_dir, self._dataset_meta["filename"].iloc[idx].replace(".parquet", ".pt")))
        return d

    # override pytorch geometric base class method
    # use our own process function
    def _process(self) -> None:
        """
        entry point of preprocessing (caching) samples
        """
        if self._generate_cache:
            self.process(force_cache=self._force_cache, 
                        multi_processing=self._multiprocessing_cache,
                        num_cache_workers=self._num_cache_workers)
    

    # override pytorch geometric base class method
    # ignore this function
    def _download(self) -> None:
        return
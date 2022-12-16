
import os, time


import sqlite3
import pandas as pd
from typing import Optional, Dict, List, Union

# from nuplan.database.common.blob_store.creator import BlobStoreCreator
from _db_converter.blob_store_creator import BlobStoreCreator
from _db_converter.blob_store.local_store import LocalStore


def pd_read_db(log_file, query):
    con = sqlite3.connect(log_file)
    df = pd.read_sql_query(query, con)
    return df

def get_log_metadata_from_db(log_file):
    query = f"""
        SELECT * FROM log
    """
    return pd_read_db(log_file, query)

def get_all_scenario_tag_from_db(log_file):
    query = f"""
        SELECT * FROM scenario_tag
    """
    return pd_read_db(log_file, query)

def get_all_scene_from_db(log_file):
    query = f"""
        SELECT * FROM scene
    """
    return pd_read_db(log_file, query)


def get_all_lidar_pc_from_db(log_file):
    query = f"""
            SELECT *
            FROM lidar_pc
            INNER JOIN scene
            ORDER BY scene_token ASC, timestamp ASC;
            """
    return pd_read_db(log_file, query)


def get_all_ego_from_db(log_file):
    query = f"""
            WITH
            ordered AS
            (
                SELECT  lp.token,
                        lp.next_token,
                        lp.prev_token,
                        lp.ego_pose_token,
                        lp.lidar_token,
                        lp.scene_token,
                        lp.filename,
                        lp.timestamp,
                        ROW_NUMBER() OVER (ORDER BY lp.timestamp ASC) AS row_num
                FROM lidar_pc AS lp
            )
            SELECT  ep.x,
                    ep.y,
                    ep.z,
                    ep.qw,
                    ep.qx,
                    ep.qy,
                    ep.qz,
                    -- ego_pose and lidar_pc timestamps are not the same, even when linked by token!
                    -- use the lidar_pc timestamp for compatibility with older code.
                    o.timestamp,
                    ep.vx,
                    ep.vy,
                    ep.vz,
                    ep.acceleration_x,
                    ep.acceleration_y,
                    ep.acceleration_z,
                    ep.token,
                    o.lidar_token,
                    s.type
            FROM ego_pose AS ep
            INNER JOIN ordered AS o
                ON o.ego_pose_token = ep.token
            LEFT JOIN scenario_tag AS s
                ON s.lidar_pc_token = o.lidar_token

            ORDER BY o.timestamp ASC;
            """
    # do not use timestamp in ego_pose, use lidar_pc's
    return pd_read_db(log_file, query)

def get_all_lidar_box_with_attributes_from_db(log_file):
    query = f"""
            SELECT
                anchor_c.name AS category_name,
                lb.x,
                lb.y,
                lb.z,
                lb.yaw,
                anchor_t.width,
                anchor_t.length,
                anchor_t.height,
                lb.vx,
                lb.vy,
                lb.vz,
                lb.track_token,
                anchor_lp.timestamp
            FROM lidar_box AS lb
            INNER JOIN track AS anchor_t
                ON anchor_t.token = lb.track_token
            INNER JOIN category AS anchor_c
                ON anchor_c.token = anchor_t.category_token
            INNER JOIN lidar_pc AS anchor_lp
                ON anchor_lp.token = lb.lidar_pc_token
            ORDER BY anchor_lp.timestamp ASC;
    """
    return pd_read_db(log_file, query)


def get_scenarios_from_db(
    log_file: str,
    filter_tokens: Optional[List[str]] = None,
    filter_types: Optional[List[str]] = None,
    filter_map_names: Optional[List[str]] = None,
    include_invalid_mission_goals: bool = True,
):
    """
    Get the scenarios present in the db file that match the specified filter criteria.
    If a filter is None, then it will be elided from the query.
    Results are sorted by timestamp ascending
    :param log_file: The log file to query.
    :param filter_tokens: If provided, the set of allowable tokens to return.
    :param filter_types: If provided, the set of allowable scenario types to return.
    :param map_names: If provided, the set of allowable map names to return.
    :param include_invalid_mission_goals: If true, then scenarios without a valid mission goal will be included
        (i.e. get_mission_goal_for_lidarpc_token_from_db(token) returns None)
        If False, then these scenarios will be filtered.
    :return: A sqlite3.Row object with the following fields:
        * token: The initial lidar_pc token of the scenario.
        * timestamp: The timestamp of the initial lidar_pc of the scenario.
        * map_name: The map name from which the scenario came.
        * scenario_type: One of the mapped scenario types for the scenario.
            This can be None if there are no matching rows in scenario_types table.
            If there are multiple matches, then one is selected from the set of allowable filter clauses at random.
    """
    filter_clauses = []
    args: List[Union[str, bytearray]] = []
    if filter_types is not None:
        filter_clauses.append(
            f"""
        st.type IN ({('?,'*len(filter_types))[:-1]})
        """
        )
        args += filter_types

    if filter_tokens is not None:
        filter_clauses.append(
            f"""
        lp.token IN ({('?,'*len(filter_tokens))[:-1]})
        """
        )
        args += [bytearray.fromhex(t) for t in filter_tokens]

    if filter_map_names is not None:
        filter_clauses.append(
            f"""
        l.map_version IN ({('?,'*len(filter_map_names))[:-1]})
        """
        )
        args += filter_map_names

    if len(filter_clauses) > 0:
        filter_clause = "WHERE " + " AND ".join(filter_clauses)
    else:
        filter_clause = ""

    if include_invalid_mission_goals:
        invalid_goals_joins = ""
    else:
        invalid_goals_joins = """
        ---Join on ego_pose to filter scenarios that do not have a valid mission goal
        INNER JOIN scene AS invalid_goal_scene
            ON invalid_goal_scene.token = lp.scene_token
        INNER JOIN ego_pose AS invalid_goal_ego_pose
            ON invalid_goal_scene.goal_ego_pose_token = invalid_goal_ego_pose.token
        """

    query = f"""
        WITH ordered_scenes AS
        (
            SELECT  token,
                    ROW_NUMBER() OVER (ORDER BY name ASC) AS row_num
            FROM scene
        ),
        num_scenes AS
        (
            SELECT  COUNT(*) AS cnt
            FROM scene
        ),
        valid_scenes AS
        (
            SELECT  o.token
            FROM ordered_scenes AS o
            CROSS JOIN num_scenes AS n

            -- Define "valid" scenes as those that have at least 2 before and 2 after
            -- Note that the token denotes the beginning of a scene
            WHERE o.row_num >= 3 AND o.row_num < n.cnt - 1
        )
        SELECT  lp.token,
                lp.timestamp,
                l.map_version AS map_name,

                -- scenarios can have multiple tags
                -- Pick one arbitrarily from the list of acceptable tags
                MAX(st.type) AS scenario_type
        FROM lidar_pc AS lp
        LEFT OUTER JOIN scenario_tag AS st
            ON lp.token = st.lidar_pc_token
        INNER JOIN lidar AS ld
            ON ld.token = lp.lidar_token
        INNER JOIN log AS l
            ON ld.log_token = l.token
        INNER JOIN valid_scenes AS vs
            ON lp.scene_token = vs.token
        {invalid_goals_joins}
        {filter_clause}
        GROUP BY    lp.token,
                    lp.timestamp,
                    l.map_version
        ORDER BY lp.timestamp ASC;
    """

    return pd_read_db(log_file, query)


def absolute_path_to_log_name(absolute_path: str) -> str:
    """
    Gets the log name from the absolute path to a log file.
    E.g.
        input: data/sets/nuplan/nuplan-v1.1/mini/2021.10.11.02.57.41_veh-50_01522_02088.db
        output: 2021.10.11.02.57.41_veh-50_01522_02088
        input: /tmp/abcdef
        output: abcdef
    :param absolute_path: The absolute path to a log file.
    :return: The log name.
    """
    filename = os.path.basename(absolute_path)

    # Files generated during caching do not end with ".db"
    # They have no extension.
    if filename.endswith(".db"):
        filename = os.path.splitext(filename)[0]
    return filename

def download_file_if_necessary(data_root: str, potentially_remote_path: str, verbose: bool = False) -> str:
    """
    Downloads the db file if necessary.
    :param potentially_remote_path: The path from which to download the file.
    :param verbose: Verbosity level.
    :return: The local path for the file.
    """
    # If the file path is a local directory and exists, then return that.
    # e.g. /data/sets/nuplan/nuplan-v1.1/file.db
    if os.path.exists(potentially_remote_path):
        return potentially_remote_path

    log_name = absolute_path_to_log_name(potentially_remote_path)
    download_name = log_name + ".db"

    # print (log_name)

    # TODO: CacheStore seems to be buggy.
    # Behavior seems to be different on our cluster vs locally regarding downloaded file paths.
    #
    # Use the underlying stores manually.

    blob_store = BlobStoreCreator.create_nuplandb("/data/cache/", verbose=verbose) # the first arg is cache dir
    local_store = LocalStore(data_root)

    # Only trigger the download if we have not already acquired the file.
    download_path_name = os.path.join(data_root, download_name)

    took_time = 0
    if not local_store.exists(download_name):
        # If we have no matches, download the file.
        # print("DB path not found. Downloading to %s..." % download_name)
        start_time = time.time()
        content = blob_store.get(potentially_remote_path)
        local_store.put(download_name, content)
        took_time = time.time() - start_time
        # print("Downloading db file took %.2f seconds." % (took_time))

    return download_path_name, took_time


def put_file_to_s3(s3_path, fn, content, verbose=False):
    blob_store = BlobStoreCreator.create_s3(s3_path, verbose=verbose)
    blob_store.put(os.path.join(s3_path, fn), content)

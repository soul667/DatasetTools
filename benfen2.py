import argparse
import gc
import shutil
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from pathlib import Path

import numpy as np
import ray
import torch
from agibot_utils.agibot_utils import get_task_info, load_local_dataset
from agibot_utils.config import AgiBotWorld_TASK_TYPE
from agibot_utils.lerobot_utils import compute_episode_stats, generate_features_from_config
from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    check_timestamps_sync,
    get_episode_data_index,
    validate_episode_buffer,
    validate_frame,
    write_episode,
    write_episode_stats,
    write_info,
)
from lerobot.datasets.video_utils import get_safe_default_codec
from ray.runtime_env import RuntimeEnv

from dataclasses import dataclass
import json
import os

@dataclass
class Dataset:
    # 从一个json文件中读取的数据信息
    RootPath: str = None  # 数据集根目录
    Dirs: list[str] = None  # 数据集目录列表
    files: list[str] = None  # 数据集文件列表
    DataInfo: dict = None
    MetaInfo: dict = None  # 元数据信息
    StateInfo: dict = None  # 状态信息
    # 一个从FrameIndex:int 映射到task：String的映射
    TaskIndexToText: dict = None  # 任务索引到任务文本的映射
    FrameIndexToTask: dict = None  # 帧索引到任务的映射
    VideoPath: Path = None  # 视频路径

class AgiBotDatasetMetadata(LeRobotDatasetMetadata):
    def save_episode(
        self,
        episode_index: int,
        episode_length: int,
        episode_tasks: list[str],
        episode_stats: dict[str, dict],
        action_config: list[dict],
    ) -> None:
        self.info["total_episodes"] += 1
        self.info["total_frames"] += episode_length

        # chunk = self.get_episode_chunk(episode_index)
        
        # if chunk >= self.total_chunks:
        #     self.info["total_chunks"] += 1

        # self.info["splits"] = {"train": f"0:{self.info['total_episodes']}"}
        # self.info["total_videos"] += len(self.video_keys)
        # if len(self.video_keys) > 0:
        #     self.update_video_info()
        # print('video info:', self.info['video_info'])
        # write_info(self.info, self.root)

        # episode_dict = {
        #     "episode_index": episode_index,
        #     "tasks": episode_tasks,
        #     "length": episode_length,
        #     "action_config": action_config,
        # }
        # self.episodes[episode_index] = episode_dict
        # write_episode(episode_dict, self.root)

        # self.episodes_stats[episode_index] = episode_stats
        # self.stats = aggregate_stats([self.stats, episode_stats]) if self.stats else episode_stats
        # write_episode_stats(episode_index, episode_stats, self.root)


class AgiBotDataset(LeRobotDataset):
    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        features: dict,
        root: str | Path | None = None,
        robot_type: str | None = None,
        use_videos: bool = True,
        tolerance_s: float = 1e-4,
        image_writer_processes: int = 0,
        image_writer_threads: int = 0,
        video_backend: str | None = None,
    ) -> "LeRobotDataset":
        """Create a LeRobot Dataset from scratch in order to record data."""
        obj = cls.__new__(cls)
        obj.meta = AgiBotDatasetMetadata.create(
            repo_id=repo_id,
            fps=fps,
            robot_type=robot_type,
            features=features,
            root=root,
            use_videos=use_videos,
        )
        obj.repo_id = obj.meta.repo_id
        obj.root = obj.meta.root
        obj.revision = None
        obj.tolerance_s = tolerance_s
        obj.image_writer = None

        if image_writer_processes or image_writer_threads:
            obj.start_image_writer(image_writer_processes, image_writer_threads)

        # TODO(aliberts, rcadene, alexander-soare): Merge this with OnlineBuffer/DataBuffer
        obj.episode_buffer = obj.create_episode_buffer()

        obj.episodes = None
        obj.hf_dataset = obj.create_hf_dataset()
        obj.image_transforms = None
        obj.delta_timestamps = None
        obj.delta_indices = None
        obj.episode_data_index = None
        obj.video_backend = video_backend if video_backend is not None else get_safe_default_codec()
        return obj

    def add_frame(self, frame: dict, task: str, timestamp: float | None = None) -> None:
        """
        This function only adds the frame to the episode_buffer. Apart from images — which are written in a
        temporary directory — nothing is written to disk. To save those frames, the 'save_episode()' method
        then needs to be called.
        """
        # Convert torch to numpy if needed
        for name in frame:
            if isinstance(frame[name], torch.Tensor):
                frame[name] = frame[name].numpy()

        features = {key: value for key, value in self.features.items() if key in self.hf_features}  # remove video keys
        validate_frame(frame, features)

        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()

        # Automatically add frame_index and timestamp to episode buffer
        frame_index = self.episode_buffer["size"]
        if timestamp is None:
            timestamp = frame_index / self.fps
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)
        self.episode_buffer["task"].append(task)

        # Add frame features to episode_buffer
        for key, value in frame.items():
            if key not in self.features:
                raise ValueError(
                    f"An element of the frame is not in the features. '{key}' not in '{self.features.keys()}'."
                )

            self.episode_buffer[key].append(value)

        self.episode_buffer["size"] += 1

    def save_episode(self, videos: dict, action_config: list, episode_data: dict | None = None) -> None:
        """
        This will save to disk the current episode in self.episode_buffer.

        Args:
            episode_data (dict | None, optional): Dict containing the episode data to save. If None, this will
                save the current episode in self.episode_buffer, which is filled with 'add_frame'. Defaults to
                None.
        """
        if not episode_data:
            episode_buffer = self.episode_buffer

        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

        # size and task are special cases that won't be added to hf_dataset
        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]
        # print(f"Saving episode {episode_index} with {episode_length} frames and tasks {episode_tasks}")

        episode_buffer["index"] = np.arange(self.meta.total_frames, self.meta.total_frames + episode_length)
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        # Add new tasks to the tasks dictionary
        for task in episode_tasks:
            task_index = self.meta.get_task_index(task)
            if task_index is None:
                self.meta.add_task(task)

        # Given tasks in natural language, find their corresponding task indices
        episode_buffer["task_index"] = np.array([self.meta.get_task_index(task) for task in tasks])

        for key, ft in self.features.items():
            # index, episode_index, task_index are already processed above, and image and video
            # are processed separately by storing image path and frame info as meta data
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["video"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key]).squeeze()

        for key in self.meta.video_keys:
            video_path = self.root / self.meta.get_video_file_path(episode_index, key)
            episode_buffer[key] = str(video_path)  # PosixPath -> str
            video_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(videos[key], video_path)

        ep_stats = compute_episode_stats(episode_buffer, self.features)

        self._save_episode_table(episode_buffer, episode_index)

        # `meta.save_episode` be executed after encoding the videos
        # add action_config to current episode
        self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats, action_config)

        ep_data_index = get_episode_data_index(self.meta.episodes, [episode_index])
        ep_data_index_np = {k: t.numpy() for k, t in ep_data_index.items()}
        check_timestamps_sync(
            episode_buffer["timestamp"],
            episode_buffer["episode_index"],
            ep_data_index_np,
            self.fps,
            self.tolerance_s,
        )

        if not episode_data:  # Reset the buffer
            self.episode_buffer = self.create_episode_buffer()

# DatasetLists: list[Dataset] = []
def get_all_tasks_sim(src_path: Path, output_path: Path):
    for root, dirs, files in os.walk(src_path):
            # print(f"Checking folder: {root}")
        if("task_train.json" not in files):
            continue
        json_file = Path(root) / "task_train.json"
        local_dir = output_path / "agibotworld" / "sim" / Path(root).name
        if not local_dir.exists():
            local_dir.mkdir(parents=True, exist_ok=True)
        # 处理的任务ID不对
        yield (json_file, local_dir.resolve())

def save_as_lerobot_dataset(agibot_world_config, task: tuple[Path, Path], num_threads, save_depth, debug):
    json_file, local_dir = task
    task_id= "2810137"  # TODO: FIX
    print(f"processing {task_id}, saving to {local_dir}")
    
    src_path = json_file.parent.parent
    task_info = get_task_info(json_file)
    task_name = task_info[0]["task_name"]
    task_init_scene = task_info[0]["init_scene_text"]
    task_instruction = f"{task_name} | {task_init_scene}"
    # task_id = json_file.stem.split("_")[-1]
    task_info = {episode["episode_id"]: episode for episode in task_info}

    features = generate_features_from_config(agibot_world_config)

    if local_dir.exists():
        shutil.rmtree(local_dir)

    # print(f'features: {features}')
    # if not save_depth:
    #     features.pop("observation.images.head_depth")

    dataset: AgiBotDataset = AgiBotDataset.create(
        repo_id= f"{task_id}",
        root=local_dir,
        fps=30,
        robot_type="a2d",
        features=features,
    )
    for root, dirs, files in os.walk(src_path):
        if "data_info.json" not in files:
            continue
        # 剩下的都是可以加入数据集的数据
        DataInfoPath = Path(root) / "data_info.json"
        MetaInfoPath = Path(root) / "meta_info.json"
        StateInfoPath = Path(root) / "parameters/camera/state.json"
        DatasetUse= Dataset(
            RootPath=root,
            Dirs=dirs,
            files=files,
            DataInfo=json.load(DataInfoPath.open()) if DataInfoPath.exists() else {},
            MetaInfo=json.load(MetaInfoPath.open()) if MetaInfoPath.exists() else {},
            StateInfo=json.load(StateInfoPath.open()) if StateInfoPath.exists() else {},
            TaskIndexToText={},
            FrameIndexToTask={},
            VideoPath=Path(root)
        )
        frames = DatasetUse.StateInfo.get("frames", [])
        #     pass
        AgiBotWorld_CONFIG=agibot_world_config
        state = {}
        action = {}

        # for key in AgiBotWorld_CONFIG["states"]:
        #     state[f"observation.states.{key}"] = np.array(f["state/" + key.replace(".", "/")], dtype=np.float32)
        # for key in AgiBotWorld_CONFIG["actions"]:
        #     action[f"actions.{key}"] = np.array(f["action/" + key.replace(".", "/")], dtype=np.float32)
        for frame_index, frame in enumerate(frames):
            # 处理每一帧的state和action
            DatasetUse.FrameIndexToTask = {}
            DatasetUse.TaskIndexToText = {}
            TaskIndex= 0
            print(f"Processing frame {frame_index+1}/{len(frames)} in {json_file.stem}")
            if "label_info" in DatasetUse.DataInfo and "action_config" in DatasetUse.DataInfo["label_info"]:
                for action in DatasetUse.DataInfo["label_info"]["action_config"]:
                    start_frame = action.get("start_frame", 0)
                    end_frame = action.get("end_frame", 0)
                    english_action_text = action.get("english_action_text", "Unknown")
                    DatasetUse.TaskIndexToText[TaskIndex] = english_action_text
                    for frame_index in range(start_frame, end_frame + 1):
                        DatasetUse.FrameIndexToTask[frame_index] = TaskIndex
                    TaskIndex += 1
            
            if(DatasetUse.FrameIndexToTask[frame_index] is None): # 如果这帧没任务就不添加
                continue
            
            joints = frame.get("robot", {}).get("joints", {})
            joint_names = joints.get("joint_name", [])
            joint_positions = joints.get("joint_position", [])
            joint_name_to_pos = dict(zip(joint_names, joint_positions))
            motor_joint_map = {
                "left_arm_0": "idx21_arm_l_joint1",
                "left_arm_1": "idx22_arm_l_joint2",
                "left_arm_2": "idx23_arm_l_joint3",
                "left_arm_3": "idx24_arm_l_joint4",
                "left_arm_4": "idx25_arm_l_joint5",
                "left_arm_5": "idx26_arm_l_joint6",
                "left_arm_6": "idx27_arm_l_joint7",
                "left_gripper": "idx31_gripper_l_inner_joint1",
                "right_arm_0": "idx61_arm_r_joint1",
                "right_arm_1": "idx62_arm_r_joint2",
                "right_arm_2": "idx63_arm_r_joint3",
                "right_arm_3": "idx64_arm_r_joint4",
                "right_arm_4": "idx65_arm_r_joint5",
                "right_arm_5": "idx66_arm_r_joint6",
                "right_arm_6": "idx67_arm_r_joint7",
                "right_gripper": "idx71_gripper_r_inner_joint1",
            }
        
            motor_names = list(motor_joint_map.keys())
            motor_positions = [joint_name_to_pos.get(motor_joint_map[m], 0.0) for m in motor_names]
            Timestamp = frame.get("time_stamp", 0.0)
            
            task_name = DatasetUse.TaskIndexToText.get(DatasetUse.FrameIndexToTask[frame_index], "Unknown")
            dataset.add_frame({
                    "observation.states.joint_position": np.array(motor_positions, dtype=np.float32),
            }, 
            task_name,
            Timestamp
            )
        videos = {}
        print(agibot_world_config)
        for key in agibot_world_config["images"]:
            print(f"Processing video for key: {key}")
            field_name = f"observation.images.{key}"
            if "sensor" in key:
                video_path = Path(DatasetUse.RootPath) / f"{key}.mp4"
            else:
                video_path = Path(DatasetUse.RootPath) / f"{key}_color.mp4"
            videos[field_name] = video_path
        action_config = DatasetUse.DataInfo["label_info"].get("action_config", [])
        print('videos:', videos)
        try:
            dataset.save_episode(videos=videos, action_config=action_config)
        except Exception as e:
            print(
                f"there are some corrupted mp4s\nException details: {str(e)}"
            )
            dataset.episode_buffer = None
            continue
        gc.collect()
        # 创建videos变量
            #         try:
            #             dataset.save_episode(videos=videos, action_config=action_config)
            #         except Exception as e:
            #             print(f"{json_file.stem}, episode_{eid}: there are some corrupted mp4s\nException details: {str(e)}")
            #             dataset.episode_buffer = None
            #             continue
            #         gc.collect()

            
            # print(motor_positions)


            # print(joint_position)
            # pass
            # state[f"observation.states.joint_position"] =
        # frames = [
        #     {
        #         **({}),
        #         **{
        #             key: value[i]
        #             if value.size
        #             else np.zeros(
        #                 AgiBotWorld_CONFIG["states"][key[state_key_prefix_len:]]["shape"],
        #                 dtype=AgiBotWorld_CONFIG["states"][key[state_key_prefix_len:]]["dtype"],
        #             )
        #             for key, value in state.items()
        #         },
        #         **{
        #             key: value[i]
        #             if value.size
        #             else np.zeros(
        #                 AgiBotWorld_CONFIG["actions"][key[action_key_prefix_len:]]["shape"],
        #                 dtype=AgiBotWorld_CONFIG["actions"][key[action_key_prefix_len:]]["dtype"],
        #             )
        #             for key, value in action.items()
        #         },
        #     }
        #     # for i in range(num_frames)
        # ]


    # print(f"{DatasetUse}")
    # # print(f"Found {len(all_subdir)} subdirectories in {src_path / 'observations' / task_id}")
    # print(f"Found {len(all_subdir)} subdirectories in {src_path}")
    # # all_subdir_eids = [int(subdir.name.split("_")[-1]) for subdir in all_subdir if subdir.name.startswith("episode_")]
    # all_subdir_eids = sorted([int(Path(path).name) for path in all_subdir])

    # print(f"Found {len(all_subdir_eids)} episode IDs in {src_path}")
    # for eid in all_subdir:
    #     path_str= str(eid)
    #     print(f"Processing {path_str}")
    #     DatasetUse= Dataset(
    #         RootPath=src_path,
    #         Dirs=[path_str],
    #         files=[],
    #         DataInfo={},
    #         MetaInfo={},
    #         StateInfo={},
    #         TaskIndexToText={},
    #         FrameIndexToTask={}
    #     )
    # if debug or not save_depth:
    #     for eid in all_subdir_eids:
    #         if eid not in task_info:
    #             print(f"{json_file.stem}, episode_{eid} not in task_info.json, skipping...")
    #             continue
    #         print(f"Processing episode_{eid} in {json_file.stem}")
    #         action_config = task_info[eid]["label_info"]["action_config"]
    #         raw_dataset = load_local_dataset(
    #             eid,
    #             src_path=src_path,
    #             task_id=task_id,
    #             save_depth=save_depth,
    #             AgiBotWorld_CONFIG=agibot_world_config,
    #         )
    #         _, frames, videos = raw_dataset
            
    #         if not all([video_path.exists() for video_path in videos.values()]):
    #             print(f"{json_file.stem}, episode_{eid}: some of the videos does not exist, skipping...")
    #             continue

    #         for frame_data in frames:
    #             dataset.add_frame(frame_data, task_instruction)
    #         try:
    #             dataset.save_episode(videos=videos, action_config=action_config)
    #         except Exception as e:
    #             print(f"{json_file.stem}, episode_{eid}: there are some corrupted mp4s\nException details: {str(e)}")
    #             dataset.episode_buffer = None
    #             continue
    #         gc.collect()
    #         print(f"process done for {json_file.stem}, episode_id {eid}, len {len(frames)}")
    # else:
    #     with ThreadPoolExecutor(max_workers=num_threads) as executor:
    #         futures = []
    #         for eid in all_subdir_eids:
    #             if eid not in task_info:
    #                 print(f"{json_file.stem}, episode_{eid} not in task_info.json, skipping...")
    #                 continue
    #             futures.append(
    #                 executor.submit(
    #                     load_local_dataset,
    #                     eid,
    #                     src_path=src_path,
    #                     task_id=task_id,
    #                     save_depth=save_depth,
    #                     AgiBotWorld_CONFIG=agibot_world_config,
    #                 )
    #             )

    #         for raw_dataset in as_completed(futures):
    #             eid, frames, videos = raw_dataset.result()
    #             if not all([video_path.exists() for video_path in videos.values()]):
    #                 print(f"{json_file.stem}, episode_{eid}: some of the videos does not exist, skipping...")
    #                 continue
    #             action_config = task_info[eid]["label_info"]["action_config"]
    #             for frame_data in frames:
    #                 dataset.add_frame(frame_data, task_instruction)
    #             try:
    #                 dataset.save_episode(videos=videos, action_config=action_config)
    #             except Exception as e:
    #                 print(
    #                     f"{json_file.stem}, episode_{eid}: there are some corrupted mp4s\nException details: {str(e)}"
    #                 )
    #                 dataset.episode_buffer = None
    #                 continue
    #             gc.collect()
    #             print(f"process done for {json_file.stem}, episode_id {eid}, len {len(frames)}")


def main(
    src_path: str,
    output_path: str,
    eef_type: str,
    task_ids: list,
    cpus_per_task: int,
    num_threads_per_task: int,
    save_depth: bool,
    debug: bool = False,
):
    #print(f'debug mode: {debug}, save_depth: {save_depth}')
    tasks = list(get_all_tasks_sim(src_path, output_path)) # 原来返回的是一次性迭代器
    print(f"Total tasks found: {len(tasks)}")

    agibot_world_config, type_task_ids = (
        AgiBotWorld_TASK_TYPE[eef_type]["task_config"],
        AgiBotWorld_TASK_TYPE[eef_type]["task_ids"],
    )
    # sim 不用筛选
    if eef_type == "sim":
        pass
    else:
        print(f"Using AgiBotWorld config for {eef_type}: {agibot_world_config}")
        if eef_type == "gripper":
            remaining_ids = AgiBotWorld_TASK_TYPE["dexhand"]["task_ids"] + AgiBotWorld_TASK_TYPE["tactile"]["task_ids"]
            tasks = filter(lambda task: task[0].stem not in remaining_ids, tasks)
        else:
            tasks = filter(lambda task: task[0].stem in type_task_ids, tasks)

        if task_ids:
            tasks = filter(lambda task: task[0].stem in task_ids, tasks)

    if debug:
        # print(f"Debug mode enabled, processing only the first task: {tasks}")
        save_as_lerobot_dataset(agibot_world_config,tasks[0], num_threads_per_task, save_depth, debug)
    else:
        runtime_env = RuntimeEnv(
            env_vars={
                "HDF5_USE_FILE_LOCKING": "FALSE",
                "HF_DATASETS_DISABLE_PROGRESS_BARS": "TRUE",
                "LD_PRELOAD": str(Path(__file__).resolve().parent / "libtcmalloc.so.4.5.3"),
            }
        )
        ray.init(runtime_env=runtime_env)
        resources = ray.available_resources()
        cpus = int(resources["CPU"])
        # print(f"Total tasks found: {len(list(tasks))}")

        print(f"Available CPUs: {cpus}, num_cpus_per_task: {cpus_per_task}")

        remote_task = ray.remote(save_as_lerobot_dataset).options(num_cpus=cpus_per_task)
        futures = []
        
        for task in tasks:
            futures.append(
                (task[0].stem, remote_task.remote(agibot_world_config, task, num_threads_per_task, save_depth, debug))
            )

        for task, future in futures:
            try:
                ray.get(future)
            except Exception as e:
                print(f"Exception occurred for {task}")
                with open("output.txt", "a") as f:
                    f.write(f"{task}, exception details: {str(e)}\n")

        ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--eef-type", type=str, choices=["gripper", "dexhand", "tactile" , "sim"], default="gripper")
    parser.add_argument("--task-ids", type=str, nargs="+", help="task_327 task_351 ...", default=[])
    parser.add_argument("--cpus-per-task", type=int, default=3)
    parser.add_argument("--num-threads-per-task", type=int, default=2)
    parser.add_argument("--save-depth", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    main(**vars(args))
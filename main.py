from dataclasses import dataclass
import tyro
import toml
import json
import os
from typing import Optional
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata,LeRobotDataset
@dataclass
class Config:
    SourceDatasetPath: str = "dataset"
    OutDatasetPath: str = "outdataset"
    ImageWriterThreads: int = 4 # 处理线程数
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


def load_config(path: Optional[str] = None) -> Config:
    if path:
        data = toml.load(path)
        cli_args = []
        for k, v in data.items():
            cli_args += [f"--{k}", str(v)]
        return tyro.cli(Config, args=cli_args)
    else:
        return tyro.cli(Config)

config = load_config("config.toml")

# 每一个数据集的路径
DatasetLists: list[Dataset] = []

def find_folders(base_path: str):
    # ExpectFolders = ["dataset", ".venv"]
    global DatasetLists
    for root, dirs, files in os.walk(base_path):
        # print(f"Checking folder: {root}")
        if("camera" not in dirs or "parameters" not in dirs or "head_color.mp4" not in files):
            continue
        print(f"Found folder Root: {os.path.join(root)}")
        DatasetUse=Dataset(RootPath=root, Dirs=dirs, files=files)
        try:
            DatasetUse.DataInfo=json.load(open(os.path.join(root, "data_info.json"), "r", encoding="utf-8"))
            DatasetUse.MetaInfo=json.load(open(os.path.join(root, "meta_info.json"), "r", encoding="utf-8"))
            DatasetUse.StateInfo=json.load(open(os.path.join(root, "parameters/camera/state.json"), "r", encoding="utf-8"))
            # print(f"{DatasetUse.StateInfo.get('schema')}")
            # print(f"{DatasetUse.StateInfo['cameras']}")
            # print(f"{DatasetUse.StateInfo.get('cameras')}")
            # 从DataInfo中获取帧索引到任务的映射
            """
             "label_info": {
            "error_label": "",
            "action_config": [
                {
                    "start_frame": 18,
                    "end_frame": 254,
                    "action_text": "右臂拿起台面上的葡萄汁",
                    "skill": "Pick",
                    "english_action_text": "Pick up the grape juice on the table with the right arm."
                },
                {
                    "start_frame": 255,
                    "end_frame": 553,
                    "action_text": "右臂将拿着的葡萄汁放进台面上的毛毡袋内",
                    "skill": "Place",
                    "english_action_text": "Put the grape juice into the felt bag on the table with the right arm."
                }
            ],
            "key_frame": {
                "signal": [],
                "dual": []
            },
            "cloud_post_processing_result": {
                "data_valid": true,
                "drop_frame_rate": 0.0,
                "filter_frame_rate": 0.0
            }
        },
            """
            DatasetUse.FrameIndexToTask = {}
            DatasetUse.TaskIndexToText = {}
            TaskIndex= 0
            if "label_info" in DatasetUse.DataInfo and "action_config" in DatasetUse.DataInfo["label_info"]:
                for action in DatasetUse.DataInfo["label_info"]["action_config"]:
                    start_frame = action.get("start_frame", 0)
                    end_frame = action.get("end_frame", 0)
                    english_action_text = action.get("english_action_text", "Unknown")
                    DatasetUse.TaskIndexToText[TaskIndex] = english_action_text
                    for frame_index in range(start_frame, end_frame + 1):
                        DatasetUse.FrameIndexToTask[frame_index] = TaskIndex
                    TaskIndex += 1
            # 输出 DatasetUse.FrameIndexToTask
            # print(f"FrameIndexToTask: {DatasetUse.FrameIndexToTask}")
        except FileNotFoundError:
            print(f"data and meta json not found in {root}, skipping...")
            continue
        DatasetLists.append(DatasetUse)

def check_create_folder(path: str) -> None:
    """Check if a folder exists, if not, create it."""
    try:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created folder: {path}")
        else:
            print(f"Folder already exists: {path}")
    except Exception as e:
        print(f"Error creating folder {path}: {e}")
    return

def create_metainfo(dataset: Dataset)->bool:
    """Convert Agibot Dataset to LeRobotDatasetMetadata format"""
    dataset_name = os.path.basename(dataset.RootPath)
    check_create_folder(config.OutDatasetPath)
    check_create_folder(config.OutDatasetPath + "/" + dataset_name)

    # print(f'root path: {dataset_name}')

def convert_to_lerobot_dataset(dataset: Dataset, config: Config) -> None:
    """Convert source dataset to LeRobotDataset format"""
    # Define features based on source data
    # print(Dataset.StateInfo)
    # print(f'camera info: {Dataset.StateInfo["cameras"]}')

    features = {
        # Camera parameters
        "observation.images.head": {
            "dtype": "video",
            "shape": (3, 
                      dataset.StateInfo['cameras']['head']['intrinsic']['height'], 
                      dataset.StateInfo['cameras']['head']['intrinsic']['width']),
            "names": ('channel', 'height', 'width'),
            'video_info':{
                "video.fps": dataset.MetaInfo.get("camera_fps",[30,30,30])[0],  # From meta_info.json
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False
            }
        },
        "observation.images.hand_right": {
            "dtype": "video",
            "shape": (3, 
                      dataset.StateInfo['cameras']['hand_right']['intrinsic']['height'], 
                      dataset.StateInfo['cameras']['hand_right']['intrinsic']['width']),
            "names": ('channel', 'height', 'width'),
            'video_info':{
                "video.fps": dataset.MetaInfo.get("camera_fps",[30,30,30])[1],  # From meta_info.json
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False
            }
        },
        "observation.images.hand_left": {
            "dtype": "video",
            "shape": (3, 
                      dataset.StateInfo['cameras']['hand_left']['intrinsic']['height'], 
                      dataset.StateInfo['cameras']['hand_left']['intrinsic']['width']),
            "names": ('channel', 'height', 'width'),
            'video_info':{
                "video.fps": dataset.MetaInfo.get("camera_fps",[30,30,30])[2],  # From meta_info.json
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False
            }
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [
                14
            ],
            "names": {
                "motors": [
                    "left_waist",
                    "left_shoulder",
                    "left_elbow",
                    "left_forearm_roll",
                    "left_wrist_angle",
                    "left_wrist_rotate",
                    "left_gripper",
                    "right_waist",
                    "right_shoulder",
                    "right_elbow",
                    "right_forearm_roll",
                    "right_wrist_angle",
                    "right_wrist_rotate",
                    "right_gripper"
                ]
            },
            "fps": 30
        },
        "action": {
            "dtype": "float32",
            "shape": [
                14
            ],
            "names": {
                "motors": [
                    "arm_l_joint1",
                    "arm_l_joint2",
                    "arm_l_joint3",
                    "arm_l_joint4",
                    "arm_l_joint5",
                    "arm_l_joint6",
                    "arm_l_joint7",
                    "arm_r_joint1",
                    "arm_r_joint2",
                    "arm_r_joint3",
                    "arm_r_joint4",
                    "arm_r_joint5",
                    "arm_r_joint6",
                    "arm_r_joint7"
                ]
            },
            "fps": 30.0
        },
        "timestamp": {
            "dtype": "float32",
            "shape": [
                1
            ],
            "names": None,
            "fps": 30.0
        },
        "episode_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": None,
            "fps": 30.0
        },
        "frame_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": None,
            "fps": 30.0
        },
        "next.done": {
            "dtype": "bool",
            "shape": [
                1
            ],
            "names": None,
            "fps": 30.0
        },
        "index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": None,
            "fps": 30.0
        },
        "task_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": None,
            "fps": 30.0
        }
    }
    
    # Creates metadata for a LeRobotDataset
    lerobot_dataset = LeRobotDataset.create(
        repo_id=f"robot/{dataset.DataInfo['sn_code']}", 
        fps=30,  # From meta_info.json
        features=features,
        root=config.OutDatasetPath,
        robot_type=dataset.DataInfo.get("robot_type", None),
        use_videos=True,
        image_writer_threads=config.ImageWriterThreads,
    )

    frames = dataset.StateInfo.get("frames", [])
    print(f"Total frames: {len(frames)}")
    # for frame in frames:
    for frame_index, frame in enumerate(frames):
        # 使用 add_frames?
        Timestamp = frame.get("time_stamp", 0.0)
        TaskName = dataset.FrameIndexToTask.get(frame_index, None)
        if TaskName is not None:
            TaskIndex = dataset.TaskIndexToText.get(TaskName, None)
            if TaskIndex is None:
                continue 
        else:
            continue        
        
        lerobot_dataset.add_frames(
            observation={
                "images.head": frame.get("head_image", None),
                "images.hand_right": frame.get("hand_right_image", None),
                "images.hand_left": frame.get("hand_left_image", None),
                "state": frame.get("state", None)
            },
            action=frame.get("action", None),
            timestamp=Timestamp,
            episode_index=0,  # Assuming single episode
            frame_index=frame_index,
            next_done=frame.get("next_done", False),
            index=frame_index,
            task_index=TaskIndex
        )
    
    # Process data and add frames
    # Handle video files and action data

# def add_task

if __name__ == "__main__":
    find_folders(config.SourceDatasetPath)
    for DatasetUse in DatasetLists:

        # print(f"Processing dataset: {DatasetUse.RootPath}")
        convert_to_lerobot_dataset(DatasetUse, config)
        create_metainfo(DatasetUse)
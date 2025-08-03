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
    ProcessCount: int = 4 # 处理进程数

@dataclass
class Dataset:
    # 从一个json文件中读取的数据信息
    RootPath: str = None  # 数据集根目录
    Dirs: list[str] = None  # 数据集目录列表
    files: list[str] = None  # 数据集文件列表
    DataInfo: dict = None
    MetaInfo: dict = None  # 元数据信息


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
        except FileNotFoundError:
            print(f"data and meta json not found in {root}, skipping...")
            continue
        DatasetLists.append(DatasetUse)

def convert_to_lerobot_dataset(dataset: Dataset, config: Config) -> None:
    """Convert source dataset to LeRobotDataset format"""
    # Define features based on source data
    features = {
        "observation.images.head": {
            "dtype": "video",
            "shape": (3, 720, 1280),  # Assuming standard RGB camera resolution
            "names": None
        },
        "observation.images.hand_right": {
            "dtype": "video",
            "shape": (3, 480, 848),
            "names": None
        },
        "observation.images.hand_left": {
            "dtype": "video",
            "shape": (3, 480, 848),
            "names": None
        }
    }
    
    # Create LeRobotDataset instance
    lerobot_dataset = LeRobotDataset.create(
        repo_id=f"robot/{dataset.DataInfo['sn_code']}", 
        fps=30,  # From meta_info.json
        features=features,
        root=config.OutDatasetPath,
        robot_type=dataset.DataInfo.get("robot_type", None),
        use_videos=True
    )
    
    # Process data and add frames
    # Handle video files and action data

if __name__ == "__main__":
    find_folders(config.SourceDatasetPath)
    for DatasetUse in DatasetLists:
        print(f"Processing dataset: {DatasetUse.RootPath}")
        convert_to_lerobot_dataset(DatasetUse, config)
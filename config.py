from pathlib import Path
from typing import Tuple
from typing import Union
import logging
import os


class Config:
    ROOT_PATH: str = Path(__file__).resolve().parent
    CLI_LOG_LEVEL: int = logging.DEBUG
    FILE_LOG_LEVEL: int = logging.DEBUG
    MODEL_NAME: str = "model92"
    CONF_THRESHOLD: float = .35
    SORT_MAX_AGE: int = 50
    SORT_MIN_HITS: int = 1
    SORT_IOU_THRESHOLD: float = .3
    THRESHOLD_SCALE_FACTOR: float = .8
    SYSTEM_DEBUG_MODE: float = True
    FRAME_RESIZE: Tuple[int, int] = (900, 720)
    RAW_FRAME_RESIZE: Tuple[int, int] = (900, 720)


    RAW_DATA_PATH: Union[str, os.PathLike, Path] = fr"{ROOT_PATH}/RawData"
    CLEANED_DATA_PATH: Union[str, os.PathLike, Path] = fr"{ROOT_PATH}/DataCleaned"
    TRAIN_IMAGES_FOLDER: Union[str, os.PathLike, Path] = fr"{ROOT_PATH}/train_data/images/train"
    VAL_IMAGES_FOLDER: Union[str, os.PathLike, Path] = fr"{ROOT_PATH}/train_data/images/val"
    TRAIN_LABELS_FOLDER: Union[str, os.PathLike, Path] = fr"{ROOT_PATH}/train_data/labels/train"
    VAL_LABELS_FOLDER: Union[str, os.PathLike, Path] = fr"{ROOT_PATH}/train_data/labels/val"
    TEST_DATA_FOLDER: Union[str, os.PathLike, Path] = fr"{ROOT_PATH}/TestData"
    VIDEOS_FOLDER: Union[str, os.PathLike, Path] =  fr"{ROOT_PATH}/Vidoes"
    MODELS_PATH: Union[str, os.PathLike, Path] = fr"{ROOT_PATH}/models"
    CLASSES_PATH: Union[str, os.PathLike, Path] = fr"{MODELS_PATH}/classes.txt"
    MIN_HEIGHT: int = 200
    MIN_WIDTH: int = 200
    MAX_HEIGHT: int = 4000
    MAX_WIDTH: int = 4000
    ALLOWED_EXTENSIONS: tuple = (".jpg", ".png", ".jpeg")

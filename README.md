# Forklift Safety Monitoring System

A computer vision system for monitoring forklift and pedestrian interactions in industrial environments to enhance workplace safety. The system uses YOLOv5 for object detection, SORT algorithm for object tracking, and Kalman filtering for motion prediction.

[![video](https://img.youtube.com/vi/d5-nabm2KUc/0.jpg)](https://www.youtube.com/watch?v=d5-nabm2KUc)
## Features

- Real-time detection of forklifts and pedestrians
- Object tracking with unique IDs
- Distance monitoring between forklifts and pedestrians
- Motion prediction using Kalman filtering
- Support for both image and video processing
- Configurable safety thresholds
- Visual alerts for safety violations
- Performance metrics display (FPS)
- Debug mode for development

## System Components

### Core Modules

- `forklift_system.py`: Main system implementation
- `detector.py`: YOLOv5-based object detection
- `sort_tracker.py`: SORT tracking algorithm implementation
- `kalman_filter.py`: Motion prediction using Kalman filtering
- `custom_logger.py`: Logging functionality
- `config.py`: System configuration settings

### Dataset Preparation Tools

Tools in the `DatasetPrepTools` directory:
- `setup_dataset_folder.py`: Creates training dataset folder structure
- `move_files.py`: Splits data into training/validation/test sets
- `dataset_cleaner.py`: Filters and processes raw dataset
- `class_counts.py`: Analyzes class distribution

### Testing & Evaluation

- `testingModel/compare_models.py`: Tool for comparing different model versions

## Installation

1. Clone the repository
2. Install python 3.11 (or 3.9+ I used 3.11 but 3.9+ should be fine :P)
3. Install dependencies:
```bash
pip install torch numpy opencv-python filterpy pandas
```

## Configuration

Key settings in `config.py`:

```python
MODEL_NAME = "model92"
CONF_THRESHOLD = 0.35
SORT_MAX_AGE = 50
SORT_MIN_HITS = 1
SORT_IOU_THRESHOLD = 0.3
THRESHOLD_SCALE_FACTOR = 0.8
```

## Usage

### Basic Usage

```python
from forklift_system import ForkliftSystem
from config import Config

# Initialize system
forklift_system = ForkliftSystem(model_name=Config.MODEL_NAME)

# Process video
forklift_system.run_on_video(
    video_input="path/to/video.mp4",
    resize=(1280, 720)
)

# Process image
img = cv2.imread("path/to/image.jpg")
forklift_system.run_on_image(img=img)
```

### Dataset Preparation

#### Setting up Dataset Structure
```python
python DatasetPrepTools/setup_dataset_folder.py
```

#### Cleaning Dataset
```python
from DatasetPrepTools.dataset_cleaner import DatasetCleaner, SizeRange
from config import Config

dc = DatasetCleaner(
    dest_folder=Config.CLEANED_DATA_PATH,
    allowed_size_range=SizeRange(
        Config.MIN_HEIGHT,
        Config.MIN_WIDTH,
        Config.MAX_HEIGHT,
        Config.MAX_WIDTH
    ),
    allowed_extensions=Config.ALLOWED_EXTENSIONS
)
dc.clean_data(dataset_path="path/to/data", file_base_name="prefix")
```

#### Splitting Dataset
```python
python DatasetPrepTools/move_files.py
```

#### Analyzing Class Distribution
```python
python DatasetPrepTools/class_counts.py
```

### Model Comparison

```python
from testingModel.compare_models import ModelCompare
from config import Config

model_compare = ModelCompare(
    models_path=Config.MODELS_PATH,
    images_path=Config.TEST_DATA_FOLDER,
    videos_path=Config.VIDEOS_FOLDER,
    video_image_size=(640, 480)
)
model_compare.compare_videos()
```

## Project Structure

```
├── DatasetPrepTools/
│   ├── setup_dataset_folder.py
│   ├── move_files.py
│   ├── dataset_cleaner.py
│   └── class_counts.py
├── models/
│   ├── model92/
│   │   └── best.pt
│   └── classes.txt
├── RawData/
├── DataCleaned/
├── TestData/
├── Videos/
├── train_data/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
├── forklift_system.py
├── detector.py
├── sort_tracker.py
├── kalman_filter.py
├── custom_logger.py
├── config.py
└── custom_data.yaml
```

## System Features

### Detection
- Uses YOLOv5 for object detection
- Supports two classes: 'forklift' and 'person'
- Configurable confidence threshold

### Tracking
- SORT algorithm for object tracking
- Unique ID assignment for each tracked object
- Persistence across frames

### Safety Monitoring
- Real-time distance calculation between forklifts and pedestrians
- Dynamic safety threshold based on forklift size
- Visual indicators for safety violations
- Logging of safety events

### Motion Prediction
- Kalman filtering for motion prediction
- Velocity vector visualization
- Future position estimation

## Development

### Debug Mode
Enable debug mode in `config.py`:
```python
SYSTEM_DEBUG_MODE = True
```

Debug features include:
- Detailed logging
- Visualization of tracking data
- Safety threshold visualization
- Motion prediction vectors

### Logging
Configure logging levels in `config.py`:
```python
CLI_LOG_LEVEL = logging.DEBUG
FILE_LOG_LEVEL = logging.DEBUG
```

### Train your own model
https://github.com/Koks-creator/HowToTrainCustomYoloV5Model


from dataclasses import dataclass
from collections import defaultdict
from random import randint
from math import sqrt
import os
from time import time
from typing import Union
from pathlib import Path
from typing import Tuple
from glob import glob
import numpy as np
import cv2
import pandas as pd

from detector import Detector
from config import Config
from sort_tracker import Sort
from kalman_filter import KalmanFilter2D
from custom_logger import CustomLogger

logger = CustomLogger(
    logger_log_level=Config.CLI_LOG_LEVEL,
    file_handler_log_level=Config.FILE_LOG_LEVEL,
).create_logger()


@dataclass
class ForkliftSystem:
    model_name: str
    classes_path: str = Config.CLASSES_PATH
    conf_threshold: float = Config.CONF_THRESHOLD
    ultralitycs_path: str = "ultralytics/yolov5"
    model_type: str = "custom"
    force_reload: bool = True
    sort_max_age: int = Config.SORT_MAX_AGE
    sort_min_hits: int = Config.SORT_MIN_HITS
    sort_iou_threshold: float = Config.SORT_IOU_THRESHOLD
    threshold_scale_factor: float = Config.THRESHOLD_SCALE_FACTOR
    debug: bool = Config.SYSTEM_DEBUG_MODE

    def __post_init__(self) -> None:
        logger.info("Initing ForkliftSystem with: \n"
            f"{self.model_name=} \n"
            f"{self.classes_path=} \n"
            f"{self.conf_threshold=} \n"
            f"{self.ultralitycs_path=} \n"
            f"{self.model_type=} \n"
            f"{self.force_reload=} \n"
            f"{self.sort_max_age=} \n"
            f"{self.sort_min_hits=} \n"
            f"{self.sort_iou_threshold=} \n"
            f"{self.threshold_scale_factor=} \n"
            f"{self.debug=}"
            )
        
        # print(glob(fr"{Config.MODELS_PATH}/{self.model_name}/*.pt"))
        # import sys; sys.exit()
        self.model_path = glob(fr"{Config.MODELS_PATH}/{self.model_name}/*.pt")[0]
        logger.info(f"{self.model_path}")

        self.detector = Detector(
            model_path=self.model_path,
            conf_threshold=self.conf_threshold,
            ultralitycs_path=self.ultralitycs_path,
            model_type=self.model_type,
            force_reload=self.force_reload
        )
        logger.info("YOLO detector set up and ready to detect some stuff buddy boyo")

        self.sorttr = Sort(
            max_age=self.sort_max_age,
            min_hits=self.sort_min_hits,
            iou_threshold=self.sort_iou_threshold
        )
        logger.info("Turbo SORT tracker set up ok sar")

        self.classes_mapping = self.load_classes()
        logger.info(f"{self.classes_mapping=}")

    @staticmethod
    def random_bgr():
        b = randint(0, 255)
        g = randint(0, 255)
        r = randint(0, 255)
        return (b, g, r)
    
    def load_classes(self) -> defaultdict[dict]:
        classes_mapping = defaultdict(dict)
        colors = [(200, 20, 200), (0, 200, 0)]
        with open(self.classes_path) as f:
            classes = f.read().strip().split("\n")
            for ind, (cl, color) in enumerate(zip(classes, colors)):
                classes_mapping[cl] = {
                    "Id": ind,
                    "Color": color
                    # "Color": self.random_bgr()
                }
        return classes_mapping

    @staticmethod
    def get_center(bbox: Tuple[int, int, int, int], img: np.ndarray = None) -> Tuple[int, int]:
        center_p = bbox[0] + (abs(bbox[2]-bbox[0])//2), bbox[1] + (abs(bbox[3]-bbox[1])//2)
        if img is not None:
            cv2.circle(img, center_p, 5, (0, 0, 200), -1)
        return center_p
    
    @staticmethod
    def distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        x1, y1 = p1
        x2, y2 = p2

        return sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def draw_threshold_line(self, frame, threshold_px: float, dist: float, forklift_center: Tuple[int, int],
                            person_center: Tuple[int, int], line_color=(0, 255, 255), line_thickness=2
                            ) -> None:
        if dist == 0:
            return

        dx = (person_center[0] - forklift_center[0]) / dist
        dy = (person_center[1] - forklift_center[1]) / dist

        threshold_point = (
            int(forklift_center[0] + dx * threshold_px),
            int(forklift_center[1] + dy * threshold_px)
        )

        cv2.line(frame, forklift_center, threshold_point, line_color, line_thickness)

    @staticmethod
    def draw_bbox(img: np.ndarray, bbox: Tuple[int, int, int, int], class_name: str, 
                  obj_id: int, conf: float, color: Tuple[int, int, int] = (255, 0, 255)
                  ) -> None:
        
        cv2.rectangle(img, bbox[:2], bbox[2:4], color, 2)
        cv2.putText(img, f"{class_name} {conf} - {obj_id}", (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_PLAIN, 1.4, color, 2)

    @staticmethod
    def draw_summary(img: np.ndarray, too_close_objects_dict: defaultdict[list], 
                     x: int = 20, y: int = 30, step: int = 30
                    ) -> np.ndarray:
        
        cv2.putText(img, "Summary:", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.4, (200, 200, 20), 2)
        for forklift, person_list in too_close_objects_dict.items():
            cv2.putText(img, f"- {forklift}: {person_list}", (x, y+30), cv2.FONT_HERSHEY_PLAIN, 1.4, (200, 200, 20), 2)
            y += step
        return img


    def process_detections(self, detections: pd.DataFrame, img: np.ndarray = None) -> defaultdict[list]:
        detections[['xmin', 'ymin', 'xmax', 'ymax']] = detections[['xmin', 'ymin', 'xmax', 'ymax']].astype(int)
        track_data = []

        # Prepare goddam data boyo
        for detection in detections.iterrows():
            detection = detection[1]
            logger.debug(f"process_detections: {detection=}")

            x1, y1, x2, y2, conf, class_id, class_name = detection
            track_data.append([x1, y1, x2, y2, conf, class_id])

        # Update tracker
        updated_tracks = self.sorttr.update(track_data)

        res_data = defaultdict(list)
        for track_data in updated_tracks:
            logger.debug(f"process_detections: {track_data=}")

            conf = round(track_data[-3], 2)
            x1, y1, x2, y2, _, obj_id, class_id = track_data.astype(int)
            class_name = list(self.classes_mapping.keys())[class_id]

            res_data[class_name].append([x1, y1, x2, y2, conf, obj_id, class_id])

            if img is not None:
                self.draw_bbox(img=img, 
                               bbox=(x1, y1, x2, y2), 
                               class_name=class_name, 
                               obj_id=obj_id,
                               color=self.classes_mapping[class_name]["Color"],
                               conf=conf
                            )
        logger.debug(f"{res_data=}")
        return res_data
    
    def predict_on_image(self, img: np.array) -> Tuple[np.array, np.array]:
        converted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_draw, res = self.detector.detect(img=converted)
        image_draw = cv2.cvtColor(image_draw, cv2.COLOR_RGB2BGR)

        logger.debug(f"predict_on_image: {res=}")

        return image_draw, res
    
    def run_on_image(self, img: np.ndarray, show: bool = True,
                     resize: Union[Tuple[int, int], None] = (1280, 720),
                     resize_raw: Union[Tuple[int, int], None] = (1280, 720)
                    ) -> None:
        too_close_objects_dict = defaultdict(list)
        last_points_dict = defaultdict(list)

        draw_frame, detections = self.predict_on_image(img=img)
        objects_data = self.process_detections(
            detections=detections,
            img=img
        )
        logger.debug(f"run_on_image: {objects_data=}")
        if "forklift" in list(objects_data.keys()):
            forklift_objects = objects_data["forklift"]
            person_objects = objects_data["person"]

            for forklift_obj in forklift_objects:
                f_bbox, f_conf, f_obj_id, f_class_id = forklift_obj[:4], *forklift_obj[4:]
                f_cp = self.get_center(bbox=f_bbox, img=img)
                last_points_dict[f_obj_id].append(f_cp)

                for person_obj in person_objects:
                    p_bbox, p_conf, p_obj_id, p_class_id = person_obj[:4], *person_obj[4:]
                    p_cp = self.get_center(bbox=p_bbox, img=img)

                    forklift_width = forklift_obj[2] - forklift_obj[0]
                    threshold_px = self.threshold_scale_factor * forklift_width
                    dist = self.distance(f_cp, p_cp)

                    if dist <= threshold_px:
                        too_close_objects_dict[f"Forklift-{f_obj_id}"].append(p_obj_id)
                        if self.debug:
                            cv2.line(img, f_cp, p_cp, (0, 0, 200), 20)
                        else:
                            cv2.line(img, f_cp, p_cp, (0, 0, 200), 4)

                        p_conf = round(p_conf, 2)
                        class_name = list(self.classes_mapping.keys())[p_class_id]
                        self.draw_bbox(img=img, bbox=p_bbox, class_name=class_name, obj_id=p_obj_id,
                                       conf=p_conf, color=(0, 0, 200)
                                       )
                    else:
                        cv2.line(img, f_cp, p_cp, (0, 200, 0), 2)
                    # cv2.line(img, f_cp, p_cp, (0, 0, 0), 1)
                    if self.debug:
                        self.draw_threshold_line(frame=img, threshold_px=threshold_px,
                                                forklift_center=f_cp, person_center=p_cp,
                                                dist=dist,
                                                )
        if resize:
            img = cv2.resize(img, resize)
        if resize_raw:
            draw_frame = cv2.resize(img, resize_raw)

        if show:
            img = self.draw_summary(img=img, too_close_objects_dict=too_close_objects_dict)
            cv2.imshow("res", img)
            cv2.imshow("resRaw", draw_frame)

            key = cv2.waitKey(0)
            if key == 27:
                logger.info("image going mstow")
        else:
            return img, draw_frame, too_close_objects_dict, last_points_dict
    
    def run_on_video(self, video_input: Union[str, os.PathLike, Path, int], 
                     resize: Tuple[int, int] = (1280, 720),
                     resize_raw: Tuple[int, int] = (1280, 720)
                    ) -> None:
        cap = cv2.VideoCapture(video_input)
        p_time = 0
        kalman_filters = defaultdict(KalmanFilter2D)

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("bye bye boyo")
                break
            
            frame, draw_frame, too_close_objects_dict, last_points_dict = self.run_on_image(
                img=frame, show=False,
                resize_raw=None,
                resize=None
                )
            logger.debug(f"run_on_video: {last_points_dict=}")
            for f_obj_id, cp_list in last_points_dict.items():
                if f_obj_id not in kalman_filters:
                    kalman_filters[f_obj_id] = KalmanFilter2D(dt=1.0)
                kalman_filter = kalman_filters[f_obj_id]

                logger.debug(f"run_on_video (kalman filter predictions): {f_obj_id=}, {cp_list=}")
                for cp in cp_list:
                    kalman_filter.predict()
                    updated_state = kalman_filter.update(cp)
                    
                    x_est, y_est, vx_est, vy_est = updated_state.flatten()
                    end_pt = (int(x_est + 20 * vx_est), int(y_est + 20 * vy_est))

                    logger.debug(f"run_on_video (kalman filter predictions) {x_est=}, {y_est=}, {vx_est=}, {vy_est=}")
                    logger.debug(f"run_on_video (kalman filter predictions) {end_pt=}")
                    
                    if self.debug:
                        cv2.circle(frame, cp, 5, (0, 0, 255), -1)
                        cv2.circle(frame, (int(x_est), int(y_est)), 5, (255, 0, 255), -1)
                        cv2.circle(frame, end_pt, 5, (255, 0, 0), -1)

                    cv2.arrowedLine(
                        frame,
                        (int(x_est), int(y_est)),
                        end_pt,
                        (0, 255, 0),
                        thickness=2,
                        line_type=cv2.LINE_AA,
                        tipLength=0.3
                    )
            logger.debug(kalman_filters)

            c_time = time()
            fps = int(1 / (c_time - p_time))
            p_time = c_time
            

            if resize:
                frame = cv2.resize(frame, resize)
            if resize_raw:
                draw_frame = cv2.resize(draw_frame, resize_raw)

            cv2.putText(frame, f"FPS: {fps}", (10, 25), cv2.FONT_HERSHEY_PLAIN, 1.4, (100, 0, 255), 2)
            if self.debug:
                cv2.putText(frame, f"DEBUG", (100, 25), cv2.FONT_HERSHEY_PLAIN, 1.4, (0, 200, 200), 2)

            cv2.imshow("res", frame)
            # cv2.imshow("res2", draw_frame)
            key = cv2.waitKey(1)
            if key == 27:
                logger.info("video going mstow")
                break
        
        cv2.destroyAllWindows()
        cap.release()

    

if __name__ == "__main__":
    #2745883-hd_1280_720_25fps.mp4
    # 16106604-hd_1280_720_30fps.mp4
    forklift_system = ForkliftSystem(model_name=Config.MODEL_NAME)
    forklift_system.run_on_video(
        video_input=fr"{Config.VIDEOS_FOLDER}/20654625-hd_720_1280_30fps.mp4",
        resize=Config.FRAME_RESIZE,
        resize_raw=Config.RAW_FRAME_RESIZE
    )
    # img = cv2.imread(r"C:\Users\table\PycharmProjects\MojeCos\MocneWozki\TestData\mocne.png")
    # forklift_system.run_on_image(img=img)

    # image = cv2.imread(f"{Config.TEST_DATA_FOLDER}/mocne.png")
    # image_draw, res = forklift_system.predict_on_image(img=image)

    # cv2.imshow("res", image_draw)
    # # cv2.imshow("res2", res)
    # cv2.waitKey(0)

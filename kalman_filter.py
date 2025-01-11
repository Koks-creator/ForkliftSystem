import cv2
import numpy as np
from typing import Union, List

class KalmanFilter2D:
    def __init__(self, dt: float = 1.0):
        self.dt = dt
        self.kf = cv2.KalmanFilter(dynamParams=4, measureParams=2, controlParams=0)
    
        # State transition matrix
        self.kf.transitionMatrix = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1,    0],
            [0, 0, 0,    1]
        ], dtype=np.float32)
        
        # Measurement matrix
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 10.0
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1000.0
        # Initial state estimate
        self.kf.statePost = np.zeros((4,1), dtype=np.float32)

    def predict(self) -> np.ndarray:
        """Predict the next state."""
        pred = self.kf.predict()
        return pred

    def update(self, z: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Update the filter with a new measurement.
        
        Args:
            z: Measurement vector [x, y]
        
        Returns:
            Corrected state vector
        """
        meas = np.array(z, dtype=np.float32).reshape(2, 1)
        corrected = self.kf.correct(meas)  # shape (4,1)
        return corrected

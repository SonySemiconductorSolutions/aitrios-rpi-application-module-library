import cv2
import numpy as np

from modlib.models.results import Detections
from modlib.devices.frame import Frame
from typing import Tuple, List


class Motion:
    """
    Calculates the change in pixel values from frame to frame and these changes
    represent motion. All motion is grouped to sort for real motion and calculates
    bboxes of the motion.

    For example, declare motion like this:
    ```
    from modlib.apps import Motion
    motion = Motion()

    with device as stream:
        for frame in stream:
            motion_bboxes = motion.detect(frame, detections)
    ```
    """

    size_threshold: int  #: Minimum size of detected motion to be considered significant. Default is 300.
    motion_threshold: int  #: Minimum threshold for motion detection. Default is 20.

    def __init__(self, size_threshold: int = 300, motion_threshold: int = 20):
        """
        Initializes the Motion class with thresholds for size and motion detection.

        Args:
            size_threshold: Minimum size of detected motion to be considered significant. Default is 300.
            motion_threshold: Minimum threshold for motion detection. Default is 20.

        Example:
            ```python
            motion = Motion()
            ```
        """
        self.constant_motion = 0
        self.image_IDs = []
        self.previous_frame = []
        self.size_threshold = size_threshold
        self.motion_threshold = motion_threshold

    def detect(self, frame: Frame) -> List[Tuple[int, int, int, int]]:
        """
        Detects motion from the current frame compared to the previous frame.

        Args:
            frame: The current frame to analyze for motion.

        Returns:
            A list of bounding boxes representing detected motion.

        Example:
            ```python
            motion_bboxes = motion.detect(frame)
        """
        gray_frame = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
        if len(self.previous_frame) <= 0:
            self.previous_frame = gray_frame
        frame_diff = cv2.subtract(gray_frame, self.previous_frame)
        contours = self.__capture_contours(frame_diff)  # Create change in motion image

        motion_bboxes = self.__motion_bboxes(contours, frame.width, frame.height)
        self.previous_frame = gray_frame
        return motion_bboxes

    def __capture_contours(self, frame_diff: np.ndarray) -> List[np.ndarray]:
        """
        Takes a grayscale image and uses image techniques to enhance edges in the image and find the contours

        Args:
            frame_diff: subtracted frame of previous and current frames

        Returns:
            masks of contours of detected motion
        """
        frame_diff = cv2.medianBlur(frame_diff, 3)  # Remove noise from image
        frame_mask = cv2.adaptiveThreshold(
            frame_diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3
        )  # inverse pixel values to define edges more
        frame_mask = cv2.medianBlur(frame_mask, 3)  # Remove noise from image
        frame_mask = cv2.morphologyEx(
            frame_mask, cv2.MORPH_CLOSE, np.array((9, 9), dtype=np.uint8), iterations=1
        )  # applies erosion and dilation then computes difference
        cnts, _ = cv2.findContours(
            frame_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1
        )  # finds the line contours and returns lsit of masks
        return cnts

    def __motion_bboxes(self, contours: Tuple, width: int, height: int) -> Detections:
        """
        Calculates the bboxes from contour masks and filters out small changes.

        Args:
            contours: contains the masks of motion detected
            width: width of
            height: height of Frame

        Returns:
            Detections of motion bboxes that can be used in other Modules
        """
        bbox_conts = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h > self.size_threshold:
                bbox_conts.append([x, y, x + w, y + h, w * h])

        bbox_conts = np.array(bbox_conts)
        if len(bbox_conts) != 0:
            bboxes = bbox_conts[:, :4]
            scores = bbox_conts[:, -1]
            bbox_conts = self.__non_max_suppression(bboxes, scores, 0.1)
            if sum(scores) > self.size_threshold * 5:  # check for motion but no detections
                self.constant_motion += 1
            else:
                self.constant_motion = 0
        else:
            self.constant_motion = 0
        bbox_conts = np.array(bbox_conts).astype(np.float32)

        if len(bbox_conts) > 0:
            bbox_conts[:, 0] = (bbox_conts[:, 0] / width).astype(np.float32)
            bbox_conts[:, 1] = (bbox_conts[:, 1] / height).astype(np.float32)
            bbox_conts[:, 2] = (bbox_conts[:, 2] / width).astype(np.float32)
            bbox_conts[:, 3] = (bbox_conts[:, 3] / height).astype(np.float32)
        else:
            bbox_conts = np.empty((0, 4))
        return Detections(
            bbox=np.array(bbox_conts), confidence=np.ones(len(bbox_conts)), class_id=np.empty(len(bbox_conts))
        )

    def __non_max_suppression(self, bboxes: np.ndarray, scores: np.ndarray, threshold: float = 1e-1) -> np.ndarray:
        """
        Post processing technique to combine overlapping motion detections
        into relevant detected motion.

        Args:
            bboxes: array of bboxes
            scores: array of confidence score of the bboxes
            threshold: minimum threshold of the size of bboxes

        Returns:
            filtered array of bboxes
        """
        # Sort the boxes by score in descending order
        bboxes = bboxes[np.argsort(scores)[::-1]]

        # remove all contained bounding boxes and get ordered index
        order = []
        check_array = np.array([True, True, False, False])
        order = list(range(0, len(bboxes)))
        for i in order:
            for j in range(0, len(bboxes)):
                if np.all((np.array(bboxes[j]) >= np.array(bboxes[i])) == check_array):
                    if j in order:
                        order.remove(j)
                    else:
                        continue
        filtered = []
        while order:
            itr_box = order.pop(0)
            filtered.append(itr_box)
            for i in order:
                # Calculate the IoU between the two bboxes
                itr = max(0, min(bboxes[itr_box][2], bboxes[i][2]) - max(bboxes[itr_box][0], bboxes[i][0])) * max(
                    0, min(bboxes[itr_box][3], bboxes[i][3]) - max(bboxes[itr_box][1], bboxes[i][1])
                )
                uni = (
                    (bboxes[itr_box][2] - bboxes[itr_box][0]) * (bboxes[itr_box][3] - bboxes[itr_box][1])
                    + (bboxes[i][2] - bboxes[i][0]) * (bboxes[i][3] - bboxes[i][1])
                    - itr
                )
                iou = itr / uni
                # Remove boxes with IoU greater than the threshold
                if iou > threshold:
                    order.remove(i)
        return bboxes[filtered]

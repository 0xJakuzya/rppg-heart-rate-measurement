import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from src import config

class FaceDetector:
    def __init__(self):
        base_options = mp_python.BaseOptions(model_asset_path=str(config.FACE_MODEL_PATH))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=config.FACE_MAX_NUM,
            min_face_detection_confidence=config.FACE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.FACE_MIN_TRACKING_CONFIDENCE,
            output_face_blendshapes=False,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def get_landmarks(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None
        height, width = frame.shape[:2]
        return [(int(point.x * width), int(point.y * height)) for point in result.face_landmarks[0]]

    def make_mask(self, frame, landmarks, indices,crop_top_frac, crop_bottom_frac):
        points = np.array([landmarks[index] for index in indices], np.int32)
        mask = np.zeros(frame.shape[:2], np.uint8)
        cv2.fillPoly(mask, [points], 255)
        if crop_top_frac is None and crop_bottom_frac is None:
            return mask
        y_min = int(points[:, 1].min())
        y_max = int(points[:, 1].max())
        roi_height = y_max - y_min
        if crop_top_frac is not None:
            mask[: y_min + int(roi_height * crop_top_frac)] = 0
        if crop_bottom_frac is not None:
            mask[y_min + int(roi_height * crop_bottom_frac) :] = 0
        return mask

    def get_multi_roi_patches(self, frame, landmarks, patch_size,):
        resolved_patch_size = patch_size or config.ROI_PATCH_SIZE
        return [self.extract_patch_from_mask(frame, roi_mask, resolved_patch_size) for roi_mask in self.get_multi_roi_masks(frame, landmarks)]

    def get_multi_roi_masks(self, frame, landmarks):
        forehead = self.make_mask(frame, landmarks, config.FOREHEAD_IDX, crop_bottom_frac=config.FOREHEAD_CROP_BOTTOM_FRAC)
        left_cheek = self.make_mask(frame, landmarks, config.LEFT_CHEEK_IDX)
        right_cheek = self.make_mask(frame, landmarks, config.RIGHT_CHEEK_IDX)
        masks: list[np.ndarray] = []
        masks.extend(self.split_mask(forehead, axis=1, parts=config.FOREHEAD_SPLIT_PARTS))
        masks.extend(self.split_mask(left_cheek, axis=0, parts=config.CHEEK_SPLIT_PARTS))
        masks.extend(self.split_mask(right_cheek, axis=0, parts=config.CHEEK_SPLIT_PARTS))
        return masks

    def extract_patch_from_mask(self, frame, mask,patch_size):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        cropped_frame = frame[y0:y1, x0:x1]
        cropped_mask = mask[y0:y1, x0:x1]
        patch = cv2.bitwise_and(cropped_frame, cropped_frame, mask=cropped_mask)
        return cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_AREA)

    def split_mask(self, mask, axis, parts):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return [mask.copy() for _ in range(parts)]
        coords = xs if axis == 1 else ys
        low, high = int(coords.min()), int(coords.max()) + 1
        edges = np.linspace(low, high, parts + 1).astype(int)
        split_masks = []
        for index in range(parts):
            part = np.zeros_like(mask)
            if axis == 1:
                selector = (xs >= edges[index]) & (xs < edges[index + 1])
            else:
                selector = (ys >= edges[index]) & (ys < edges[index + 1])
            part[ys[selector], xs[selector]] = 255
            split_masks.append(part)
        return split_masks

    def close(self):
        self.landmarker.close()

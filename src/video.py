import threading
from queue import Empty, Queue
import cv2
import numpy as np
from src import config

class VideoCapture:
    def __init__(self, camera_index=config.CAMERA_INDEX, width=config.FRAME_WIDTH,
                height=config.FRAME_HEIGHT, fps=config.FPS_TARGET): 
        self.cam = cv2.VideoCapture(camera_index)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cam.set(cv2.CAP_PROP_FPS, fps)
        self.fps = fps
        self.running = False
        self.frame_queue: Queue[np.ndarray] = Queue(maxsize=config.CAMERA_QUEUE_SIZE)
        self.latest_frame: np.ndarray | None = None
        self.thread: threading.Thread | None = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.capture, daemon=True)
        self.thread.start()

    def capture(self):
        while self.running:
            ok, frame = self.cam.read()
            if not ok:
                self.running = False
                break

            self.latest_frame = frame
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except Empty:
                    pass
            try:
                self.frame_queue.put_nowait(frame)
            except Exception:
                pass

    def read(self):
        return self.latest_frame

    def get_frame(self, timeout=1.0):
        try:
            return self.frame_queue.get(timeout=timeout)
        except Empty:
            return None

    def stop(self):
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cam.release()

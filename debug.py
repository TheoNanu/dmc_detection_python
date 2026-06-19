from typing import Protocol

import cv2 as cv
import numpy as np


class DebugSink(Protocol):
    def show(self, name: str, image: np.ndarray) -> None: ...
    def log(self, msg: str) -> None: ...
    def pause(self) -> None: ...

class NullSink:
    def show(self, name, image): pass
    def log(self, msg): pass
    def pause(self): pass

class CvDebugSink:
    def show(self, name, image): cv.imshow(name, image)
    def log(self, msg): print(msg)
    def pause(self): cv.waitKey(0)
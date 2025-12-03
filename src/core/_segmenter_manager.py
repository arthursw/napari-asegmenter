from multiprocessing import shared_memory
from typing import cast

import numpy as np
from napari_wsegmenter.core._memory_manager import (
    create_shared_array,
    release_shared_memory,
    share_array,
    wrap,
)
from napari_wsegmenter.core._segmenter_manager_base import SegmenterManagerBase


class SegmenterManager(SegmenterManagerBase):

    _shared_image: np.ndarray | None = None
    _shm_image: shared_memory.SharedMemory | None = None
    _shared_segmentation: np.ndarray | None = None
    _shm_segmentation: shared_memory.SharedMemory | None = None

    def _initialize_shared_memory(self, image: np.ndarray):
        if (
            self._shared_image is not None
            and self._shm_image is not None
            and self._shared_segmentation is not None
            and self._shm_segmentation is not None
        ):
            if (
                self._shared_image.dtype == image.dtype
                and self._shared_image.shape == image.shape
            ):
                return
            else:
                self.release_shared_memory()
        self._shared_image, self._shm_image = share_array(image)
        self._shared_segmentation, self._shm_segmentation = (
            create_shared_array(image.shape[:2], dtype="uint8")
        )

    def perform_segmentation(
        self, image: np.ndarray, segmenter: str, shared_memory: bool = True
    ):
        if not shared_memory:
            return super().perform_segmentation(image, segmenter)
        segmenter_module = self._initialize_environment(segmenter)
        self._initialize_shared_memory(image)
        if self._shared_image is None or self._shared_segmentation is None:
            return
        if self._shm_image is None or self._shm_segmentation is None:
            return
        segmenter_module.segment_shared_memory(
            self.config[segmenter]["module_name"],
            wrap(self._shared_image, self._shm_image),
            wrap(self._shared_segmentation, self._shm_segmentation),
            self.config[segmenter]["default_parameters"],
        )
        return self._shared_segmentation

    def release_shared_memory(self):
        release_shared_memory(self._shm_image)
        release_shared_memory(self._shm_segmentation)
        self._shm_image = None
        self._shm_segmentation = None

    def exit(self):
        self.release_shared_memory()
        self.exit_environments()


if __name__ == "__main__":

    segmenter_manager = SegmenterManager()
    result = cast(
        np.ndarray,
        segmenter_manager.perform_segmentation(
            np.random.random((100, 100)), "StarDist", True
        ),
    )
    print(result.shape)
    segmenter_manager.exit()

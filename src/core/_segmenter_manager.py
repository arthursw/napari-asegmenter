from typing import cast

import appose
import numpy as np

from core._segmenter_manager_base import SegmenterManagerBase


class SegmenterManager(SegmenterManagerBase):

    _shared_image: appose.NDArray | None = None
    _shm_image: appose.SharedMemory | None = None
    _shared_segmentation: appose.NDArray | None = None
    _shm_segmentation: appose.SharedMemory | None = None

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
        self._shm_image = appose.SharedMemory(
            create=True,
            rsize=int(np.prod(image.shape) * np.dtype(image.dtype).itemsize),
        )
        self._shared_image = appose.NDArray(
            str(image.dtype), list(image.shape), self._shm_image
        )

        segmentation_shape = image.shape[:2]
        self._shm_segmentation = appose.SharedMemory(
            create=True,
            rsize=int(
                np.prod(segmentation_shape) * np.dtype("uint8").itemsize
            ),
        )
        self._shared_segmentation = appose.NDArray(
            "uint8", list(segmentation_shape), self._shm_segmentation
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
            self._shared_image,
            self._shared_segmentation,
            self.config[segmenter]["default_parameters"],
        )
        return self._shared_segmentation

    def release_shared_memory(self):
        if self._shm_image:
            self._shm_image.dispose()
            self._shm_image = None
        if self._shm_segmentation:
            self._shm_segmentation.dispose()
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

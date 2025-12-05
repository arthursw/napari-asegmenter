from pathlib import Path
from typing import cast

import numpy as np
from appose import NDArray, SharedMemory
from appose.builder.pixi import PixiBuilder

from core._segmenter_manager_base import SegmenterManagerBase


class SegmenterManager(SegmenterManagerBase):

    _shared_image: NDArray | None = None
    _shm_image: SharedMemory | None = None
    _shared_segmentation: NDArray | None = None
    _shm_segmentation: SharedMemory | None = None

    def _initialize_environment(self, name: str):
        config = self.config[name]
        environment = (
            PixiBuilder()
            .conda(config["dependencies"]["conda"] + ["appose"])
            .pypi(config["dependencies"]["pip"])
            .base("envs/" + name)
            .log_debug()
            .build()
        )
        import sys

        service = environment.python()
        self._services.append(service)
        service.debug(lambda msg: print(msg, file=sys.stderr, end=""))

        segmenter_path = (
            Path(__file__).resolve().parent / "segmenters" / f"{name}.py"
        )
        segmenter_module = (
            service.task(str(segmenter_path)).wait_for().result()
        )

        with open(segmenter_path) as f:
            segmenters_script = f.read()
            segmenter_module = (
                service.task(segmenters_script).wait_for().result()
            )

        return segmenter_module

    def _initialize_shared_memory(self, image: np.ndarray):
        segmentation_shape = image.shape[:2]
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

                self._shared_image = NDArray(
                    str(image.dtype), list(image.shape), self._shm_image
                )
                self._shared_segmentation = NDArray(
                    "uint8", list(segmentation_shape), self._shm_segmentation
                )
                return
            else:
                self.release_shared_memory()
        self._shm_image = SharedMemory(
            create=True,
            rsize=int(np.prod(image.shape) * np.dtype(image.dtype).itemsize),
        )
        self._shared_image = NDArray(
            str(image.dtype), list(image.shape), self._shm_image
        )

        self._shm_segmentation = SharedMemory(
            create=True,
            rsize=int(
                np.prod(segmentation_shape) * np.dtype("uint8").itemsize
            ),
        )
        self._shared_segmentation = NDArray(
            "uint8", list(segmentation_shape), self._shm_segmentation
        )

    def perform_segmentation(
        self, image: np.ndarray, segmenter: str, shared_memory: bool = True
    ):
        if not shared_memory:
            return super().perform_segmentation(image, segmenter)
        segmenter_module = self._initialize_environment(segmenter)
        self._initialize_shared_memory(image)

        segmenter_module.segment(
            self._shared_image,
            self.config[segmenter]["default_parameters"],
            self._shared_segmentation,
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

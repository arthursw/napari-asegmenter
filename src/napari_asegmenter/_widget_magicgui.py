"""
This is a minimalist example using magicgui, but I cannot find the close event to exit environments properly
"""

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from appose import NDArray, SharedMemory
from appose.builder.pixi import PixiBuilder
from magicgui import magicgui
from napari.qt.threading import thread_worker

if TYPE_CHECKING:
    import napari
    import napari.types

WETLANDS_INSTALL_DIR = Path.home() / ".local" / "share" / "wetlands"
WETLANDS_INSTALL_DIR.mkdir(parents=True, exist_ok=True)
PYTHON_VERSION = "3.10"
SEGMENTERS_PATH = Path(__file__).resolve().parent / "core" / "segmenters"

config = {
    "cellpose": {
        "dependencies": {
            "python": PYTHON_VERSION,
            "conda": ["cellpose==3.1.0"],
        },
        "segmenter_script_name": SEGMENTERS_PATH / "_cellpose.py",
        "default_parameters": {
            "model_type": "cyto3",
            "use_gpu": False,
            "diameter": 30.0,
        },
    },
    "stardist": {
        "dependencies": {
            "python": PYTHON_VERSION,
            "pip": ["tensorflow==2.16.1", "csbdeep==0.8.1", "stardist==0.9.1"],
            "conda": [
                {
                    "name": "nvidia::cudatoolkit=11.0.*",
                    "platforms": ["win-64", "linux-64"],
                    "optional": True,
                },
                {
                    "name": "nvidia::cudnn=8.0.*",
                    "platforms": ["win-64", "linux-64"],
                    "optional": True,
                },
            ],
        },
        "segmenter_script_name": SEGMENTERS_PATH / "_stardist.py",
        "default_parameters": {"model_name": "2D_versatile_fluo"},
    },
    "sam": {
        "dependencies": {
            "python": PYTHON_VERSION,
            "conda": ["sam2==1.1.0", "huggingface_hub==0.29.2"],
        },
        "segmenter_script_name": SEGMENTERS_PATH / "_sam.py",
        "default_parameters": {
            "use_gpu": False,
            "points_per_side": 32,
            "pred_iou_thresh": 0.88,
            "stability_score_thresh": 0.95,
        },
    },
}

environment_manager = None


@thread_worker
def log_output(process: subprocess.Popen) -> None:
    if process.stdout is None:
        return
    for line in iter(process.stdout.readline, ""):
        print(line.strip())


_services = []


def _initialize_environment(name: str):
    econfig = config[name]
    environment = (
        PixiBuilder()
        .conda(econfig["dependencies"]["conda"] + ["appose"])
        .pypi(econfig["dependencies"]["pip"])
        .base("envs/" + name)
        .log_debug()
        .build()
    )
    import sys

    service = environment.python()
    _services.append(service)
    service.debug(lambda msg: print(msg, file=sys.stderr, end=""))

    segmenter_path = (
        Path(__file__).resolve().parent / "segmenters" / f"{name}.py"
    )
    segmenter_module = service.task(str(segmenter_path)).wait_for().result()

    with open(segmenter_path) as f:
        segmenters_script = f.read()
        segmenter_module = service.task(segmenters_script).wait_for().result()

    return segmenter_module


_shared_image = None
_shm_image = None
_shared_segmentation = None
_shm_segmentation = None


def _release_shared_memory():
    global _shared_image, _shm_image, _shared_segmentation, _shm_segmentation
    if _shm_image:
        _shm_image.dispose()
        _shm_image = None
    if _shm_segmentation:
        _shm_segmentation.dispose()
        _shm_segmentation = None


def _initialize_shared_memory(image: np.ndarray):
    global _shared_image, _shm_image, _shared_segmentation, _shm_segmentation
    segmentation_shape = image.shape[:2]
    if (
        _shared_image is not None
        and _shm_image is not None
        and _shared_segmentation is not None
        and _shm_segmentation is not None
    ):
        if (
            _shared_image.dtype == image.dtype
            and _shared_image.shape == image.shape
        ):

            _shared_image = NDArray(
                str(image.dtype), list(image.shape), _shm_image
            )
            _shared_segmentation = NDArray(
                "uint8", list(segmentation_shape), _shm_segmentation
            )
            return
        else:
            _release_shared_memory()
    _shm_image = SharedMemory(
        create=True,
        rsize=int(np.prod(image.shape) * np.dtype(image.dtype).itemsize),
    )
    _shared_image = NDArray(str(image.dtype), list(image.shape), _shm_image)

    _shm_segmentation = SharedMemory(
        create=True,
        rsize=int(np.prod(segmentation_shape) * np.dtype("uint8").itemsize),
    )
    _shared_segmentation = NDArray(
        "uint8", list(segmentation_shape), _shm_segmentation
    )


@magicgui(
    segmenter={"choices": ["stardist", "cellpose", "sam"]},
)
def segmenter_widget(
    img: "napari.types.ImageData",
    segmenter: "str",
) -> "napari.types.LabelsData":
    segmenter_module = _initialize_environment(segmenter)
    _initialize_shared_memory(img)
    segmenter_module.segment(
        _shared_image,
        config[segmenter]["default_parameters"],
        _shared_segmentation,
    )
    return _shared_segmentation.ndarray()  # type: ignore


def exit_environments():
    _release_shared_memory()
    global _services
    for service in _services:
        service.close()
    _services = []


# I need something like this
segmenter_widget.closed.connect(exit_environments)

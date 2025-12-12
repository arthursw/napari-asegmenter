"""
This is a minimalist example using magicgui, but I cannot find the close event to exit environments properly
"""

import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from appose import Environment, NDArray
from appose.builder.pixi import PixiBuilder
from magicgui import magic_factory
from napari.types import LayerDataTuple

if TYPE_CHECKING:
    import napari
    import napari.types

PYTHON_VERSION = "3.10"
ENVS = Path("envs")


def init_service(env: Environment, task_path):
    service = env.python().init("import numpy")
    service.debug(lambda msg: print(msg, file=sys.stderr, end=""))
    # service.task(Path(__file__).parent.resolve() / task_path).wait_for()
    with open(Path(__file__).parent.resolve() / task_path) as f:
        module = service.task(f.read()).wait_for().result()
    return service, module


env_cellpose = (
    PixiBuilder()
    .conda("appose", "cellpose==3.1.0")
    .base(ENVS / "cellpose")
    .log_debug()
    .build()
)
service_cellpose, module_cellpose = init_service(env_cellpose, "_cellpose.py")

env_stardist = (
    PixiBuilder()
    .conda("appose")
    .pypi("tensorflow==2.16.1", "csbdeep==0.8.1", "stardist==0.9.1")
    .base(ENVS / "stardist")
    .log_debug()
    .build()
)
service_stardist, module_stardist = init_service(env_stardist, "_stardist.py")

env_sam = (
    PixiBuilder()
    .conda("appose")
    .pypi("appose", "sam2==1.1.0", "huggingface_hub==0.29.2")
    .base(ENVS / "sam")
    .log_debug()
    .build()
)
service_sam, module_sam = init_service(env_sam, "_sam.py")

_shared_image = None
_shared_segmentation = None


# Helper to return a LayerDataTuple
def layer(
    ndarray: NDArray | None, name: str, layer_type: str = "labels"
) -> LayerDataTuple:
    if ndarray is None:
        raise Exception("NDArray is undefined.")
    return cast(
        LayerDataTuple, (ndarray.ndarray().copy(), {"name": name}, layer_type)
    )


def _release_shared_memory():
    global _shared_image, _shared_segmentation
    if _shared_image:
        _shared_image.shm.dispose()
        _shared_image = None
    if _shared_segmentation:
        _shared_segmentation.shm.dispose()
        _shared_segmentation = None


def _initialize_shared_memory(image: np.ndarray):
    global _shared_image, _shared_segmentation
    segmentation_shape = image.shape[:2]
    if _shared_image is not None and _shared_segmentation is not None:
        if (
            _shared_image.dtype == image.dtype
            and _shared_image.shape == image.shape
        ):

            _shared_image = NDArray(
                str(image.dtype), list(image.shape), _shared_image.shm
            )
            _shared_segmentation = NDArray(
                "uint8", list(segmentation_shape), _shared_segmentation.shm
            )
            return
        else:
            _release_shared_memory()
    _shared_image = NDArray(str(image.dtype), list(image.shape))
    _shared_segmentation = NDArray("uint8", list(segmentation_shape))


# Computes the Cellpose segmentation using the global shared memory
@magic_factory(model_type={"choices": ["cyto3", "cyto2", "nuclei"]})
def cellpose(
    image: "napari.types.ImageData",
    model_type="cyto3",
    use_gpu: bool = False,
    diameter: float = 30.0,
) -> LayerDataTuple:
    _initialize_shared_memory(image)
    module_cellpose.segment(
        _shared_image,
        _shared_segmentation,
        {
            "model_type": model_type,
            "use_gpu": use_gpu,
            "diameter": diameter,
            "channels": [0, 0],
        },
    )

    return layer(_shared_segmentation, "Cellpose segmentation")


# Same as cellpose() but simpler because it creates a new shared memory each time
# Instead, it uses the context manager which frees the shared memory on return
@magic_factory(model_type={"choices": ["cyto3", "cyto2", "nuclei"]})
def cellpose_simple(
    image: "napari.types.ImageData",
    model_type="cyto3",
    use_gpu: bool = False,
    diameter: float = 30.0,
) -> LayerDataTuple:
    segmentation_shape = image.shape[:2]
    _shared_image = NDArray(str(image.dtype), list(image.shape))
    _shared_segmentation = NDArray("uint8", list(segmentation_shape))

    module_cellpose.segment(
        _shared_image,
        _shared_segmentation,
        {
            "model_type": model_type,
            "use_gpu": use_gpu,
            "diameter": diameter,
            "channels": [0, 0],
        },
    )
    _shared_image.shm.close()
    _shared_image.shm.unlink()
    _shared_segmentation.shm.close()
    _shared_segmentation.shm.unlink()
    return layer(_shared_segmentation, "Cellpose segmentation")


# Computes the StarDist segmentation using the global shared memory
@magic_factory(
    model_name={"choices": ["2D_versatile_fluo", "2D_paper_dsb2018"]}
)
def stardist(
    image: "napari.types.ImageData", model_name="2D_versatile_fluo"
) -> LayerDataTuple:
    _initialize_shared_memory(image)
    module_stardist.segment(
        _shared_image, _shared_segmentation, {"model_name": model_name}
    )

    return layer(_shared_segmentation, "Cellpose segmentation")


# Computes the SAM segmentation using the global shared memory
@magic_factory()
def sam(
    image: "napari.types.ImageData",
    use_gpu: bool = False,
    points_per_side: int = 8,
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.95,
) -> LayerDataTuple:
    _initialize_shared_memory(image)
    module_sam.segment(
        _shared_image,
        _shared_segmentation,
        {
            "use_gpu": use_gpu,
            "points_per_side": points_per_side,
            "pred_iou_thresh": pred_iou_thresh,
            "stability_score_thresh": stability_score_thresh,
        },
    )

    return layer(_shared_segmentation, "Cellpose segmentation")


@magic_factory(
    call_button="Exit plugin",
)
def exit_button():
    exit_environments()


def exit_environments():
    _release_shared_memory()
    for service in [service_stardist, service_cellpose, service_sam]:
        service.close()


# I need something like this
# widget.closed.connect(exit_environments)

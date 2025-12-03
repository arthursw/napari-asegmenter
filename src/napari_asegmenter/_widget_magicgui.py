"""
This is a minimalist example using magicgui, but I cannot find the close event to exit environments properly
"""

import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from magicgui import magicgui
from napari.qt.threading import thread_worker
from wetlands.environment_manager import EnvironmentManager

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


def initialize_environment(name: str):
    global environment_manager
    if environment_manager is None:
        environment_manager = EnvironmentManager(debug=True)
    environment = environment_manager.create(
        name, config[name]["dependencies"]
    )
    # launched = environment.launched()
    # if not launched:
    environment.launch()
    segmenter_module = environment.importModule(
        str(config[name]["segmenter_script_name"])
    )
    # if not launched:
    #     worker = cast(WorkerBase, log_output(segmenter_module.process))
    #     worker.start()
    return segmenter_module


@magicgui(
    segmenter={"choices": ["stardist", "cellpose", "sam"]},
)
def segmenter_widget(
    img: "napari.types.ImageData",
    segmenter: "str",
) -> "napari.types.LabelsData":
    segmenter_module = initialize_environment(segmenter)
    with tempfile.TemporaryDirectory() as tempdir:
        input_path = Path(tempdir) / "image.npy"
        output_path = Path(tempdir) / "segmentation.npy"
        np.save(input_path, cast(np.ndarray, img.data))
        segmenter_module.segment(
            input_path, output_path, config[segmenter]["default_parameters"]
        )
        return np.load(output_path)


def exit_environments():
    global environment_manager
    if environment_manager is None:
        return
    for _, environment in environment_manager.environments.items():
        environment.exit()


# I need something like this
segmenter_widget.closed.connect(exit_environments)

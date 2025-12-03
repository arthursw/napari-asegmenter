import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))


def _segment(module_name, image, parameters) -> np.ndarray | None:
    import importlib

    module = importlib.import_module(module_name)
    return module.segment(image, parameters)


def segment_files(module_name, image_path, segmentation_path, parameters):
    image = np.load(image_path)
    segmentation = _segment(module_name, image, parameters)
    if segmentation is not None:
        np.save(segmentation_path, segmentation)
        return segmentation_path


def segment_shared_memory(
    module_name, shared_image, shared_segmentation, parameters
):
    from _memory_manager import get_shared_array  # type: ignore

    with get_shared_array(shared_image) as image:
        labels = _segment(module_name, image, parameters)
        with get_shared_array(shared_segmentation) as segmentation:
            segmentation[:] = labels
    return shared_segmentation


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "Segmenters",
        description="Segment an image using StarDist, Cellpose or SAM",
        argument_default=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-s",
        "--segmenter",
        help="Name of the segmenter",
        choices=["stardist", "cellpose", "sam"],
        default="cellpose",
    )
    parser.add_argument(
        "-i", "--image", help="Input image path", type=Path, required=True
    )
    parser.add_argument(
        "-seg",
        "--segmentation",
        help="Output image path",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    if args.segmenter != "sam":
        import skimage

        image = skimage.io.imread(str(args.image))
        image = skimage.color.rgb2gray(image)
    else:
        from PIL import Image

        image = Image.open(args.image)
        # Convert to grayscale, then back to RGB (clone gray component)
        image = np.array(image.convert("L").convert("RGB"))

    parameters = {}
    match args.segmenter:
        case "stardist":
            parameters = {"model_name": "2D_versatile_fluo"}
        case "cellpose":
            parameters = {
                "model_type": "cyto3",
                "use_gpu": False,
                "diameter": 30.0,
                "channels": [0, 0],
            }
        case "sam":
            parameters = {
                "use_gpu": False,
                "points_per_side": 32,
                "pred_iou_thresh": 0.88,
                "stability_score_thresh": 0.95,
            }

    segmentation = _segment(
        f"_{args.segmenter}",
        image,
        parameters,
    )
    if segmentation is not None:
        segmentation_path = (
            args.image.parent
            / f"{args.image.stem}_segmentation{args.image.suffix}"
            if args.segmentation is None
            else args.segmentation
        )

        if args.segmenter != "sam":
            import skimage

            skimage.io.imsave(str(segmentation_path), segmentation)
        else:
            from PIL import Image

            segmentation_image = Image.fromarray(segmentation)
            segmentation_image.save(segmentation_path)

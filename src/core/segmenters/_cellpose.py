model, last_parameters = None, None


def segment(image, parameters):

    print("Loading libraries...")
    import cellpose.io  # type: ignore
    import cellpose.models  # type: ignore

    global model, model_type, last_parameters

    if (
        last_parameters is None
        or last_parameters["model_type"] != parameters["model_type"]
        or last_parameters["use_gpu"] != parameters["use_gpu"]
    ):
        print("Loading model...")
        model_type = parameters["model_type"]
        model = cellpose.models.Cellpose(
            gpu=parameters["use_gpu"], model_type=model_type
        )
    last_parameters = parameters

    if model is None:
        return
    print("Computing segmentation...")
    masks, flows, styles, diams = model.eval(
        image, diameter=parameters["diameter"], channels=parameters["channels"]
    )
    return masks

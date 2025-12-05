from napari import Viewer, run

viewer = Viewer()
dock_widget, plugin_widget = viewer.window.add_plugin_dock_widget(
    "napari-asegmenter", "Segmenter"
)
run()

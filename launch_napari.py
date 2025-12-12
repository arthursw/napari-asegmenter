from napari import Viewer, run

viewer = Viewer()
dock_widget, plugin_widget = viewer.window.add_plugin_dock_widget(
    "napari-asegmenter", "SAM segmenter"
)
dock_widget, plugin_widget = viewer.window.add_plugin_dock_widget(
    "napari-asegmenter", "Cellpose segmenter"
)
dock_widget, plugin_widget = viewer.window.add_plugin_dock_widget(
    "napari-asegmenter", "Stardist segmenter"
)
dock_widget, plugin_widget = viewer.window.add_plugin_dock_widget(
    "napari-asegmenter", "Exit"
)

run()

# napari-wsegmenter

[![License MIT](https://img.shields.io/pypi/l/napari-wsegmenter.svg?color=green)](https://github.com/arthursw/napari-wsegmenter/raw/main/LICENSE)
<!-- [![PyPI](https://img.shields.io/pypi/v/napari-wsegmenter.svg?color=green)](https://pypi.org/project/napari-wsegmenter) -->
<!-- [![Python Version](https://img.shields.io/pypi/pyversions/napari-wsegmenter.svg?color=green)](https://python.org) -->
<!-- [![tests](https://github.com/arthursw/napari-wsegmenter/workflows/tests/badge.svg)](https://github.com/arthursw/napari-wsegmenter/actions) -->
<!-- [![codecov](https://codecov.io/gh/arthursw/napari-wsegmenter/branch/main/graph/badge.svg)](https://codecov.io/gh/arthursw/napari-wsegmenter) -->
<!-- [![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-wsegmenter)](https://napari-hub.org/plugins/napari-wsegmenter) -->
[![npe2](https://img.shields.io/badge/plugin-npe2-blue?link=https://napari.org/stable/plugins/index.html)](https://napari.org/stable/plugins/index.html)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-purple.json)](https://github.com/copier-org/copier)

Segment images using either Stardist, Cellpose or SAM.

The plugin uses [Appose-python](https://github.com/apposed/appose-python) to install each tool (Stardist, Cellpose and SAM) in isolated environment, and execute segmentations in those environments.

----------------------------------

## Installation, usage & development

You can launch napari with the plugin by running `uv run python launch_napari.py`.

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

### Tests

To test the project locally, use uv to install the testing optional dependencies: `uv pip install ".[testing]"`
Then run tox: `uv run tox run`

Test with `uv` and `ipdb`: `uv run pytest --pdb --pdbcls=IPython.terminal.debugger:TerminalPdb tests`
Use `--last-failed` to only re-run the failures: `uv run pytest --pdb --pdbcls=IPython.terminal.debugger:TerminalPdb --last-failed tests`

## License

Distributed under the terms of the [MIT] license,
"napari-wsegmenter" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[tox]: https://tox.readthedocs.io/en/latest/
[file an issue]: https://github.com/arthursw/napari-wsegmenter/issues

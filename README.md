# sm4sh_model_py
Python bindings to [sm4sh_model](https://github.com/ScanMountGoat/sm4sh_lib) for high level and efficient data access to model files for Smash 4 for Wii U.

## Installation
The compiled extension module can be imported just like any other Python file. On Windows, rename `sm4sh_model_py.dll` to `sm4sh_model_py.pyd`. If importing `sm4sh_model_py` fails, make sure the import path is specified correctly and the current Python version matches the version used when building. For installing in the current Python environment, install [maturin](https://github.com/PyO3/maturin) and use `maturin develop --release`.

## Building
Build the project with `cargo build --release`. This will compile a native python module for the current Python interpreter. For use with Blender, make sure to build for the Python version used by Blender. This can be achieved by activating a virtual environment with the appropriate Python version or setting the Python interpeter using the `PYO3_PYTHON` environment variable. See the [PyO3 guide](https://pyo3.rs) for details.

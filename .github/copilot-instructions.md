# Spinor Visualization Project

This Python project provides interactive visualization of spinors using the hyperchord representation.

## Project Structure

- `spinor_viz/` - Core Python package
  - `core.py` - Spinor mathematics (Pauli matrices, rotations)
  - `visualization.py` - 2D and 3D plotting utilities
  - `utils.py` - Helper functions
- `notebooks/` - Interactive Jupyter notebooks
- `tests/` - Unit tests

## Development Guidelines

- Use NumPy for numerical operations
- Use Matplotlib for 2D plots, Plotly for 3D interactive visualizations
- Use ipywidgets for interactive notebook controls
- Follow PEP 8 style guidelines
- Document functions with docstrings

## Running the Project

```bash
pip install -e .
jupyter lab notebooks/
```

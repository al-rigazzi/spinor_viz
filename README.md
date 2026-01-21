# Spinor Visualization

Interactive visualization of spinors using the **hyperchord representation**.

This project provides tools to visualize and understand spinors — mathematical objects that represent quantum mechanical spin states and rotations in 3D space. The hyperchord representation offers a geometric interpretation where spinor components are encoded as chords on a sphere.

## Features

- **Spinor Mathematics**: Core functions for spinor operations, Pauli matrices, and SU(2) rotations
- **2D Visualization**: Projection of hyperchord circles onto a plane
- **3D Visualization**: Interactive 3D plots showing the full spinor geometry
- **Interactive Notebooks**: Jupyter notebooks with widgets to explore:
  - Rotation axis and angle controls
  - Animated spinor evolution
  - Complex number representations

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/spinor-viz.git
cd spinor-viz

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
```

## Quick Start

```python
import numpy as np
from spinor_viz import Spinor, plot_spinor_3d

# Create a spinor
s = Spinor.from_angles(theta=np.pi/3, phi=np.pi/4)

# Visualize it
plot_spinor_3d(s)
```

## Notebooks

Launch Jupyter Lab to explore the interactive notebooks:

```bash
jupyter lab notebooks/
```

Available notebooks:
1. **01_spinor_basics.ipynb** - Introduction to spinors and the hyperchord representation
2. **02_rotations.ipynb** - Interactive exploration of spinor rotations
3. **03_animations.ipynb** - Animated spinor evolution under rotation

## Mathematical Background

A spinor $\psi = \begin{pmatrix} \psi_\uparrow \\ \psi_\downarrow \end{pmatrix}$ can be parameterized by three half-angles:

- $\theta/2$: Polar angle (determines ratio of up/down components)
- $\phi/2$: Azimuthal angle (overall phase difference)
- $\alpha/2$: Overall phase

The hyperchord representation maps these to geometric objects on a sphere, where:
- The equator circle represents the base sphere
- Two inclined circles (hyperchords) encode the spinor components
- Arrows on the circles show the complex values

## License

MIT License - see [LICENSE](LICENSE) for details.

## References

- Based on MATLAB code for spinor visualization
- Geometric interpretation of spinors and the Hopf fibration

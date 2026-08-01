# Installation

REST3D uses `rest3d` for scene reconstruction (Stages 1–2) and keeps `gym` for
the original Isaac Gym Stage 3 backend. The optional Isaac Lab Stage 3 backend
uses a third, independent `isaaclab` environment; do not upgrade either existing
environment to install it.

**1. Clone this repo**
```bash
git clone https://github.com/ShirleyMaxx/REST3D.git REST3D
cd REST3D
export REST3D_ROOT="$(pwd)"
```

**2. Set up the `rest3d` environment** — follow [environments/install_rest3d.md](environments/install_rest3d.md)

**3. Set up the `gym` environment** — follow [environments/install_gym.md](environments/install_gym.md)

**4. Optional: set up the `isaaclab` environment** — follow
[environments/install_isaaclab.md](environments/install_isaaclab.md). The tested
RTX 5090/WSL headless baseline and its known upstream dependency conflict are
recorded there. Runtime commands and regression gates are in
[doc/isaac-lab-stage3.md](doc/isaac-lab-stage3.md).

**5. Set your API key** (for the `rest3d` environment):
```bash
# default Gemini backend
export GEMINI_API_KEY=...
```

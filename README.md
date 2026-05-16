> [!WARNING]
> Warning, uses experimental package `comfy-env` to attempt a one click isolated install. Will download and use pixi package manager.

# ComfyUI-Sharp

## Installation

Three options, in order of speed → reliability:

1. **ComfyUI Manager (recommended)** — search for `Sharp` in the Manager and click Install from the highest version displayed. If that doesn't work, try nightly.
2. **Manager via Git URL** — in ComfyUI Manager: "Install via Git URL" with `https://github.com/PozzettiAndrea/ComfyUI-Sharp.git`.
3. **Manual (most reliable)**:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/PozzettiAndrea/ComfyUI-Sharp.git
   cd ComfyUI-Sharp
   pip install -r requirements.txt --upgrade
   python install.py
   ```

> **Please report any problems** you hit during installation or use of my nodes — open a [Discussion](https://github.com/PozzettiAndrea/ComfyUI-Sharp/discussions) or [Issue](https://github.com/PozzettiAndrea/ComfyUI-Sharp/issues). Very grateful for your help! 🙏

---


<div align="center">
<a href="https://pozzettiandrea.github.io/ComfyUI-Sharp/">
<img src="https://pozzettiandrea.github.io/ComfyUI-Sharp/gallery-preview.png" alt="Workflow Test Gallery" width="800">
</a>
<br>
<b><a href="https://pozzettiandrea.github.io/ComfyUI-Sharp/">View Live Test Gallery →</a></b>
</div>

ComfyUI wrapper for [SHARP](https://arxiv.org/abs/2512.10685) by [Apple](https://github.com/apple/ml-sharp) - monocular 3D Gaussian Splatting in under 1 second.

2 Example workflows.

Workflow 1: standard/user input focal length.
![Workflow](docs/no_exif.png)


https://github.com/user-attachments/assets/479fb066-4d40-4d7c-a8d4-d1224fc22efa


Workflow 2: focal length extraction from exif data.

![Workflow_exif](docs/with_exif.png)


https://github.com/user-attachments/assets/b0c3e196-aa93-4380-8f8b-9c19b833b818

Note: for PLY inference this model is good on its own, but for the Gaussian Viewer node, you're going to need to install this node as well! https://github.com/PozzettiAndrea/ComfyUI-GeometryPack

Model auto-downloads on first run. For offline use, place `sharp_2572gikvuh.pt` in `ComfyUI/models/sharp/`.



## Nodes

- **Load SHARP Model** - (down)Load the SHARP model
- **SHARP Predict** - Generate 3D Gaussians from a single image
- **Load Image with EXIF** - Load image and auto-extract focal length from EXIF (35mm equivalent)

Images with EXIF data get focal length auto-calculated when using the Load Image with EXIF node.

## Community

Questions or feature requests? Open a [Discussion](https://github.com/PozzettiAndrea/ComfyUI-Sharp/discussions) on GitHub.

Join the [Comfy3D Discord](https://discord.gg/bcdQCUjnHE) for help, updates, and chat about 3D workflows in ComfyUI.

## Credits

Thanks to Apple for releasing SHARP as open source.

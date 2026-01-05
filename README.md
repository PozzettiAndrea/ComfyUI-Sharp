# ComfyUI-Sharp

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

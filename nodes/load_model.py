"""LoadSharpModel node for ComfyUI-Sharp."""

import os
from urllib.request import urlopen

import torch
from tqdm import tqdm

# Try to get ComfyUI models directory
try:
    import folder_paths
    MODELS_DIR = os.path.join(folder_paths.models_dir, "sharp")
except ImportError:
    MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "sharp")

# Model cache
_MODEL_CACHE = {}

# Default model URL and filename
DEFAULT_MODEL_URL = "https://huggingface.co/apple/Sharp/resolve/main/sharp_2572gikvuh.pt"
DEFAULT_MODEL_FILENAME = "sharp_2572gikvuh.pt"


class LoadSharpModel:
    """Load and cache the SHARP model for 3D Gaussian prediction."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device": (["auto", "cuda", "mps", "cpu"], {
                    "default": "auto",
                    "tooltip": "Device to run inference on. 'auto' selects best available."
                }),
            },
            "optional": {
                "checkpoint_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to .pt checkpoint. Leave empty to auto-download from Hugging Face."
                }),
            }
        }

    RETURN_TYPES = ("SHARP_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "SHARP"
    DESCRIPTION = "Load the SHARP model for monocular 3D Gaussian Splatting prediction."

    def load_model(self, device: str, checkpoint_path: str = ""):
        """Load and cache the SHARP model."""
        from sharp.models import PredictorParams, create_predictor

        # Resolve device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch, "mps") and torch.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        # Create cache key
        cache_key = f"{checkpoint_path or 'default'}_{device}"

        if cache_key in _MODEL_CACHE:
            return (_MODEL_CACHE[cache_key],)

        # Load or download checkpoint
        if checkpoint_path and os.path.exists(checkpoint_path):
            model_path = checkpoint_path
        else:
            # Use default model in ComfyUI models/sharp directory
            os.makedirs(MODELS_DIR, exist_ok=True)
            model_path = os.path.join(MODELS_DIR, DEFAULT_MODEL_FILENAME)

            if not os.path.exists(model_path):
                print(f"[SHARP] Downloading model to {model_path}")
                response = urlopen(DEFAULT_MODEL_URL)
                total_size = int(response.headers.get('content-length', 0))

                with open(model_path, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading SHARP model") as pbar:
                        while True:
                            chunk = response.read(8192)
                            if not chunk:
                                break
                            f.write(chunk)
                            pbar.update(len(chunk))
                print(f"[SHARP] Download complete.")
            else:
                print(f"[SHARP] Loading model from {model_path}")

        print(f"[SHARP] Loading checkpoint...")
        state_dict = torch.load(model_path, weights_only=True)

        # Create predictor
        print(f"[SHARP] Initializing model...")
        predictor = create_predictor(PredictorParams())
        predictor.load_state_dict(state_dict)
        predictor.eval()
        print(f"[SHARP] Moving model to {device}...")
        predictor.to(device)
        print(f"[SHARP] Model ready!")

        model_dict = {
            "predictor": predictor,
            "device": device,
        }

        _MODEL_CACHE[cache_key] = model_dict

        return (model_dict,)


NODE_CLASS_MAPPINGS = {
    "LoadSharpModel": LoadSharpModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadSharpModel": "Load SHARP Model",
}

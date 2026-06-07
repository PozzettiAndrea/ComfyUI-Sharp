"""LoadSharpModel node for ComfyUI-Sharp.

Returns a lightweight config dict (JSON-serializable) so it works across
comfy-env subprocess isolation boundaries. Actual model loading happens
on-demand in _load_sharp_model(), called by inference nodes.
"""

import os
import sys
import time
import logging

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

from comfy_api.latest import io

log = logging.getLogger("sharp")


def _p(msg: str) -> None:
    """Print to stderr so the line surfaces in ComfyUI's worker log
    (`log.info` on the 'sharp' logger doesn't route through the
    comfy-env subprocess pipe — direct print to stderr does)."""
    print(f"[LoadSharpModel] {msg}", file=sys.stderr, flush=True)


def _comfy_tqdm():
    """tqdm that shows download progress in ComfyUI's UI."""
    try:
        import comfy.utils
        import tqdm as _tqdm_mod
    except ImportError:
        return None
    holder = {"pbar": None, "total": 0, "done": 0}
    class _T(_tqdm_mod.tqdm):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            if self.total and self.total > 0 and holder["pbar"] is None:
                holder["total"] = self.total
                holder["done"] = 0
                holder["pbar"] = comfy.utils.ProgressBar(self.total)
        def update(self, n=1):
            ret = super().update(n)
            if n and holder["pbar"] and holder["total"] > 0:
                holder["done"] = min(holder["done"] + n, holder["total"])
                holder["pbar"].update_absolute(holder["done"], holder["total"])
            return ret
    return _T


# Try to get ComfyUI models directory
try:
    import folder_paths
    MODELS_DIR = os.path.join(folder_paths.models_dir, "sharp")
    os.makedirs(MODELS_DIR, exist_ok=True)
    folder_paths.add_model_folder_path("sharp", MODELS_DIR)
except ImportError:
    MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "sharp")

SHARP_REPO_ID = "apple/Sharp"
SHARP_FILENAME = "sharp_2572gikvuh.pt"


def _find_local_checkpoint():
    """Return the path to an already-present SHARP checkpoint, or None.

    Prefers folder_paths.get_full_path("sharp", ...) so a checkpoint pre-staged
    in ANY registered "sharp" model folder (including extra_model_paths.yaml or
    a CI-mounted models dir) is found and re-used -- no re-download. Falls back
    to the plain MODELS_DIR join when folder_paths is unavailable.
    """
    try:
        import folder_paths
        hit = folder_paths.get_full_path("sharp", SHARP_FILENAME)
        if hit and os.path.exists(hit):
            return hit
    except Exception:
        pass
    direct = os.path.join(MODELS_DIR, SHARP_FILENAME)
    return direct if os.path.exists(direct) else None


def _hf_token():
    """Use a HF token if the environment provides one. Authenticated requests
    get a much higher rate limit, which matters in CI where many jobs hit HF at
    once (anonymous 429s). Returns None when unset -> anonymous download."""
    for var in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        tok = os.environ.get(var)
        if tok:
            return tok
    return None


def _download_with_retry(log_fn, **kwargs):
    """hf_hub_download with backoff on transient HTTP failures (429 rate limit,
    5xx). The SHARP checkpoint is large and CI hammers HF in parallel, so a
    single 429 shouldn't fail the whole run. Retries up to 5x with exponential
    backoff; re-raises on the final attempt or on non-retryable errors."""
    token = _hf_token()
    if token:
        kwargs.setdefault("token", token)
    delays = [5, 15, 30, 60]  # seconds; len+1 = total attempts
    for attempt in range(len(delays) + 1):
        try:
            return hf_hub_download(**kwargs)
        except HfHubHTTPError as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            retryable = status == 429 or (status is not None and 500 <= status < 600)
            if not retryable or attempt == len(delays):
                raise
            wait = delays[attempt]
            log_fn(f"HF download got HTTP {status} (attempt {attempt + 1}/"
                   f"{len(delays) + 1}); retrying in {wait}s...")
            time.sleep(wait)

# -- Module-level model cache (persists across subprocess calls) ----------

_model_patcher = None   # Single ModelPatcher instance
_model_config = None     # Config dict that built the current patcher

_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def _load_sharp_model(config):
    """Load model on first call, reuse on subsequent calls.

    Returns (predictor, device). The model is wrapped in ModelPatcher for
    ComfyUI-native VRAM management and loaded to GPU via load_models_gpu().
    """
    global _model_patcher, _model_config
    import comfy.model_management
    import comfy.model_patcher
    import comfy.ops
    import comfy.utils
    from .sharp import PredictorParams, create_predictor

    if _model_patcher is None or _model_config != config:
        # Config changed or first load — build from scratch
        model_path = config["model_path"]
        dtype = _DTYPE_MAP[config["dtype"]]

        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()

        # Select optimal operations class (enables fp8, nvfp4, CublasOps, etc.)
        manual_cast_dtype = comfy.model_management.unet_manual_cast(dtype, load_device)
        operations = comfy.ops.pick_operations(dtype, manual_cast_dtype)

        # Load state dict
        _p(f"loading checkpoint from {model_path} (dtype={config['dtype']})")
        log.info(f"Loading checkpoint from {model_path}")
        state_dict = comfy.utils.load_torch_file(model_path)

        # Build model on meta device (zero memory allocation) then load weights
        # directly with assign=True, avoiding 2x RAM peak from CPU construction.
        _p("initializing model on meta device...")
        log.info("Initializing model on meta device...")
        with torch.device("meta"):
            predictor = create_predictor(
                PredictorParams(),
                dtype=dtype,
                device=None,
                operations=operations,
            )

        # Load weights with assign=True — replaces meta Parameters with real tensors
        # without ever allocating the full model on CPU first.
        # strict=False because registered buffers (e.g. normalization stats) may not
        # be in the checkpoint and will be fixed up below.
        predictor.load_state_dict(state_dict, strict=False, assign=True)

        # Fix any leftover meta-device buffers (e.g. register_buffer constants not
        # present in the checkpoint) by materializing them as real zero tensors.
        for name, buf in list(predictor.named_buffers()):
            if buf.device.type == "meta":
                parts = name.split(".")
                parent = predictor
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                parent._buffers[parts[-1]] = torch.zeros_like(buf, device="cpu")
        predictor.eval()
        if comfy.model_management.force_channels_last():
            predictor.to(memory_format=torch.channels_last)
        comfy.model_management.archive_model_dtypes(predictor)
        _p(f"model ready ({dtype})")
        log.info(f"Model ready ({dtype})")

        # Wrap with ModelPatcher — ComfyUI manages VRAM from here
        patcher = comfy.model_patcher.ModelPatcher(
            predictor,
            load_device=load_device,
            offload_device=offload_device,
        )
        _model_patcher = patcher
        _model_config = config

    return _model_patcher


# -- Node -----------------------------------------------------------------

class LoadSharpModel(io.ComfyNode):
    """Download the SHARP checkpoint and return a config for inference nodes.

    Returns a lightweight config dict (model path + dtype) that is
    JSON-serializable for comfy-env IPC. The actual model is loaded
    on-demand by inference nodes via _load_sharp_model().
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadSharpModel",
            display_name="(Down)Load SHARP Model",
            category="SHARP",
            description="Download/configure the SHARP model for monocular 3D Gaussian Splatting prediction.",
            inputs=[
                io.Combo.Input("precision", options=["auto", "bf16", "fp16", "fp32"],
                               default="auto", optional=True,
                               tooltip="Model precision. auto: best for your GPU (bf16 on Ampere+, fp16 on Volta/Turing, fp32 on older)."),
            ],
            outputs=[
                io.Custom("SHARP_MODEL_CONFIG").Output(display_name="model_config"),
            ],
        )

    @classmethod
    def execute(cls, precision: str = "auto"):
        """Download checkpoint and return config dict."""
        import comfy.model_management

        load_device = comfy.model_management.get_torch_device()

        # Resolve dtype
        if precision == "auto":
            if comfy.model_management.should_use_bf16(load_device):
                dtype = torch.bfloat16
            elif comfy.model_management.should_use_fp16(load_device):
                dtype = torch.float16
            else:
                dtype = torch.float32
        else:
            dtype = _DTYPE_MAP[precision]

        dtype_str = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}[dtype]
        _p(f"precision={precision!r} -> dtype={dtype_str} (device={load_device})")

        # Locate the checkpoint, downloading only if it isn't already on disk.
        os.makedirs(MODELS_DIR, exist_ok=True)
        model_path = _find_local_checkpoint()
        if model_path is not None:
            # Already present (in any registered "sharp" model folder, incl.
            # extra_model_paths.yaml) -> use directly, no network call. The SHARP
            # checkpoint is immutable, so a present file is valid.
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            _p(f"checkpoint already present, reusing (no download): {model_path} ({size_mb:.1f} MB)")
        else:
            _p(f"checkpoint not found — downloading from HuggingFace "
               f"repo '{SHARP_REPO_ID}' file '{SHARP_FILENAME}' -> {MODELS_DIR}")
            model_path = _download_with_retry(
                _p,
                repo_id=SHARP_REPO_ID,
                filename=SHARP_FILENAME,
                local_dir=MODELS_DIR,
                tqdm_class=_comfy_tqdm(),
            )

        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        _p(f"checkpoint ready: {model_path} ({size_mb:.1f} MB)")
        _p(f"config: precision={precision} -> dtype={dtype_str}")
        log.info(f"SHARP config: precision={precision} -> dtype={dtype_str}, path={model_path}")

        config = {
            "model_path": model_path,
            "precision": precision,
            "dtype": dtype_str,
        }
        return io.NodeOutput(config)


NODE_CLASS_MAPPINGS = {
    "LoadSharpModel": LoadSharpModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadSharpModel": "(Down)Load SHARP Model",
}

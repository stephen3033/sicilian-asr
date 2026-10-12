import json
import os
import sys
from copy import deepcopy
from pathlib import Path

from common.errors import ResourceError
from common.resources import verify_file
from common.utils import local_path


def external_paths(root: Path, manifest: dict) -> None:
    for name in ("icefall_source", "zipa_source"):
        spec = manifest["resources"][name]
        archive = local_path(root, spec["path"])
        actual = verify_file(archive, spec)
        directory = local_path(root, spec["extract_to"])
        marker = directory / ".sicilian-asr-source.json"
        if not marker.is_file() or json.loads(marker.read_text()) != {
            "revision": spec["revision"],
            "archive_sha256": actual["sha256"],
        }:
            raise ResourceError(
                f"Acquire the pinned {name} with scripts/download_resources.py"
            )
        sys.path.insert(0, str(directory))
    sys.path.insert(0, str(root / ".cache/external/zipa/zipformer_crctc"))
    os.environ["WANDB_MODE"] = "disabled"


def acoustic_worker(root: Path, manifest: dict, mode: str, output: Path) -> dict:
    import resource

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    external_paths(root, manifest)
    import numpy as np
    import torch
    from lhotse.features.kaldi.extractors import Fbank
    from zipa_ctc_inference import large_params
    from zipformer_crctc.train import get_model

    device = "mps" if mode.endswith("mps") else "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        return {
            "status": "unavailable",
            "reason": "MPS is not visible to this process; check hardware and sandbox permissions.",
        }
    torch.manual_seed(0)
    torch.set_num_threads(4)
    spec = manifest["resources"]["zipa_checkpoint"]
    checkpoint = local_path(root, spec["path"])
    verify_file(checkpoint, spec)
    model = get_model(deepcopy(large_params))
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model.eval().requires_grad_(False)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    # Generated signal exercises numerical/device compatibility, never linguistic accuracy.
    audio = [
        0.05 * torch.sin(2 * torch.pi * 220 * torch.arange(n) / 16000)
        for n in (16000, 24000)
    ]
    features = Fbank().extract_batch(audio, sampling_rate=16000)
    feature_lengths = torch.tensor(
        [feature.shape[0] for feature in features], dtype=torch.int64
    )
    features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
    if mode.startswith("scripted"):
        # Use the existing upstream inference conversion and encoder export wrapper.
        from export import EncoderModel
        from scaling_converter import convert_scaled_to_non_scaled

        convert_scaled_to_non_scaled(model, inplace=True)
        encoder = torch.jit.script(
            EncoderModel(model.encoder, model.encoder_embed).eval()
        )
        encoder = encoder.to(device)
        ctc_head = model.ctc_output.to(device)
        forward = encoder
    else:
        model.to(device)
        forward = model.forward_encoder
        ctc_head = model.ctc_output
    with torch.inference_mode():
        encoder_out, lengths = forward(features.to(device), feature_lengths.to(device))
        log_probs = ctc_head(encoder_out).cpu()
        lengths = lengths.cpu()
    if log_probs.ndim != 3 or log_probs.shape[0] != 2 or log_probs.shape[-1] != 127:
        raise ResourceError(f"Unexpected frame output shape: {tuple(log_probs.shape)}")
    if not torch.isfinite(log_probs).all():
        raise ResourceError("Nonfinite CTC frame output")
    if (
        not torch.all((lengths > 0) & (lengths <= log_probs.shape[1]))
        or lengths[0] == lengths[1]
    ):
        raise ResourceError(f"Invalid unequal-batch frame lengths: {lengths}")
    valid = torch.arange(log_probs.shape[1])[None, :] < lengths[:, None]
    normalization_error = log_probs.logsumexp(-1)[valid].abs().max().item()
    if normalization_error > 1e-4:
        raise ResourceError(
            f"CTC outputs are not normalized log probabilities: {normalization_error}"
        )
    np.savez(
        output.with_suffix(".npz"), log_probs=log_probs.numpy(), lengths=lengths.numpy()
    )
    return {
        "status": "ok",
        "device": device,
        "mode": mode,
        "parameter_count": parameter_count,
        "shape": list(log_probs.shape),
        "lengths": lengths.tolist(),
        "normalization_max_abs_error": normalization_error,
        "checkpoint_sha256": spec["sha256"],
        "cpu_fallback_enabled": False,
    }


def compare_frames(reference: Path, candidate: Path) -> dict:
    import numpy as np

    with np.load(reference) as left, np.load(candidate) as right:
        if left["log_probs"].shape != right["log_probs"].shape or not np.array_equal(
            left["lengths"], right["lengths"]
        ):
            return {"status": "error", "reason": "Output shape or lengths differ"}
        mask = np.arange(left["log_probs"].shape[1])[None, :] < left["lengths"][:, None]
        a, b = left["log_probs"][mask], right["log_probs"][mask]
        equivalent = bool(np.allclose(a, b, rtol=1e-3, atol=1e-3))
        return {
            "status": "ok" if equivalent else "error",
            "max_abs_error": float(np.abs(a - b).max()),
            "rtol": 1e-3,
            "atol": 1e-3,
        }

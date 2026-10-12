"""Strict CPU adapter for the pinned upstream ZIPA architecture."""

import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import torch

from common.errors import ResourceError
from common.resources import verify_file
from common.settings import MIN_AUDIO_SAMPLES, SAMPLE_RATE
from common.utils import local_path
from ipa.inventory import inventory
from ipa.types import AcousticOutput


def external_paths(root: Path, manifest: dict) -> None:
    for name in ("icefall_source", "zipa_source"):
        spec = manifest["resources"][name]
        actual = verify_file(local_path(root, spec["path"]), spec)
        directory = local_path(root, spec["extract_to"])
        marker = directory / ".sicilian-asr-source.json"
        if not marker.is_file() or json.loads(marker.read_text()) != {
            "revision": spec["revision"],
            "archive_sha256": actual["sha256"],
        }:
            raise ResourceError(
                f"Missing or incompatible extracted {name}; run prepare"
            )
        sys.path.insert(0, str(directory))
    sys.path.insert(0, str(root / ".cache/external/zipa/zipformer_crctc"))
    os.environ["WANDB_MODE"] = "disabled"


class ZipaAcoustic:
    def __init__(self, root: Path, config: dict, manifest: dict):
        if config["device"] != "cpu":
            raise ResourceError("ZIPA supports CPU only on the pinned stack")
        torch.set_num_threads(config.get("threads", 4))
        external_paths(root, manifest)
        from lhotse.features.kaldi.extractors import Fbank
        from zipa_ctc_inference import large_params
        from zipformer_crctc.train import get_model

        for key in ("checkpoint", "vocabulary", "tokenizer"):
            verify_file(
                local_path(root, config[key + "_path"]),
                manifest["resources"][config[key + "_resource"]],
            )
        self.tokens, self.tokenizer = inventory(
            local_path(root, config["vocabulary_path"]),
            local_path(root, config["tokenizer_path"]),
        )
        self.model = get_model(deepcopy(large_params))
        self.model.load_state_dict(
            torch.load(
                local_path(root, config["checkpoint_path"]),
                weights_only=True,
                map_location="cpu",
            ),
            strict=True,
        )
        self.model.float().eval().requires_grad_(False)
        self.fbank = Fbank()
        self.preprocessing = {
            "sample_rate": SAMPLE_RATE,
            "channels": 1,
            "dtype": "float32",
            "fbank": self.fbank.to_dict(),
            "waveform_normalization": "none",
        }

    def infer(self, audio: list[torch.Tensor]) -> AcousticOutput:
        if not audio or any(
            a.ndim != 1
            or a.numel() < MIN_AUDIO_SAMPLES
            or a.dtype != torch.float32
            or not torch.isfinite(a).all()
            for a in audio
        ):
            raise ResourceError("Expected finite mono float32 audio of at least 100 ms")
        features = self.fbank.extract_batch(audio, sampling_rate=SAMPLE_RATE)
        lengths = torch.tensor([len(f) for f in features], dtype=torch.int64)
        features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
        with torch.inference_mode():
            encoded, lengths = self.model.forward_encoder(features, lengths)
            output = AcousticOutput(
                self.tokens,
                0,
                self.preprocessing,
                self.model.ctc_output(encoded).cpu(),
                lengths.cpu(),
            )
        output.validate()
        return output

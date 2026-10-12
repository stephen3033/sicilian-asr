"""Strict CPU adapter for the pinned upstream ZIPA architecture."""

from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys

import torch

from sicilian_asr.lexicon.prepare import inventory
from sicilian_asr.resources import ResourceError, local_path, verify_file


@dataclass
class AcousticOutput:
    tokens: list[str]
    blank_id: int
    preprocessing: dict
    log_probs: torch.Tensor
    lengths: torch.Tensor

    def validate(self) -> None:
        p, lengths = self.log_probs, self.lengths
        if p.device.type != "cpu" or p.dtype != torch.float32 or p.ndim != 3 or p.shape[-1] != len(self.tokens):
            raise ResourceError("Expected float32 CPU [batch, frames, vocabulary] probabilities")
        if len(set(self.tokens)) != len(self.tokens) or not 0 <= self.blank_id < len(self.tokens):
            raise ResourceError("Invalid token inventory or blank ID")
        if lengths.device.type != "cpu" or lengths.dtype not in (torch.int32, torch.int64) or lengths.shape != (p.shape[0],):
            raise ResourceError("Invalid frame lengths shape/type")
        if not p.shape[0] or not torch.all((lengths > 0) & (lengths <= p.shape[1])):
            raise ResourceError("Invalid frame lengths")
        if not torch.isfinite(p).all():
            raise ResourceError("Nonfinite acoustic probabilities")
        valid = torch.arange(p.shape[1])[None, :] < lengths[:, None]
        if p.logsumexp(-1)[valid].abs().max().item() > 1e-4:
            raise ResourceError("Acoustic outputs are not normalized natural-log probabilities")


def external_paths(root: Path, manifest: dict) -> None:
    for name in ("icefall_source", "zipa_source"):
        spec = manifest["resources"][name]
        actual = verify_file(local_path(root, spec["path"]), spec)
        directory = local_path(root, spec["extract_to"])
        marker = directory / ".sicilian-asr-source.json"
        if not marker.is_file() or json.loads(marker.read_text()) != {"revision": spec["revision"], "archive_sha256": actual["sha256"]}:
            raise ResourceError(f"Missing or incompatible extracted {name}; run prepare")
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
            verify_file(local_path(root, config[key + "_path"]), manifest["resources"][config[key + "_resource"]])
        self.tokens, self.tokenizer = inventory(local_path(root, config["vocabulary_path"]), local_path(root, config["tokenizer_path"]))
        self.model = get_model(deepcopy(large_params))
        self.model.load_state_dict(torch.load(local_path(root, config["checkpoint_path"]), weights_only=True, map_location="cpu"), strict=True)
        self.model.float().eval().requires_grad_(False)
        self.fbank = Fbank()
        self.preprocessing = {"sample_rate": 16000, "channels": 1, "dtype": "float32", "fbank": self.fbank.to_dict(), "waveform_normalization": "none"}

    def infer(self, audio: list[torch.Tensor]) -> AcousticOutput:
        if not audio or any(a.ndim != 1 or a.numel() < 1600 or a.dtype != torch.float32 or not torch.isfinite(a).all() for a in audio):
            raise ResourceError("Expected finite mono float32 audio of at least 100 ms")
        features = self.fbank.extract_batch(audio, sampling_rate=16000)
        lengths = torch.tensor([len(f) for f in features], dtype=torch.int64)
        features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
        with torch.inference_mode():
            encoded, lengths = self.model.forward_encoder(features, lengths)
            output = AcousticOutput(self.tokens, 0, self.preprocessing, self.model.ctc_output(encoded).cpu(), lengths.cpu())
        output.validate()
        return output


def greedy_ctc(output: AcousticOutput) -> list[str]:
    """Ordinary CTC collapse: merge repeats, then remove blanks."""
    output.validate()
    result = []
    for sequence, length in zip(output.log_probs.argmax(-1), output.lengths):
        previous = None
        ids = []
        for identifier in sequence[:int(length)].tolist():
            if identifier != previous and identifier != output.blank_id:
                ids.append(identifier)
            previous = identifier
        result.append("".join(output.tokens[i] for i in ids).replace("▁", " ").strip())
    return result

import pytest
import torch

from common.errors import ResourceError
from ipa.diagnostics import greedy_ctc
from ipa.types import AcousticOutput

TOKENS = ["<blk>", "<sos/eos>", "<unk>", "▁", "a", "b"]


def acoustic(ids, size=6):
    p = torch.full((1, len(ids), size), -30.0, dtype=torch.float32)
    p[0, range(len(ids)), ids] = 0
    return AcousticOutput(
        TOKENS[:size], 0, {}, p.log_softmax(-1), torch.tensor([len(ids)])
    )


def test_ordinary_greedy_collapse_and_valid_lengths():
    output = acoustic([3, 4, 4, 0, 4, 5])
    assert greedy_ctc(output) == ["aab"]
    output.lengths[0] = 5
    assert greedy_ctc(output) == ["aa"]


@pytest.mark.parametrize(
    "change",
    [
        "nonfinite",
        "unnormalized",
        "length_zero",
        "length_large",
        "length_float",
        "dtype",
    ],
)
def test_invalid_acoustic_outputs(change):
    output = acoustic([4, 5])
    if change == "nonfinite":
        output.log_probs[0, 0, 0] = float("nan")
    if change == "unnormalized":
        output.log_probs += 1
    if change == "length_zero":
        output.lengths[0] = 0
    if change == "length_large":
        output.lengths[0] = 3
    if change == "length_float":
        output.lengths = output.lengths.float()
    if change == "dtype":
        output.log_probs = output.log_probs.double()
    with pytest.raises(ResourceError):
        output.validate()

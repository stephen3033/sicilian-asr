"""Full-recording decode from verified, frozen ZIPA posteriors."""

import gc
import resource
import time

from common.artifacts import read
from common.errors import ResourceError


def decode_video(directory, acoustic_dir, acoustic, settings):
    import numpy as np
    import torch

    from asr.wfst.cache import load_graph
    from asr.wfst.compact import decode
    from ipa.types import AcousticOutput

    started = time.perf_counter()
    graph, words = load_graph(directory)
    if not torch.isfinite(graph.scores).all():
        raise ResourceError("Nonfinite graph costs")
    results = []
    tokens = read(directory / "compile_input.json")["tokens"]
    for segment in acoustic["segments"]:
        with np.load(acoustic_dir / segment["path"], allow_pickle=False) as saved:
            if saved["tokens"].tolist() != tokens:
                raise ResourceError("Acoustic and graph label inventories differ")
            values = torch.from_numpy(saved["log_probs"].copy()).unsqueeze(0)
            output = AcousticOutput(
                tokens, 0, {}, values, torch.tensor([values.shape[1]])
            )
        results.append({**segment, **decode(graph, output, words, settings)})
    del graph
    gc.collect()
    return {
        "text": " ".join(r["text"] for r in results),
        "segments": results,
        "decode_seconds": time.perf_counter() - started,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }

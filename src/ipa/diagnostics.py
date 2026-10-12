from ipa.types import AcousticOutput


def greedy_ctc(output: AcousticOutput) -> list[str]:
    """Ordinary CTC collapse: merge repeats, then remove blanks."""
    output.validate()
    result = []
    for sequence, length in zip(output.log_probs.argmax(-1), output.lengths):
        previous = None
        ids = []
        for identifier in sequence[: int(length)].tolist():
            if identifier != previous and identifier != output.blank_id:
                ids.append(identifier)
            previous = identifier
        result.append("".join(output.tokens[i] for i in ids).replace("▁", " ").strip())
    return result

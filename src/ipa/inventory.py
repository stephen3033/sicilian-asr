from pathlib import Path

from common.errors import ResourceError


def inventory(path: Path, tokenizer_path: Path) -> tuple[list[str], object]:
    import sentencepiece as spm

    table = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        token, value = line.rsplit(" ", 1)
        identifier = int(value)
        if identifier in table or token in table.values():
            raise ResourceError("Duplicate acoustic token/ID")
        table[identifier] = token
    if set(table) != set(range(127)) or [table[i] for i in range(4)] != [
        "<blk>",
        "<sos/eos>",
        "<unk>",
        "▁",
    ]:
        raise ResourceError("Expected pinned ZIPA 127-token inventory")
    tokens = [table[i] for i in range(len(table))]
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    if tokenizer.get_piece_size() != len(tokens) or any(
        tokenizer.id_to_piece(i) != t for i, t in enumerate(tokens)
    ):
        raise ResourceError("Tokenizer and acoustic IDs disagree")
    return tokens, tokenizer

"""Conservative whole-word label observations from training-only alignments."""

import math
from collections import Counter, defaultdict


def interior_word_labels(words, seed_labels, alignment):
    """Assign labels only to unambiguous interior words of an aligned cue.

    Boundaries are provisional seed-alignment boundaries, not forced audio
    boundaries. Insertions internal to a word are retained. An insertion at a
    word boundary excludes both adjacent words. Cue-edge words are excluded.
    """
    if len(words) != len(seed_labels) or any(not labels for labels in seed_labels):
        raise ValueError("Each word must have a nonempty seed-label sequence")
    flattened = [label for labels in seed_labels for label in labels]
    owners = [index for index, labels in enumerate(seed_labels) for _ in labels]
    observed = [[] for _ in words]
    ambiguous = set()
    cursor = 0
    for pair in alignment:
        reference, hypothesis = pair["reference"], pair["hypothesis"]
        if reference is not None:
            if cursor >= len(flattened) or reference != flattened[cursor]:
                raise ValueError("Alignment does not match the word seed sequences")
            if hypothesis is not None:
                observed[owners[cursor]].append(hypothesis)
            cursor += 1
        elif hypothesis is not None:
            left = owners[cursor - 1] if cursor else None
            right = owners[cursor] if cursor < len(owners) else None
            if left is not None and left == right:
                observed[left].append(hypothesis)
            else:
                ambiguous.update(index for index in (left, right) if index is not None)
    if cursor != len(flattened):
        raise ValueError("Alignment omits reference labels")
    return [
        {"word": word, "labels": observed[index]}
        for index, word in enumerate(words)
        if 0 < index < len(words) - 1 and index not in ambiguous and observed[index]
    ]


def recurring_word_labels(observations):
    """Keep paths observed >=3 times, in >=3 recordings, and >=20% per word."""
    counts, totals, support = Counter(), Counter(), defaultdict(set)
    for row in observations:
        key = row["word"], tuple(row["labels"])
        counts[key] += 1
        totals[row["word"]] += 1
        support[key].add(row["video"])
    return [
        {
            "word": word,
            "labels": list(labels),
            "count": count,
            "total": totals[word],
            "recordings": sorted(support[word, labels]),
        }
        for (word, labels), count in sorted(counts.items())
        if count >= 3
        and len(support[word, labels]) >= 3
        and count / totals[word] >= 0.2
    ]


def add_observed_word_paths(lexicon, accepted, tokens, mode, strength):
    """Add exact observed paths while retaining scored seed-based fallbacks.

    Empirical priors center at the highest retained word-specific count, with
    0.5 smoothing. The general seed mapping is a fallback with pseudo-count 0.5;
    observed paths carry their own exact labels, avoiding a second application
    of seed-to-ZIPA substitutions. Vocabulary cannot expand in this control.
    """
    if (
        mode not in ("append", "empirical")
        or not math.isfinite(strength)
        or strength <= 0
    ):
        raise ValueError("Invalid word-path mode or strength")
    result = {word: [list(path) for path in paths] for word, paths in lexicon.items()}
    settings, evidence = {}, []
    by_word = defaultdict(list)
    identifiers = {label: index for index, label in enumerate(tokens)}
    for row in accepted:
        if row["word"] in result:
            by_word[row["word"]].append(row)
    for word, rows in sorted(by_word.items()):
        maximum = max(row["count"] for row in rows)
        fallback = (
            strength * math.log((maximum + 0.5) / 0.5) if mode == "empirical" else 0.0
        )
        settings[word] = [{"cost": fallback, "exact": False} for _ in result[word]]
        for row in rows:
            ids = [identifiers.get(label, -1) for label in row["labels"]]
            if not ids or any(identifier < 4 for identifier in ids):
                raise ValueError("Observed word paths require nonempty lexical labels")
            cost = (
                strength * math.log((maximum + 0.5) / (row["count"] + 0.5))
                if mode == "empirical"
                else strength
            )
            # A duplicate token sequence is intentional when the old path has
            # substitutions/omissions and this observed path must remain exact.
            result[word].append(ids)
            settings[word].append({"cost": cost, "exact": True})
            evidence.append({**row, "cost": cost, "fallback_cost": fallback})
    return result, settings, evidence

"""Estimate decoder label preferences from aligned training observations.

Seed labels are coordinates for the existing lexicon, not IPA ground truth.
Counts measure ZIPA/seed alignment agreement and can include alignment noise.
"""

import math
from collections import Counter, defaultdict


def empirical_label_costs(records, tokens, strength=1.0):
    """Return supported alternatives and seed costs, with equal smoothing.

    Retain substitutions seen >=20 times, for >=10% of a seed's occurrences,
    in >=3 recordings. Both seed and alternative counts receive 0.5. Center
    log costs at the most frequent retained label (zero cost); this avoids
    imposing an uncertainty penalty on every occurrence of an ambiguous seed.
    Deletions and insertions are counted for support but are not graph options.
    """
    if not math.isfinite(strength) or strength <= 0:
        raise ValueError("Mapping strength must be finite and positive")
    observations, totals, support = Counter(), Counter(), defaultdict(set)
    for record in records:
        for pair in record["alignment"]:
            seed, observed = pair["reference"], pair["hypothesis"]
            if seed is None:
                continue
            totals[seed] += 1
            if observed is not None:
                observations[seed, observed] += 1
                support[seed, observed].add(record["video"])
    accepted = defaultdict(list)
    for (seed, observed), count in sorted(observations.items()):
        if (
            seed != observed
            and count >= 20
            and count / totals[seed] >= 0.1
            and len(support[seed, observed]) >= 3
            and seed in tokens[4:]
            and observed in tokens[4:]
        ):
            accepted[seed].append(observed)
    alternatives, seed_costs, evidence = {}, {}, []
    for seed, observed in sorted(accepted.items()):
        labels = [seed, *observed]
        maximum = max(observations[seed, label] for label in labels)
        options = []
        for label in labels:
            count = observations[seed, label]
            cost = strength * math.log((maximum + 0.5) / (count + 0.5))
            options.append(
                {
                    "label": label,
                    "count": count,
                    "recordings": len(support[seed, label]),
                    "cost": cost,
                }
            )
            if label == seed:
                seed_costs[str(tokens.index(seed))] = cost
            else:
                alternatives.setdefault(str(tokens.index(seed)), []).append(
                    {"token": tokens.index(label), "cost": cost}
                )
        evidence.append({"seed": seed, "total": totals[seed], "options": options})
    return alternatives, seed_costs, evidence


def positional_label_costs(rows, tokens, strength=1.0):
    """Estimate final/nonfinal costs from aggregated training word evidence.

    Each accepted positional substitution needs the same 20 occurrences,
    10% fraction, and three-recording support as the global estimate. Missing
    positions fall back to the caller's global mapping; seed fallbacks receive
    identical 0.5 smoothing. Do not use the number of aggregated rows as counts.
    """
    if not math.isfinite(strength) or strength <= 0:
        raise ValueError("Mapping strength must be finite and positive")
    counts, totals, support = Counter(), Counter(), defaultdict(set)
    for row in rows:
        position = (
            "final"
            if row["seed_label_offset"] == row["seed_label_length"] - 1
            else "nonfinal"
        )
        seed, observed = row["seed"], row["observed"]
        totals[position, seed] += row["count"]
        if observed is not None:
            counts[position, seed, observed] += row["count"]
            support[position, seed, observed].update(row["recordings"])
    accepted = defaultdict(list)
    for (position, seed, observed), count in sorted(counts.items()):
        if (
            seed != observed
            and count >= 20
            and count / totals[position, seed] >= 0.1
            and len(support[position, seed, observed]) >= 3
            and seed in tokens[4:]
            and observed in tokens[4:]
        ):
            accepted[position, seed].append(observed)
    mapping, evidence = {}, []
    for (position, seed), observed in sorted(accepted.items()):
        labels = [seed, *observed]
        maximum = max(counts[position, seed, label] for label in labels)
        options, measured = [], []
        for label in labels:
            count = counts[position, seed, label]
            cost = strength * math.log((maximum + 0.5) / (count + 0.5))
            options.append({"token": tokens.index(label), "cost": cost})
            measured.append(
                {
                    "label": label,
                    "count": count,
                    "cost": cost,
                    "recordings": len(support[position, seed, label]),
                }
            )
        mapping.setdefault(position, {})[str(tokens.index(seed))] = options
        evidence.append(
            {
                "position": position,
                "seed": seed,
                "total": totals[position, seed],
                "options": measured,
            }
        )
    return mapping, evidence

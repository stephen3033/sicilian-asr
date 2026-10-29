"""Bounded full-recording repair experiments; never change production selection."""

from copy import deepcopy
from pathlib import Path

from common.artifacts import digest, read, sha, verified_files
from common.errors import ResourceError
from common.utils import write_json
from pipeline.experiment import aggregate, decode_bundle, improves
from pipeline.scn_experiment import candidate, rank
from pipeline.scn_residual_data import candidate_effects


def semantic_settings(settings):
    def canonical(value):
        if isinstance(value, dict):
            return {k: canonical(v) for k, v in value.items()}
        if isinstance(value, list):
            return [canonical(v) for v in value]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return value

    return {
        k: canonical(v)
        for k, v in settings.items()
        if k not in ("compress_artifacts", "graph_path")
    }


def candidate_key(lexicon, arpa, tokens, settings, decoder):
    return digest(
        {
            "lexicon": digest(lexicon),
            "arpa": sha(arpa),
            "tokens": tokens,
            "settings": semantic_settings(settings),
            "decoder": semantic_settings(decoder),
        }
    )


class Trials:
    def __init__(self, root, bundle, output, recipe):
        self.root, self.bundle, self.output, self.recipe = root, bundle, output, recipe
        inputs = read(bundle / "graph/compile_input.json")
        self.tokens, self.settings = inputs["tokens"], inputs["settings"]
        self.lexicon = read(bundle / "graph/lexicon_selected.json")
        self.decoder = read(bundle / "decoder.json")
        self.baseline = read(output / "baseline-replay/result.json")
        self.inventory = read(output / "inventory.json")
        self.references = {
            r["video"]: r["reference"] for r in read(bundle / "recordings.json")
        }
        self.known = {}
        baseline = {
            "id": "baseline",
            "graph": str(bundle / "graph"),
            "decoder": self.decoder,
            "compile_settings": self.settings,
            "metrics": self.baseline["metrics"],
            "result": str(output / "baseline-replay/result.json"),
            "status": "complete",
        }
        prior = [baseline]
        for filename in ("lm-results.json", "pronunciation-results.json"):
            path = root / recipe["corpus_study"] / filename
            if path.exists():
                prior += read(path)["results"]
        for row in prior:
            graph = Path(row["graph"])
            if not (graph / "graph.json").exists() or not Path(row["result"]).exists():
                continue
            graph_receipt = read(graph / "graph.json")
            verified_files(graph, graph_receipt)
            result_path = Path(row["result"])
            decode_receipt = read(result_path.parent / "result.receipt.json")
            verified_files(result_path.parent, decode_receipt)
            identity = decode_receipt["identity"]
            if identity.get("bundle") != sha(bundle / "bundle.json") or any(
                sha(root / "src" / name) != expected
                for name, expected in identity.get("runtime", {}).items()
            ):
                continue
            if row["id"] != "baseline" and graph_receipt["identity"].get(
                "compiler"
            ) != sha(root / "src/asr/wfst/graph.py"):
                continue
            key = candidate_key(
                read(graph / "lexicon_selected.json"),
                graph / "restricted.arpa",
                read(graph / "compile_input.json")["tokens"],
                row["compile_settings"],
                row["decoder"],
            )
            self.known[key] = row

    def evaluate(
        self, label, arpa, settings=None, decoder=None, lexicon=None, **metadata
    ):
        settings = deepcopy(settings if settings is not None else self.settings)
        decoder = deepcopy(decoder if decoder is not None else self.decoder)
        lexicon = lexicon if lexicon is not None else self.lexicon
        key = candidate_key(lexicon, arpa, self.tokens, settings, decoder)
        if key in self.known:
            original = self.known[key]
            row = {
                **original,
                "id": label,
                "reused_from": original["id"],
                "reuse_identity": key,
            }
        else:
            print("Residual candidate", label, flush=True)
            row = candidate(
                self.bundle,
                self.output,
                label,
                lexicon,
                arpa,
                {**settings, "compress_artifacts": True},
                decoder,
            )
            self.known[key] = row
        result = read(row["result"])
        if {r["video"] for r in result["recordings"]} != set(
            self.references
        ) or aggregate(result["recordings"]) != result["metrics"]:
            raise ResourceError("Repair trial did not score all complete recordings")
        effects = candidate_effects(
            self.inventory["errors"],
            self.baseline["recordings"],
            result["recordings"],
            self.references,
        )
        return {
            **row,
            **metadata,
            "semantic_identity": key,
            "effects": effects,
            "improves_baseline": improves(row["metrics"], self.baseline["metrics"]),
            "per_recording": [
                {
                    "video": r["video"],
                    "metrics": {k: r["metrics"][k] for k in ("wer", "cer")},
                }
                for r in result["recordings"]
            ],
        }


def calibrate(root, bundle, output, recipe):
    study = read(output / "lm/models.json")
    if study["status"] != "complete":
        raise ResourceError("Complete the repaired language models before calibration")
    trials = Trials(root, bundle, output, recipe)
    old = read(root / recipe["corpus_study"] / "lm-results.json")["results"]
    strongest = min((r for r in old if r["kind"] == "converted"), key=rank)
    families = [
        {
            "id": "baseline",
            "arpa": bundle / "graph/restricted.arpa",
            "kind": "baseline",
        },
        {
            "id": "converted-original",
            "arpa": Path(strongest["graph"]) / "restricted.arpa",
            "kind": "converted",
            "parent": strongest["lm"],
        },
    ]
    families += [
        {**m, "arpa": Path(m["directory"]) / "model.arpa"} for m in study["models"]
    ]
    results = []
    for family in families:
        for scale in recipe["calibration"]["lm_scales"]:
            for bonus in recipe["calibration"]["word_bonuses"]:
                label = f"{family['id']}-lm{scale:g}-b{bonus:g}"
                settings = {**trials.settings, "lm_scale": scale, "word_bonus": bonus}
                decoder = {**trials.decoder, "lm_scale": scale, "word_bonus": bonus}
                row = trials.evaluate(
                    label,
                    family["arpa"],
                    settings,
                    decoder,
                    family=family["id"],
                    kind=family["kind"],
                    intervention="LM construction and global calibration",
                )
                results.append(row)
                write_json(
                    output / "calibration.json",
                    {
                        "status": "running",
                        "results": results,
                        "planned": len(families)
                        * len(recipe["calibration"]["lm_scales"])
                        * len(recipe["calibration"]["word_bonuses"]),
                    },
                )
    comparisons = []
    for row in results:
        if row["family"] != "converted-w50-o3-support":
            continue
        raw = next(
            r for r in results if r["id"] == row["id"].replace("converted", "raw")
        )
        comparisons.append(
            {
                "converted": row["id"],
                "raw": raw["id"],
                "conversion_improves": improves(row["metrics"], raw["metrics"]),
                "wer_error_delta": row["metrics"]["wer"]["errors"]
                - raw["metrics"]["wer"]["errors"],
                "cer_error_delta": row["metrics"]["cer"]["errors"]
                - raw["metrics"]["cer"]["errors"],
            }
        )
    result = {
        "status": "complete",
        "results": results,
        "matched_controls": comparisons,
        "baseline": trials.baseline["metrics"],
        "production_changed": False,
        "policy": "Every changed LM receives the complete declared calibration grid, regardless of its default-setting score. Same frozen complete recordings, vocabulary and pronunciation paths. Reuse requires checksummed graph and result plus matching semantic inputs.",
        "historical_sweeps": "Retired historical results are evidence, not executable cache entries. Their different parents/scales/bonuses do not establish equivalence to these trials.",
    }
    write_json(output / "calibration.json", result)
    return result


def scale_pronunciation_costs(settings, scale):
    result = deepcopy(settings)

    def costs(value):
        if isinstance(value, list):
            return [costs(v) for v in value]
        if isinstance(value, dict):
            return {k: v * scale if k == "cost" else costs(v) for k, v in value.items()}
        return value

    for key in (
        "phone_alternatives",
        "phone_position_costs",
        "phone_context_costs",
        "pronunciation_settings",
    ):
        if result.get(key):
            result[key] = costs(result[key])
    for key in ("phone_seed_costs", "phone_optional_costs"):
        if result.get(key):
            result[key] = {k: v * scale for k, v in result[key].items()}
    return result


def boundary_outcomes(trials, directory):
    import copy

    rows = []
    for recording in trials.baseline["recordings"]:
        manifest = read(directory / recording["video"] / "acoustic.json")
        verified_files(directory / recording["video"], manifest)
        for probe in manifest["segments"]:
            pair = probe["pair"]
            candidate_recordings = copy.deepcopy(trials.baseline["recordings"])
            replaced = next(
                r for r in candidate_recordings if r["video"] == recording["video"]
            )
            original = replaced["segments"]
            replaced["text"] = " ".join(
                [
                    *(s["text"] for s in original[: pair["left"]]),
                    probe["text"],
                    *(s["text"] for s in original[pair["right"] + 1 :]),
                ]
            )
            from pipeline.experiment import score

            replaced["metrics"] = score(
                replaced["text"], trials.references[recording["video"]]
            )
            effects = candidate_effects(
                trials.inventory["errors"],
                trials.baseline["recordings"],
                candidate_recordings,
                trials.references,
            )
            rows.append(
                {
                    "pair": pair,
                    "joined_text": probe["text"],
                    "baseline_text": " ".join(
                        s["text"] for s in original[pair["left"] : pair["right"] + 1]
                    ),
                    "metrics": aggregate(candidate_recordings),
                    "effects": effects,
                    "diagnostic_only": True,
                    "posterior_receipt": str(
                        directory
                        / recording["video"]
                        / f"{probe['index']:05d}.receipt.json"
                    ),
                }
            )
    write_json(
        trials.output / "boundary-outcomes.json",
        {
            "status": "complete",
            "pairs": rows,
            "policy": "Each probe replaces just one adjacent pair in the full baseline transcription. ZIPA was rerun on the continuous waveform; no posterior concatenation. Overlapping probes are not jointly deployable and never supply training labels.",
        },
    )
    return rows


def alternate_segmentation(trials, acoustic_bundle, parent, label):
    result = decode_bundle(
        acoustic_bundle,
        Path(parent["graph"]),
        trials.output / "decodes" / label,
        parent["decoder"],
    )
    effects = candidate_effects(
        trials.inventory["errors"],
        trials.baseline["recordings"],
        result["recordings"],
        trials.references,
    )
    return {
        **parent,
        "id": label,
        "family": "segmentation",
        "intervention": "30-second target / 36-second maximum segmentation with frozen ZIPA",
        "metrics": result["metrics"],
        "result": str(trials.output / "decodes" / label / "result.json"),
        "acoustic_bundle": str(acoustic_bundle),
        "effects": effects,
        "improves_baseline": improves(result["metrics"], trials.baseline["metrics"]),
        "per_recording": [
            {
                "video": r["video"],
                "metrics": {k: r["metrics"][k] for k in ("wer", "cer")},
            }
            for r in result["recordings"]
        ],
    }


def run_repairs(root, bundle, output, recipe):
    from pipeline.scn_residual_audio import prepare_audio
    from pipeline.scn_residual_lm import BackoffModel
    from pipeline.scn_residual_pronunciation import audit as pronunciation_audit
    from pipeline.scn_residual_search import audit as search_audit

    state = output / "active-scoring.json"
    if state.exists() and read(state)["status"] != "complete":
        raise ResourceError(
            "Counterfactual scoring is active or failed; old scores cannot authorize repair tests"
        )
    scores = read(output / "scores.json")
    if scores["status"] != "complete" or not scores["factor_parity_verified"]:
        raise ResourceError(
            "Complete and verify all residual counterfactual scores first"
        )
    trials = Trials(root, bundle, output, recipe)
    results = []
    baseline = trials.evaluate(
        "baseline-control", bundle / "graph/restricted.arpa", family="baseline"
    )

    def record(row):
        results.append(row)
        write_json(
            output / "repair-results.json", {"status": "running", "results": results}
        )

    # Isolate the normalization defect in the historical LM, preserving every
    # original context and lexical path. Tiny support mass changes no count.
    normal_dir = output / "baseline-normalized"
    normal_dir.mkdir(parents=True, exist_ok=True)
    normalized = BackoffModel(bundle / "graph/restricted.arpa")
    normalization = normalized.supported(trials.lexicon, 1e-12)
    normalized.write(normal_dir / "model.arpa")
    write_json(normal_dir / "audit.json", normalization)
    for scale in recipe["calibration"]["lm_scales"]:
        for bonus in recipe["calibration"]["word_bonuses"]:
            record(
                trials.evaluate(
                    f"baseline-normalized-lm{scale:g}-b{bonus:g}",
                    normal_dir / "model.arpa",
                    {**trials.settings, "lm_scale": scale, "word_bonus": bonus},
                    {**trials.decoder, "lm_scale": scale, "word_bonus": bonus},
                    family="baseline-normalized",
                    intervention="Normalize historical ARPA distributions; unchanged observed n-grams",
                )
            )
    pronunciation = pronunciation_audit(bundle, output, recipe)
    analysis = read(output / "pronunciation/analysis.json")
    global_settings = {
        **trials.settings,
        "phone_alternatives": analysis["global_alternatives"],
        "phone_seed_costs": analysis["global_seed_costs"],
        "phone_position_costs": None,
    }
    # Historical controls are checksummed and reused when fully equivalent.
    for name, settings in (
        ("channel-global", global_settings),
        (
            "channel-positional",
            {**global_settings, "phone_position_costs": analysis["positional_costs"]},
        ),
        (
            "channel-context",
            {**trials.settings, "phone_context_costs": analysis["phone_context_costs"]},
        ),
    ):
        record(
            trials.evaluate(
                name,
                bundle / "graph/restricted.arpa",
                settings,
                family="pronunciation-control",
                intervention="Training-only label mapping control",
            )
        )
    if pronunciation["missing_accepted_paths"]:
        lexicon, settings = deepcopy(trials.lexicon), deepcopy(trials.settings)
        for row in pronunciation["missing_accepted_paths"]:
            word = row["word"]
            if word not in lexicon:
                continue
            existing = settings.setdefault("pronunciation_settings", {}).setdefault(
                word, [{} for _ in lexicon[word]]
            )
            lexicon[word].append([trials.tokens.index(t) for t in row["labels"]])
            existing.append({"cost": 0.0, "exact": True})
        record(
            trials.evaluate(
                "recurring-channel-paths",
                bundle / "graph/restricted.arpa",
                settings,
                lexicon=lexicon,
                family="pronunciation-repair",
                intervention="Add missing recurring exact channel paths; preserve all baseline alternatives",
            )
        )
    flags = {
        f
        for segment in scores["segments"]
        for c in segment["cases"]
        for f in c["mechanistic_flags"]
    }
    if "pronunciation_cost_reversal" in flags:
        for factor in recipe["pronunciation"].get("cost_scales", (0.0, 0.5, 1.5)):
            record(
                trials.evaluate(
                    f"pronunciation-cost-x{factor:g}",
                    bundle / "graph/restricted.arpa",
                    scale_pronunciation_costs(trials.settings, factor),
                    family="pronunciation-calibration",
                    intervention="Global lexical cost calibration; no label/path learning",
                    cost_scale=factor,
                )
            )
    search = search_audit(bundle, output)
    if search["proven_search_errors"]:
        for search_beam, output_beam, maximum in ((40, 16, 20000), (60, 20, 40000)):
            decoder = {
                **trials.decoder,
                "search_beam": search_beam,
                "output_beam": output_beam,
                "max_active_states": maximum,
            }
            record(
                trials.evaluate(
                    f"search-{search_beam}-{output_beam}-{maximum}",
                    bundle / "graph/restricted.arpa",
                    decoder=decoder,
                    family="search",
                    intervention="Wider search after a better compiled path was demonstrated",
                )
            )
    probes = prepare_audio(root, bundle, output, recipe, "boundaries")
    boundary_outcomes(trials, probes)
    resegmented = prepare_audio(root, bundle, output, recipe, "segmentation")
    segmentation = alternate_segmentation(
        trials, resegmented, baseline, "segmentation-30-36"
    )
    record(segmentation)
    result = {
        "status": "complete",
        "results": results,
        "search": search,
        "baseline": baseline,
        "segmentation": segmentation,
        "resegmented": str(resegmented),
    }
    write_json(output / "repair-results.json", result)
    return result


def run_tests(root, bundle, output, recipe):
    calibration = read(output / "calibration.json")
    if calibration["status"] != "complete":
        calibration = calibrate(root, bundle, output, recipe)
    repairs = run_repairs(root, bundle, output, recipe)
    trials = Trials(root, bundle, output, recipe)
    results = [*calibration["results"], *repairs["results"]]
    baseline = next(r for r in results if r["id"] == "baseline-lm0.8-b0")
    search, segmentation, resegmented = (
        repairs["search"],
        repairs["segmentation"],
        Path(repairs["resegmented"]),
    )

    def record(row):
        results.append(row)
        write_json(
            output / "interventions.json", {"status": "running", "results": results}
        )

    # Combine only independently Pareto-improving changes. No production file
    # or historical best bundle is updated by this research selection.
    lm_winners = [
        r
        for r in results
        if r.get("family")
        not in (
            "segmentation",
            "search",
            "pronunciation-control",
            "pronunciation-calibration",
            "pronunciation-repair",
        )
        and r["improves_baseline"]
    ]
    pron_winners = [
        r
        for r in results
        if r.get("family", "").startswith("pronunciation") and r["improves_baseline"]
    ]
    if lm_winners and pron_winners:
        language, pron = min(lm_winners, key=rank), min(pron_winners, key=rank)
        combined = {
            **language["compile_settings"],
            **{
                k: v
                for k, v in pron["compile_settings"].items()
                if k.startswith("phone_") or k == "pronunciation_settings"
            },
        }
        record(
            trials.evaluate(
                "combined-lm-pronunciation",
                Path(language["graph"]) / "restricted.arpa",
                combined,
                language["decoder"],
                read(Path(pron["graph"]) / "lexicon_selected.json"),
                family="combined",
                intervention="Combine independently supported LM and pronunciation changes",
                parents=[language["id"], pron["id"]],
            )
        )
    if segmentation["improves_baseline"]:
        eligible = [
            r
            for r in results
            if r["improves_baseline"] and r["family"] != "segmentation"
        ]
        if eligible:
            record(
                alternate_segmentation(
                    trials,
                    resegmented,
                    min(eligible, key=rank),
                    "combined-segmentation",
                )
            )
    winners = [r for r in results if r["improves_baseline"]]
    selected = min(winners, key=rank) if winners else baseline
    replay_bundle = Path(selected.get("acoustic_bundle", bundle))
    fresh = decode_bundle(
        replay_bundle,
        Path(selected["graph"]),
        output / "selected-fresh-replay",
        selected["decoder"],
        fresh=True,
    )
    old = read(selected["result"])
    if fresh["metrics"] != selected["metrics"] or {
        r["video"]: r["text"] for r in fresh["recordings"]
    } != {r["video"]: r["text"] for r in old["recordings"]}:
        raise ResourceError(
            "Research selection failed a fresh complete-recording replay"
        )
    from pipeline.scn_experiment import export_candidate

    export_candidate(
        bundle,
        Path(selected["graph"]),
        selected["decoder"],
        selected,
        output / "research-candidate",
    )
    import yaml

    configuration = {
        "languages": ["scn"],
        "output_language": "scn",
        "resources": {
            "scn": {
                "candidate_bundle_path": str(
                    (output / "research-candidate").relative_to(root)
                )
            }
        },
        "decoder": {
            **selected["decoder"],
            "graph_path": str((output / "prepared-research").relative_to(root)),
        },
    }
    if selected.get("acoustic_bundle"):
        configuration["audio"] = recipe["segmentation"]
    (output / "experimental-scn.yaml").write_text(
        yaml.safe_dump(configuration, sort_keys=False, allow_unicode=True)
    )
    result = {
        "status": "complete",
        "results": results,
        "selected": selected,
        "fresh_replay": str(output / "selected-fresh-replay/result.json"),
        "search": search,
        "production_changed": False,
        "policy": "Development selection only. Reference-constrained scores never become training evidence. Improvement requires lower WER without worse CER, or lower CER at equal WER. Independent acceptance on fresh recordings remains pending.",
    }
    write_json(output / "interventions.json", result)
    return result

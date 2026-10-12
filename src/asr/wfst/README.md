# Shared WFST layer

Consumes IPA frame distributions and a prepared graph; returns words, path scores, and optional diagnostics. Language preparation supplies the lexicon, word table, and restricted ARPA.

| Module | Responsibility |
| --- | --- |
| `compact.py` | Compact topology, frame adapter, search and word recovery |
| `graph.py` | Lexicon disambiguation and static TLG composition |
| `arpa.py` | Read original ARPA weights and retain selected n-grams |
| `cache.py` | Verify completed graph receipts and artifact hashes |
| `__main__.py` | Isolated native compiler worker |

## Compact CTC

The topology and tensor construction follow [Laptev et al. (2022)](https://www.isca-archive.org/interspeech_2022/laptev22_interspeech.pdf) and its [NeMo implementation](https://github.com/NVIDIA/NeMo/blob/v1.23.0/nemo/collections/asr/parts/k2/topologies.py). This is not k2's `modified=True` topology.

The initial hub emits each acoustic label to its own state, including a separate blank state. Each such state can self-loop on that label with epsilon output or return to the hub on a decoder-only auxiliary label. Only the hub has a final transition. Separating the blank state prevents a neutral frame following a real blank from emitting a new phone. In ZIPA, blank remains ID 0, auxiliary input is 127, and lexicon disambiguators begin at 128. FST output epsilon uses 0; this is distinct from consuming an acoustic blank on the input side.

The adapter creates a new tensor with twice as many frames and one extra column. Even frames contain the original scores (times the acoustic scale), with auxiliary probability zero (`-inf` log score). Odd frames contain zero log score for every column. Valid lengths double. Original matrices and optional exports are unchanged. The neutral scores are not renormalized: they implement transitions, not new acoustic evidence.

Compact CTC is not ordinary greedy CTC. In particular, two consecutive identical acoustic labels can yield one or two output labels, depending on self-loop versus auxiliary return. Blank-separated identical labels yield two labels. The tests explicitly preserve this published distinction and check final-frame termination.

## Static composition and weights

The lexicon permits one optional `▁` at segment start. Its word loop never inserts a boundary between phones or words. Distinct pronunciations are preserved. Optional Italian compatibility rules add doubled-consonant and prevelar-nasal alternatives and derive MLS-attested `l'`/`c'` contractions before vocabulary selection; their provenance is recorded separately from the source audit. Homophones and pronunciations that prefix another pronunciation receive disambiguators; `#0` propagates grammar backoff. `<unk>` maps to acoustic ID 2.

`kaldilm==1.15.4` compiles the restricted original ARPA to OpenFst text with `#0` backoff. k2 imports costs as natural-log scores. Following [Icefall's pinned recipe](https://github.com/k2-fsa/icefall/blob/c1eda5427c46502b2cfdcbeb910bf35f2134a0d4/egs/librispeech/ASR/local/compile_hlg.py), composition performs `L∘G`, connection, determinization, disambiguator removal, epsilon removal, connection/arc sorting, then `T∘LG`. Intermediate graphs, word/token tables, restricted ARPA, audits, and stage statistics remain cached.

The graph uses tropical best-path backoff as in the Icefall recipe. Controlled fixtures compare resulting path scores against KenLM, including sentence boundaries, unknowns, and missing n-grams. This does not establish equivalence for every arbitrary ARPA: epsilon backoff paths can compete with explicit n-grams under tropical decoding. Original weights are preserved; no word insertion penalty, LM training, pruning, or renormalization is added.

The original defaults were acoustic/LM scales 1/1, search/output beams 20/8, and active-state bounds 30/10000; `src/asr/ita/config.yaml` records the active settings. LM scale is baked into `G` and therefore changes the cache key; runtime acoustic scale acts only inside the frame adapter. Search consumes complete distributions through `k2.DenseFsaVec`, `intersect_dense_pruned(allow_partial=False)`, and shortest-path word recovery. Callers processing many recordings can retain a one-element graph vector, avoiding a multi-GB graph copy for every call.

Every decoded segment records the chosen pronunciation and separates its scaled acoustic score from its scaled grammar score. `--diagnostics` additionally samples 32 lattice paths with a fixed seed, includes the actual best path, and retains the highest sampled path score per distinct sentence. These are competing sampled paths, not exact N-best sentence scores or confidence estimates. Diagnostic mode exports original probabilities and includes OOV causes; references never enter candidate generation.


## Cache integrity

Graph identity includes source hashes/revisions, original ARPA hash, rule versions, vocabulary and pronunciation settings, package versions, and compiler source fingerprints. Every prepared artifact has a SHA-256. Stale, modified, or incomplete caches fail verification. Native compilation runs in its own process so an abort cannot leave a successful preparation receipt. `prepare.py` under Italian ASR owns the language-specific identity and workflow.

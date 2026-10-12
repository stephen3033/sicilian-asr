# Minimum path to a first DOS24 Sicilian WFST

**Resource preparation is implemented; the Sicilian decoder is not.** The
[expanded audit](../dos24-audit.md) now supplies a small contextually verified
DOS24 slice and a larger supported/inferred pool. A sufficiently broad reviewed
training corpus and validated word–IPA seed remain to be developed. ZIPA remains fixed.

## Compatibility with the actual repository

| Component | Current implementation | Reuse / future Sicilian work |
|---|---|---|
| Audio and ZIPA inference | `media_input/media.py`, `audio/segmentation.py`, `ipa/zipa.py`; existing resampling, chunking and 127-label frame scores | Reuse unchanged; validate Sicilian phones on independent speech before claiming accuracy |
| IPA identity | `ipa/inventory.py::inventory`; `tokens.txt` must match `unigram_127.model` exactly | Reuse inventory checks; retain broad and regional phonetic distinctions in source records |
| Italian lexicon conversion | PhonItalia TSV, Italian normalization, concatenation, `ɡ→g`, rejection of unknown/non-round-trip sequences | New SCN conversion entry point and DOS24 normalization; preserve Italian converter |
| Pronunciation variants | Italian `adapt`: length/closure and prevelar nasal alternatives; LM-attested Italian elisions | Reuse idea and audit format; do not apply Italian contraction rules or incomplete consonant lists to SCN |
| LM acquisition | Existing published MLS Italian trigram archive | No reusable Sicilian weights; final SCN estimation remains future work |
| LM handling | `asr/wfst/arpa.py` and `asr/ita/lm.py` parses/restricts ARPA and selects lexicon overlap; it **does not estimate an LM** | Generic ARPA parsing useful; normalization, vocabulary selection and order handling need separate SCN support |
| Order constraints | `restrict_arpa` requires orders 1–3; `graph.py` calls `kaldilm.arpa2fst(max_order=3)` | Unigram/bigram/four-gram work requires future parameterization and tests; do not fabricate empty higher orders |
| CTC topology | `asr/wfst/compact.py`; auxiliary graph label after 127 acoustic labels | Reuse unchanged for the same acoustic inventory; auxiliary/disambiguation labels are not new acoustic phones |
| Lexicon FST | `lexicon_fst`: phone-token input, word-ID output, homophone/prefix disambiguation; optional once-only `▁`, no interword marker | Core mechanics reusable with validated SCN pronunciations and separate word table |
| Grammar composition | ARPA to G via kaldilm; k2 L∘G, determinization, disambiguation removal, T∘LG | Reuse algorithms; new resource identities and controlled graph-size checks |
| Preparation/configuration | `asr/ita/prepare.py` explicitly accepts enabled Italian only; required resources come from `resources.ita` | Add future SCN dispatch; `src/asr/scn/config.yaml` and `ita_scn.yaml` remain disabled now |
| Decoding/search | Existing k2 search and frame-score interface | Reuse core search; measure SCN beam/LM-scale behavior later |
| Evaluation | Strict and Italian spoken-form normalization | Retain strict scoring mechanics; create SCN reference conventions, number expansion and regional scoring separately |
| Provenance/cache | Manifest, fingerprints, graph identity and prepared-artifact checks | Reuse; this phase adds reviewed HTTP snapshots and isolated SCN acquisition routing |
| Bilingual words/grammar | No bilingual graph implemented | Future language-tagged lexical IDs, pronunciation variants and explicit switch policy |

The existing Italian graph identity hashes its selected resources, compiler files, settings and package versions. SCN additions are not dependencies of that identity. The Python corpus extra adds extraction packages; existing research package pins, decoder code and configurations are preserved. See [validation evidence](../../resources/sources/validation.json) for final checks.

## Priorities and acceptance gates

| Priority | Already available | Programmatic work available next | Manual / external requirement | Gate |
|---|---|---|---|---|
| P0: certify a text slice | Licensed localization, UNESCO text, Google text, regional Wikipedia/UD; versioned originals and extraction | Export reviewer samples and source-stratified frequency lists | A DOS24-competent reviewer certifies or corrects an explicitly licensed slice; PDF columns and language contamination resolved | Positive DOS24 evidence, independent document splits, provenance for every retained record |
| P1: seed pronunciations | DOS24/2017 descriptions; WikiPron candidates; morphology and toponym lists; ZIPA token audit | Join candidate spellings without silently converting them; generate symbol-support and missing-vocabulary reports | Validate 500–2,000 high-frequency/in-domain spellings and pronunciations, with regional coverage and exceptions | Gold seed + held-out lexicon; no unexplained unknown-token projections |
| P2: auditable G2P rules | Spelling/phonology analysis and source inventory | Implement deterministic rule families and explicit projection, tagged as generated | Resolve stress/hiatus, `z`, retroflex/dental distinctions, loans and region-conditioned outcomes | Held-out error analysis and reviewer approval of each rule family |
| P3: choose a baseline LM | Count probes and source-group splits | Re-run orders 1–4 on the **certified DOS24** subset with train-only vocabulary | Obtain more independent DOS24 text if coverage remains inadequate | Select on development perplexity and OOV/sparsity; reserve test material; document vocabulary–lexicon overlap |
| P4: construct the first graph | Fixed ZIPA, k2, topology and L/G composition mechanics | Add separate SCN preparation and order-aware ARPA support; compile small graph in a future task | Review lexical ambiguity and vocabulary scope | Token round trips; tiny known paths; normalization/probability checks; bounded graph memory; fresh held-out speech evaluation |
| P5: bilingual baseline | Working Italian model, future SCN model, switching design below | Build language-tagged word inventory and controlled grammar composition later | Collect authentic bilingual references and resolve borrowing labels | Monolingual regressions plus separate mixed-speech evaluation |

For an engineering smoke graph, a manually authored and licensed small DOS24 corpus plus a few hundred gold words can exercise composition; it cannot establish useful general ASR. For a meaningful initial study, aim for tens of thousands of reviewed tokens from many independent documents and a few thousand domain-relevant words, then let held-out measurements determine whether more data is needed. These are planning targets, not dataset sizes or guarantees. No final LM should be selected from auxiliary Wikipedia results alone.

## Italian–Sicilian switching resources

[Alfonzetti (2015)](https://www.athensjournals.gr/philology/2015-2-1-2-Alfonzetti.pdf), *Age-related Variation in Code-switching between Italian and the Sicilian Dialect*, reports transcribed spontaneous conversations and written electronic interaction. It establishes that authentic switching exists. It does not establish an openly licensed bulk corpus for this repository. SoPhISM addresses Sicilian and Sicilian Italian variation, but style/language contrasts are not automatically an annotated word-level switching dataset.

The acquired corpus contains Italian material and written mixing. A qualitative [review of nine candidates](../../resources/corpus/review-notes.json) found two clearly Italian prose records, one Italian clause followed by a Sicilian clause, and six “switching” flags attributable to titles/quotations. These are model-assisted triage notes, not native-speaker gold or prevalence estimates. The acquired 1,508 Omnilingual transcript rows are a useful future speech-review source, with original partitions retained; no audio was acquired or word-level switching validated here. UD's Italian translations are parallel sentences, not code switches.

| Phenomenon | Modeling / annotation consequence |
|---|---|
| Utterance-level language choice | One language model for the whole utterance; evaluate language selection separately |
| Word-level switching | Language changes inside an utterance; requires word/span labels for evaluation |
| Italian borrowing | A lexical item can belong to Sicilian usage without an active switch; evidence and community judgment needed |
| Shared written form | Spelling alone may not identify language; keep ambiguity until context/phonology helps |
| Sicilian adaptation of Italian | May use Sicilian morphology/phonology; a separate lexical pronunciation, not automatic Italian IPA |
| Quotation, title or parallel translation | Attested multilingual text, but not proof of spontaneous conversational switching |

## Initial switching design without mixed training text

First compare whole-utterance alternatives using separately weighted Italian and Sicilian grammars. A future word-switching baseline can use states `(language, monolingual_history)` and internal word identities `(language, spelling)`. Keep identical displayed words separate internally until after decoding; homophones and alternative pronunciations need consistent disambiguation.

At each word boundary, staying in language ℓ has mass `1−qℓ`; switching has mass `qℓ` and consumes a word from the target language's unigram distribution, then enters that language's one-word history. Thus a switch loses unsupported cross-language history rather than inventing mixed n-gram counts. Treat EOS and termination probabilities explicitly when making the distributions stochastic. Language prior, switch mass and decoder LM scale must not be double-counted. An additive switch-cost approximation is also possible, but must be documented as a decoding heuristic rather than a normalized LM.

There must be no free epsilon cycle between languages. Keep switching arcs word-consuming, and put every word ID in the appropriate L and G tables. Apply acoustic scores only to the fixed ZIPA labels; language tags are lexical metadata, not acoustic outputs. Generate artificial mixed paths only for structural unit tests, never label them authentic training/evaluation speech. Tune penalties on later authentic bilingual development data and report monolingual regressions, switch precision/recall and ASR errors separately. No bilingual implementation was made in this phase.

## External blockers and remaining uncertainty

Rights or access remain unresolved for site-wide Cademia articles/Scola, Telegram exports, several glossary/CLDR forks, Arba Sicula journal bitext, Dieli data, ARTESIA/VSM/ALS bulk access, and row-level rights in large web-mined corpora. Pontoon and the original FLORES download were not collected through crawler-disallowed routes. Earlier UNESCO text is available under its text-only share-alike license, but images and mixed-license derived distributions require separate treatment.

The strongest uncertainty is how much already acquired contemporary prose can be validated as DOS24 with modest review. Others are regional pronunciation coverage, consistency of WikiPron spellings/IPA, speech transcript orthography, PDF extraction quality, domain mismatch and additional open sources outside this bounded search. None is resolved by training a larger neural model. Prioritize linguistic validation and a small reproducible WFST baseline before neural rescoring or bilingual expansion.

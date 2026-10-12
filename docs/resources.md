# Resource acquisition and provenance

The source of truth is [`resources/manifest.yaml`](../resources/manifest.yaml). Every available artifact has an official URL, pinned Git revision or checksum-pinned academic release identity, format, citation, license evidence, size, and SHA-256. Unavailable/planned entries explicitly explain why paths, licenses, or revisions cannot be supplied. A source repository's license is not automatically the license of its datasets or model weights.

## Reproduction and integrity

After `uv sync --frozen --extra research`, run:

```sh
.venv/bin/python scripts/download_resources.py --all
```

Use resource IDs to select individual downloads, or `--dry-run` to inspect the acquisition list. Omitting `--all` skips the large checkpoint. The downloader streams to temporary files, checks HTTP body length and manifest fingerprints, and atomically renames only complete verified artifacts. It checks existing files before reusing them and refuses to overwrite corrupt existing artifacts. Investigate a reported mismatch before removing that file and retrying.

Source archives are extracted with Python's restrictive tar `data` filter into a temporary directory, then moved into an ignored source cache. Existing source directories must have the matching revision/archive marker. No setup code from a downloaded dataset is executed.

The model's SHA-256 comes from its published Git LFS metadata. For GitHub raw files, upstream Git blob identities were checked where available. Other SHA-256 values were measured from the pinned official bytes during initialization; they are integrity pins, not claims of independently signed publisher checksums. The manifest freezes them for subsequent acquisitions. Local receipt timestamps are recorded separately in `.cache/resource-status.json`.

All downloaded linguistic data, weights, archives, and extracted sources are Git-ignored, including resources whose redistribution terms are unresolved. The Python wheel includes project code only. Unit-test fixtures are artificial validation inputs and never populate resource directories.

## Italian pronunciation resources

Source: [phonItaliaR](https://github.com/stefanocoretta/phonItaliaR/tree/2ee9d610e806bba6d9268d544eeb9295d5d13a78), commit `2ee9d610e806bba6d9268d544eeb9295d5d13a78`. Academic provenance: Goslin, Galluzzi, and Romani, *PhonItalia: a phonological lexicon for Italian*, Behavior Research Methods 46, 872–886 (2014), [DOI](https://doi.org/10.3758/s13428-013-0400-8). Stefano Coretta provides the IPA additions.

| Artifact preserved as supplied | Content | Measured size/count |
| --- | --- | --- |
| `phonItalia-1.10-wordforms.tsv` | Original PhonItalia 1.10 table with `word`, `Phones`, `PhoneSyll`, and lexical statistics | 46,276,243 bytes; 120,000 rows; 94,505 distinct `word` strings |
| `phonitalia.rda` | Author's IPA-enriched R data object | 10,698,980 bytes; preserved without conversion or requiring R |
| `italian.dict` | Author's headerless word/IPA TSV dictionary | 2,699,644 bytes; 94,356 rows; 94,290 distinct written forms; 94,346 distinct word/pronunciation pairs |
| Information PDF and package license | Original documentation and license declaration | Preserved beside the data |

The original TSV uses a custom ASCII phone representation; it must not be labeled Unicode IPA. PhonItaliaR's [data documentation](https://stefanocoretta.github.io/phonItaliaR/reference/phonitalia.html) describes IPA fields, tokenized phone sequences, syllable boundaries, and geminates represented with `ː`. Its supplied `italian.dict` uses those tokenized geminate pronunciations. The [author's preparation code](https://github.com/stefanocoretta/phonItaliaR/blob/2ee9d610e806bba6d9268d544eeb9295d5d13a78/data-raw/phonitalia.R) also handles spelling accents, filters entries, adds a few contractions, and applies selected assimilation/voicing rules. This project downloads that existing result unchanged; it does not run the R conversion or generate pronunciations. Duplicate rows remain in the original file. The derived decoder lexicon removes only duplicate normalized word–pronunciation pairs and reports exclusions, alternatives, and spelling collisions; see [linguistic conversion](linguistic-conversion.md).

**License evidence:** phonItaliaR declares [CC BY 4.0](https://github.com/stefanocoretta/phonItaliaR/blob/2ee9d610e806bba6d9268d544eeb9295d5d13a78/LICENSE.md). The original author's [2013 announcement](https://linguistlist.org/issues/24/3960/) describes noncommercial research use under a Creative Commons license without identifying the exact variant there. The discrepancy remains unresolved. Keep attribution and original notices, and do not infer unrestricted commercial or redistribution rights from the wrapper repository alone.

## Italian language model: pretrained MLS

The selected resource is the Italian LM archive from [Multilingual LibriSpeech / OpenSLR 94](https://www.openslr.org/94/), the single academic release described by Pratap et al. (2020). MLS publishes pretrained 3-gram and 5-gram models under CC BY 4.0. Only the trigram and its accompanying documentation are extracted; there is no LM training, PAISÀ fallback, or adaptation against references.

| Integrity field | Value |
| --- | --- |
| Archive | `mls_lm_italian.tar.gz`, 1,795,621,889 bytes |
| Publisher MD5 | `c2f4d28ffcb60be7ac3ac3012a9abb09`, from [md5sum.txt](https://dl.fbaipublicfiles.com/mls/md5sum.txt) |
| Measured archive SHA-256 | `758feb0965eec33be2e04dbe0388e77be4ef4877b9cc2ea82e2f0110c1aa9a91` |
| Original trigram | 1,249,663,557 bytes |
| Trigram SHA-256 | `19fd8f382a1ccde71984e0a102ea7086c5abef719efc55eaaa2b9da6dfa1a5f7` |
| Original ARPA counts | 700,630 unigrams; 10,502,874 bigrams; 27,413,663 trigrams |
| Default restricted ARPA counts | 20,003 unigrams including specials; 3,729,469 bigrams; 13,365,926 trigrams |

The publisher README explicitly identifies `3-gram_lm.arpa` as trained on the archive's normalized `data.txt`. The archive also supplies a Gutenberg book list, word counts, license, and README. Extraction receipts bind each retained member's SHA-256 to the verified archive hash. Existing derived files are compared with the archive or a verified receipt, rather than certifying their own provenance.

The [MLS paper, Sections 3.4.1 and 5](https://arxiv.org/html/2012.03411) describes Gutenberg text, NFKC normalization, punctuation/unwanted-character filtering and dehyphenation, with books filtered against development/test material by titles and 5-gram overlap. LMs were estimated with KenLM. The Italian corpus table reports 814 initial books, 640 retained books, 42.28 million words, 700.63 thousand unique words, and 2.14 million sentences. These are rounded publisher statistics, separate from measured ARPA counts. An exact LM training command, smoothing/pruning flags, and exact unrounded sentence/token totals are not supplied by the extracted README or paper; they are recorded as unavailable rather than inferred.

The retained original ARPA is never changed. Selection uses unigram probability, not the optional word-count table. Graph vocabulary restriction preserves retained LM weights and sentence boundaries. Full/shared vocabulary statistics and the limitations of tropical backoff decoding are documented in [architecture.md](architecture.md).

### Historical FBK investigation

The earlier [FBK public resource](https://github.com/SpeechTechLab/FBK_ASR_public/tree/9845fffdb8eb6e5f51a20e870bd9acb5f9000c84) linked a Drive folder that returned HTTP 404 during initialization on 2026-10-08. Its exact model bytes, license, and vocabulary were not obtained. The `fbk_ita_lm` manifest entry remains unavailable for historical traceability; Italian configuration now selects `mls_ita_lm`. No fallback silently replaces either artifact.

## Acceptance speech

[Emozionalmente / OpenSLR 161](https://www.openslr.org/161/) links [Zenodo record 12616095, version 1.1](https://zenodo.org/records/12616095). The academic release is CC BY 4.0. The ZIP is pinned by publisher MD5 `398c6502dd2b0e1e810f927c12897547` and measured SHA-256 `6a1fed885668374e2802ae4cfd27c6cdf1db652b8856c6e91e74fe61add3d2b9` (558,910,248 bytes).

`scripts/acceptance_italian.py` chooses the lexically first filename labeled `neutrality` in the publisher test CSV, before decoding. It extracts only that recording and documentation, keeps the supplied sentence unchanged, creates an MP4 with AAC audio and black video, and records source/container/reference hashes. See [implementation-report.md](implementation-report.md). This selection is deterministic and independent of recognition output.

## ZIPA and Sicilian resources

ZIPA checkpoint access, feature conventions, model license, and CTC interfaces are documented in [acoustic-model.md](acoustic-model.md). Its weight checksum is publisher-provided, and the 127-label inventory is verified against the paired tokenizer.

The Sicilian resource discovery phase has acquired reviewed reference, text and lexical sources, with source-specific permissions and checksums. See [the inventory and measurements](scn-resource-inventory.md) and [reproduction commands](../resources/scn/sources/README.md). DOS24 text certification, the final pronunciation lexicon and LM remain pending; Sicilian and bilingual decoding are disabled.

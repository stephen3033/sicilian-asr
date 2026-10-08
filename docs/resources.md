# Resource acquisition and provenance

The source of truth is [`resources/manifest.yaml`](../resources/manifest.yaml). Every available artifact has an official URL, immutable source revision, format, citation, license evidence, size, and SHA-256. Unavailable/planned entries explicitly explain why paths, licenses, or revisions cannot be supplied. A source repository's license is not automatically the license of its datasets or model weights.

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

The original TSV uses a custom ASCII phone representation; it must not be labeled Unicode IPA. PhonItaliaR's [data documentation](https://stefanocoretta.github.io/phonItaliaR/reference/phonitalia.html) describes IPA fields, tokenized phone sequences, syllable boundaries, and geminates represented with `ː`. Its supplied `italian.dict` uses those tokenized geminate pronunciations. The [author's preparation code](https://github.com/stefanocoretta/phonItaliaR/blob/2ee9d610e806bba6d9268d544eeb9295d5d13a78/data-raw/phonitalia.R) also handles spelling accents, filters entries, adds a few contractions, and applies selected assimilation/voicing rules. This project downloads that existing result unchanged; it does not run the R conversion or generate pronunciations. Duplicate rows are retained.

**License evidence:** phonItaliaR declares [CC BY 4.0](https://github.com/stefanocoretta/phonItaliaR/blob/2ee9d610e806bba6d9268d544eeb9295d5d13a78/LICENSE.md). The original author's [2013 announcement](https://linguistlist.org/issues/24/3960/) describes noncommercial research use under a Creative Commons license without identifying the exact variant there. The discrepancy remains unresolved. Keep attribution and original notices, and do not infer unrestricted commercial or redistribution rights from the wrapper repository alone.

## Italian language model

Source: [SpeechTechLab/FBK_ASR_public](https://github.com/SpeechTechLab/FBK_ASR_public/tree/9845fffdb8eb6e5f51a20e870bd9acb5f9000c84), documentation revision `9845fffdb8eb6e5f51a20e870bd9acb5f9000c84`, by Marco Matassoni and Roberto Gretter, February 2023.

The README points to a [Google Drive folder](https://drive.google.com/drive/folders/10kL8G-SU5meqXw0H3b0OLEOVhSvIyk5q) for `wavlm-large-it-cv10.zip`, including the pruned generic LM variant `wavlm-large-it-cv10_tutto.3.256K.msb.p2`. This folder returned HTTP 404 on 2026-10-08. A check of the official SpeechTek Hugging Face model list did not locate a corresponding downloadable n-gram model.

The README describes generic trigrams based on normalized news text (`EN20*`), an Italian Wikipedia 2018 dump, and dictionary material. “256K” is a nominal vocabulary label, not a measured vocabulary size. ARPA generation and a KenLM binary-building utility are documented, but the exact distributed LM file, format, checksum, vocabulary, pruning details, complete training sources, and license could not be verified without the bundle. The repository's vocabulary list and WavLM template are **not** substitute language models; raw `.3ngt` counts are not a pretrained ARPA model either.

The manifest therefore records `unavailable`, and YAML uses `lm_path: null`. No LM file is created. To recover access, request the existing pruned trigram, corresponding vocabulary, ARPA export if available, version/checksum, and resource license from `matasso@fbk.eu` or `gretter@fbk.eu` (contacts supplied by the repository). No messages have been sent automatically. Once an authorized official artifact exists, pin it in the manifest and implement acquisition for its verified packaging; do not run adaptation or training scripts as a replacement.

## ZIPA and deferred Sicilian resources

ZIPA checkpoint access, feature conventions, model license, and CTC interfaces are documented in [acoustic-model.md](acoustic-model.md). Its weight checksum is publisher-provided, and the 127-label inventory is verified against the paired tokenizer.

Sicilian resource entries are planned, with no downloads or generated data. See [sicilian-resources.md](sicilian-resources.md). Future resources must get their own provenance and license entries before acquisition.

# Italian resource provenance

## Italian pronunciation resources

Source: [phonItaliaR](https://github.com/stefanocoretta/phonItaliaR/tree/2ee9d610e806bba6d9268d544eeb9295d5d13a78), commit `2ee9d610e806bba6d9268d544eeb9295d5d13a78`. Academic provenance: Goslin, Galluzzi, and Romani, *PhonItalia: a phonological lexicon for Italian*, Behavior Research Methods 46, 872–886 (2014), [DOI](https://doi.org/10.3758/s13428-013-0400-8). Stefano Coretta provides the IPA additions.

| Artifact preserved as supplied | Content | Measured size/count |
| --- | --- | --- |
| `phonItalia-1.10-wordforms.tsv` | Original PhonItalia 1.10 table with `word`, `Phones`, `PhoneSyll`, and lexical statistics | 46,276,243 bytes; 120,000 rows; 94,505 distinct `word` strings |
| `phonitalia.rda` | Author's IPA-enriched R data object | 10,698,980 bytes; preserved without conversion or requiring R |
| `italian.dict` | Author's headerless word/IPA TSV dictionary | 2,699,644 bytes; 94,356 rows; 94,290 distinct written forms; 94,346 distinct word/pronunciation pairs |
| Information PDF and package license | Original documentation and license declaration | Preserved beside the data |

The original TSV uses a custom ASCII phone representation; it must not be labeled Unicode IPA. PhonItaliaR's [data documentation](https://stefanocoretta.github.io/phonItaliaR/reference/phonitalia.html) describes IPA fields, tokenized phone sequences, syllable boundaries, and geminates represented with `ː`. Its supplied `italian.dict` uses those tokenized geminate pronunciations. The [author's preparation code](https://github.com/stefanocoretta/phonItaliaR/blob/2ee9d610e806bba6d9268d544eeb9295d5d13a78/data-raw/phonitalia.R) also handles spelling accents, filters entries, adds a few contractions, and applies selected assimilation/voicing rules. This project downloads that existing result unchanged; it does not run the R conversion or generate pronunciations. Duplicate rows remain in the original file. The derived decoder lexicon removes only duplicate normalized word–pronunciation pairs and reports exclusions, alternatives, and spelling collisions; see [linguistic conversion](#conversion-and-evaluation-rules).

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
| Original baseline restricted ARPA counts | 20,003 unigrams including specials; 3,729,469 bigrams; 13,365,926 trigrams |

The publisher README explicitly identifies `3-gram_lm.arpa` as trained on the archive's normalized `data.txt`. The archive also supplies a Gutenberg book list, word counts, license, and README. Extraction receipts bind each retained member's SHA-256 to the verified archive hash. Existing derived files are compared with the archive or a verified receipt, rather than certifying their own provenance.

The [MLS paper, Sections 3.4.1 and 5](https://arxiv.org/html/2012.03411) describes Gutenberg text, NFKC normalization, punctuation/unwanted-character filtering and dehyphenation, with books filtered against development/test material by titles and 5-gram overlap. LMs were estimated with KenLM. The Italian corpus table reports 814 initial books, 640 retained books, 42.28 million words, 700.63 thousand unique words, and 2.14 million sentences. These are rounded publisher statistics, separate from measured ARPA counts. An exact LM training command, smoothing/pruning flags, and exact unrounded sentence/token totals are not supplied by the extracted README or paper; they are recorded as unavailable rather than inferred.

The retained original ARPA is never changed. Selection uses unigram probability, not the optional word-count table. Graph vocabulary restriction preserves retained LM weights and sentence boundaries. Full/shared vocabulary statistics and the limitations of tropical backoff decoding are documented in [WFST mechanics](../architecture.md#compact-ctc).

### Historical FBK investigation

The earlier [FBK public resource](https://github.com/SpeechTechLab/FBK_ASR_public/tree/9845fffdb8eb6e5f51a20e870bd9acb5f9000c84) linked a Drive folder that returned HTTP 404 during initialization on 2026-10-08. Its exact model bytes, license, and vocabulary were not obtained. The `fbk_ita_lm` manifest entry remains unavailable for historical traceability; Italian configuration now selects `mls_ita_lm`. No fallback silently replaces either artifact.

## Acceptance speech

[Emozionalmente / OpenSLR 161](https://www.openslr.org/161/) links [Zenodo record 12616095, version 1.1](https://zenodo.org/records/12616095). The academic release is CC BY 4.0. The ZIP is pinned by publisher MD5 `398c6502dd2b0e1e810f927c12897547` and measured SHA-256 `6a1fed885668374e2802ae4cfd27c6cdf1db652b8856c6e91e74fe61add3d2b9` (558,910,248 bytes).

`scripts/acceptance_italian.py` chooses the lexically first filename labeled `neutrality` in the publisher test CSV, before decoding. It extracts only that recording and documentation, keeps the supplied sentence unchanged, creates an MP4 with AAC audio and black video, and records source/container/reference hashes. See [implementation-report.md](README.md#measured-results). This selection is deterministic and independent of recognition output.


## Conversion and evaluation rules

Rules are versioned in `LEXICON_RULES`, `VOCAB_RULES`, and `TEXT_RULES`. Original data remain unchanged. Each preparation saves a complete normalized lexicon, selected lexicon, vocabulary, and row-level audit.

## Phones and token IDs

The pinned ZIPA [training-data generator](https://github.com/lingjzhu/zipa/blob/21e9e8aa4cc6c130cb05d836a1979886f9245302/scripts/generate_training_data.py) removes spaces between source phones and maps `ɡ` to `g`. Its [training recipe](https://github.com/lingjzhu/zipa/blob/21e9e8aa4cc6c130cb05d836a1979886f9245302/zipformer_crctc/train.py) then uses the supplied SentencePiece model. We apply this convention to each pronunciation, requiring a lossless reconstruction from the returned token IDs after removing only the initial dummy `▁`.

| Source | ZIPA representation | Policy |
| --- | --- | --- |
| `ɡ` | `g` | Training spelling convention |
| `ɛ`, `ɔ` | same symbols | Preserve vowel quality; never merge with `e`, `o` |
| `tʃ`, `dʒ`, `ts`, `dz` | inventory components in order | Preserve affricate content; no replacement with an approximate consonant |
| `fː`, `tː`, etc. | original plus optional `ff`, `tt` | Preserve the source and add a distinct doubled alternative; never collapse length to a singleton |
| `tz` | excluded | Ambiguous source phone; no inferred correction |
| `bv` | excluded | Ambiguous source phone; no inferred correction |
| Explicit stress | none | Supplied dictionary lacks explicit stress and ZIPA lacks stress labels |
| `▁` | optional segment prefix | Never inserted between dictionary phones or required at each word |

Original ZIPA IDs 0–126 remain fixed. ID 0 is blank, 1 is `<sos/eos>`, 2 is acoustic `<unk>`, and 3 is `▁`. ID 1 has no lexical pronunciation and cannot produce a word. Decoder auxiliary ID 127 and disambiguators 128+ are not acoustic classes.

The current audit retains 93,802 unique normalized word–pronunciation pairs over 93,746 words, including 56 words with alternatives. Of 94,356 source rows, 543 are excluded (542 with `tz`, one with `bv`), and 11 duplicate normalized pairs are removed. One spelling collision is recorded: `un` and `un'` normalize to `un`. Coverage is 99.424% of the 94,289 normalized source words. Excluded rows and words, source-phone counts, and all collisions are listed in `lexicon_audit.json`.

## Optional Italian compatibility alternatives

`decoder.pronunciation_variants` enables `ita-length-doubled-closure-nasal-l-c-elisions-v1`. It adds complete-pronunciation alternatives while retaining originals. Long consonants gain doubled forms; the long affricates `tʃː`, `dʒː`, `tsː`, and `dzː` gain `ttʃ`, `ddʒ`, `tts`, and `ddz`. Velar nasal `ŋ` immediately before `k` or `g` gains an `n` alternative. Applying the two transformations separately and together yields at most four forms per original. Vowels, stress, and singleton consonants are not globally merged.

`decoder.contractions` considers exact MLS unigram spellings containing an apostrophe. Only `l'` and `c'` plus a supported, vowel-initial suffix pronunciation are derived, with `l` and `tʃ` prefixes respectively. Thus the decoder can retain `l'ho` and `c'è` without using evaluation references or estimating LM probabilities. Other apostrophized forms remain unsupported and are listed explicitly. Derivation precedes pronunciation alternatives and vocabulary selection.

`lexicon_adaptation.json` records every added pronunciation, source spelling/pronunciation, applied rules, added words, and unsupported LM contractions. `lexicon_audit.json` continues to describe the unmodified source conversion. `lexicon_full.json` includes enabled derivatives. Both options are part of graph cache identity; neither alters source dictionary or ARPA bytes.

On the pinned sources, enabling both rules yields 100,343 supported words, including 6,597 derived spellings and 51,021 added pronunciation pairs. The 20,000-word selection includes 641 of those spellings. These are resource-coverage counts, not an accuracy claim.

## Words, vocabulary restriction, and unknowns

Words use NFC, lowercase, standardized apostrophes, punctuation removal, and whitespace normalization. Only word-internal apostrophes survive. Accents and digits are preserved; neither phonetic nor orthographic words are rewritten. Source rows normalizing to multiple words are rejected explicitly.

The selected vocabulary is the highest-unigram-probability subset of exact shared normalized lexicon/MLS word spellings, with lexical ordering breaking ties. The original LM is not rewritten to collapse spelling variants: 385 noncanonical unigrams are counted diagnostically. Before compatibility additions there are 72,286 shared supported words; the default retains 20,000 plus `<unk>`. Setting the limit to null retains all shared words. No evaluation reference is read during selection.

The derived ARPA retains every n-gram whose tokens are in the selected set or the original `<s>`, `</s>`, `<unk>` symbols. Probabilities and backoff weights are copied verbatim; this is a decoding-vocabulary restriction, not a normalized retrained LM. The LM unknown symbol maps to ZIPA `<unk>` and remains visible in evaluation. Spoken OOV words can still become in-vocabulary substitutions. In the original acceptance graph, `l'ho` and the digit `2` are OOV even in the full supported lexicon; the contraction rule restores `l'ho`, while number formatting remains a separate evaluation concern.

## Evaluation text

SRT/VTT are parsed as cues before normalization; cue indices, timestamps, VTT headers/NOTE/STYLE/REGION blocks, and formatting are excluded from spoken text. Numeric lines within cue bodies remain spoken text. Cue timing and metadata are saved separately. Plain UTF-8 `.txt` is accepted for independently corrected transcripts.

Hypothesis and reference receive the same `nfc-lower-apostrophes-punctuation-v1` normalization. Top-level WER uses whitespace-separated tokens; CER uses Unicode characters after whitespace removal. `<unk>` is retained literally and scored. Digits stay digits in these strict scores.

The additive `spoken_form` metrics apply `ita-standalone-cardinals-0-999999-v1` to the original reference/hypothesis before the shared normalizer. Standalone integers gain Italian cardinal spellings (including elision and final `-tré` accents). Signs, decimals, dates, identifiers, and leading-zero forms remain unconverted. Each side records transformed character spans and unsupported numbers; contractions and grammatical errors remain scored. Both views include vocabulary OOV counts. `oov_causes` distinguishes absent lexicon entries from words outside the selected decoding vocabulary. No PER is computed without an IPA reference.

Subtitle paraphrases, omissions, overlapping cues, and timing offsets can affect whole-recording comparisons; this milestone does not align words to cues or de-duplicate subtitle speech. Publisher `E’` spellings are not silently changed to `è`; this annotation issue remains visible in the regression corpus.

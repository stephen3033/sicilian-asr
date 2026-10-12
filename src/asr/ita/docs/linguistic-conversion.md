# PhonItalia, ZIPA, and Italian orthography

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

# DOS24 and earlier Sicilian writing

The corpus classifications below describe the 8 October snapshot. The subsequent
[expanded DOS24 audit](dos24-audit.md) adds contextual verified decisions,
supported/inferred counts, current Wikipedia and LiITA measurements. Its independent
conformity fields supersede the older four-way corpus grouping below.

Research snapshot: 8 October 2026 (America/Los_Angeles). **No spelling conversion is implemented, and no acquired prose has been certified as DOS24.** Source identity, publication date and orthographic conformity are separate facts.

## Authority and version history

The target is Salvatore Matteo Baiamonte / Cademia Siciliana's *Documento per l'Ortografia del Siciliano*, DOS24 v2.2. The [official version history](https://cademiasiciliana.org/pruposta-ortugrafica/) dates v1 to 1 November 2017, v2 and v2.1 to 7 April 2024, and v2.2 to 9 April 2024. The older PDF's upload directory says October 2017; that directory is not its authoritative publication date. Separate v2/v2.1 files were not located or treated as interchangeable with v2.2.

The [2024 PDF](https://cademiasiciliana.org/wp-content/uploads/Cademia-Siciliana-DOS24-v.2.2.pdf) and [2017 PDF](https://cademiasiciliana.org/wp-content/uploads/2017/10/Proposta-di-normalizzazione-ortografica-comune-della-lingua-siciliana-I-edizione-2017.pdf) are acquired unchanged in the ignored cache. Their explanatory language is Italian: neither PDF is counted as a Sicilian prose corpus. Printed terms are CC BY-NC-ND 4.0. DOS24's legalcode hyperlink omits NC, conflicting with its declaration; the inventory retains this discrepancy. This project distributes neither adapted reference PDFs nor their full extracted text.

DOS24 is a writing system across varieties, not a prescription to pronounce all words identically. Its introduction expressly includes Sicily, satellite islands and the Reggio area. An implementation should associate regional pronunciations with a written lexical identity, with geographic evidence recorded for each variant.

## Verified differences and continuity

Page numbers below are PDF page numbers. This comparison is an analytical summary, not a replacement standard or an executable conversion specification.

| Feature | 2017 edition | DOS24 v2.2 | Consequence |
|---|---|---|---|
| Accent notation | Grave and acute can convey stress and vowel quality, pp.12–15 | Grave marks stress/disambiguation; circumflex marks contraction, pp.7–9 | Do not globally replace acute accents without reviewing names, quotations and stress |
| Syllabification of medial doubles | Keeps long consonant spelling together, pp.10–11 | Divides medial doubles between syllables; initial doubles remain together, p.8 | Old syllabification rules cannot be copied into a new stress engine |
| Retroflex spelling | Recommends `ḍḍ`; permits `ddh` or `dd` when characters are inconvenient, p.22 | Specifies `ḍḍ`, distinguishing dental `dd`, p.17 | `dd` is ambiguous; replacing every `dd` would corrupt dental words |
| Forms of “have” | Earlier accent practices need word-level review | Explicitly makes grave accents on specified inflected forms optional, pp.7–8 | More than one valid written form can survive standardization |
| Vowels | Describes regional outcomes and cross-variety writing | Five stressed vowel spellings and three unstressed ones, etymological choices, pp.9–11 | Written `e/o` do not license one universal regional pronunciation |
| Lost sounds / apostrophes | Distinguishes systematic loss from occasional elision, pp.14–15 | Retains this distinction, p.8 | Apostrophe deletion is not generic cleanup |
| Regional consonants | Broad grapheme/phone table and discussion, pp.9–10,18–32 | Further explanatory treatment of `d`, `v`, `g/j`, retroflexes, pp.13–17 | Preserve regional alternatives and lexical exceptions |
| Initial length and external doubling | Discusses gemination | Initial `b`, palatal `g`, `r`, `z` may be written single but pronounced long; external doubling normally leaves word spelling stable, pp.17–18 | Letter doubling alone is insufficient for pronunciation generation |

Older standards and historical texts also preserve authorial, local, literary or Italian-influenced forms. A text with `ḍḍ` is not automatically DOS24; a text without it may simply contain no applicable word. Dates after April 2024 and Cademia affiliation do not prove conformity.

## Corpus classification

The machine-readable [catalog](../resources/sources/catalog.json) supplies an evidence string for every classification. The four text categories are:

1. `dos24`: positive edition/conformance evidence plus a documented review; currently zero.
2. `cademia_pre2024`: the three UNESCO issues from 2018–2019, identified through their publisher credits and the [Cademia project description](https://cademiasiciliana.org/curreri/). They are not retrospectively promoted.
3. `other_sicilian`: the 2023 Wikipedia snapshot, regional/historical UD material and the Arba Sicula-oriented Napizia WikiMatrix selection.
4. `uncertain_mixed`: current localization, Google translations, Tatoeba and spoken transcripts without a verified standard declaration. This category expresses uncertainty, not a claim that every record mixes languages.

Pronunciation tables, lexical lists and evaluation transcripts are inventoried separately from prose. Statistics are acquisition/extraction measurements, not a certificate that every token is Sicilian or linguistically usable.

## Current transformations and a future migration procedure

Current tokenization performs Unicode NFC, case folding by lowercasing, apostrophe-character unification and punctuation separation. It retains accents, circumflexes, `ḍ`, elision apostrophes and numbers; it does not expand numbers, respell dialect words, restore missing vowels, change `dd`, or translate Italian. Original text and source locators remain in every extracted record. Markup and localization variables are structurally removed, with separate fragments on either side of unknown substitutions.

A future DOS24 conversion requires a separately versioned procedure: identify source edition; preserve its original text; annotate candidate changes with rule ID and evidence; distinguish mechanical Unicode repair from lexical respelling; resolve ambiguity with context and a reviewer; retain unchanged, rejected and unresolved records; and evaluate conversion on a held-out, independently reviewed sample. Record token-level alignments, before/after strings, reviewer and source license. Do not publish adapted material where the source disallows derivatives. No such conversion has been applied in this phase.

The first useful review target is a small, licensed sample of contemporary localization prose, followed by column-correct UNESCO article text. Reviewers should determine both DOS24 conformity and whether UI fragments form useful sentence contexts. Historical literature should remain a separate cohort even when an explicitly approved modernization is later produced.

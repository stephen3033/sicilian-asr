# Sicilian pronunciation strategy

This document records the pre-POC resource assessment. For the implemented provisional lexicon, trained LM and current measurements, see the [POC study](poc.md).

**Use a manually validated seed lexicon plus auditable rules, with regional variants.** Keep ZIPA as the phonetic recognizer. No G2P model, pronunciation rule engine or authoritative new word–IPA pairs were created in this phase.

## Evidence and coverage

The primary spelling authority is [DOS24 v2.2](https://cademiasiciliana.org/wp-content/uploads/Cademia-Siciliana-DOS24-v.2.2.pdf). Its predecessor provides explicit IPA tables and extensive consonant discussion: [Cademia 2017](https://cademiasiciliana.org/wp-content/uploads/2017/10/Proposta-di-normalizzazione-ortografica-comune-della-lingua-siciliana-I-edizione-2017.pdf), especially pp.9–10 and 18–32. Consult both without assuming their spelling prescriptions are identical.

At the user's request, the [Wikipedia phonology overview](https://en.wikipedia.org/wiki/Sicilian_language#Phonology) was used to locate further literature. Its bibliography points to Bonner (2001), *Introduction to Sicilian Grammar*; Bonner (2003), [*Principal differences among Sicilian dialects: Part I*](https://raco.cat/index.php/Ianua/article/view/113861), Ianua 29–38; Privitera (1998), *Basic Sicilian*; Cipolla (2005), *The Sound of Sicilian*; and Ledgeway & Maiden (eds., 2016), *The Oxford Guide to the Romance Languages*, pp.479–480. The Bonner endpoint returned 403, and the books were not acquired. These are validation leads, not claims of having inspected all original pages.

For regional realization, Celata, Meluzzi & Ricci (2016), [*The sociophonetics of rhotic variation in Sicilian dialects and Sicilian Italian*](https://loquens.revistas.csic.es/index.php/loquens/article/view/30), documents the SoPhISM research design around Modica and Leonforte. CSFLS's *Vocabolario Siciliano* and ALS, and [Turin LFSAG's archive](https://www.lfsag.unito.it/ark/), are important additional validation leads; downloadable, reusable phonetic gold was not established for them.

There is an existing machine-readable pronunciation starting point: [WikiPron](https://github.com/CUNY-CL/wikipron), described by [Lee et al. (LREC 2020)](https://aclanthology.org/2020.lrec-1.521/). The pinned Sicilian tables contain **1,599 broad rows / 1,397 spellings** and **1,187 narrow rows / 801 spellings**. They overlap and must not be summed as independent vocabulary. They inherit Wiktionary data licensing; the code's Apache license is not the data license. The tables are crowdsourced observations, lack consistent regional/DOS24 labels, and are not an authoritative gold lexicon. Verified new gold pairs in this phase: **zero**.

The inspected pinned eSpeak NG and goruut repository trees did not establish a dedicated DOS24 profile. This bounded search is not proof that no Sicilian G2P exists. An Italian G2P fallback would erase distinctions and is not recommended.

## Phone inventory and rule requirements

This is a candidate modeling inventory. Slash/bracket choices below distinguish broad categories from realization candidates; they are not an instruction to add every phone to every word. The [machine-readable ZIPA audit](../resources/lexicon/ipa-compatibility.json) tests the actual installed token table and tokenizer.

| Area | Broad starting point / candidate realizations | Evidence and validation needed |
|---|---|---|
| Vowels | Stressed `/a ɛ i ɔ u/`; native unstressed `/a i u/` | DOS24 pp.9–11; borrowings may fall outside the native pattern |
| Regional vowels | `[e o ɪ ʊ]`, diphthongal and metaphonic outcomes | Identify variety, stress and morphological conditioning; avoid making one regional outcome universal |
| Stops | `/p b t d k g/`, palatal `[c ɟ]`, retroflex `/ɖ/` and `[ʈ]` in clusters | 2017 table and DOS24 pp.13–17; `ch/gh` sequences need context and lexical analysis |
| Affricates | `tʃ dʒ ts dz`; retroflex sequences such as `ʈɽ ɖɽ` | Segment as the pinned tokenizer does; `z` voicing is lexically ambiguous |
| Fricatives | `/f v s ʃ/`; candidate `[z ʒ ç ʝ x χ h ɣ β ð ʂ ʐ]` | Distinguish lenition, loan phones, local reflexes and phonemic contrasts; several are optional narrow realizations |
| Nasals | `/m n ɲ/`; `[ŋ ɱ ɳ]` | Place assimilation must be context-sensitive, not a spelling replacement |
| Liquids/glides | `/l r j w/`; `[ɾ ɽ ʎ]` where independently attested | Rhotic realization and dental rhotacism vary; do not assume every written `d` is `[ɾ]` |
| Length and stress | Gemination / `ː`; stress `ˈ ˌ` in linguistic records | Initial intrinsic length and external doubling require explicit handling; ZIPA has no stress-marker labels |

DOS24 encodes a lexical and etymological standard. Pronunciation rules need longest-match handling for `gn`, `sc(i/e)`, `ci/gi`, `ch/gh`, palatal sequences, `ḍḍ`, and `tr/dr/str`; stress and hiatus must be resolved before vowel rules. Penultimate stress is a starting default subject to written accents, hiatus, lexical exceptions and loanwords. `h` can be a spelling discriminator, a digraph component or a regional fricative; a universal “delete h” rule is wrong. `j` and `i` require glide/hiatus decisions. Contractions, clitics and morphological suffixes can move stress or alter segment sequences. These are rule families to validate, not implemented mappings.

Keep variant profiles for western, central/metaphonic, eastern/southeastern, Messinese, island and Reggio varieties, with locality tags finer than those broad group names when evidence permits. The labels are sampling strata, not an assertion of uniform pronunciation inside each region. Future coverage must include speakers from each stratum and independently transcribed examples of exceptional contexts. Current Omnilingual transcript metadata supplies a Sicilian glottocode and speaker IDs, not a validated geographic inventory.

## ZIPA projection

ZIPA uses 127 fixed SentencePiece labels: blank 0, start/end 1, unknown 2, dummy word-prefix 3. Lexical phone paths must use supported labels at IDs 4 and above. The broad phones and affricate/retroflex sequences tested here are representable. IPA `ɡ` is unsupported as written, but ASCII `g` is present. The established Italian conversion already uses that explicit equivalence.

Stress marks, the affricate tie bar and several combining diacritics are unsupported. Keep full linguistic IPA and stress in source metadata, alongside a separately versioned ZIPA projection. Authorize tie-bar removal only for a documented affricate segmentation; do not silently erase every unsupported diacritic. Reject unknown-token or non-round-tripping projections. Length marks are supported, but length versus repeated closure labels must be evaluated acoustically, especially for Sicilian retroflexes. Italian adaptation rules do not cover every Sicilian case.

With only concatenation of supplied phones and `ɡ→g`, **1,431/1,599 broad** and **919/1,187 narrow** WikiPron rows round-trip through the installed tokenizer. This measures representation only; it does not measure pronunciation correctness or ZIPA recognition accuracy. Unsupported rows remain evidence, not fabricated replacements.

## Compare construction approaches

All numerical targets below are engineering proposals to test, not measured sufficiency thresholds from the literature.

| Approach | Initial data target | Work and limitations |
|---|---|---|
| A. Manual lexicon | 500–2,000 frequent, in-domain DOS24 words; multiple verified realizations where needed | Low software complexity, substantial linguistic work; immediately auditable but limited coverage |
| B. Rules plus exceptions | 300–1,000 independently verified examples spanning all rule families, plus a separate held-out lexicon and exception list | Medium implementation complexity; best initial expansion route; stress, `z`, loans and regional deletion remain lexical/conditional |
| C. Statistical/neural G2P | Investigate only after several thousand clean aligned pairs; use 5k/10k/20k learning-curve checkpoints if attainable | Highest data/engineering cost; existing WikiPron rows are neither enough evidence of DOS24 coverage nor gold regional labels |

Build A first, then B. Each lexicon record should carry DOS24 spelling, source spelling, full IPA, stress, locality, morphology/lemma, citation/page or recording locator, license, reviewer, confidence, generator/rule version if applicable, and projected token IDs. Keep observed, manually validated and rule-generated pronunciations distinguishable. Split evaluation by lemma/morphological family, so inflected relatives do not make G2P generalization look artificially easy. Freeze a gold test set before expansion and measure word accuracy, phone error rate, unsupported-symbol rate and regional coverage. No statistical G2P is justified merely by the language's resource scarcity.

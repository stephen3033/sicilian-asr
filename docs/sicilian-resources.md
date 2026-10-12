# Sicilian resource preparation

Language code: **scn**. Target: **Cademia Siciliana DOS24 v2.2**.

The reviewed acquisition set has licensed originals in an ignored cache, source-preserving extraction, corpus statistics, pronunciation-inventory checks and auxiliary n-gram diagnostics. The expanded audit adds a small contextually verified DOS24 lower bound, a separate supported/inferred pool, and explicit unreviewed coverage. No final Sicilian lexicon, LM, WFST or bilingual decoder has been built.

- [Resource inventory and corpus measurements](scn-resource-inventory.md)
- [Expanded source, rights and DOS24 audit](scn-dos24-audit.md)
- [DOS24 and earlier orthographies](scn-orthography-analysis.md)
- [Pronunciation lexicon strategy](scn-pronunciation-strategy.md)
- [N-gram feasibility](scn-lm-feasibility.md)
- [Compatibility matrix, code switching and implementation plan](scn-implementation-plan.md)
- [Acquisition and reproduction commands](../resources/scn/sources/README.md)

`configs/scn.yaml` and `configs/ita_scn.yaml` remain disabled. The existing Italian acoustic recognizer, lexical resources, LM and decoder behavior are preserved. Further implementation starts with a reviewed DOS24 corpus slice and gold pronunciations, while ZIPA remains fixed.

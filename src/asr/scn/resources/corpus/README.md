# Corpus measurements

`extraction.json` identifies original checksums, archive members and extraction counts. `statistics.json` describes the acquired text by source and orthographic cohort. `review-notes.json` contains a small, purposive qualitative review; it is not a population estimate or expert annotation.

Full source-preserving records, normalized tokens, duplicate pairs, split assignments and the larger review queue are ignored artifacts under `.cache/scn/corpus/`. `records.jsonl` keeps original strings alongside extracted text, source IDs/checksums, document locators, orthography evidence and data roles. `analyzed.jsonl` adds tokens, screening flags and document/duplicate-connected split groups. Source-defined sentences, heuristic sentence units, UI fragments and lexical entries have separate counts.

Counts are Unicode tokenization results, including numbers and extraction artifacts. They are not validated DOS24 word counts. The DOS24 cohort currently contains zero documents. No historical material has been respelled. The diagnostic LM step excludes evaluation/lexical roles, exact duplicates, unreviewed PDF layout, unreviewed speech transcripts, detected markup/very long records and language-screen candidates. Original and descriptive records remain intact.

Reproduction commands and rights limitations are in [sources/README.md](../sources/README.md).

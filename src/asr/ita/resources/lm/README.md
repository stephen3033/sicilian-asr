# Italian language model

The implemented decoder uses the **pretrained MLS Italian trigram**, released through [OpenSLR 94](https://www.openslr.org/94/) under CC BY 4.0. `prepare` verifies the archive's published MD5 and pinned SHA-256, then extracts the original `mls/3-gram_lm.arpa`, README, LICENSE, book list, and vocabulary counts. The original ARPA is never edited.

The [manifest](../../../../../docs/resources.md#manifests) pins the academic archive; [resource provenance](../../../../../docs/resources.md) records training information and measured counts. Graph preparation writes the selected-vocabulary ARPA under `.cache/graphs/`, preserving retained probabilities and backoff weights. No LM training or PAISÀ fallback is implemented. The earlier unavailable FBK entry remains historical only.

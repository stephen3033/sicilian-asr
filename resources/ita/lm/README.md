# Italian language model

The existing FBK Italian trigram is the intended resource. Its official Drive download currently returns HTTP 404. No LM artifact exists here, and configurations deliberately use `lm_path: null`.

The [manifest](../../manifest.yaml) records the official source, documented nominal 256K vocabulary, intended pruned variant, and unresolved format/license. See [access instructions](../../../docs/resources.md#italian-language-model). Do not substitute vocabulary lists, raw counts, dummy ARPA data, or a newly trained LM.

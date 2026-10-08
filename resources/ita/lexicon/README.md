# Italian pronunciation resources

`scripts/download_resources.py` acquires the original PhonItalia 1.10 TSV, the author-provided IPA dictionary and R data, and their documentation here. All downloaded files remain unchanged and are Git-ignored.

The initial pronunciation path is `italian.dict`, a headerless UTF-8 TSV with a written word and space-separated IPA phones. It contains 94,356 rows and 94,290 distinct written forms; the original word-form table has 120,000 rows. These counts refer to different artifacts.

See [provenance and licensing](../../../docs/resources.md) and the [manifest](../../manifest.yaml). The wrapper declares CC BY 4.0, while the original authors describe noncommercial research use. Do not assume the project MIT license covers these resources.

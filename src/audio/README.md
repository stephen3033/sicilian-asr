# Audio segmentation

`segment_audio(samples, config, origin, rate)` returns `Segment` records with sample offsets, source times, boundary reasons, and forced-cut flags. Segments partition the extracted waveform without gaps or overlap.

Defaults come from [common/settings.py](../common/settings.py): target 20 seconds, maximum 24 seconds, and a search window beginning four seconds before the target. Silence lasting at least 300 ms below −40 dBFS is preferred by proximity to the target. Otherwise, the lowest-energy 10 ms frame is selected and the cut is flagged. A final tail must remain long enough for ZIPA.

Source times are sample offsets plus the media stream's origin. They describe segment boundaries, not word alignment or subtitle synchronization. The [IPA layer](../ipa/README.md) processes one segment at a time.

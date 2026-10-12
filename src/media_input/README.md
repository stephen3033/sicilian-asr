# Media input

`media.py` probes a local media file with FFprobe, selects the requested absolute audio-stream index (or the first audio stream), and uses FFmpeg to extract float32 mono audio at 16 kHz. It returns the waveform and source metadata. Missing streams, nonfinite audio, and inputs shorter than 100 ms fail explicitly.

Extraction downmixes and resamples without amplitude normalization. Metadata retains stream/container durations and the timestamp origin. It does not repair discontinuous media timestamps.

`reference.py` independently parses UTF-8 text, SRT, and VTT references. It preserves spoken numeric lines, strips subtitle markup, keeps cue times, and rejects malformed cues. The pipeline supplies references only to evaluation.

Dependencies: FFmpeg/FFprobe, SoundFile, NumPy, and common utilities. [Audio segmentation](../audio/README.md) is the next recognition stage.

`youtube.py` acquires public audio and explicitly uploaded subtitle tracks, retaining checksummed originals and timed JSON3 cues. Automatic captions are ineligible for the Sicilian POC.

from copy import deepcopy
from pathlib import Path
import json

import numpy as np
import pytest
import soundfile as sf
import torch

from sicilian_asr.acoustic.zipa import AcousticOutput, greedy_ctc
from sicilian_asr.audio import extract_audio, segment_audio
from sicilian_asr.evaluation import edits, evaluate, read_reference
from sicilian_asr.lexicon.prepare import convert, inventory
from sicilian_asr.lm.prepare import entries, restrict_arpa, select_vocabulary
from sicilian_asr.preparation import verify_cache
from sicilian_asr.resources import ResourceError, fingerprints
from sicilian_asr.text import normalize

ROOT = Path(__file__).resolve().parents[1]
TOKENS = ['<blk>', '<sos/eos>', '<unk>', '▁', 'a', 'b']


def acoustic(ids, size=6):
    p = torch.full((1, len(ids), size), -30., dtype=torch.float32)
    p[0, range(len(ids)), ids] = 0
    return AcousticOutput(TOKENS[:size], 0, {}, p.log_softmax(-1), torch.tensor([len(ids)]))


def test_ordinary_greedy_collapse_and_valid_lengths():
    output = acoustic([3, 4, 4, 0, 4, 5])
    assert greedy_ctc(output) == ['aab']
    output.lengths[0] = 5
    assert greedy_ctc(output) == ['aa']


@pytest.mark.parametrize('change', ['nonfinite', 'unnormalized', 'length_zero', 'length_large', 'length_float', 'dtype'])
def test_invalid_acoustic_outputs(change):
    output = acoustic([4, 5])
    if change == 'nonfinite': output.log_probs[0, 0, 0] = float('nan')
    if change == 'unnormalized': output.log_probs += 1
    if change == 'length_zero': output.lengths[0] = 0
    if change == 'length_large': output.lengths[0] = 3
    if change == 'length_float': output.lengths = output.lengths.float()
    if change == 'dtype': output.log_probs = output.log_probs.double()
    with pytest.raises(ResourceError): output.validate()


def test_normalization_preserves_digits_accents_apostrophes_unknown():
    assert normalize("  È L’ANNO 2026: caffe\u0300, l'acqua! <unk> ") == "è l'anno 2026 caffè l'acqua <unk>"
    assert normalize("'ciao'—due-tre") == 'ciao due tre'


def test_subtitles_preserve_numeric_spoken_lines(tmp_path):
    path = tmp_path/'test.srt'
    path.write_text('1\n00:00:01,500 --> 00:00:03,000\n<i>Ho preso</i>\n6\n\n2\n00:00:04,000 --> 00:00:05,000\nL’acqua &amp; il tè\n')
    ref = read_reference(path)
    assert ref['normalized'] == "ho preso 6 l'acqua il tè"
    assert ref['cues'][0]['start_seconds'] == 1.5
    path = tmp_path/'test.vtt'
    path.write_text('WEBVTT\nKind: captions\nLanguage: it\n\nNOTE hidden\n123\n\nSTYLE\n::cue { color: red; }\n\nidentifier\n00:01.500 --> 00:03.000 align:start\n<v Marco><c.red>Oggi</c> 2 <00:02.000>anni\n')
    assert read_reference(path)['normalized'] == 'oggi 2 anni'


@pytest.mark.parametrize('text', ['1\nbroken\nfoo', '1\n00:00:02,000 --> 00:00:01,000\nfoo', '1\n00:61:00,000 --> 01:02:00,000\nfoo'])
def test_invalid_subtitle_rejected(tmp_path, text):
    p = tmp_path/'bad.srt'; p.write_text(text)
    with pytest.raises(ResourceError): read_reference(p)


def test_alignment_counts_and_empty_reference():
    counts, _ = edits(['a', 'b', 'c'], ['a', 'd', 'c', 'e'])
    assert (counts['substitutions'], counts['insertions'], counts['deletions'], counts['reference_units']) == (1, 1, 0, 3)
    assert edits(['a', 'b'], ['a'])[0]['deletions'] == 1
    assert edits([], ['a'])[0]['rate'] is None
    metrics, _ = evaluate('a <unk>', {'normalized': 'a b 2'}, {'a', 'b'}, {'a'})
    assert metrics['oov_full_supported_lexicon']['tokens'] == 1
    assert metrics['oov_decoding_vocabulary']['tokens'] == 2
    assert metrics['wer']['hypothesis_units'] == 2
    assert metrics['cer']['reference_units'] == 3


def test_sample_exact_segmentation():
    x = np.ones(55*16000, dtype=np.float32)*.1
    x[19*16000:20*16000] = 0
    segments = segment_audio(x, {}, origin=4.5)
    assert segments[0].boundary == 'silence'
    assert segments[0].source_start_seconds == 4.5
    assert segments[-1].end_sample == len(x)
    assert all(a.end_sample == b.start_sample for a, b in zip(segments, segments[1:]))
    assert sum(s.end_sample-s.start_sample for s in segments) == len(x)
    assert all(s.end_sample-s.start_sample <= 24*16000 for s in segments)
    assert any(s.forced_cut for s in segments)


def test_audio_conversion_stream_selection_and_empty_inputs(tmp_path):
    p = tmp_path/'stereo.wav'
    t = np.arange(24000)/48000
    sf.write(p, np.stack([.2*np.sin(2*np.pi*220*t), .1*np.sin(2*np.pi*440*t)], axis=1), 48000)
    audio, info = extract_audio(p, tmp_path/'mono.wav')
    assert audio.shape == (8000,) and info['selected_stream'] == 0
    assert info['duration_seconds'] == .5
    with pytest.raises(ResourceError, match='stream'): extract_audio(p, tmp_path/'bad.wav', 4)
    with pytest.raises(ResourceError): segment_audio(np.array([]), {})
    with pytest.raises(ResourceError): segment_audio(audio, {'max_seconds': 10})


def test_lexicon_alternatives_exclusions_and_unicode(tmp_path):
    token_path = ROOT/'.cache/models/zipa-cr-ns-large-800k/tokens.txt'
    model_path = token_path.with_name('unigram_127.model')
    if not model_path.exists(): pytest.skip('Pinned tokenizer not acquired')
    tokens, tokenizer = inventory(token_path, model_path)
    path = tmp_path/'lexicon.dict'
    path.write_text('caffè\tk a fː ɛ\ncaffe\u0300\tk a fː ɛ\ncaffè\tk a fː e\ngatto\tɡ a tː o\ntest\ttz a\naltro\tbv a\n')
    lex, audit = convert(path, tokens, tokenizer)
    assert len(lex['caffè']) == 2
    assert audit['duplicate_normalized_pairs_removed'] == 1
    assert audit['excluded_rows'] == 2
    assert audit['normalization_collisions']['caffè'] == ['caffe\u0300', 'caffè']
    assert ''.join(tokens[i] for i in lex['gatto'][0]) == 'gatːo'
    assert all(3 not in p for pronunciations in lex.values() for p in pronunciations)


def test_vocabulary_selection_and_unchanged_arpa_weights(tmp_path):
    arpa = ROOT/'tests/fixtures/tiny.arpa'
    words, _ = select_vocabulary(arpa, {'uno': [], 'una': [], 'due': [], 'assente': []}, 2)
    assert words == ['due', 'uno']
    restricted = tmp_path/'filtered.arpa'
    report = restrict_arpa(arpa, restricted, words)
    old = {(n, w): line for n, w, p, line in entries(arpa)}
    for n, w, _, line in entries(restricted): assert old[n, w] == line
    assert report['retained_ngrams'][1] == 5


def test_stale_and_corrupted_cache(tmp_path):
    p = tmp_path/'TLG.pt'; p.write_bytes(b'fixture')
    (tmp_path/'prepared.json').write_text(json.dumps({'status':'complete','key':'key','artifacts':{'TLG.pt':fingerprints(p)['sha256']}}))
    assert verify_cache(tmp_path, 'key')['status'] == 'complete'
    with pytest.raises(ResourceError, match='Stale'): verify_cache(tmp_path, 'different')
    p.write_bytes(b'changed')
    with pytest.raises(ResourceError, match='checksum'): verify_cache(tmp_path, 'key')

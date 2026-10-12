from copy import deepcopy
import io
import json
from pathlib import Path
import tarfile

import pytest
import yaml

from sicilian_asr.config import load_config
from sicilian_asr.pipeline import transcribe
from sicilian_asr.resources import ResourceError, extract_members, fingerprints, load_manifest, verify_file

ROOT = Path(__file__).resolve().parents[1]


def test_academic_archive_checksums_and_derived_integrity(tmp_path):
    archive = tmp_path/'fixture.tar.gz'
    with tarfile.open(archive, 'w:gz') as t:
        data = b'original arpa bytes\n'
        member = tarfile.TarInfo('release/lm.arpa'); member.size = len(data)
        t.addfile(member, io.BytesIO(data))
    actual = fingerprints(archive)
    spec = {'path':'fixture.tar.gz', 'revision':'release-1', 'md5':actual['md5'], 'members':{'release/lm.arpa':'derived/lm.arpa'}}
    assert verify_file(archive, spec)['sha256'] == actual['sha256']
    spec['md5'] = '0'*32
    with pytest.raises(ResourceError, match='md5 mismatch'): verify_file(archive, spec)
    spec['md5'] = actual['md5']
    receipt = extract_members(tmp_path, spec, actual)
    assert receipt['archive_sha256'] == actual['sha256']
    assert (tmp_path/'derived/lm.arpa').read_bytes() == data
    (tmp_path/'derived/lm.arpa').write_bytes(b'corrupted')
    with pytest.raises(ResourceError, match='checksum'): extract_members(tmp_path, spec, actual)


def test_existing_unreceipted_derivative_cannot_self_certify(tmp_path):
    archive = tmp_path/'fixture.tar.gz'
    with tarfile.open(archive, 'w:gz') as t:
        member = tarfile.TarInfo('x'); member.size = 1
        t.addfile(member, io.BytesIO(b'x'))
    (tmp_path/'derived').write_bytes(b'y')
    with pytest.raises(ResourceError, match='differs from archive'):
        extract_members(tmp_path, {'path':'fixture.tar.gz','members':{'x':'derived'}}, fingerprints(archive))


@pytest.mark.parametrize('section,key,value', [('decoder','device','mps'), ('acoustic','device','cuda'), ('decoder','ctc_topology','modified'),
    ('decoder','lm_scale',float('nan')), ('decoder','vocabulary_size',0), ('acoustic','sample_rate',48000)])
def test_unsupported_configurations(tmp_path, section, key, value):
    manifest = load_manifest(ROOT/'resources/manifest.yaml')
    config = yaml.safe_load((ROOT/'configs/ita.yaml').read_text())
    config[section][key] = value
    p = tmp_path/'config.yaml';p.write_text(yaml.safe_dump(config))
    with pytest.raises(ResourceError): load_config(p, ROOT, manifest)


def test_failed_run_retains_diagnostics_without_success_or_greedy_fallback(tmp_path):
    manifest = load_manifest(ROOT/'resources/manifest.yaml')
    config = load_config(ROOT/'configs/ita.yaml', ROOT, manifest)
    output = tmp_path/'output'
    with pytest.raises(FileNotFoundError):
        transcribe(ROOT, config, manifest, tmp_path/'missing.mp4', None, output)
    metadata = json.loads((output/'run_metadata.json').read_text())
    assert metadata['status'] == 'failed' and 'error' in metadata
    assert not (output/'transcription.txt').exists()
    assert not (output/'ipa_greedy.txt').exists()
    with pytest.raises(ResourceError, match='empty'):
        transcribe(ROOT, config, manifest, tmp_path/'missing.mp4', None, output)

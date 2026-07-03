# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cache-identity (``hash_attrs``) regression tests for the reference Datakit DAG.

These are pure StepSpec-construction tests -- no cluster, no data. They lock in the
cache-identity contract: every content-determining parameter enters the step hash, no
region-specific ``gs://`` path does, and external inputs are pinned by a caller
version tag rather than their absolute path.
"""

import dataclasses
import json

import pytest
from marin.execution.step_spec import StepSpec

from experiments.datakit.embeddings.luxical.pipeline import LUXICAL_REVISION
from experiments.datakit.reference_pipeline import (
    SMOKE_SCALE,
    TOKENIZER_REVISION,
    reference_datakit_steps,
)
from experiments.datakit.store.datakit_store import _quality_bucket


def test_quality_bucket_uses_supplied_thresholds():
    # bisect_right: bucket i = [thresholds[i-1], thresholds[i]); N buckets = len+1.
    thresholds = (0.2, 0.4, 0.6, 0.8)
    assert [_quality_bucket(s, thresholds) for s in (0.0, 0.2, 0.5, 0.8, 1.0)] == [0, 1, 2, 4, 4]
    # A different cutoff set re-buckets the same scores -- why it must be hashed.
    assert _quality_bucket(0.5, (0.5,)) == 1
    assert _quality_bucket(0.49, (0.5,)) == 0


@pytest.fixture(autouse=True)
def _marin_prefix(monkeypatch):
    # ``StepSpec.output_path`` resolves ``marin_prefix()``; pin it so the test never
    # depends on ambient GCS metadata. (``hash_id`` itself excludes the prefix.)
    monkeypatch.setenv("MARIN_PREFIX", "gs://marin-test-region")


def _sources() -> dict[str, StepSpec]:
    return {name: StepSpec(name=f"datakit/normalize/{name}", fn=lambda op: None) for name in ("a", "b")}


def _build(*, scale=SMOKE_SCALE, **kw):
    return reference_datakit_steps(
        _sources(),
        quality_model="gs://some-region/quality/model.bin",
        quality_model_version="sonnet46-thr05",
        scale=scale,
        **kw,
    )


def _steps_by_name(result) -> dict[str, StepSpec]:
    return {s.name: s for s in result.all_steps}


def test_no_region_path_in_hash_attrs_except_known_bloom_gap():
    # A region-specific gs:// path in a hash means byte-identical data gets a
    # different output path per region. The only remaining leak is the decontam
    # bloom's EVAL_ROOT (tracked follow-up); everything else must be clean.
    for step in _build().all_steps:
        if step.name == "datakit/bloom/_combined":
            continue
        assert "gs://" not in json.dumps(step.hash_attrs, default=str), f"{step.name} leaks a gs:// path into its hash"


def test_store_hash_tracks_content_not_resources():
    base = _build().output_buckets.hash_id
    # cluster_view is read by the store fn and NOT captured by any dep -> must re-key.
    cv = dataclasses.replace(SMOKE_SCALE.cluster, cluster_view=16)
    changed = _build(scale=dataclasses.replace(SMOKE_SCALE, cluster=cv)).output_buckets.hash_id
    # store_max_workers is execution policy -> must NOT re-key.
    resourced = _build(scale=dataclasses.replace(SMOKE_SCALE, store_max_workers=999)).output_buckets.hash_id
    assert changed != base
    assert resourced == base


def test_minhash_params_rekey_minhash_and_dedup():
    base = _steps_by_name(_build())
    mh = dataclasses.replace(SMOKE_SCALE.minhash, num_bands=13)
    changed = _steps_by_name(_build(scale=dataclasses.replace(SMOKE_SCALE, minhash=mh)))
    assert changed["datakit/minhash/a"].hash_id != base["datakit/minhash/a"].hash_id
    # dedup has no params of its own; it must re-key via its minhash deps.
    assert changed["datakit/dedup"].hash_id != base["datakit/dedup"].hash_id


def test_centroid_seed_rekeys_training():
    base = _steps_by_name(_build())["datakit/cluster/train_centroids"].hash_id
    seeded = dataclasses.replace(SMOKE_SCALE.cluster, train_seed=7)
    changed = _steps_by_name(_build(scale=dataclasses.replace(SMOKE_SCALE, cluster=seeded)))
    assert changed["datakit/cluster/train_centroids"].hash_id != base


def test_embed_and_tokenize_pin_upstream_revisions():
    steps = _steps_by_name(_build())
    assert steps["datakit/embed/a"].hash_attrs["luxical_revision"] == LUXICAL_REVISION
    assert steps["datakit/tokenize/a"].hash_attrs["tokenizer_revision"] == TOKENIZER_REVISION


def test_external_path_requires_version_tag():
    with pytest.raises(ValueError, match="quality_model_version is required"):
        reference_datakit_steps(_sources(), quality_model="gs://r/model.bin", quality_model_version=None)
    with pytest.raises(ValueError, match="centroids_version is required"):
        reference_datakit_steps(
            _sources(),
            quality_model="gs://r/model.bin",
            quality_model_version="v",
            domain_centroids="gs://r/centroids",
            centroids_version=None,
        )


def test_quality_model_version_not_path_drives_identity():
    # Same model bytes staged at two region paths, same version tag -> one output path.
    a = reference_datakit_steps(
        _sources(), quality_model="gs://region-a/model.bin", quality_model_version="sonnet46-thr05"
    )
    b = reference_datakit_steps(
        _sources(), quality_model="gs://region-b/model.bin", quality_model_version="sonnet46-thr05"
    )
    qa = {s.name: s for s in a.all_steps}["datakit/quality_model/reference"]
    qb = {s.name: s for s in b.all_steps}["datakit/quality_model/reference"]
    assert qa.hash_id == qb.hash_id

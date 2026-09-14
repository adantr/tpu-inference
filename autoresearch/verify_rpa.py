"""Real TPU attention/reference checks for the frozen decode experiment."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_inference.kernels.ragged_paged_attention.v3.kernel import (
    get_kv_cache_shape, ragged_paged_attention, ref_ragged_paged_attention)


@pytest.mark.parametrize("capacity", [7936, 8192])
@pytest.mark.parametrize("pattern", ["normal", "sharp", "zeros", "window", "softcap"])
@pytest.mark.parametrize("mode", ["decode", "prefill", "mixed"])
def test_attention_reference_and_cache(capacity, pattern, mode, record_property):
    assert jax.default_backend() == "tpu", "real TPU execution is required"
    rng = np.random.default_rng(917)
    page_size, query_heads, kv_heads, head_dim = 256, 32, 8, 128
    lengths = [1, 255, 256, 257, 4095, 4096, 4097, capacity]
    query_lengths = [1] * len(lengths)
    distribution = [len(lengths)] * 3
    if mode == "prefill":
        lengths = query_lengths = [128, 128, 128, 128]
        distribution = [0, 4, 4]
    elif mode == "mixed":
        lengths, query_lengths = [1, 257, 128, 512], [1, 1, 128, 129]
        distribution = [2, 3, 4]
    count = len(lengths)
    tokens = math.ceil(sum(query_lengths) / 128) * 128
    pages_per_seq = capacity // page_size
    used_pages = sum(math.ceil(length / page_size) for length in lengths)
    total_pages = used_pages + 3
    cache_shape = get_kv_cache_shape(total_pages, page_size, kv_heads,
                                   head_dim, jnp.bfloat16)
    cache = rng.standard_normal(cache_shape, dtype=np.float32)
    queries = rng.standard_normal((tokens, query_heads, head_dim), dtype=np.float32)
    keys = rng.standard_normal((tokens, kv_heads, head_dim), dtype=np.float32)
    values = rng.standard_normal((tokens, kv_heads, head_dim), dtype=np.float32)
    if pattern == "sharp":
        queries *= 4
    elif pattern == "zeros":
        queries.fill(0)
        keys.fill(0)
        cache.fill(0)
    permutation = rng.permutation(total_pages)
    page_indices = np.zeros((count, pages_per_seq), dtype=np.int32)
    cursor = 0
    for index, length in enumerate(lengths):
        pages = math.ceil(length / page_size)
        chosen = permutation[cursor:cursor + pages]
        page_indices[index, :pages] = chosen
        cursor += pages
        tail = length % page_size
        if tail:
            cache[chosen[-1], tail:] = np.nan
    # Unused pages and tail lanes must survive exactly, without contaminating output.
    cache[permutation[cursor:]] = np.nan

    def inputs():
        return (
            jnp.asarray(queries, dtype=jnp.bfloat16),
            jnp.asarray(keys, dtype=jnp.bfloat16),
            jnp.asarray(values, dtype=jnp.bfloat16),
            jnp.asarray(cache, dtype=jnp.bfloat16),
            jnp.asarray(lengths, dtype=jnp.int32),
            jnp.asarray(page_indices.reshape(-1)),
            jnp.asarray(np.cumsum([0] + query_lengths), dtype=jnp.int32),
            jnp.asarray(distribution, dtype=jnp.int32),
        )

    options = dict(sm_scale=head_dim ** -0.5,
                   sliding_window=128 if pattern == "window" else None,
                   soft_cap=5.0 if pattern == "softcap" else None)
    routing = dict(chunk_prefill_size=128 if mode != "decode" else None)
    # FP32 reference scores avoid rounding the dot product before scaling it.
    expected, expected_cache = ref_ragged_paged_attention(
        *inputs(), **options, out_dtype=jnp.float32)
    output, updated_cache = jax.block_until_ready(
        ragged_paged_attention(*inputs(), **options, **routing))
    expected = np.asarray(expected, dtype=np.float32)
    actual = np.asarray(output[:sum(query_lengths)], dtype=np.float32)
    assert np.isfinite(expected).all() and np.isfinite(actual).all()
    error = np.abs(actual - expected)
    record_property("max_absolute_error", float(error.max()))
    record_property("rms_error", float(np.sqrt(np.mean(error ** 2))))
    np.testing.assert_allclose(actual, expected, atol=0.02, rtol=0.02)
    np.testing.assert_array_equal(np.asarray(updated_cache).view(np.uint16),
                                  np.asarray(expected_cache).view(np.uint16))
    if pattern == "normal":
        # Shared layers must ignore their supplied K/V and leave all cache bits alone.
        shared_inputs = list(inputs())
        shared_inputs[1] = jnp.full_like(shared_inputs[1], jnp.nan)
        shared_inputs[2] = jnp.full_like(shared_inputs[2], jnp.nan)
        shared_inputs[3] = jnp.array(np.asarray(expected_cache))
        shared, shared_cache = jax.block_until_ready(ragged_paged_attention(
            *shared_inputs, **options, **routing, update_kv_cache=False))
        np.testing.assert_array_equal(np.asarray(shared[:sum(query_lengths)]),
                                      np.asarray(output[:sum(query_lengths)]))
        np.testing.assert_array_equal(np.asarray(shared_cache).view(np.uint16),
                                      np.asarray(expected_cache).view(np.uint16))

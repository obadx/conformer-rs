# muaalem_streaming_kv_cache

Streaming inference extension for the Muaalem Wav2Vec2Bert conformer ASR model.  
Enables chunked audio processing with per-layer **KV cache** + **conv cache**, bounded memory via FIFO eviction, and optional sliding-window attention.

---

## Architecture

```
Input chunk → FeatureProjection → Wav2Vec2BertEncoder (×N layers)
                                      │
                            ┌─────────┴──────────┐
                            │  SelfAttention      │ ← KV cache (past_key_values)
                            │  ConvModule         │ ← Conv cache (conv_caches)
                            └─────────┬──────────┘
                                      │
                                   LM heads (per level)
                                      │
                                   logits
```

Two state objects are threaded across chunks:

| State | Shape | Purpose |
|---|---|---|
| `past_key_values` | `list[tuple(K, V)]` — one `(batch, num_heads, seq_len, head_dim)` pair per layer | Lets self-attention attend to previous chunks |
| `conv_caches` | `list[Tensor(batch, hidden, kernel-1)]` — one per layer | Prevents information loss in the causal depthwise conv at chunk boundaries |

---

## What was fixed

### 1. `position_embeddings_type` default — `configuration_....py:56`

**Before:** `None` (no position embeddings)  
**After:** `"relative_key"` (matches pretrained weights)

Without this, loading a pretrained offline model checkpoint would silently produce wrong attention because the relative position embedding layers would never be called.

### 2. Sliding window threshold swap — `modeling_....py:~428`

**Before:**
```python
sliding_mask = (gap > sliding_window_right) | (gap < -sliding_window_left)
```

**After:**
```python
sliding_mask = (gap > sliding_window_left) | (gap < -sliding_window_right)
```

`sliding_window_left=25` should limit lookback (past context); `sliding_window_right=5` should limit lookahead. They were inverted, so only **5 frames** of past context survived the mask, making the cache nearly useless.

### 3. ConvModule had no cross-chunk state — `modeling_....py:~214`

The causal depthwise conv (`kernel_size=31`) always padded the left side with `kernel_size-1` zeros at every chunk. This zeros out temporal context at chunk boundaries.

**Fix:** ConvModule now accepts an optional `conv_cache` tensor — the last `kernel_size-1` frames of the GLU output from the previous chunk — and uses it instead of zeros.

```python
if conv_cache is not None:
    hidden_states = torch.cat([conv_cache, hidden_states], dim=-1)
else:
    hidden_states = torch.nn.functional.pad(hidden_states, (cache_frames, 0))
```

### 4. conv_cache was not threaded through the stack

Added `conv_caches` parameter at every level:
`ConvModule → EncoderLayer → Encoder → Wav2Vec2BertModel → Wav2Vec2BertForMultilevelCTCStreamingKVCache`

Returned as `outputs["conv_caches"]` alongside `outputs["past_key_values"]`.

### 5. Removed misleading auto-generated header

The file had a "DO NOT EDIT — auto-generated from modular" banner copied from HuggingFace. Removed since this is hand-written custom code.

---

## Usage

```python
from conformer_python.muaalem_streaming_kv_cache import (
    Wav2Vec2BertForMultilevelCTCStreamingKVCacheConfig,
    Wav2Vec2BertForMultilevelCTCStreamingKVCache,
)

config = Wav2Vec2BertForMultilevelCTCStreamingKVCacheConfig(
    level_to_vocab_size={"phonemes": 42, "tarf": 3, ...},
)
model = Wav2Vec2BertForMultilevelCTCStreamingKVCache(config)

# --- Full-utterance training ---
outputs = model(input_features=full_audio, labels=labels)
loss = outputs["loss"]

# --- Chunked streaming inference ---
# Chunk 1
outputs = model(input_features=chunk_1, attention_mask=mask_1)
logits = outputs["logits"]
past_kv = outputs["past_key_values"]
conv_c = outputs["conv_caches"]

# Chunk 2..N
outputs = model(
    input_features=chunk_n,
    attention_mask=mask_n,
    past_key_values=past_kv,
    conv_caches=conv_c,
)
logits = outputs["logits"]
past_kv = outputs["past_key_values"]   # updated
conv_c = outputs["conv_caches"]        # updated
```

---

## Configuration

Key streaming parameters in `Wav2Vec2BertForMultilevelCTCStreamingKVCacheConfig`:

| Param | Default | What it controls |
|---|---|---|
| `chunk_size` | 25 | Frames per chunk (dictates how often KV cache is updated) |
| `lookahead_frames` | 5 | Right context per chunk (not yet used in attention) |
| `max_context_frames` | 250 | Max cached key/value frames before FIFO eviction drops oldest |
| `use_sliding_window` | False | Whether to constrain attention within a sliding window |
| `sliding_window_left` | 25 | Max past frames a query can attend to in the cache |
| `sliding_window_right` | 5 | Max future frames a query can attend to |
| `position_embeddings_type` | `"relative_key"` | Must match the pretrained checkpoint |

---

## Known limitations

- **No tests** — The implementation is not exercised by any test in the repo. At minimum, verify with `test_sliding_window.py` or a simple forward/backward sanity check.
- **No overlap-add stitching** — The model returns logits per chunk but does not handle output overlap/stitching. The caller must implement that (see `quran-muaalem/tests/test_sliding_window.py` for the offline approach).
- `conv_caches` are `detach()`ed — Gradients do not flow across chunks during inference, which is correct for streaming but means no chunk-level gradient checkpointing.

---

## File structure

```
muaalem_streaming_kv_cache/
├── __init__.py                                         # Public exports
├── configuration_multi_level_ctc_streaming_kv_cache.py # Config class
├── modeling_multi_level_ctc_streaming_kv_cache.py     # Model + all modules
├── multi_level_tokenizer_streaming_lstm.py            # Multi-level tokenizer
├── vocab.py                                           # Vocab builder
└── README.md                                          # This file
```

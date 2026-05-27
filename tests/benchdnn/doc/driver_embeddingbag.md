# Embedding Bag Driver

## Usage
``` sh
    ./benchdnn --embedding_bag [benchdnn-knobs] [embeddingbag-knobs] [embeddingbag-desc] ...
```

where *embeddingbag-knobs* are:

 - `--dir={FWD_I [default]}` -- dnnl_prop_kind_t. Only forward inference is
            supported. Refer to [direction](knobs_dir.md) for details.
 - `--tbldt={f32 [default]}` -- embedding table data type.
            Refer to [data types](knobs_dt.md) for details.
            Supported values: `f32`.
 - `--dstdt={f32 [default]}` -- destination (output) data type.
            Refer to [data types](knobs_dt.md) for details.
            Supported values: `f32`.
 - `--wtdt={f32 [default]}` -- per-sample weights data type (used only when
            `--is_weight=true`). Supported values: `f32`.
 - `--idt={s32 [default]}` -- indices tensor data type.
            Supported values: `s32`.
 - `--odt={s32 [default]}` -- offsets tensor data type.
            Supported values: `s32`.
 - `--alg={EMBEDDING_BAG_SUM [default], EMBEDDING_BAG_MEAN, EMBEDDING_BAG_MAX,
            EMBEDDING_BAG_LOOKUP}` -- reduction algorithm applied over each bag.
            - `EMBEDDING_BAG_SUM` -- sum of selected embedding vectors.
            - `EMBEDDING_BAG_MEAN` -- mean of selected embedding vectors.
            - `EMBEDDING_BAG_MAX` -- element-wise maximum of selected embedding
              vectors.
            - `EMBEDDING_BAG_LOOKUP` -- direct lookup (single index per bag,
              no reduction). Equivalent to a simple embedding lookup.
 - `--padding_idx=INT` -- index value to treat as padding and exclude from
            aggregation. Use `-1` (default) to disable padding index handling.
 - `--is_weight=BOOL` -- enable per-sample weights for the `SUM` algorithm.
            If `true`, a weights tensor with the same shape as the indices tensor
            is used to scale each selected embedding before summing. Only valid
            with `EMBEDDING_BAG_SUM`. The default is `false`.
 - `--include_last_offset=BOOL` -- when `true`, the offsets tensor contains one
            extra element at the end that equals the total number of indices.
            This mirrors the PyTorch `include_last_offset` convention and allows
            the driver to compute each bag's size without implicit assumptions.
            The default is `false`.
 - `--match=REGEX` -- skip problems not matching the regular expression in
            `REGEX`. By default no pattern is applied (run everything).
            Note: Windows may interpret only string arguments surrounded by
            double quotation marks.
 - Any attributes options. Refer to [attributes](knobs_attr.md) for details.

and *embeddingbag-desc* is a problem descriptor. The canonical form is:
```
    ExF:G:H
```
where each field is a colon-separated tensor descriptor:

| Field | Tensor    | Shape   | Description                                      |
|-------|-----------|---------|--------------------------------------------------|
| `ExF` | table     | `E x F` | Embedding table: `E` rows (vocabulary size), `F` embedding dimension |
| `G`   | indices   | `G`     | Flat list of `G` integer indices into the table  |
| `H`   | offsets   | `H`     | `H` offsets, one per output bag; the i-th bag covers `indices[offsets[i] : offsets[i+1]]` |

The output tensor shape is derived automatically as `H x F` (number of bags by
embedding dimension).

When `--include_last_offset=true`, the offsets tensor has `H` entries where the
last entry equals `G`, so each bag boundary is fully specified.


## Tensor Memory Layout

All tensors use the `abx` (plain row-major) memory layout. The driver does not
currently expose a tag knob for embedding bag tensors.


## Constraints

- Only forward inference (`FWD_I`) is supported; no backward pass.
- `--tbldt` and `--dstdt` must be `f32`.
- `--idt` and `--odt` must be `s32`.
- `--is_weight=true` is only valid with `--alg=EMBEDDING_BAG_SUM`.
- The number of indices (`G`) must be greater than or equal to the number of
  offsets (`H`).
- Exactly three dimension groups must be supplied in the problem descriptor
  (`TABLExDIM:INDICESxDIM:OFFSETSxDIM`).


## Essence of Testing

**Table** values are filled with uniform random floats in `[1.0, 2.0)`.

**Indices** are filled with uniform random integers in `[1, E-1)` where `E` is
the vocabulary (table row) count, ensuring all indices are valid.

**Offsets** are filled with monotonically increasing values starting at 0, with
a stride computed as `(G - 2) / H` (minimum stride of 1), so that each bag
contains at least one index and no bag overruns the indices buffer.

**Weights** (when `--is_weight=true`) are filled with sequential integers
`1, 2, 3, ...` matching the indices count.

**Destination** is left zero-initialised before execution and validated against
a reference implementation after the primitive runs.

Correctness is checked on the `DST` tensor only. The reference computation
iterates over each bag's index range and applies the configured reduction
(`sum`, `mean`, `max`, or direct `lookup`) while skipping any index equal to
`padding_idx`.


## Examples

Run a basic sum embedding bag with a 1000-row, 64-dim table, 4096 indices, and
128 bags:
``` sh
    ./benchdnn --embedding_bag 1000x64:4096:128
```

Run the mean algorithm:
``` sh
    ./benchdnn --embedding_bag --alg=EMBEDDING_BAG_MEAN 1000x64:4096:128
```

Run the max algorithm:
``` sh
    ./benchdnn --embedding_bag --alg=EMBEDDING_BAG_MAX 1000x64:4096:128
```

Run a direct lookup (one index per bag):
``` sh
    ./benchdnn --embedding_bag --alg=EMBEDDING_BAG_LOOKUP 512x32:256:256
```

Run a weighted sum with padding index 0:
``` sh
    ./benchdnn --embedding_bag --alg=EMBEDDING_BAG_SUM --is_weight=true \
               --padding_idx=0 1000x64:4096:128
```

Run with `include_last_offset` enabled (offsets tensor carries the trailing
sentinel):
``` sh
    ./benchdnn --embedding_bag --include_last_offset=true 1000x64:4096:128
```

Sweep algorithms with a shared descriptor:
``` sh
    ./benchdnn --embedding_bag \
               --alg=EMBEDDING_BAG_SUM,EMBEDDING_BAG_MEAN,EMBEDDING_BAG_MAX \
               1000x128:8192:256
```

More examples with different benchdnn common options can be found at
driver_conv.md.

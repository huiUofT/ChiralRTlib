# Training Data Format

The public repository does not include the original curated data. To reproduce
training, provide derived training tables in a local directory such as
`training_tables/` with the following names:

```text
{column}_stage1.csv
{column}_stage2_rs.csv
```

where `{column}` is one of `IA`, `IB`, `IC`, or `IG`.

## Stage 1 Table

Required columns:

| Column | Description |
| --- | --- |
| `mol_id` | Stable molecule identifier. |
| `smiles` | Molecular SMILES string. |
| `stage1_label` | Binary separation label, where `1` means separated and `0` means not separated. |
| `scaffold` | Scaffold group used for scaffold-aware splitting. |

Example header:

```csv
mol_id,smiles,stage1_label,scaffold
```

## Stage 2 Rs Table

Required columns:

| Column | Description |
| --- | --- |
| `mol_id` | Stable molecule identifier matching the Stage 1 table. |
| `rs` | Measured chromatographic resolution for separated examples. |

Example header:

```csv
mol_id,rs
```

The training script left-joins `rs` onto the Stage 1 table by `mol_id`. Missing
`rs` values are expected for negative examples.

# tasks/label_heads

Label-head implementations used by `train/split_data.py` and `main.py`.

- `base.py`: common interface and label-map helpers.
- `manuscript_id.py`: default task; labels are manuscript IDs.
- `decade.py`: optional dating task; labels are coarse decade buckets.
- `factory.py`: constructs the requested head by name.

When adding a new task, implement `LabelHead`, add it to `factory.py`, then update
`system.validate_runtime_config()` so invalid task names fail early.

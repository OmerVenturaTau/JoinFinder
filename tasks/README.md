# tasks

Task abstractions for turning dataset rows into supervised labels.

`label_heads/` contains the concrete label heads. The current default in
`system.py` is `LABEL_HEAD = "manuscript_id"`. The dating/decade head exists for
experiments that use dating tables, but it is not the default training task.

Role in the page workflow: after `train/dataset.py` converts a page into model
inputs, the selected label head defines what target the fused page latent is
trained to predict. Changing the label head changes how the latent space is shaped
before it is exported for Geniza clustering.

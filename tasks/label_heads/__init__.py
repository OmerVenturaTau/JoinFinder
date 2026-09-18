"""Task label heads (what the classifier predicts).

This package centralizes how we derive a *single* class label per image/sample.
Default head is `manuscript_id`, but you can switch to other heads (e.g. `period`)
via `system.LABEL_HEAD` without touching training code.
"""



# Provisional feature descriptions

The release includes [numeric results and annotations](../feature_validation_results.json) for 36 diagnostic features, 12 per trained layer. Selection was deliberately stratified by earlier assistant review (six apparently coherent, four mixed, two insufficient-observation features per layer), not sampled to estimate dictionary-wide accuracy.

## Completed checks

The same unblinded AI assistant authored the descriptions and synthetic contrast tests. No independent human or expert review occurred. The 28 hypotheses, eight abstentions and 112 positive/close-negative pairs were frozen before inference (SHA-256 `925df19df5a2a478f177b7ada9fb4664c0449b3a112918ccb40461a242425c62`).

Across 224 synthetic sequences, the results were **24 supported-pattern, four inconclusive, eight abstained**. Each feature's annotation, stratum, disposition and numeric pair outcomes are retained, including failures to establish a clear pattern. No source excerpts, token windows or synthetic prompt text are distributed; this limits independent rerunning of the contrast experiment.

A pair passed if its positive target-span maximum was at least 0.25 × q25 and its positive-minus-negative difference at least 0.10 × q25. Here q25 is the lower quartile of saved top-example peaks, not a natural activation quantile. At least three of four passing pairs yielded supported-pattern; at most one passing pair and at least two strong reversed negatives yielded refuted-contrast-prediction; remaining nonabstained cases were inconclusive. There were no refuted-contrast-prediction dispositions.

These are custom exploratory thresholds. The outcomes do not measure semantic accuracy, exclusivity, monosemanticity, causal necessity or sufficiency. A passing pair can still have a strongly activating negative. Freezing tests before inference does not remove shared-author bias.

Original full-input replay matched 300 saved feature records across 298 sequences with maximum absolute activation error 0.0. It preserved full-dictionary TopK competition, normalization and position exclusion. Those examples were already seen during development and are not held-out semantic evidence.

## Missing evidence

Independent-model held-out assessment was not completed: **zero assessment judge calls and no independent held-out metrics** are recorded. All 280 retained positive examples for the 28 nonabstained hypotheses overlapped development evidence. A separate 300-feature assistant review is descriptive, not a semantic-accuracy benchmark.

Training used an unpinned streaming dataset revision without consumed-document IDs; complete training/evaluation disjointness cannot be established. No causal intervention result or dictionary-wide semantic-quality claim is made. Publishing the artifacts does not change these limitations.

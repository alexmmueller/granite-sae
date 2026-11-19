# Rights and attribution

Copyright 2026 Alex M. Müller.

The owner-owned custom SAE weights, training code, reference loader,
documentation, provisional annotations and numeric results in this package are
licensed under the Apache License, Version 2.0 (SPDX: `Apache-2.0`). See the
complete, unmodified canonical text in [LICENSE](LICENSE) and attribution in
[NOTICE](NOTICE). This grant does not license upstream base-model weights or
rights-uncleared corpus content, nor override third-party rights or notices.
It does not assert sole authorship of upstream material or upstream approval.

The three existing safetensors files retain the embedded literal
`LICENSE_CHOICE_PENDING` as export-time license-selection provenance only.
That legacy field is superseded by the current Apache-2.0 grant in `LICENSE`
and the external binding in `manifest.json` and each layer's `metadata.json`.
The external records identify each weight file by path and SHA-256 and record
the legacy field separately in `embedded_metadata_provenance`; their current
`license` fields are Apache-2.0. Weight bytes and hashes were not changed merely
to update this historical header. Future exports declare Apache-2.0 directly.
Licensing does not authorize publication or certify scientific quality.

The base model is `ibm-granite/granite-4.2-3b`, revision
`4e9c275e232f10211ec88644d106a1f6a1b60e06`, by the Granite Team, IBM.
Its pinned model card declares Apache-2.0; the inspected repository tree has no
separate LICENSE file. Base-model weights are not included. Applicable upstream
notice obligations require review independently of the custom-artifact license.

Training used HuggingFaceFW/FineWeb-Edu, `sample-10BT`, credited to Anton Lozhkov,
Loubna Ben Allal, Leandro von Werra and Thomas Wolf (2024), Hugging Face,
DOI 10.57967/hf/2497. The dataset card declares ODC-By 1.0 and Common Crawl terms.
Database rights do not clear the rights of every underlying webpage. The package
therefore excludes corpus excerpts, token windows and source documents.
The limited provisional description/result subset contains annotations and
numeric aggregates only, not source text or a complete label set. These custom
annotations and results are covered by Apache-2.0. Any later example release requires its own rights
and privacy review.

Source URLs and retrieval digests appear in `provenance/rights_sources.json`.

# Causal audit and frozen comparison — 9 November 2026

Francisco Angulo de Lafuente · [ORCID0009-0001-1634-7063](https://orcid.org/0009-0001-1634-7063).

The prepared historical workflow had two executed validation defects: a supervised
reservoir readout was fitted before validation splits, and validation-day reputation
features used labels unavailable to a wholly unlabelled deployment day. Corrections
were versioned and reviewed before the new target snapshot was acquired. Original
research sources in the repository root are preserved as historical material and
must not be treated as validated predictive evidence.

A fixed control (causal tabular features, LightGBM, temporal calibration) was
compared with the same model plus 96 label-free random recurrent coordinates.
Both portable alternatives were exported before results were viewed.

| Prespecified arm | Heldout days | Total idealised PnL | Purchases |
|---|---:|---:|---:|
| Corrected control |11|0.0|0|
| Control plus random reservoir |11|0.0|0|
| Always abstain anchor |11|0.0|0|

The paired candidate-minus-control mean is0.0; its circular two-day-block bootstrap
95%interval is[0.0,0.0] (10,000 draws, seed 20261109). The promotion gate required at
least 7 paired days, positive candidate total, and a strictly positive lower bound.
It failed. The interval is degenerate because both policies abstained throughout;
it provides no evidence of equivalence or of a profitable predictive signal.

Dates 28 October–7 November are historical observations newly unseen by this rescue
when the protocol was frozen, not live forward forecasts. All earlier dates were
development or threshold warmup. The protocol's original word “prospective” is
retained in the sealed record for traceability; that word does not turn these
historical dates into real-time evaluation. The future December organiser rotations
remain the official confirmation.

The source is CPU only, uses no network or pretrained fitted object, fits inside
each run, masks the complete heldout day for reputation features, and excludes
unresolved prior outcomes. The fixed last-eight-day threshold rule never ranks
test tokens or imposes a test-day purchase cap. It retains the fit-segment models
after isotonic calibration. Five unaffected sequence channels are used; the two
legacy price/depth sequence channels with an observed-zero filling defect were
excluded before this comparison.

The same run produced 38 prediction artifacts:19 paired dates including 8 warmup and
11 evaluation days. It finished successfully in about 37 minutes on four CPU cores.
The independent verifier bound all prediction identities, labels, timestamps and
source/input hashes, recomputed thresholds and official payments, and reproduced
the fixed bootstrap. Mandatory input checks covered 165 official raw files,
44,165 aligned tokens,56 tape feature columns and9,184,377 wallet edges. Raw files, labels,
per-token arrays and model objects are not redistributed.

Use portable/memecoins-causal-control.ipynb for the selected, unproven baseline.
The reservoir alternative remains available to reproduce the negative comparison;
it was not promoted. The exact selected notebook completed all six code cells on the owner Linux
server in 662.034 seconds, with four CPU cores, no GPU, about 3.1 GiB peak child
RSS and a legal 292-row CSV (zero purchases). Input transfer hashes covered all
165 official data files. This owner-server check is distinct from the organiser's
sandbox trial and acceptance; see RUNTIME_VERIFICATION.json and DELIVERY_STATUS.json. This archive alone is not a league entry.

Reproduction: obtain the organiser's authorised files separately, attach them to
the notebook, disable internet/GPU, and run all six code cells. The plain-source
notebook trains from attached train.csv and optional train_archive.csv and writes
submission.csv. study.py and the verifiers preserve their original audit-workspace
paths; recreate those source/input contracts in a separate workspace for an audit.
Do not remove missing-proof guards. No further parameter search was performed on
these 11 days. A new competitive hypothesis requires a newly frozen protocol and
genuinely fresh outcomes.

Prior work included Claude-assisted preparation; Codex executed the correction,
comparison and delivery checks. Fresh-context technical reviews were internal
checks, not external certification. No contest win, leaderboard result or real-world
financial return is established by this release.

## Historical-code and input boundaries

The preserved root src/solution.py executes loading and model training at module
import time. Do not import or run it as the current workflow. It contains the
superseded validation method described above. The selected plain-source portable
notebook is the current entry path.

acquire_snapshot_v4.py is included byte for byte to show the executed input
rebuild. Its frozen raw snapshot and CURRENT_FILE_METADATA_ALL.json, daily
derived-input manifests and original review receipts remain private. The recorded
INPUT_MANIFEST and protocol hashes allow integrity comparisons; they do not
redistribute those inputs. Consequently this is a source-and-aggregate archive,
not a standalone byte-identical rebuild of the private historical audit workspace.
The portable notebook needs only separately authorised attached competition files;
the separate audit verifier requires reconstructing the documented private proofs.
Root software retains its scoped MIT notices; the Holographic-Reservoir port also
retains its Apache-2.0 attribution, accompanied here by the full Apache license.

# SVM Classifier Benchmark

## Goal

The SVM benchmark is a standalone classifier experiment, separate from the HCA clustering work.

## Design

- Reuse the same hybrid-plus-core winner feature space.
- Keep the same chronological train / validation / test rule.
- Tune threshold on validation, refit on train + validation, then score once on test.
- Test a small SVM grid around the regularization strength `C`.
- Include one `drop_location` variant to check whether raw station identity matters for the SVM.

## Important Constraint

This benchmark uses a linear SVM so the full dataset and validation-first design remain tractable in the current
CPU-only repository.

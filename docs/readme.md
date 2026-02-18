# VSGUI10K Preprocessing Pipeline

## Overview
Official-style preprocessing pipeline for VSGUI10K visual search modeling.

## Steps implemented
- Search-phase isolation (img_type == 2)
- Fixation validity + duration filtering
- Coordinate normalization (FPOGX_scaled)
- Trial segmentation
- Target bbox alignment
- Validation-based sanity checks

## How to run

Place dataset inside:

data/
    vsgui10k_fixations.csv
    vsgui10k_targets.csv
    vsgui10k-images/
    segmentation/

Then run:

python preprocessing/preprocess_official_with_validation.py
python preprocessing/qa_vsgui10k_validation.py

## Outputs
- trials_official_with_validation.jsonl.gz
- preprocess_validation_stats.json
- validation_report.json

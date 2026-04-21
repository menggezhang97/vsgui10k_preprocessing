# VSGUI10K visual search-pipeline

## Overview
Official-style preprocessing pipeline for VSGUI10K visual search modeling.

## Steps implemented
- Search-phase isolation (img_type == 2)
- Fixation validity + duration filtering
- Coordinate normalization (FPOGX_scaled)
- Trial segmentation
- Target bbox alignment
- Validation-based sanity checks

## How to prepare the data:
Download VSGUI10K dataset via  https://osf.io/hmg9b/. Files are read from the data folder. 

Place dataset inside:
data/
    vsgui10k_fixations.csv
    vsgui10k_targets.csv
    vsgui10k-images/
    segmentation/

Then run:

python ./preprocess_official_with_validation.py
python ./qa_vsgui10k_validation.py

Outputs:
- trials_official_with_validation.jsonl.gz
- preprocess_validation_stats.json
- validation_report.json
- debug_official_with_validation/image_examples


## How to train the model and get the visualization:
Use the commit line like below to tun the model:
python UImain_patch_hybrid/[name of the train.py] --config [the diectory and name of the configs u want to use] --alpha [amount u want to run] --freeze_patch
Use this to run the visualization:
python UImain_patch_hybrid/visualize_patch_hybrid_scanpath_real.py --checkpoint UImain_patch_hybrid/outputs/[the .ptfile that u want to set as checkpoint] --trials_path trials_official_with_validation.jsonl.gz --image_dir data/vsgui10k-images --seg_root data/segmentation --output_dir [directory u set]

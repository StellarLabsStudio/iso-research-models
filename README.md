# iSO Research Models

The **iSO Research Models** repository is the central hub for all
machine learning research and model implementations within the
**StellarLabs iSO ecosystem**.\
It serves both as a **research playground** for reproducing papers and
experimenting with new architectures, and as a **production-oriented
model library** for the iSO platform.

## Overview

This repository includes:

-   Implementations of **state-of-the-art ML research papers**
-   Experimental and production iSO model prototypes:
    -   Real-time camera capture correction
    -   Media enhancement & restoration
    -   Ownership verification & provenance systems
    -   Visual similarity & reverse image search embeddings
    -   Detection, classification, and multimodal vision models
-   Modular training and evaluation pipelines
-   Dataset loaders and experiment configurations
-   Full documentation for research papers and model architectures

## Repository Structure

    iso-research-models/

    models/
        ssrw_v1/
        ssrw_v2/

    datasets/
        raw/
            coco/
            msflickr/
        split/
            coco/
                test/
            msflickr/
                train/
                val/

    training/
        ssrw_v1/
        ssrw_v2/

    scripts/
        setup_ssrwdatasets.py

    docs/
        model_cards/
        architecture_diagrams/
        research-papers/

    tests/
        ssrw_v1/
        ssrw_v2/
        
    configs/
        ssrw_v1.yaml
        ssrw_v2.yaml
    
    LICENSE
    README.md

## Documentation

The `docs/` directory contains all research and model documentation.

### docs/research-papers/

Summaries, notes, and insights from academic papers reproduced in this
repository.

### docs/model_cards/

Documentation for each model: purpose, architecture, training details,
evaluation results, and usage.

### docs/architecture_diagrams/

Visual diagrams for model architectures and system pipelines.

## Installation

    pip install -r requirements.txt

## Running Experiments

### Train a model:

    python training/train.py --config experiments/camera_correction_v1.yaml

### Evaluate a model:

    python evaluation/camera_correction_eval.py --weights path/to/weights.pt

## Tests

Run the full test suite:

    pytest tests/

## Contributing

1.  Create a new branch or fork the repository\
2.  Add your model, experiment, or paper implementation\
3.  Document it under `docs/model_cards/` or `docs/research-papers/`\
4.  Open a pull request

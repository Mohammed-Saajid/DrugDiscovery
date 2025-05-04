# Memory-Augmented Attention Model for Molecular Generation

This repository contains the implementation of a memory-augmented attention-based deep learning model designed to generate novel chemical molecules. The model is set to be trained on a combined dataset composed of ZINC, ChEMBL, and MOSES, offering a diverse and chemically valid molecular space for training.

##  Overview

Molecular generation is a crucial task in computational drug discovery and cheminformatics. This project introduces a generative model that leverages memory-augmented attention mechanisms to improve the generation of novel, valid, and diverse molecular structures.

Key features of this model:
- Memory-augmented attention mechanism for better contextual learning
- SELFIES-based molecular representation
- Combined dataset from ZINC, ChEMBL, and MOSES for enhanced diversity
- Capable of generating novel, valid, and synthesizable molecules

##  Model Architecture

The core of the model architecture is based on an attention mechanism enhanced with an external memory module, allowing it to retain long-term molecular structure dependencies. The architecture includes:

- Encoder-Decoder structure
- Self-attention layers
- Memory interaction modules (read/write mechanisms)
- SELFIES token embeddings

##  Datasets

Three major publicly available datasets were used for training:

- **[ZINC](https://zinc.docking.org/)**: A free database of commercially-available compounds.
- **[ChEMBL](https://www.ebi.ac.uk/chembl/)**: A manually curated database of bioactive molecules with drug-like properties.
- **[MOSES](https://github.com/molecularsets/moses)**: A benchmark dataset for molecular generation tasks.

These datasets were merged to form a unified training dataset.


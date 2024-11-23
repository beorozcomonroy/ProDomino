# ProDomino: *In silico* prediction of protein domain insertion sites to engineer switchable proteins

---
## Introduction

ProDomino enables the prediction of domain insertion sites based on embeddings from 
[ESM-2](https://github.com/facebookresearch/esm) 3B model. The model returns a per position score for the probability that this position tolerates 
an insert.
## Requirements

All experiments were run using python 3.10 and pytorch 2.10 using CUDA 12.1 AND CUDNN 8.9.2


## Installation

To use this repo make sure u have conda or mamba installed on your machine.
The run:
```bash
conda env create -n prodomino --file environment.yml
```

If installation fails check if you have the required CUDA versions installed or adapt them to your systems version.


## Usage

ProDomino provides two main classes: `Embedder` and `ProDomino` that allow the prediction of insertion sites.

`Embedder` provides an interface to ESM-2 generating the requirement inout data for `ProDomino`

`ProDomino` then generates the final prediction and provides various plotting utilities.

```python
from ProDomino import Embedder, ProDomino

seq = ''

embedder = Embedder()
model = ProDomino(
    chkpt,'mini_3b_mlp')

embed = embedder.predict_embedding(seq)
pred = model.predict_insertion_sites(embed)
```

A complete example can be found in `example.ipynb`




---
## Cite





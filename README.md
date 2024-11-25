# ProDomino: *In silico* prediction of protein domain insertion sites to engineer switchable proteins

<img src="img/ProDomino.png" alt="drawing" width="300"/>  

## Abstract 

Domain insertion engineering is a powerful approach to juxtapose otherwise separate biological functions, resulting in proteins with new-to-nature activities. A prominent example are switchable proteins, created by inserting receptor domains into effector proteins. Identifying suitable, allosteric sites for domain insertion, however, typically requires extensive screening and optimization, limiting the utility of this approach.
We present ProDomino, a novel machine learning pipeline for domain recombination, trained on a synthetic protein sequence dataset derived from naturally occurring intradomain insertions. We show that ProDomino can robustly identify domain insertion sites in proteins of biotechnological relevance, which we experimentally validated in E. coli and human cells. Finally, employing light- and chemically regulated receptor domains as inserts, we demonstrate the rapid, model-guided creation of potent, single-component opto- and chemogenetic protein switches, including CRISPR-Cas9 and Cas12a switches for inducible genome editing in human cells. Our work simplifies domain insertion engineering and substantially accelerates the design of customized allosteric proteins.


---

ProDomino enables the prediction of domain insertion sites  The model returns a per position probability score for insertion site tolerance.
For further details, please refer to our manuscript: LINK!!!!!

## Requirements

All experiments were run using python 3.10 and pytorch 2.10 using CUDA 12.1 AND CUDNN 8.9.2
For insertion site prediction, ProDomino uses embeddings from the [ESM-2](https://github.com/facebookresearch/esm) 3B model.

## Installation

To use this repo make sure tha you have installed conda or mamba on your device.
Then run:
```bash
conda env create -n prodomino --file environment.yml
```

A likely point of installation failure are CUDA version conflicts. If installation fails install the required CUDA or adapt them to your systems version.


## Usage

ProDomino provides two main classes: `Embedder` and `ProDomino`.

`Embedder` provides an interface to ESM-2 to generate the required input data for `ProDomino`

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





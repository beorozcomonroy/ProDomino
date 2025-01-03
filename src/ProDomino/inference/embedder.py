import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from ..models import MODELS
from ..utils.tracker import Hparams

import esm
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.wrap import enable_wrap, wrap

from Bio.PDB.Polypeptide import three_to_one
from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB import PDBParser, PDBIO
from io import StringIO
import py3Dmol

class Embedder:
    def __init__(self, model_name = "esm2_t36_3B_UR50D", num_layers=36):
        url = "tcp://localhost:8891"
        torch.distributed.init_process_group(backend="nccl", init_method=url, world_size=1, rank=0)
        model_data, regression_data = esm.pretrained._download_model_and_regression_data(model_name)
        # initialize the model with FSDP wrapper
        fsdp_params = dict(
            mixed_precision=True,
            flatten_parameters=True,
            state_dict_device=torch.device("cpu"),  # reduce GPU mem usage
            cpu_offload=True,  # enable cpu offloading
        )
        with enable_wrap(wrapper_cls=FSDP, **fsdp_params):
            model, vocab = esm.pretrained.load_model_and_alphabet_core(
                model_name, model_data, regression_data
            )
            self.batch_converter = vocab.get_batch_converter()
            model.eval()

            # Wrap each layer in FSDP separately
            for name, child in model.named_children():
                if name == "layers":
                    for layer_name, layer in child.named_children():
                        wrapped_layer = wrap(layer)
                        setattr(child, layer_name, wrapped_layer)
            self.model = wrap(model)
    def predict_embedding(self,sequence,name=None):
        name = 'Sequence' if name is None else name
        print(len(sequence))
        batch_labels, batch_strs, batch_tokens = self.batch_converter([(name,sequence)])
        batch_tokens = batch_tokens.cuda()
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[num_layers], return_contacts=True)
        token_representations = results["representations"][num_layers][0, 1:-1]
        return token_representations.cpu().numpy().squeeze().astype(np.float32)

    def batch_predict_embedding(self,processing_list):
        """
        processing_list: list = [[(name,sequence)],[...],[(name,sequence)]]
        """
        proceesed_results = {}
        for data in processing_list:
            batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
            batch_tokens = batch_tokens.cuda()
            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[3num_layers], return_contacts=True)
            token_representations = results["representations"][num_layers][0, 1:-1].cpu().numpy().squeeze().astype(np.float32)
            proceesed_results[batch_labels] = token_representations
        return proceesed_results

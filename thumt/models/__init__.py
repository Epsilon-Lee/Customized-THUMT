# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import thumt.models.seq2seq
import thumt.models.rnnsearch
import thumt.models.transformer
import thumt.models.c2f_transformer
import thumt.models.c2f_sv_transformer
import thumt.models.prune_l0_transformer
import thumt.models.prune_l01_transformer
import thumt.models.layer_fused_transformer
import thumt.models.c2f_4l_transformer
import thumt.models.c2f_4l_sc_transformer


def get_model(name):
    name = name.lower()

    if name == "rnnsearch":
        return thumt.models.rnnsearch.RNNsearch
    elif name == "seq2seq":
        return thumt.models.seq2seq.Seq2Seq
    elif name == "transformer":
        print("Model type: TRANSFORMER")
        return thumt.models.transformer.Transformer
    elif name == "c2f-transformer":
        print("Model type: C2F-TRANSFORMER")
        return thumt.models.c2f_transformer.Transformer
    elif name == "c2f-sv-transformer":
        print("Model type: C2F-SV-TRANSFORMER")
        return thumt.models.c2f_sv_transformer.Transformer
    elif name == "prune-l0-transformer":
        print("Model type: PRUNE-L0-TRANSFORMER")
        return thumt.models.prune_l0_transformer.Transformer
    elif name == "prune-l1-transformer":
        print("Model type: PRUNE-L1-TRANSFORMER")
        return thumt.models.prune_l01_transformer.Transformer
    elif name == "layer-fused-transformer":
        print("Model type: LAYER-FUSED-TRANSFORMER")
        return thumt.models.layer_fused_transformer.Transformer
    elif name == "c2f-4l-transformer":
        print("Model type: C2F-4L-TRANSFORMER")
        return thumt.models.c2f_4l_transformer.Transformer
    elif name == "c2f-4l-sc-transformer":
        print("Model type: C2F-4L-SC-TRANSFORMER")
        return thumt.models.c2f_4l_sc_transformer.Transformer
    else:
        raise LookupError("Unknown model %s" % name)

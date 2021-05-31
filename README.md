# Transglot: Learning Local Parts of 3D Structure from Language

![representative](doc/images/teaser.png)


## Introduction
Transglot is a extended version of Shapeglot. It can extracts local regions corresponding given a sentence with a cross-attention mechanism.
This work explores if our model can find a local part of the 3D structure from natural language.
From attention maps, we can see that Transglot focuses on a local part related with a given sentence.
It is similar with the part segmentation of shapes even Transglot isn't provided any part labels or part segmentation supervision.

## Requirements
Install with `pip install -e .`. This implementation is based on some libraries, pytorch-lightning and hydra.

CUDA: 11.2
## Train
1. For preprocessing the data, run `transglot/notebooks/prepare_chairs_in_context_data.ipynb`.
2. Train Transglot
```
# You can select some options. 
# pn: PointNet, pn2: PointNet++, pt: Point Transformer, pct: Point Cloud Transformer
python train.py embedding_dim=100 hidden_dim=256 attn_layers=1 num_heads=1 \
pc_encoder_type=["pn", "pn2", "pt", "pct"] \
batch_size=96 epochs=35 lr=1e-3 weight_decay=5e-3 \
```

## Test
`python test.py`

I attached a checkpoint of trained model. It runs the uploaded model.
If you want to test your own trained model, you should edit a sub_dir path in test.py.
For more details, please refer to comments in that.

trained model: embedding_dim=100, hidden_dim=256, attn_layers=1, num_heads=1, pc_encoder_type=pn


## License
This provided code is licensed under the terms of the MIT license (see LICENSE for details).

Repository for training &amp; serving a classification model for moth species.

# Installation

- poetry install
- wandb login

# Prepare data

Run the following command:

```console
python moths/scripts/valid_data_paths_symbolic.py test_data/ test_data/source_a test_data/source_b
```

The result will look as follows:

```console
.
├── image_folder
│   ├── Species A
│   ├── Species B
│   └── Species C
├── source_a
│   ├── Species A
│   └── Species B
└── source_b
    ├── Species B
    └── Species C
```

The script serves two purposes:

1. it combines multiple data sources into a single source, and
2. it hides corrupted images.

The files are written as symbolic links, and thus take (almost) no additional space.

# Configure and run

All configuration can be found in `config`. The used config file is set via the
environment variable `MOTHS_CONFIG` (by default set to `default`).

Train by running:

```console
python moths/scripts/train.py
MOTHS_CONFIG=optim python moths/scripts/train.py -m  # specify multi run when using optim.yaml
```

There are 3 predefined configuration files:

1. default: run the default training on the 'Vlinderstichting' machine
2. dev: run a development (cpu & single process) run locally on the `test_data`
3. optim: a multi run hyperparameter optimization

Configuration and hyperparameter optimization is set up
with [hydra](https://hydra.cc/docs/intro/). This allows to easily override certain
hyperparameters via the console:

```console
python moths/scripts/train.py data.batch_size=4 trainer.loggers.wandb.project=dev
```

The training loop is set up
with [pytorch-lightning](https://pytorch-lightning.readthedocs.io/en/latest/). Some
classes are directly instantiated from the config. This allows for very expressive
configuration files. They are recognizable by the hydra reserved key `_target_` for the
class reference.

If wandb is set up and the `WandbLogger` is specified in the `training.loggers` config,
then all metrics are automatically logged to [wandb](https://wandb.ai/butterflies).

# Data transformations

Data transformations (including augmentation) are configured entirely via the
instantiated classes:

```yaml
train_transforms:
  - _target_: torchvision.transforms.Resize
    size: 224
  - _target_: torchvision.transforms.ToTensor
```

They are instantiated and composed into a final transformation like this:

```python
Compose([instantiate(c) for c in config.data.train_transforms])
```

For more control you can build your own transformations and instantiate them directly.

To override or hyperparameter optimize a value in a list, you have to specify the
index (0 based) in the name:

```console
python moths/scripts/train.py data.train_transforms.0.size=64
```

# Multi label

In addition to the species the framework is set up to classify the genus, group, and
family as well. This information is provided via the `data/family.csv` file. Classes
that are missing from this file are bundled in the `other` class.

There are a couple configurations specifically important for this multi label
classification:

1. `data.min_samples`: specifies how many samples a class (species) should have, the
   classes that do not meet this lower bound are bundled into the `other` class
2. `lit.loss_weights`: a list of 4 values used to weigh the loss of each label

# Unfreezing

By default, the backbone is completely frozen and only the classification layer trained.
This can be changed via 3 hyperparameters:

1. `lit.unfreeze_backbone_epoch_start`: at what epoch start unfreezing
2. `lit.unfreeze_backbone_epoch_duration`: how many epochs it takes to unfreeze
3. `lit.unfreeze_backbone_percentage`: how many layers are unfrozen at the end

# Predict & Inference

The term predict is used to run the model on one of the data module data loaders (data
for which the label is known) outside the context of the training loop.

The term inference will be used to indicate running the model on data for which no label
is known in a more production setting.

## todo: create better evaluation metric and tables (confusion matrices)

## save the production label hierarchy file:

## todo: save this together with the checkpoint file

```python
import pickle
from pathlib import Path
from moths.label_hierarchy import label_hierarchy_from_file

label_hierarchy_path = Path("/home/vlinderstichting/Data/moths/data/family.csv")
data_source_path = Path("/home/vlinderstichting/Data/moths/artifacts/image_folder")
label_hierarchy = label_hierarchy_from_file(label_hierarchy_path, data_source_path, 50)

with Path("/tmp/label_hierarchy.pkl").open('wb') as f:
    pickle.dump(label_hierarchy_path, f, protocol=4)
```
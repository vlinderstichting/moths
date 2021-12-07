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
python moths/scripts/train.py -m  # specify multi run when using optim.yaml
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
then all metrics are automatically logged to wandb.

# Data transformations

Data transformations (including augmentation) are configured entirely via the
instantiated classes:

```yaml
train_transforms:
  - _target_: torchvision.transforms.Resize
    size: 224
  - _target_: torchvision.transforms.ToTensor
```

```python
Compose([instantiate(c) for c in config.data.train_transforms])
```

More a more control you can build your own transformations and instantiate them
via `moths.my_transformation.MyTransformation`.

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
3. `data.weighted_sampling`: specifies how to sample during training
    1. `NONE`: no weighted sampling, every sample is presented once each epoch
    2. `FREQ`: weights correct for the class frequency to create a perfectly balanced
       dataset
    3. `ROOT`: weights are the inverse of the root of the class frequency to create a
       'freq-lite' sampling

# Selected features

Some non-obvious configuration options are:

- gradual unfreezing: By default the backbone is completely frozen and only the
  classification layer trained. This can be changed via 3 hyperparameters:
    1. `lit.unfreeze_backbone_epoch_start`: at what epoch start unfreezing
    2. `lit.unfreeze_backbone_epoch_duration`: how many epochs it takes to unfreeze
    3. `lit.unfreeze_backbone_percentage`: how many are unfrozen at the end

- todo: mixup, cutmix

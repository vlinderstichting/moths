Repository for training &amp; serving a classification model for moth species.

# Installation

- poetry install
- wandb login

# Prepare data

Run the following command:

```bash
python scripts/valid_data_paths_symbolic.py test_data/ test_data/source_a test_data/source_b
```

The result will look as follows:

```bash
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

```bash
python moths/cmd/train.py
MOTHS_CONFIG=optim python moths/cmd/train.py -m  # specify multi run when using optim.yaml
```

There are 3 predefined configuration files:

1. default: run the default training on the 'Vlinderstichting' machine
2. dev: run a development (cpu & single process) run locally on the `test_data`
3. optim: a multi run hyperparameter optimization

Configuration and hyperparameter optimization is set up
with [hydra](https://hydra.cc/docs/intro/). This allows to easily override certain
hyperparameters via the console:

```bash
python moths/cmd/train.py data.batch_size=4 trainer.loggers.wandb.project=dev
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

```bash
python moths/cmd/train.py data.train_transforms.0.size=64
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
is known in a production setting.

## Prediction

A prediction must be run with 2 extra parameters; `training_output_path`
and `lit.prediction_output_path`, like this:

```bash
python moths/cmd/predict.py training_output_path=/training/output lit.prediction_output_path=/prediction/output
```

It expects `label_hierarchy.pkl` and `best.ckpt` in `/training/output`. These are
created by training, thus can be pointed directly at the output folder of a training
run.

A folder is created with the following contents:

```bash
.
├── best.ckpt
├── label_hierarchy.pkl
├── images/
└── arrays/
```

The `images` folder contains images organized by their *prediction* label. Images will
be named as `{correct or wrong}-{random id}-{correct label if wrong}`
eg, `wrong-40cc1705554542e1a8a38e85e807382c-Pammene ochsenheimeriana.jpg`.

The `arrays` folder contains numpy arrays with the truth (int), prediction (int), and
the logits (list of floats), per sample.

Extra information on the prediction can be generated with:

```bash
python moths/scripts/evaluate_predictions.py /prediction/output
```

It will print some general statistics on the performance and create an additional
folder `cms` that contains many confusion matrices.

## Inference

Not implemented.

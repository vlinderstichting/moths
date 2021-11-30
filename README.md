Repository for training &amp; serving a classification model for moth species.

# Installation

- poetry install
- configure wandb

# Prepare data

Run the following command:

```console
python moths/scripts/valid_data_paths_symbolic.py test_data/ test_data/source_a test_data/source_b
```

The script serves two purposes:

1. it will combine multiple data sources into a single source, and
2. it hides corrupted images.

The files are written as symbolic links, and thus do not take additional space.

# Configure and run

All configuration can be found in `moths/config`.

Train by running:

```console
python moths/scripts/train.py
```

# Development

Unfortunately, in the code we took the order as specified in the family.csv file, ie.
species, group, family, genus. But the hierarchy goes as group > family > genus >
species, where group is the largest and has the least classes, and species is the
smallest with the most classes.


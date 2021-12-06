Repository for training &amp; serving a classification model for moth species.

# Installation

- poetry install
- configure wandb

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
```

# Hyperparameter optimization



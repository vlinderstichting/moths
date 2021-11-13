Repository for training &amp; serving a classification model for moth species.

# Installation

- poetry venv
- configure wandb

# Prepare data

There are two matters that complicate the loading and fetching of data:

1. multiple paths as image data folder sources
2. a label hierarchy of species (the class), group, family, and genus

Point (1) is self-explanatory. Point (2) refers to the fact that the image data folders
indicate what species an image belongs to, but we have an additional file that indicates
what genus the species belongs to, what family the genus belongs to, and what group the
family belongs to.

The PyTorch library does not facilitate having multiple paths as image data folder
sources.

Two things need to happen as data preparation:

1. generate a list valid (ie. non-corrupt) image files to use as cache
2. generate a label hierarchy file to make sure the same classes from different sources
   get assigned the same label index

```console
python scripts/label_hierarchy.py test_output/ data/family.csv test_data/source_a test_data/source_b
python scripts/valid_data_paths.py test_output/ test_data/source_a test_data/source_b
```

This will generate 3 files:

      test_output/
      ├── invalid-paths.txt
      ├── valid-paths.txt
      └── label-hierarchy.csv

We need these files later for training and testing.

Note: the PyTorch `is_valid_file` method takes only the file name as parameter. Since it
by default only checks the extension, this is enough. The caching system assumes (
abuses) that file names are unique over all sources. For this project this is true.

It can be checked by running (taken
from [StackOverflow](https://stackoverflow.com/a/45971199)):

```console
find /path/to/data -type f -printf '%p/ %f\n' | sort -k2 | uniq -f1 --all-repeated=separate
```

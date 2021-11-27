from pathlib import Path
from typing import cast

from hydra import compose, initialize_config_dir
from typer.testing import CliRunner

from moths.data_module import DataModule
from moths.scripts.label_hierarchy import LABEL_HIERARCHY_FILE_NAME, label_hierarchy_app
from moths.scripts.train import Config, train
from moths.scripts.valid_data_paths import (
    INVALID_PATH_FILE_NAME,
    VALID_PATH_FILE_NAME,
    valid_path_app,
)

# todo: make independent from test file location
PROJECT_ROOT_PATH = Path(__file__).parent.parent
FAMILY_PATH = str(PROJECT_ROOT_PATH / "data" / "family.csv")
TEST_SOURCE_A_PATH = str(PROJECT_ROOT_PATH / "test_data" / "source_a")
TEST_SOURCE_B_PATH = str(PROJECT_ROOT_PATH / "test_data" / "source_b")
HYDRA_CONFIG_PATH = str(PROJECT_ROOT_PATH / "moths" / "config")

typer_runner = CliRunner()


def test_data_preparation(tmp_path):
    label_hierarchy_app_result = typer_runner.invoke(
        label_hierarchy_app,
        [str(tmp_path), FAMILY_PATH, TEST_SOURCE_A_PATH, TEST_SOURCE_B_PATH],
    )
    tmp_label_hierarchy_path = tmp_path / LABEL_HIERARCHY_FILE_NAME
    assert label_hierarchy_app_result.exit_code == 0
    assert tmp_label_hierarchy_path.exists()

    hierarchy = hierarchy_from_path(tmp_label_hierarchy_path)
    assert len(hierarchy.classes) == 4
    assert len(hierarchy.groups) == 1
    assert len(hierarchy.families) == 1
    assert len(hierarchy.genuses) == 3

    valid_path_app_result = typer_runner.invoke(
        valid_path_app,
        [str(tmp_path), TEST_SOURCE_A_PATH, TEST_SOURCE_B_PATH],
    )
    tmp_valid_data_path = tmp_path / VALID_PATH_FILE_NAME
    tmp_invalid_data_path = tmp_path / INVALID_PATH_FILE_NAME
    assert valid_path_app_result.exit_code == 0
    assert tmp_valid_data_path.exists()
    assert tmp_invalid_data_path.exists()
    with tmp_valid_data_path.open("r") as fp:
        assert len(fp.readlines()) == 18
    with tmp_invalid_data_path.open("r") as fp:
        assert len(fp.readlines()) == 0

    initialize_config_dir(config_dir=HYDRA_CONFIG_PATH)
    config = compose(
        config_name="config",
        overrides=[
            f"data.valid_path_file={str(tmp_valid_data_path)}",
            f"data.label_hierarchy_file={str(tmp_label_hierarchy_path)}",
        ],
    )
    data_module = DataModule(config.data)
    data_module.setup()

    assert len(data_module.test_dataset) == 5
    assert len(data_module.val_dataset) == 4
    assert len(data_module.train_dataset) == 9


def test_train():
    initialize_config_dir(config_dir=HYDRA_CONFIG_PATH)
    config = compose(config_name="config")
    train(cast(Config, config))

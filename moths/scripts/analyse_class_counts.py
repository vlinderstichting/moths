import logging

import hydra
from hydra.core.config_store import ConfigStore
from pytorch_lightning import seed_everything

from moths.config import resolve_config_path
from moths.label_hierarchy import label_hierarchy_from_file
from moths.scripts.train import CONFIG_NAME, Config

log = logging.getLogger("MOTHS")


cs = ConfigStore.instance()
cs.store(name="code_config", node=Config)


@hydra.main(config_path="../../config", config_name=CONFIG_NAME)
def analyse(config: Config) -> None:
    seed_everything(config.seed, workers=True)

    label_hierarchy_path = resolve_config_path(config.data.label_hierarchy_file)
    data_source_path = resolve_config_path(config.data.data_path)

    label_hierarchy = label_hierarchy_from_file(
        label_hierarchy_path, data_source_path, config.min_samples
    )

    log.info(
        f"Final class count: "
        f"{len(label_hierarchy.classes) - 1} "
        f"{len(label_hierarchy.groups) - 1} "
        f"{len(label_hierarchy.families) - 1} "
        f"{len(label_hierarchy.genuses) - 1}."
    )


if __name__ == "__main__":
    analyse()

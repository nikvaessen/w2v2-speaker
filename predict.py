################################################################################
#
# This run script encapsulates making predictions with a particular network
# on data without labels.
#
# Author(s): Nik Vaessen
################################################################################

import hydra

from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from src.hydra_resolvers import (
    division_resolver,
    integer_division_resolver,
    random_uuid,
)

################################################################################
# set custom resolvers

OmegaConf.register_new_resolver("divide", division_resolver)
OmegaConf.register_new_resolver("idivide", integer_division_resolver)
OmegaConf.register_new_resolver("random_uuid", random_uuid)

################################################################################
# wrap around main hydra script


@hydra.main(config_path="config", config_name="predict")
def run(cfg: DictConfig):
    # we import here such that tab-completion in bash
    # does not need to import everything (which slows it down
    # significantly)
    from src.main import run_predictions

    return run_predictions(cfg)


################################################################################
# execute hydra application

if __name__ == "__main__":
    load_dotenv()
    run()


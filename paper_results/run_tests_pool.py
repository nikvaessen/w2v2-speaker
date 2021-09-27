import pathlib

from collections import defaultdict

root_folder = "/home/nvaessen/data/transfer/paper_n4/aam/ablation/"
experiment_name = "speaker_wav2vec2_aam"
tag_prefix = "ablation"
test_run = False
test_sets_to_use = [
    # "voxceleb2",
    "voxceleb2_test_everyone",
    # "voxceleb2_test_hard"
]

postfix_map = {
    "voxceleb2": "o",
    "voxceleb2_test_hard": "h",
    "voxceleb2_test_everyone": "e",
}


path_dict = defaultdict(set)
for ckpt in sorted(pathlib.Path(root_folder).glob("*.ckpt")):
    first_underscore = ckpt.stem.find("_")
    first_dot = ckpt.stem.find(".")

    ablation_name = ckpt.stem[first_underscore+1:first_dot]

    path_dict[ablation_name].add(ckpt.absolute())

assert len(path_dict) == 10
for key, v in path_dict.items():
    assert len(v) == 3

for test_set in test_sets_to_use:
    for ablation_name, ckpt_set in path_dict.items():
        for ckpt in ckpt_set:
            command_template = (
                f"python run.py -m +experiment={experiment_name} "
                f"data/module={test_set} "
                f"fit_model=False "
                f"network.stat_pooling_type=first+cls "
                f"tag={ablation_name}_eval_{postfix_map[test_set]} "
                f"load_network_from_checkpoint={ckpt} "
                f"network.explicit_num_speakers=5994 "
                f"hydra/launcher=slurm "
            )

            print(f"{command_template} & ")

            if test_run:
                exit()

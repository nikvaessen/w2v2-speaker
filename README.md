# Fine-tuning wav2vec2 for speaker recognition

This is the code used to run the experiments in https://arxiv.org/abs/2109.15053. Detailed logs of each training run can be found here:

* x-vector: https://www.comet.ml/nik-vaessen-ru-nl/xvector-sv-ce?shareable=eVKNVMJUEV0WZd5FxXKeOFl2B
* ecapa-tdnn: https://www.comet.ml/nik-vaessen-ru-nl/ecapa-tdnn?shareable=Cis9Pp3RMwgirkJQp8wvHWlfs
* w2v2-ce: https://www.comet.ml/nik-vaessen-ru-nl/wav2vec2-sv-ce?shareable=8B3IJfHK4T1UF1jG0wasPHJb5
* w2v2-aam: https://www.comet.ml/nik-vaessen-ru-nl/wav2vec2-sv-aam?shareable=wGswdkZvqET0Iy7lmC2ORpMhO
* w2v2-bce: https://www.comet.ml/nik-vaessen-ru-nl/wav2vec2-sv-bce?shareable=SPGxZGBNAmFaxVsmsDBisFfrU

## Installing dependencies 

If poetry is not installed, see https://python-poetry.org/docs/. We also
expect at least python 3.8 on the system. If this is not the case, look into
https://github.com/pyenv/pyenv for an easy tool to install a specific
python version on your system. 

The python dependencies can be installed (in a project-specific virtual environment) by:

```bash
$ poetry shell  # enter project-specific virtual environment
```

From now on, every command which should be run under the virtual environment
(which looks like `(wav2vec-speaker-identification-<ID>-py<VERSION>) $`)
which is shortened to `(xxx) $ `.

Then install all required python packages:

```bash
(xxx) $ pip install -U pip
(xxx) $ poetry update # install dependencies 
```

Because PyTorch is currently serving the packages on PiPY incorrectly,
we need to use pip to install the specific PyTorch versions we need.

```bash
(xxx) $ pip install -r requirements/requirements_cuda101.txt # if CUDA 10.1
(xxx) $ pip install -r requirements/requirements_cuda110.txt # if CUDA 11.0
```

Make sure to modify/create a requirements file for your operating system and
CUDA version. 

Finally, install the local package in the virtual environment by running

```bash
(xxx) $ poetry install
```

## Setting up the environment

Copy the example environment variables:

```bash
$ cp .env.example .env 
```

You can then fill in `.env` accordingly. 

### Downloading and using voxceleb1 and 2

I've experienced that the download links for voxceleb1/2 can be unstable.
I recommend manually downloading the dataset from the google drive link displayed 
on https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html.

You should end up 4 zip files, which should be placed in `$DATA_FOLDER/voxceleb_archives`. 
1. `vox1_dev_wav.zip` 
2. `vox1_test_wav.zip`
3. `vox2_dev_aac.zip`
4. `vox2_test_aac.zip`

You should also download the meta files of voxceleb. You can use 
`preparation_scripts/download_pretrained_models.sh` to download them
to the expected location `$DATA_FOLDER/voxceleb_meta`.

#### Converting voxceleb2 data from .m4a to .wav

This requires ffmpeg to be installed on the machine. Check with `ffmpeg -version`.
Assuming the voxceleb2 data is placed at `$DATA_FOLDER/voxceleb_archives/vox2_dev_aac.zip`
and `$DATA_FOLDER/voxceleb_archives/vox2_test_aac.zip`, run the following commands, starting
from the root project directory. 

```bash
source .env

PDIR=$PWD # folder where this README is located
D=$DATA_FOLDER # location of data - should be set in .env file 
WORKERS=$(nproc --all) # number of CPUs available 

# extract voxceleb 2 data
cd $D
mkdir -p convert_tmp/train convert_tmp/test

unzip voxceleb_archives/vox2_dev_aac.zip -d convert_tmp/train
unzip voxceleb_archives/vox2_test_aac.zip -d convert_tmp/test

# run the conversion script
cd $PDIR
poetry run python preparation_scripts/voxceleb2_convert_to_wav.py $D/convert_tmp --num_workers $WORKERS

# rezip the converted data
cd $D/convert_tmp/train
zip $D/voxceleb_archives/vox2_dev_wav.zip wav -r

cd $D/convert_tmp/test
zip $D/voxceleb_archives/vox2_test_wav.zip wav -r

# delete the unzipped .m4a files
cd $D
rm -r convert_tmp
```

Note that this process can take a few hours on a fast machine and day(s) on a single (slow) cpu.
Make sure to save the `vox2_dev_wav.zip` and `vox2_test_wav.zip` files somewhere secure, so you don't have redo
this process :).

### Downloading pre-trained models.

You can run `./preparation_scripts/download_pretrained_models.sh` to download the pre-trained models of wav2vec2 to the required `$DATA_DIRECTORY/pretrained_models` directory.

## Running the experiments

Below we show all the commands for training the specified network. They should reproduce the results in the paper.
Note that we used a SLURM GPU cluster and each command therefore includes `hydra/launcher=slurm`. 
If you want to reproduce these locally these lines need to be removed.

### wav2vec2-sv-ce

#### auto_lr_find

```
python run.py +experiment=speaker_wav2vec2_ce \
tune_model=True data/module=voxceleb1 \
trainer.auto_lr_find=auto_lr_find tune_iterations=5000
```

5k iters, visually around 1e-4

#### grid search

grid = 1e-5, 5e-5, 9e-5, 1e-4, 2e-4, 5e-4, 1e-3

```
python run.py -m +experiment=speaker_wav2vec2_ce \
data.dataloader.train_batch_size=66 \
optim.algo.lr=1e-5,5e-5,9e-5,1e-4,2e-4,5e-4,1e-3 \
hydra/launcher=slurm hydra.launcher.exclude=cn104 hydra.launcher.array_parallelism=7
```

#### best performance n=3
```
python run.py -m +experiment=speaker_wav2vec2_ce \
data.dataloader.train_batch_size=66 optim.algo.lr=9e-5 \
seed=26160,79927,90537 \
hydra/launcher=slurm hydra.launcher.exclude=cn104 hydra.launcher.array_parallelism=3
```

#### best pooling n=3

```
python run.py -m +experiment=speaker_wav2vec2_ce \
data.dataloader.train_batch_size=66 optim.algo.lr=9e-5 \
seed=168621,597558,440108 \
network.stat_pooling_type=mean,mean+std,attentive,quantile,first,first+cls,last,middle,random,max \
hydra/launcher=slurm hydra.launcher.exclude=cn104 hydra.launcher.array_parallelism=4
```

### wav2vec2-sv-aam

aam with m=0.2 and s=30

#### auto_lr_find
```
python run.py +experiment=speaker_wav2vec2_ce \
tune_model=True data/module=voxceleb1 \
trainer.auto_lr_find=auto_lr_find tune_iterations=5000 \
optim/loss=aam_softmax
```

#### grid search

```
python run.py -m +experiment=speaker_wav2vec2_aam \
data.dataloader.train_batch_size=66 \
optim.algo.lr=1e-5,5e-5,9e-5,1e-4,2e-4,5e-4,1e-3 \
hydra/launcher=slurm hydra.launcher.exclude=cn104 hydra.launcher.array_parallelism=7
```
same grid 

#### best performance n=3
```
python run.py -m +experiment=speaker_wav2vec2_aam \
data.dataloader.train_batch_size=66 optim.algo.lr=0.00005 \
seed=29587,14352,70814 \
hydra/launcher=slurm hydra.launcher.exclude=cn104 hydra.launcher.array_parallelism=3
```

#### best pooling n=3

```
python run.py -m +experiment=speaker_wav2vec2_aam \
data.dataloader.train_batch_size=66 optim.algo.lr=0.00005 \
seed=392401,39265,62634  \
network.stat_pooling_type=mean,mean+std,attentive,quantile,first,first+cls,last,middle,random,max \
hydra/launcher=slurm hydra.launcher.exclude=cn104 hydra.launcher.array_parallelism=4
```

### wav2vec2-sv-bce

#### auto_lr_find
```
python run.py +experiment=speaker_wav2vec2_pairs \
tune_model=True data/module=voxceleb1_pairs \
trainer.auto_lr_find=auto_lr_find tune_iterations=5000
```

#### grid search

5e-6,7e6,9e-6,1e-5,2e-5,3e-5,4e-5,1e-4
```
python run.py -m +experiment=speaker_wav2vec2_pairs \
optim.algo.lr=5e-6,7e-6,9e-6,1e-5,2e-5,3e-5,4e-5,1e-4 \
data.dataloader.train_batch_size=32 \
hydra/launcher=slurm hydra.launcher.exclude=cn104 hydra.launcher.array_parallelism=8
```

#### best performance n=4

```
python run.py -m +experiment=speaker_wav2vec2_pairs \
optim.algo.lr=0.00003 data.dataloader.train_batch_size=32 \
seed=154233,979426,971817,931201 \
hydra/launcher=slurm hydra.launcher.exclude=cn104 hydra.launcher.array_parallelism=4 
```

### xvector

#### auto_lr_find

```
python run.py +experiment=speaker_xvector \
tune_model=True data/module=voxceleb1 \
trainer.auto_lr_find=auto_lr_find tune_iterations=5000
```

#### grid search
1e-5,6e-5,1e-4,2e-4,3e-4,4e-4,8e-4,1e-3

```
python run.py -m +experiment=speaker_xvector \
optim.algo.lr=1e-5,6e-5,1e-4,2e-4,3e-4,4e-4,8e-4,1e-3 \
data.dataloader.train_batch_size=66 \
hydra/launcher=slurm hydra.launcher.exclude=cn105 hydra.launcher.array_parallelism=8
```

#### best performance n=3
```
python run.py -m +experiment=speaker_xvector \
optim.algo.lr=0.0004 trainer.max_steps=100_000 \
data.dataloader.train_batch_size=66 \
seed=82713,479728,979292 \
hydra/launcher=slurm hydra.launcher.exclude=cn105 hydra.launcher.array_parallelism=6 \
```

### ecapa-tdnn

#### auto_lr_find

```
python run.py +experiment=speaker_ecapa_tdnn \
tune_model=True data/module=voxceleb1 \
trainer.auto_lr_find=auto_lr_find tune_iterations=5000
```

#### grid search

5e-6,1e-5,5e-4,1e-4,5e-3,7e-4,9e-4,1e-3

```
python run.py -m +experiment=speaker_ecapa_tdnn \
optim.algo.lr=5e-6,1e-5,5e-4,1e-4,5e-3,7e-4,9e-4,1e-3 \
data.dataloader.train_batch_size=66 \
hydra/launcher=slurm hydra.launcher.exclude=cn105 hydra.launcher.array_parallelism=8
```

#### best performance n=3
```
python run.py -m +experiment=speaker_ecapa_tdnn \
optim.algo.lr=0.001 trainer.max_steps=100_000 \
data.dataloader.train_batch_size=66 \
seed=494671,196126,492116 \
hydra/launcher=slurm hydra.launcher.exclude=cn105 hydra.launcher.array_parallelism=6
```

### Ablation

#### baseline

```
python run.py -m +experiment=speaker_wav2vec2_aam \
data.dataloader.train_batch_size=66 optim.algo.lr=0.00005 \
seed=392401,39265,62634 network.stat_pooling_type=first+cls \
hydra/launcher=slurm hydra.launcher.array_parallelism=3
```

#### unfrozen feature extractor
```
python run.py -m +experiment=speaker_wav2vec2_aam \
data.dataloader.train_batch_size=66 optim.algo.lr=0.00005 \
seed=914305,386390,865459 network.stat_pooling_type=first+cls \
network.completely_freeze_feature_extractor=False tag=no_freeze \
hydra/launcher=slurm hydra.launcher.array_parallelism=3 hydra.launcher.exclude=cn104
```

#### no pre-trained weights

```
python run.py -m +experiment=speaker_wav2vec2_aam \
data.dataloader.train_batch_size=66 optim.algo.lr=0.00005 \
seed=517646,414321,137524 network.stat_pooling_type=first+cls \
network.completely_freeze_feature_extractor=False network.reset_weights=True tag=no_pretrain \
hydra/launcher=slurm hydra.launcher.array_parallelism=3 hydra.launcher.exclude=cn104
```

#### no layerdrop
```
python run.py -m +experiment=speaker_wav2vec2_aam \
data.dataloader.train_batch_size=66 optim.algo.lr=0.00005 \
seed=15249,728106,821754 network.stat_pooling_type=first+cls \
network.layerdrop=0.0 tag=no_layer \
hydra/launcher=slurm hydra.launcher.array_parallelism=3
```

#### no dropout

```
python run.py -m +experiment=speaker_wav2vec2_aam \
data.dataloader.train_batch_size=66 optim.algo.lr=0.00005 \
seed=627687,883727,154405 network.stat_pooling_type=first+cls \
network.layerdrop=0.0 network.attention_dropout=0 \ 
network.feat_proj_dropout=0 network.hidden_dropout=0 tag=no_drop \
hydra/launcher=slurm hydra.launcher.array_parallelism=3 
```

#### no time masking
```
python run.py -m +experiment=speaker_wav2vec2_aam \
data.dataloader.train_batch_size=66 optim.algo.lr=0.00005 \
seed=602400,553540,419322 network.stat_pooling_type=first+cls \
network.layerdrop=0.0 network.attention_dropout=0 network.feat_proj_dropout=0 \
network.hidden_dropout=0 network.mask_time_prob=0 tag=no_mask \
hydra/launcher=slurm hydra.launcher.array_parallelism=3 
```


#### batch size 32
```
python run.py -m +experiment=speaker_wav2vec2_aam \
data.dataloader.train_batch_size=32 trainer.max_steps=200_000 \
optim.algo.lr=0.00005 network.stat_pooling_type=first+cls \
tag=bs_32 seed=308966,753370,519822 \
hydra/launcher=slurm hydra.launcher.array_parallelism=3 
```

#### batch size 128
```
python run.py -m +experiment=speaker_wav2vec2_aam \
data.dataloader.train_batch_size=128 trainer.max_steps=50_000 \
optim.algo.lr=0.00005 seed=54375,585956,637400 \
network.stat_pooling_type=first+cls tag=bs_128 \
hydra/launcher=slurm hydra.launcher.array_parallelism=3 hydra.launcher.exclude=cn104
```

#### constant lr=3e-6

```
python run.py -m +experiment=speaker_wav2vec2_aam \
data.dataloader.train_batch_size=66 optim.algo.lr=3e-6 \
seed=549686,190215,637679 network.stat_pooling_type=first+cls \
optim/schedule=constant tag=lr_low \
hydra/launcher=slurm hydra.launcher.array_parallelism=3 
```

#### constant lr=5e-5
```
python run.py -m +experiment=speaker_wav2vec2_aam \
data.dataloader.train_batch_size=66 optim.algo.lr=0.00005 \
seed=419703,980724,124995 network.stat_pooling_type=first+cls \
optim/schedule=constant tag=lr_same \
hydra/launcher=slurm hydra.launcher.array_parallelism=3  
```

#### tri_stage

```
python run.py -m +experiment=speaker_wav2vec2_aam \
data.dataloader.train_batch_size=66 optim.algo.lr=0.00005 \
seed=856797,952324,89841 network.stat_pooling_type=first+cls \
optim/schedule=tri_stage tag=lr_3stage \
optim.schedule.scheduler.lr_lambda.initial_lr=1e-7 optim.schedule.scheduler.lr_lambda.final_lr=1e-7 \
hydra/launcher=slurm hydra.launcher.array_parallelism=3
```

#### exp decay
```
python run.py -m +experiment=speaker_wav2vec2_aam \
data.dataloader.train_batch_size=66 optim.algo.lr=0.00005 seed=962764,682423,707761 \
network.stat_pooling_type=first+cls optim/schedule=exp_decay tag=lr_exp_decay \
optim.schedule.scheduler.lr_lambda.final_lr=1e-7 \
hydra/launcher=slurm hydra.launcher.array_parallelism=3  
```

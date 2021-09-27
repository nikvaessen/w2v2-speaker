### set environment variables
source ../.env 2> /dev/null || source .env

### create folder to store models in
PRETRAINED="$DATA_FOLDER"/pretrained_models/wav2vec
mkdir -p "$PRETRAINED"

### download pretrained models

# wav2vec1 large
echo "wav2vec1 - https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_large.pt"
curl -C - https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_large.pt --output "$PRETRAINED"/wav2vec_large.pt

# wav2vec2 small - no ft
echo "# wav2vec2 small - no ft - https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt"
curl -C - https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt --output "$PRETRAINED"/wav2vec2_small_noft.pt

# wav2vec2 small - 10 minutes
echo "# wav2vec2 small - 10m ft - https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small_10m.pt"
curl -C - https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small_10m.pt --output "$PRETRAINED"/wav2vec2_small_ft10m.pt

# wav2vec2 small - 100 hours
echo "# wav2vec2 small - 100h ft - https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small_100h.pt"
curl -C - https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small_100h.pt --output "$PRETRAINED"/wav2vec2_small_ft100h.pt

# wav2vec2 small - 960h ft
echo "wav2vec2 small - 960h ft - https://dl.fbaipublicfiles.com/fairseq/wav2vec/libri960_big.pt"
curl -C - https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small_960h.pt --output "$PRETRAINED"/wav2vec2_small_ft960h.pt

# wav2vec2 large - no ft
echo "wav2vec2 large - no ft - https://dl.fbaipublicfiles.com/fairseq/wav2vec/libri960_big.pt"
curl -C - https://dl.fbaipublicfiles.com/fairseq/wav2vec/libri960_big.pt --output "$PRETRAINED"/wav2vec2_large_noft.pt

# wav2vec2 large - 10 minutes
echo "# wav2vec2 base - 10m ft - https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_big_10m.pt"
curl -C - https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_big_10m.pt --output "$PRETRAINED"/wav2vec2_large_10m.pt

# wav2vec2 large - 100 hours
echo "# wav2vec2 large - 100h ft - https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_big_100h.pt"
curl -C - https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_big_100h.pt --output "$PRETRAINED"/wav2vec2_large_100h.pt

# wav2vec2 large - 960 ft
echo "wav2vec2 large - 960h ft - https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_big_960h.pt"
curl -C - https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_big_960h.pt --output "$PRETRAINED"/wav2vec2_large_960h.pt
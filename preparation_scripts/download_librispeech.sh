set -e

### set environment variables
source ../.env 2> /dev/null || source .env

# default directory to save files in
DIR="$DATA_FOLDER"/librispeech
mkdir -p "$DIR"

## download files
curl -C - https://www.openslr.org/resources/12/dev-clean.tar.gz --output "$DIR"/dev-clean.tar.gz
curl -C - https://www.openslr.org/resources/12/dev-other.tar.gz --output "$DIR"/dev-other.tar.gz
curl -C - https://www.openslr.org/resources/12/test-clean.tar.gz --output "$DIR"/test-clean.tar.gz
curl -C - https://www.openslr.org/resources/12/test-other.tar.gz --output "$DIR"/test-other.tar.gz
curl -C - https://www.openslr.org/resources/12/train-clean-100.tar.gz --output "$DIR"/train-clean-100.tar.gz
curl -C - https://www.openslr.org/resources/12/train-clean-360.tar.gz --output "$DIR"/train-clean-360.tar.gz
curl -C - https://www.openslr.org/resources/12/train-other-500.tar.gz --output "$DIR"/train-other-500.tar.gz

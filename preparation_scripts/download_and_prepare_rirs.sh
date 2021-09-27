set -e

### set environment variables
source ../.env 2> /dev/null || source .env

# default directory to save files in
DIR="$DATA_FOLDER"
mkdir -p "$DIR"

## download files
curl -C - https://www.openslr.org/resources/28/rirs_noises.zip --output "$DIR"/rirs_noises.zip

# extract file and remove zip
cd "$DIR"
unzip rirs_noises.zip -d "$DIR"
rm rirs_noises.zip

# create tar for webdataset compatability
mkdir -p "$DIR"/rirs_shards/
tar --sort=name -cf rirs_shards/pointsource_noises.tar RIRS_NOISES/pointsource_noises
tar --sort=name -cf rirs_shards/real_rirs_isotropic_noises.tar RIRS_NOISES/real_rirs_isotropic_noises
tar --sort=name -cf rirs_shards/simulated_rirs.tar RIRS_NOISES/simulated_rirs

# remove extracted dir
rm -r "$DIR"/RIRS_NOISES
source .env

PDIR=$PWD # folder where this README is located
D=$DATA_FOLDER # location of data - should be set in .env file
WORKERS=$(nproc --all) # number of CPUs available

# extract voxceleb 2 data
cd "$D" || exit
mkdir -p convert_tmp/train convert_tmp/test

unzip voxceleb_archives/vox2_dev_aac.zip -d convert_tmp/train
unzip voxceleb_archives/vox2_test_aac.zip -d convert_tmp/test

# run the conversion script
cd "$PDIR" || exit
poetry run python preparation_scripts/voxceleb2_convert_to_wav.py "$D"/convert_tmp --num_workers "$WORKERS"

# rezip the converted data
cd "$D"/convert_tmp/train || exit
zip "$D"/voxceleb_archives/vox2_dev_wav.zip wav -r

cd "$D"/convert_tmp/test || exit
zip "$D"/voxceleb_archives/vox2_test_wav.zip wav -r

# delete the unzipped .m4a files
cd "$D" || exit
rm -r convert_tmp
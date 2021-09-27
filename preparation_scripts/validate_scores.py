################################################################################
#
# This file creates a CLI for validating a score text file given a
# text file with pairs (without gt labels). If validation is successfull a 
# zipfile will be created which can be submitted to voxceleb challenge on
# codalab.
#
# pair text file format:
# 'FILEa FILEb\n'
# ...
# 'FILEc FILEd\n'
# 
# score text file format:
# 'SCORE_FLOAT FILEa FILEb\n'
# ...
# 'SCORE_FLOAT FILEc FILEd\n'
#
# where SCORE_FLOAT is a string representing a float between 0 and 1.
# 
# Author(s): Nik Vaessen
################################################################################

import pathlib
import argparse
import tqdm
import zipfile

from typing import List, Tuple

################################################################################
# validation function

def _load_pair_file(file: pathlib.Path) -> List[Tuple[str, str]]:
    with file.open('r') as f:
        lines = f.readlines()

    loaded_list = []

    for l in lines:
        l = l.strip()

        assert l.count(" ") == 1
        
        split_line = l.split(" ")
        assert len(split_line) == 2

        key1, key2 = split_line
        loaded_list.append((key1, key2))
    
    return loaded_list

def _load_score_file(file: pathlib.Path) -> List[Tuple[float, str, str]]:
    with file.open('r') as f:
        lines = f.readlines()

    loaded_list = []
    
    for l in lines:
        l = l.strip()

        assert l.count(" ") == 2
        
        split_line = l.split(" ")
        assert len(split_line) == 3

        score, key1, key2 = split_line

        try:
            score = float(score)
        except:
            raise ValueError(f"could not convert {score} to float")
        
        assert isinstance(score, float)
        loaded_list.append((score, key1, key2))
    
    return loaded_list

def validate(pair_file: pathlib.Path, score_file: pathlib.Path):
    # load data in file
    pairs = _load_pair_file(pair_file)
    scores = _load_score_file(score_file)

    # ensure each float is between 0 and 1
    print("validate each score is valid")
    for score_tuple in tqdm.tqdm(scores):
        score = score_tuple[0]

        assert score <= 1
        assert score >= 0

    # ensure each pair is present
    print("validate each pair is present")
    for score_tuple in tqdm.tqdm(scores):
        pair_tuple = (score_tuple[1], score_tuple[2])

        assert pair_tuple in pairs
    


################################################################################
# creation of submission file.

SCORE_FILE_NAME = 'scores.txt'
ZIPFILE_NAME = 'submission.zip'

def create_submission(score_file: pathlib.Path):
    zipfile_path = score_file.parent / ZIPFILE_NAME

    with zipfile.ZipFile(str(zipfile_path), mode='w') as f:
        f.write(str(score_file), SCORE_FILE_NAME)

################################################################################
# entrypoint of CLI

def main():
    # set CLI arguments 
    parser = argparse.ArgumentParser()

    parser.add_argument("--score_file", required=True)
    parser.add_argument("--pair_file", required=True)

    # load arguments
    args = parser.parse_args()
    
    score_file = pathlib.Path(args.score_file)
    pair_file = pathlib.Path(args.pair_file)

    # validate score file
    # validate(pair_file, score_file)

    # create submission zipfile
    create_submission(score_file)


if __name__ == "__main__":
    main()
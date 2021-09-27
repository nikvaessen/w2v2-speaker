################################################################################
#
# Converts the voxceleb2 data from .m4a to .wav files by using FFMPEG.
#
# Author(s): Nik Vaessen
################################################################################

import argparse
import multiprocessing
import pathlib
import subprocess
import time

from tqdm import tqdm
from yaspin import yaspin

################################################################################
# methods for converting


def subprocess_convert_to_wav(infile: str, outfile: str):
    """
    Use a subprocess calling FFMPEG to convert a file to 16 KHz .wav file.

    Parameters
    ----------
    infile: path to file which needs to be converted
    outfile: path where converted file needs to be stored
    """
    subprocess.check_output(
        [
            "ffmpeg",
            "-y",
            "-i",
            infile,
            "-ac",
            "1",
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            outfile,
        ],
        stderr=subprocess.PIPE,
    )


def convert_to_wav(
    directory_path: pathlib.Path,
    num_workers: int = 1,
    delete_m4a_files: bool = True,
    fix_folder_structure: bool = True,
):
    """
    Convert all ".m4a" files in a specified directory to ".wav" by using FFMPEG.
    Also potentially fixes the name structure so that voxceleb1 and voxceleb2
    have the same directory structure.

    Parameters
    ----------
    directory_path: the directory to scan for .m4a files to convert
    num_workers: the number of threads to use (in order to speed up conversions)
    delete_m4a_files: whether to remove the converted .m4a files after conversion has completed.
    fix_folder_structure: whether to fix the directory names so be consistent with voxceleb1.
    """
    # rename /train/dev/aac to train/wav and test/aac to test/wav to be consistent
    # with voxceleb1
    if fix_folder_structure:
        train_dir_aac = directory_path / "train" / "dev" / "aac"
        test_dir_aac = directory_path / "test" / "aac"

        if train_dir_aac.exists():
            train_dir_aac.rename(directory_path / "train" / "wav")
            train_dir_aac.parent.rmdir()

        if test_dir_aac.exists():
            test_dir_aac.rename(directory_path / "test" / "wav")

    # find all files in the train and test subdirectories
    all_train_test_files = []

    with yaspin(text=f"recursively finding all files in {directory_path}"):
        all_train_test_files.extend(
            [f for f in directory_path.rglob("*") if f.is_file()]
        )

    # filter the files on their extension
    wav_files = [f for f in all_train_test_files if f.suffix == ".wav"]
    m4a_files = [f for f in all_train_test_files if f.suffix == ".m4a"]
    other_files = [
        f
        for f in all_train_test_files
        if f.suffix != ".wav" and f.suffix != ".m4a"
    ]

    # sanity check existence of other file extensions
    if len(wav_files) > 0:
        print(
            "WARNING: folder already contains wav files."
            " Is this intended? Continuing in 10 seconds..."
        )
        time.sleep(10)
    if len(other_files) > 0:
        print(
            "WARNING: folder contains out-of-dataset files: ",
            *other_files,
            sep="\n",
        )

    # use multiple workers to call FFMPEG and convert the .m4a files to .wav
    with tqdm(total=len(m4a_files)) as p, multiprocessing.Pool(
        processes=num_workers
    ) as workers:
        for m4a_file in sorted(m4a_files):
            wav_file = m4a_file.parent / (m4a_file.stem + ".wav")

            if wav_file.exists():
                p.update(1)
                continue

            def cb_success(_):
                p.update(1)

            def cb_error(e):
                print(e)

                if isinstance(e, subprocess.CalledProcessError):
                    print("stdout:\n", e.stdout.decode("utf-8"))

            workers.apply_async(
                subprocess_convert_to_wav,
                args=(str(m4a_file), str(wav_file)),
                callback=cb_success,
                error_callback=cb_error,
            )

        workers.close()
        workers.join()

    # optionally delete all .m4a files
    if delete_m4a_files:
        for f in m4a_files:
            if f.suffix == ".m4a":
                f.unlink()


################################################################################
# CLI

parser = argparse.ArgumentParser(
    description="Convert all .m4a files in a directory to 16 KHz .wav files"
)

parser.add_argument(
    "directory_path",
    type=pathlib.Path,
    help="directory containing the audio files to be converted",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=1,
    help="Number of threads to use for converting the dataset files.",
)
parser.add_argument(
    "--no_delete_m4a_files",
    action="store_true",
    default=False,
    help="do not delete the .m4a files after they have been converted",
)
parser.add_argument(
    "--no_fix_folder_structure",
    action="store_true",
    default=False,
    help="do not rename the train and test folder to `wav`. "
    "This is normally done for consistency with voxceleb1.",
)


################################################################################
# script execution

if __name__ == "__main__":
    args = parser.parse_args()

    convert_to_wav(
        directory_path=args.directory_path,
        num_workers=args.num_workers,
        delete_m4a_files=not args.no_delete_m4a_files,
        fix_folder_structure=not args.no_fix_folder_structure,
    )

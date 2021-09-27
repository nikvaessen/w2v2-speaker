################################################################################
#
# Lightning data module for librispeech (using webdataset shards)
#
# Author(s): Nik Vaessen
################################################################################

import pathlib
import json
import random
import subprocess

from typing import Optional, Union, List, Dict, Callable, Generator

import torch
import torchaudio
import yaspin
import torch as t
import webdataset as wds

from dataclasses import dataclass
from torch.utils.data import DataLoader

from src.config_util import CastingConfig
from src.data.common import (
    WebDataSetShardConfig,
    BatchDebugInfo,
    SpeechDataLoaderConfig,
)
from src.data.modules.speech.speech_data_module import SpeechLightningDataModule
from src.data.preprocess.base import Preprocessor
from src.data.modules.speech.training_batch_speech import (
    SpeechRecognitionDataSample,
    default_collate_fn,
    SpeechRecognitionDataBatch,
)
from src.data.util import load_raw_audio
from src.tokenizer.base import BaseTokenizer
from src.util import extract_archive, remove_directory


################################################################################
# config for lightning data module


@dataclass
class LibriSpeechLightningDataModuleConfig(CastingConfig):
    use_train_clean_100: bool
    use_train_clean_360: bool
    use_train_other_500: bool

    train_clean_100_path: pathlib.Path
    train_clean_360_path: pathlib.Path
    train_other_500_path: pathlib.Path

    dev_clean_path: pathlib.Path
    dev_other_path: pathlib.Path

    test_clean_path: pathlib.Path
    test_other_path: pathlib.Path

    shards_folder: pathlib.Path
    extraction_folder: pathlib.Path

    train_collate_fn: str
    val_collate_fn: str
    test_collate_fn: str

    add_side_info: bool
    limit_samples: int


################################################################################
# data module


class LibriSpeechLightningDataModule(SpeechLightningDataModule):
    def __init__(
        self,
        cfg: LibriSpeechLightningDataModuleConfig,
        shard_cfg: WebDataSetShardConfig,
        dl_cfg: SpeechDataLoaderConfig,
        tokenizer: BaseTokenizer,
        train_pipeline: List[Preprocessor],
        val_pipeline: List[Preprocessor],
        test_pipeline: List[Preprocessor],
    ):
        super().__init__()

        self.cfg = cfg
        self.shard_cfg = shard_cfg
        self.dl_cfg = dl_cfg
        self.train_pipeline = train_pipeline
        self.val_pipeline = val_pipeline
        self.test_pipeline = test_pipeline
        self._tokenizer = tokenizer

        # paths to files
        self.train_shards_folder = self.cfg.shards_folder / "train"
        self.dev_shards_folder = self.cfg.shards_folder / "val"
        self.test_shards_folder = self.cfg.shards_folder / "test"

        self.cfg.shards_folder.mkdir(exist_ok=True)

        # values set in self#setup()
        self._vocabulary: List[str] = None

        self.train_ds: torch.utils.data.Dataset = None

        self.val_ds_clean: torch.utils.data.Dataset = None
        self.val_ds_other: torch.utils.data.Dataset = None

        self.test_ds_clean: torch.utils.data.Dataset = None
        self.test_ds_other: torch.utils.data.Dataset = None

    def summary(self):
        print("librispeech is ready for use")

    def vocabulary(self) -> List[str]:
        if self._vocabulary is None:
            raise ValueError("vocabulary is accessible after setup() is called")

        return self._vocabulary

    def tokenizer(self) -> BaseTokenizer:
        return self._tokenizer

    def prepare_data(self) -> None:
        if self._is_prepared():
            return

        # define paths to temporary
        extraction_folder = self.cfg.extraction_folder

        tc100 = extraction_folder / "train_clean_100"
        tc360 = extraction_folder / "train_clean_360"
        to500 = extraction_folder / "train_other_500"

        dev_clean = extraction_folder / "dev_clean"
        dev_other = extraction_folder / "dev_other"

        test_clean = extraction_folder / "test_clean"
        test_other = extraction_folder / "test_other"

        # step 1): extract data
        extract_archive(self.cfg.train_clean_100_path, tc100)
        extract_archive(self.cfg.train_clean_360_path, tc360)
        extract_archive(self.cfg.train_other_500_path, to500)

        extract_archive(self.cfg.dev_clean_path, dev_clean)
        extract_archive(self.cfg.dev_other_path, dev_other)

        extract_archive(self.cfg.test_clean_path, test_clean)
        extract_archive(self.cfg.test_other_path, test_other)

        # step 2) determine and save vocabulary
        char_vocabulary = self._determine_char_vocabulary([tc100, tc360, to500])

        with self._get_vocabulary_file().open("w") as f:
            json.dump(
                {
                    "vocabulary": char_vocabulary,
                },
                f,
            )

        # step 3) write shards
        write_librispeech_shards(
            tc100 / "LibriSpeech",
            self.train_shards_folder,
            self.shard_cfg.use_gzip_compression,
            self.shard_cfg.samples_per_shard,
            shard_name_pattern="train_clean_100_%06d.tar",
        )
        write_librispeech_shards(
            tc360 / "LibriSpeech",
            self.train_shards_folder,
            self.shard_cfg.use_gzip_compression,
            self.shard_cfg.samples_per_shard,
            shard_name_pattern="train_clean_360_%06d.tar",
        )
        write_librispeech_shards(
            to500 / "LibriSpeech",
            self.train_shards_folder,
            self.shard_cfg.use_gzip_compression,
            self.shard_cfg.samples_per_shard,
            shard_name_pattern="train_other_500_%06d.tar",
        )

        write_librispeech_shards(
            dev_clean / "LibriSpeech",
            self.dev_shards_folder,
            self.shard_cfg.use_gzip_compression,
            self.shard_cfg.samples_per_shard,
            shard_name_pattern="dev_clean_%06d.tar",
        )
        write_librispeech_shards(
            dev_other / "LibriSpeech",
            self.dev_shards_folder,
            self.shard_cfg.use_gzip_compression,
            self.shard_cfg.samples_per_shard,
            shard_name_pattern="dev_other_%06d.tar",
        )

        write_librispeech_shards(
            test_clean / "LibriSpeech",
            self.test_shards_folder,
            self.shard_cfg.use_gzip_compression,
            self.shard_cfg.samples_per_shard,
            shard_name_pattern="test_clean_%06d.tar",
        )
        write_librispeech_shards(
            test_other / "LibriSpeech",
            self.test_shards_folder,
            self.shard_cfg.use_gzip_compression,
            self.shard_cfg.samples_per_shard,
            shard_name_pattern="test_other_%06d.tar",
        )

        # step 4) clean up
        remove_directory(extraction_folder)

        self._set_is_prepared()

    def _verify_tokenizer_matches_vocabulary(self):
        tokenizer_vocab = self._tokenizer.vocabulary_dictionary()
        for char in self.vocabulary():
            if char == " ":
                # space characters are always supported :)
                continue

            if char not in tokenizer_vocab:
                raise ValueError(
                    f"given tokenizer cannot handle char {char} in vocabulary"
                )

    @staticmethod
    def _determine_char_vocabulary(subset_directories: List[pathlib.Path]):
        transcription_file_paths = []

        for subset_dir in subset_directories:
            for f in subset_dir.rglob("*.trans.txt"):
                transcription_file_paths.append(f)

        vocab = set()

        for transcription_file in transcription_file_paths:
            with transcription_file.open("r") as f:
                lines = [
                    " ".join(line.strip().split(" ")[1:]) for line in f.readlines()
                ]

                for line in lines:
                    for char in line:
                        vocab.update(char)

        sorted_vocab = sorted(list(vocab))

        return sorted_vocab

    def _load_vocabulary(self):
        with self._get_vocabulary_file().open("r") as f:
            return json.load(f)["vocabulary"]

    def _get_vocabulary_file(self):
        return self.cfg.shards_folder / "vocabulary.json"

    def _get_shard_info_file(self):
        return self.cfg.shards_folder / ".info"

    def _is_prepared(self):
        return self._get_shard_info_file().exists()

    def _set_is_prepared(self):
        with self._get_shard_info_file().open("w") as f:
            json.dump(
                {
                    "samples_per_shard": self.shard_cfg.samples_per_shard,
                    "use_gzip_compression": self.shard_cfg.use_gzip_compression,
                },
                f,
            )

    def setup(self, stage: Optional[str] = None) -> None:
        # verify vocabulary
        self._vocabulary = self._load_vocabulary()
        self._verify_tokenizer_matches_vocabulary()

        # setup train dataset
        versions = []

        if self.cfg.use_train_clean_100:
            versions.append("clean_100")

        if self.cfg.use_train_clean_360:
            versions.append("clean_360")

        if self.cfg.use_train_other_500:
            versions.append("other_500")

        if len(versions) == 0:
            raise ValueError("unable to prepare any training data subset")

        self.train_ds = self._prepare_train_dataset(versions)

        # setup val datasets
        self.val_ds_clean = self._prepare_val_datasets(prepare_clean=True)
        self.val_ds_other = self._prepare_val_datasets(prepare_clean=False)

        # setup test datasets
        self.test_ds_clean = self._prepare_test_datasets(prepare_clean=True)
        self.test_ds_other = self._prepare_test_datasets(prepare_clean=False)

    @staticmethod
    def _find_shard_paths(
        folder: pathlib.Path, patterns: List[str], only_1_shard=False
    ):
        shards = []

        for p in patterns:
            shards.extend([str(f) for f in folder.glob(p) if f.is_file()])

        shards = sorted(shards)

        if len(shards) == 0:
            raise ValueError(
                f"unable to find any shards in {folder} matching patterns `{patterns}`"
            )

        if only_1_shard:
            return shards[0]
        else:
            return shards

    def _prepare_train_dataset(self, versions: List[str]):
        # version is one of 'clean_100', 'clean_360', 'other_500'
        for version in versions:
            if version not in ["clean_100", "clean_360", "other_500"]:
                raise ValueError(
                    f"version {version} not one of "
                    f"['clean_100', 'clean_360', 'other_500']"
                )

        # train dataloader
        train_ds: wds.Processor = (
            wds.WebDataset(
                self._find_shard_paths(
                    self.train_shards_folder,
                    patterns=[f"train_{version}_*.tar*" for version in versions],
                    only_1_shard=True if self.cfg.limit_samples > 0 else False,
                ),
                # due to the preprocessing pipes we cannot determine
                # what the actual length is
                length=float("inf"),
            )
            .shuffle(
                self.shard_cfg.shuffle_shards,
            )
            .decode("pil")
            .pipe(self._pipe_to_speech_recognition_data_sample())
        )

        for p in self.train_pipeline:
            train_ds = train_ds.pipe(self._pipe_preprocessors(p))

        train_ds = train_ds.pipe(
            DynamicSpeechBatchProcessor(
                max_samples_in_batch=self.dl_cfg.train_max_num_samples,
                max_queue_size=self.shard_cfg.queue_size,
                collate_fn=self._determine_collate_fn(self.cfg.train_collate_fn),
            )
        )

        if self.cfg.limit_samples > 0:
            train_ds = train_ds.slice(self.cfg.limit_samples)

        return train_ds

    def _prepare_val_datasets(self, prepare_clean: bool = True):
        # dataloader
        val_ds: wds.Processor = (
            wds.WebDataset(
                self._find_shard_paths(
                    self.dev_shards_folder,
                    patterns=[f"dev_{'clean' if prepare_clean else 'other'}_*.tar*"],
                    only_1_shard=True if self.cfg.limit_samples > 0 else False,
                ),
                # due to the preprocessing pipes we cannot determine
                # what the actual length is
                length=float("inf"),
            )
            .decode("pil")
            .pipe(self._pipe_to_speech_recognition_data_sample())
        )

        for p in self.val_pipeline:
            val_ds = val_ds.pipe(self._pipe_preprocessors(p))

        return val_ds

    def _prepare_test_datasets(self, prepare_clean: bool = True):
        # dataloader
        test_ds: wds.Processor = (
            wds.WebDataset(
                self._find_shard_paths(
                    self.test_shards_folder,
                    patterns=[f"test_{'clean' if prepare_clean else 'other'}_*.tar*"],
                    only_1_shard=True if self.cfg.limit_samples > 0 else False,
                ),
                # due to the preprocessing pipes we cannot determine
                # what the actual length is
                length=float("inf"),
            )
            .decode("pil")
            .pipe(self._pipe_to_speech_recognition_data_sample())
        )

        for p in self.test_pipeline:
            test_ds = test_ds.pipe(self._pipe_preprocessors(p))

        return test_ds

    def _pipe_to_speech_recognition_data_sample(self):
        def apply_pipeline(x: dict) -> SpeechRecognitionDataSample:
            # x is a dict with keys 'wav.pyd` and `meta.json`
            key: str = x["__key__"]
            audio_sample: torch.Tensor = x["wav.pyd"]
            transcription: str = x["meta.json"]["transcription"]

            # check for NaN value in audio sample
            if torch.any(torch.isnan(audio_sample)):
                raise ValueError(f"NaN value in audio sample of {key=}")

            # convert transcription into sequence of integers
            transcription_int_sequence = self._tokenizer.encode_string(transcription)

            if len(transcription_int_sequence.shape) == 0:
                # 1 letter transcriptions
                transcription_int_sequence = t.stack([transcription_int_sequence])

            return SpeechRecognitionDataSample(
                key=key,
                ground_truth=transcription_int_sequence,
                network_input=audio_sample,
                ground_truth_string=transcription,
                input_length=audio_sample.shape[1],
                ground_truth_sequence_length=transcription_int_sequence.shape[0],
                side_info=BatchDebugInfo(
                    original_tensor=audio_sample,
                    pipeline_progress=[],
                    meta=x["meta.json"],
                )
                if self.cfg.add_side_info
                else None,
            )

        def pipe_fn(data_iterator):
            for x in data_iterator:
                yield apply_pipeline(x)

        return pipe_fn

    @staticmethod
    def _pipe_preprocessors(preprocessor: Preprocessor):
        def pipe_fn(data_iter):
            for x in data_iter:
                sample = preprocessor.process(x)

                if isinstance(sample, list):
                    for y in sample:
                        yield y
                else:
                    yield sample

        return pipe_fn

    def train_dataloader(
        self,
    ) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return torch.utils.data.DataLoader(
            dataset=self.train_ds,
            batch_size=None,
            num_workers=self.dl_cfg.num_workers,
            pin_memory=self.dl_cfg.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        def wrap_dataset(dataset):
            return torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=self.dl_cfg.val_batch_size,
                collate_fn=self._determine_collate_fn(self.cfg.val_collate_fn),
                num_workers=self.dl_cfg.num_workers,
                pin_memory=self.dl_cfg.pin_memory,
                drop_last=False,
            )

        return [wrap_dataset(self.val_ds_clean), wrap_dataset(self.val_ds_other)]

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        def wrap_dataset(dataset):
            return torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=self.dl_cfg.test_batch_size,
                collate_fn=self._determine_collate_fn(self.cfg.test_collate_fn),
                num_workers=self.dl_cfg.num_workers,
                pin_memory=self.dl_cfg.pin_memory,
                drop_last=False,
            )

        return [wrap_dataset(self.test_ds_clean), wrap_dataset(self.test_ds_other)]

    @staticmethod
    def _determine_collate_fn(name: str):
        if name == "default":
            return default_collate_fn
        else:
            raise ValueError(f"cannot determine collate_fn {name}")


################################################################################
# custom preprocessor which creates dynamically-sized batches with a certain
# length of samples


class DynamicSpeechBatchProcessor:
    def __init__(
        self,
        max_samples_in_batch: int,
        max_queue_size: int,
        collate_fn: Callable[
            [List[SpeechRecognitionDataSample]], SpeechRecognitionDataBatch
        ],
    ):
        self.max_samples_in_batch = max_samples_in_batch
        self.max_queue_size = max_queue_size
        self.collate_fn = collate_fn

        self.queue: List[SpeechRecognitionDataSample] = []

    def __call__(
        self, batch_iterator: Generator[SpeechRecognitionDataSample, None, None]
    ) -> Generator[SpeechRecognitionDataBatch, None, None]:
        self.queue.clear()

        for batch in batch_iterator:
            if not isinstance(batch, SpeechRecognitionDataSample):
                raise ValueError(
                    f"batch is expected to be of type {SpeechRecognitionDataSample}"
                )

            self.queue.append(batch)

            if len(self.queue) == self.max_queue_size:
                yield self.get_batch()

        while len(self.queue) > 0:
            yield self.get_batch()

    def get_batch(self) -> SpeechRecognitionDataBatch:
        if len(self.queue) == 0:
            raise ValueError("cannot get a batch while queue is empty")
        if len(self.queue) == 1:
            batch = self.collate_fn(self.queue)
            self.queue.clear()
            return batch

        # sort the queue by length of the samples
        self.queue = sorted(self.queue, key=lambda b: b.input_length)

        # create state for batch creation
        sample_indexes = []
        current_batch_size = 0

        # select a random sample to start with
        prime_sample_idx = random.randint(0, len(self.queue) - 1)
        prime_sample: SpeechRecognitionDataSample = self.queue[prime_sample_idx]

        sample_indexes.append(prime_sample_idx)
        current_idx_min = prime_sample_idx
        current_idx_max = prime_sample_idx
        current_batch_size += 1
        current_max_sample_length = prime_sample.input_length
        current_min_sample_length = prime_sample.input_length

        # keep selecting samples while limiting the distance between
        # `min_sample_length` and `max_sample_length` until adding another sample
        # would results in a batch with more than `max_samples_in_batch`
        # or the whole queue is selected
        while True:
            # retrieve the two potential candidate samples
            candidate_idx_min = current_idx_min - 1
            candidate_idx_max = current_idx_max + 1

            if candidate_idx_min >= 0:
                min_sample: SpeechRecognitionDataSample = self.queue[candidate_idx_min]
            else:
                min_sample = None
            if candidate_idx_max < len(self.queue):
                max_sample: SpeechRecognitionDataSample = self.queue[candidate_idx_max]
            else:
                max_sample = None

            if min_sample is None and max_sample is None:
                break

            # check which candidate minimizes the distance between
            # max_target_length and min_target_length
            min_sample_distance = (
                current_max_sample_length - min_sample.input_length
                if min_sample is not None
                else float("inf")
            )
            max_sample_distance = (
                max_sample.input_length - current_min_sample_length
                if max_sample is not None
                else float("inf")
            )

            if min_sample_distance < max_sample_distance:
                # add min_sample to batch
                sample_to_add = min_sample
                sample_idx_to_add = candidate_idx_min
                current_idx_min = candidate_idx_min
            else:
                # add max_sample to batch
                sample_to_add = max_sample
                sample_idx_to_add = candidate_idx_max
                current_idx_max = candidate_idx_max

            # check if adding this sample to the batch
            # would exceed the max length; if it does we break instead
            if (current_batch_size + 1) * max(
                sample_to_add.input_length, current_max_sample_length
            ) > self.max_samples_in_batch:
                break

            # add the sample to the batch
            sample_indexes.append(sample_idx_to_add)
            current_batch_size += 1
            current_max_sample_length = max(
                current_max_sample_length, sample_to_add.input_length
            )
            current_min_sample_length = min(
                current_min_sample_length, sample_to_add.input_length
            )

        # retrieve the samples and remove them from the queue
        # in reverse order (high to low) in order to keep indexes valid
        # as samples are removed
        batch_samples = [
            self.queue.pop(idx) for idx in sorted(sample_indexes, reverse=True)
        ]

        return self.collate_fn(batch_samples)


################################################################################
# method to write shards based on a extracted librispeech folder


def write_librispeech_shards(
    librispeech_folder_path: pathlib.Path,
    shards_path: pathlib.Path,
    compress_in_place: bool,
    samples_per_shard: int,
    shard_name_pattern: str = "shard-%06d.tar",
):
    """
    Transform a librispeech-structured folder of .flac files to WebDataset shards.

    :param librispeech_folder_path: folder where extracted librespeech data is located
    :param shards_path: folder to write shards of data to
    :param compress_in_place: boolean value determining whether the shards will
                              be compressed with the `gpig` utility.
    :param samples_per_shard: number of data samples to store in each shards.
    :param shard_name_pattern: pattern of name to give to each shard
    """
    # make sure output folder exist
    shards_path.mkdir(parents=True, exist_ok=True)

    # find all audio files
    audio_files = sorted([f for f in librispeech_folder_path.rglob("*.flac")])

    # store statistics
    all_reader_ids = set()
    all_chapter_ids = set()
    all_keys = set()

    # create tuples
    # (unique_sample_id, transcription string, path_to_audio_file, num_audio_frames)
    data_tuples = []

    for file in audio_files:
        # path should be
        # ${librispeech_folder_path}/<reader_id>/<chapter_id>/<reader_id>-<chapter_id>-<utterance_id>.wav
        reader_id = file.parent.parent.name
        chapter_id = file.parent.name

        # create a unique key for this sample
        key = file.stem

        # store statistics
        all_reader_ids.add(reader_id)
        all_chapter_ids.add(chapter_id)

        if key in all_keys:
            raise ValueError(f"duplicate key {key}")
        else:
            all_keys.add(key)

        # load the transcription of the audio file
        with (file.parent / f"{reader_id}-{chapter_id}.trans.txt").open("r") as f:
            lines = [line.strip() for line in f.readlines()]
            transcription = None

            for line in lines:
                split_line = line.split(" ")
                utterance_key = split_line[0]

                if utterance_key == key:
                    transcription = " ".join(split_line[1:])
                    break

        if transcription is None:
            raise ValueError(f"unable to find transcription for {file}")

        # load num_frames in audio file
        num_frames = torchaudio.info(str(file)).num_frames

        tup = (key, transcription, file, num_frames)
        data_tuples.append(tup)

    # write a meta.json file which contains statistics on the data
    # which will be written to shards
    meta_dict = {
        "reader_ids": list(all_reader_ids),
        "chapter_ids": list(all_chapter_ids),
        "keys": list(all_keys),
        "num_samples": len(data_tuples),
        "num_speakers": len(all_reader_ids),
    }

    with (
        shards_path / f"meta_{'_'.join(shard_name_pattern.split('_')[0:-1])}.json"
    ).open("w") as f:
        json.dump(meta_dict, f)

    # sort the tuples by length of audio file so that a batch does not need
    # a lot of padding
    data_tuples = sorted(data_tuples, key=lambda tupl: tupl[3])

    # write shards
    all_keys = set()
    shards_path.mkdir(exist_ok=True, parents=True)
    pattern = str(shards_path / shard_name_pattern)

    # optionally compress the .tar shards
    def compress(file_name: str):
        if compress_in_place:
            with yaspin.yaspin() as spinner:
                spinner.write(f"> compressing {file_name}")
                subprocess.call(
                    ["pigz", file_name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

    with wds.ShardWriter(
        pattern, maxsize=5e9, maxcount=samples_per_shard, post=compress
    ) as sink:
        for key, transcription, f, num_frames in data_tuples:
            # load the audio tensor
            tensor, sample_rate = torchaudio.load(str(f))

            if torch.any(torch.isnan(tensor)):
                raise ValueError(f"NaN value in wav file of {key=} at {f=}")

            # verify key is unique
            assert key not in all_keys
            all_keys.add(key)

            # extract speaker_id, youtube_id and utterance_id from key
            reader_id, chapter_id, utterance_id = key.split("-")

            # create sample to write
            sample = {
                "__key__": f"{reader_id}/{chapter_id}/{utterance_id}",
                "wav.pyd": tensor,
                "meta.json": {
                    "reader_id": reader_id,
                    "chapter_id": chapter_id,
                    "utterance_id": utterance_id,
                    "transcription": transcription,
                    "num_frames": num_frames,
                    "sampling_rate": sample_rate,
                },
            }

            # write sample to sink
            sink.write(sample)

            # delete audio file on disk
            f.unlink()

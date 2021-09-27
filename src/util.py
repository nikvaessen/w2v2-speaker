################################################################################
#
# Utility functions useful throughout the source code
#
# Author(s): Nik Vaessen
################################################################################

import argparse
import inspect
import pathlib
import shutil
import subprocess
import zipfile

from typing import Type

import psutil
import torch as t
import hurry.filesize

from pytorch_lightning import Trainer


################################################################################
# sane handing of boolean arguments in argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


################################################################################
# automatically extract the input arguments of a class constructor from
# the argparse Namespace


def process_input_arguments(cls: Type, input_namespace: argparse.Namespace):
    arguments = {}
    input_namespace = vars(input_namespace)

    for init_arg in inspect.signature(cls.__init__).parameters:
        if init_arg not in input_namespace:
            continue

        if input_namespace[init_arg] is not None:
            arguments[init_arg] = input_namespace[init_arg]

    return arguments


################################################################################
# device management


def get_gpu_device(fallback_to_cpu=True):
    if t.cuda.is_available():
        device = t.device("cuda")
    elif fallback_to_cpu:
        device = t.device("cpu")
        print(
            f"WARNING: tried to get GPU device but CUDA is unavailable."
            f" Falling back to CPU."
        )
    else:
        raise ValueError("CUDA is unavailable")

    return device


def get_cpu_device():
    return t.device("cpu")


################################################################################
# resource information


def print_cpu_info():
    proc = psutil.Process()
    num_cpus = len(proc.cpu_affinity())

    print(f"process has been allocated {num_cpus} cpu(s)")
    pass


def print_memory_info():
    proc = psutil.Process()
    mem_info = proc.memory_info()

    print(f"process has the following memory constraints:")
    for name, value in mem_info._asdict().items():
        print(name, hurry.filesize.size(value))


################################################################################
# debug a tensor


def debug_tensor_content(
    tensor: t.Tensor,
    name: str = None,
    save_dir: pathlib.Path = None,
    print_full_tensor: bool = False,
):
    if isinstance(save_dir, pathlib.Path):
        if name is None:
            raise ValueError("name cannot be None and save_dir is specified")
        file = save_dir / (name + ".txt")
        file.parent.mkdir(exist_ok=True, parents=True)

        file = file.open("w")
    else:
        file = None

    with t.no_grad():
        if name is not None:
            print(f"### {name} ###", file=file)

        print(tensor, file=file)
        print(tensor.shape, file=file)
        print(
            "min",
            t.min(tensor),
            "argmin",
            t.argmin(tensor),
            "max",
            t.max(tensor),
            "argmax",
            t.argmax(tensor),
            "mean",
            t.mean(tensor * 1.0),  # force float values for mean calculation
            "std",
            t.std(tensor * 1.0),  # force float values for std calculation
            file=file,
            sep="\n",
        )
        print("nan", t.any(t.isnan(tensor)), file=file)
        print("inf+", t.any(t.isposinf(tensor)), file=file)
        print("inf-", t.any(t.isneginf(tensor)), file=file)

        if print_full_tensor:
            t.set_printoptions(profile="full")
            print(tensor, file=file)
            t.set_printoptions(profile="default")

        if save_dir is not None:
            t.save(tensor, str(save_dir / (name + ".tensor")))

        print(file=file)
        if file is not None:
            file.close()


################################################################################
# extract save dir from a Trainer instance


def extract_save_dir_path(trainer: Trainer):
    return (
        pathlib.Path(trainer.default_root_dir)
        / "lightning_logs"
        / f"version_{trainer.logger.version}"
    )


################################################################################
# remove a directory with all it's (recursive) content


def remove_directory(dir_path: pathlib.Path):
    for child in dir_path.glob("*"):
        if child.is_file():
            child.unlink()
        else:
            remove_directory(child)

    dir_path.rmdir()


################################################################################
# Extract a zipfile to a certain location


def extract_archive(path_to_archive: pathlib.Path, extract_to: pathlib.Path):
    print(f"extracting {path_to_archive} into: \n--> {extract_to}")

    if extract_to.exists():
        shutil.rmtree(extract_to)

    shutil.unpack_archive(path_to_archive, extract_to)


def extract_archive_7z(path_to_archive: pathlib.Path, extract_to: pathlib.Path):
    subprocess.call(["7z", "x", path_to_archive, f"-o{extract_to}", "-y"])


def extract_archive_unzip(path_to_archive: pathlib.Path, extract_to: pathlib.Path):
    print(f"extracting {path_to_archive} into: \n--> {extract_to}")
    extract_to.mkdir(exist_ok=True, parents=True)
    subprocess.call(["unzip", "-oq", path_to_archive, "-d", extract_to])


################################################################################
# reset the weights of a nn module


def reset_model(model: t.nn.Module, top=True):
    if top:
        print("resetting weights of model:")
        print(model)

    for layer in model.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
        else:
            if hasattr(layer, "children"):
                reset_model(layer, top=False)
            else:
                print(f"{layer} cannot be reset")

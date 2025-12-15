import concurrent.futures
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from beat_this.dataset.augment import (
    augment_mask_,
    augment_pitchtempo,
    precomputed_augmentation_filenames,
)
from beat_this.utils import index_to_framewise

from .mmnpz import MemmappedNpzFile


class PhraseBoundaryDataset(Dataset):
    """
    A PyTorch Dataset for phrase boundary detection using HarmonixSet annotations.
    """

    def __init__(
        self,
        item_names: list[str],
        data_folder,
        spect_fps=50,
        train_length=None,
        deterministic=False,
        augmentations={},
        length_based_oversampling_factor=0,
    ):
        self.spect_basepath = data_folder / "audio" / "spectrograms"
        self.annotation_basepath = data_folder / "annotations"
        self.fps = spect_fps
        self.train_length = train_length
        self.deterministic = deterministic
        self.augmentations = augmentations
        self.length_based_oversampling_factor = length_based_oversampling_factor
        datasets = sorted(set(name.split("/", 1)[0] for name in item_names))
        # load dataset info
        self.dataset_info = self._load_dataset_infos(datasets)
        # load .npz spectrogram bundles, if any
        self.spects = self._load_spect_bundles(datasets)
        # load the annotations in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            items = executor.map(self._load_dataset_item, item_names)
        items = [item for item in items if item is not None]
        if self.length_based_oversampling_factor and self.train_length is not None:
            # oversample the dataset according to the audio lengths, so that long pieces are sampled more often
            oversampled_items = []
            for item in items:
                oversampling_factor = np.round(
                    self.length_based_oversampling_factor
                    * len(self._get_spect(item))
                    / self.train_length
                ).astype(int)
                oversampling_factor = max(oversampling_factor, 1)
                oversampled_items.extend(itertools.repeat(item, oversampling_factor))
            print(
                f"Training set oversampled from {len(items)} to {len(oversampled_items)} excerpts."
            )
            items = oversampled_items
        self.items = items

    def _load_dataset_infos(self, datasets):
        dataset_info = {}
        for dataset in datasets:
            with open(self.annotation_basepath / dataset / "info.json") as f:
                dataset_info[dataset] = json.load(f)
        return dataset_info

    def _load_spect_bundles(self, datasets):
        spects = {}
        for dataset in datasets:
            npz_file = (self.spect_basepath / dataset).with_suffix(".npz")
            if npz_file.exists():
                spects[dataset] = MemmappedNpzFile(npz_file)
        return spects

    def _load_dataset_item(self, item_name):
        # stop if not all the augmented audio files are there
        dataset, remainder = item_name.split("/", 1)
        for aug_filename in precomputed_augmentation_filenames(self.augmentations):
            if (f"{remainder}/{aug_filename[:-4]}") not in self.spects.get(
                dataset, ()
            ) and not (self.spect_basepath / item_name / aug_filename).exists():
                print(
                    f"Skipping {item_name} because not all necessary spectrograms are there."
                )
                return

        # load phrase boundary annotations
        dataset, stem = item_name.split("/", 1)
        annotation_path = (
            self.annotation_basepath
            / dataset
            / "annotations"
            / "phrase_changes"
            / (stem + ".txt")
        )
        if not annotation_path.exists():
            print(f"Skipping {item_name} because phrase annotations are missing.")
            return

        boundary_times = np.loadtxt(annotation_path, ndmin=1)
        return {
            "spect_path": Path(item_name) / "track.npy",
            "boundary_time": boundary_times,
            "dataset": dataset,
        }

    def _get_spect(self, item):
        try:
            dataset, filename = str(item["spect_path"]).split("/", 1)
            spect = self.spects[dataset][filename[:-4]]
        except KeyError:
            spect = np.load(self.spect_basepath / item["spect_path"], mmap_mode="r")
        return spect

    def get_frame_count(self, index):
        """Return number of frames of given item."""
        return len(self._get_spect(self.items[index]))

    def get_boundary_count(self, index):
        """Return number of phrase changes of given item."""
        return len(self.items[index]["boundary_time"])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        if isinstance(index, (int, np.int64)):  # when index is a single int
            item = self.items[index]

            # select a pitch shift and time stretch
            item = augment_pitchtempo(item, self.augmentations)

            # load spectrogram
            spect = self._get_spect(item)

            # define the excerpt to use
            original_length = len(spect)
            if self.train_length is not None:
                longer = original_length - self.train_length
            else:
                longer = 0
            if longer > 0:  # if the piece is longer than the desired length
                if self.deterministic:
                    # select the middle of the excerpt
                    start_frame = longer // 2
                else:
                    start_frame = np.random.randint(0, longer)
                end_frame = start_frame + self.train_length
            else:
                start_frame = 0
                end_frame = original_length

            # obtain a view of the excerpt
            spect = spect[start_frame:end_frame]

            if "mask" in self.augmentations:
                # copy the spectrogram and apply mask augmentation
                spect = np.copy(spect)
                spect = augment_mask_(spect, self.augmentations, self.fps)
            else:
                # only ensure we have a writeable array (so PyTorch is happy)
                spect = np.require(spect, requirements="W")

            # prepare annotations
            (
                framewise_truth_boundary,
                truth_orig_boundary,
            ) = prepare_annotations(item, start_frame, end_frame, self.fps)

            # restructure the item dict with the correct training information
            item = {
                "spect": spect,
                "spect_path": str(item["spect_path"]),
                "dataset": item["dataset"],
                "start_frame": start_frame,
                "truth_boundary": framewise_truth_boundary,
                "padding_mask": (
                    np.ones(self.train_length, dtype=bool)
                    if self.train_length is not None
                    else np.ones(original_length, dtype=bool)
                ),
                "truth_orig_boundary": truth_orig_boundary,
            }

            # pad all framewise tensors if needed
            if longer < 0:
                item["spect"] = np.pad(
                    item["spect"], [(0, -longer), (0, 0)], constant_values=0
                )
                item["truth_boundary"] = np.pad(
                    item["truth_boundary"], [(0, -longer)], constant_values=0
                )
                item["padding_mask"][longer:] = 0
            return item

        else:  # when index is a list of ints
            return [self[i] for i in index]


class PhraseDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for phrase-boundary detection on HarmonixSet using full-track excerpts.

    Args:
        data_dir (Path or str): The parent directory where the data (spectrograms and labels) is stored.
        batch_size (int, optional): The size of the batches to be generated by the DataLoader. Defaults to 8.
        train_length (int, optional): The length of the subsequences in frames. If None, the entire pieces are returned. Defaults to None.
        num_workers (int, optional): The number of worker processes to use for data loading. Defaults to 20.
        augmentations (dict, optional): A dictionary of data augmentations to apply. Defaults to {"pitch": {"min": -5, "max": 6}, "tempo": {"min": -20, "max": 20, "stride": 4}}.
        test_dataset (str, optional): The name of the dataset to use for testing. Defaults to "harmonix".
        hung_data (bool, optional): If True, only use HarmonixSet entries. Defaults to False.
        no_val (bool, optional): If True, train on all train+val data and do not use a validation set. Defaults to False.
        spect_fps (int, optional): The frames per second of the spectrograms. Defaults to 50.
        length_based_oversampling_factor (int, optional): The factor by which to oversample the train dataset based on sequence length. Defaults to 0.
        fold (int, optional): The fold number for cross-validation. If None, the single split is used. Defaults to None.
        predict_datasplit (str, optional): The split to use for prediction. Prediction dataset is always full pieces. Defaults to "test".
    """

    def __init__(
        self,
        data_dir,
        batch_size=8,
        train_length=None,
        num_workers=20,
        augmentations={
            "pitch": {"min": -5, "max": 6},
            "tempo": {"min": -20, "max": 20, "stride": 4},
        },
        test_dataset="harmonix",
        hung_data=False,
        no_val=False,
        spect_fps=50,
        length_based_oversampling_factor=0,
        fold=None,
        predict_datasplit="test",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.initialized = {}
        # remember all arguments
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.train_length = train_length
        self.num_workers = num_workers
        if not set(augmentations.keys()).issubset({"mask", "pitch", "tempo"}):
            raise ValueError(f"Unsupported augmentations: {augmentations.keys()}")
        self.augmentations = augmentations
        self.test_set_name = test_dataset
        self.hung_data = hung_data
        self.no_val = no_val
        self.spect_fps = spect_fps
        self.length_based_oversampling_factor = length_based_oversampling_factor
        self.fold = fold
        self.predict_datasplit = predict_datasplit

    def setup(self, stage):
        if self.initialized.get(stage, False):
            return

        # set up the paths
        annotation_dir = self.data_dir / "annotations"

        # load train/val splits
        if stage in ("fit", "validate"):
            self.val_items = []
            self.train_items = []
            split_file = "8-folds.split" if self.fold is not None else "single.split"
            for dataset_dir in annotation_dir.iterdir():
                if not dataset_dir.is_dir() or not (dataset_dir / split_file).exists():
                    continue
                dataset = dataset_dir.name
                if dataset != "harmonix":
                    continue
                if dataset == self.test_set_name:
                    continue
                split = pd.read_csv(
                    dataset_dir / split_file,
                    header=None,
                    names=["piece", "part"],
                    sep="\t",
                )
                if self.fold is not None:
                    # CV: use given fold for validation, rest for training
                    self.val_items.extend(
                        f"{dataset}/{stem}"
                        for stem in split.piece[split.part == self.fold]
                    )
                    self.train_items.extend(
                        f"{dataset}/{stem}"
                        for stem in split.piece[split.part != self.fold]
                    )
                else:
                    # single split: marked as val and train
                    self.val_items.extend(
                        f"{dataset}/{stem}" for stem in split.piece[split.part == "val"]
                    )
                    self.train_items.extend(
                        f"{dataset}/{stem}"
                        for stem in split.piece[split.part == "train"]
                    )
            if self.no_val:
                # Train on all available data (excluding the test set).
                # For compatibility, validation metrics are still computed
                # on the original validation set now included in training.
                self.train_items.extend(self.val_items)
            if self.hung_data:
                self.train_items = [item for item in self.train_items if item.startswith("harmonix/")]
            self.val_items.sort()
            self.train_items.sort()

        # load validation set
        if stage in ("fit", "validate"):
            self.val_dataset = PhraseBoundaryDataset(
                self.val_items,
                deterministic=True,
                augmentations={},
                train_length=self.train_length,
                data_folder=self.data_dir,
                spect_fps=self.spect_fps,
            )
            print(
                "Validation set:",
                len(self.val_dataset),
                "items from:",
                *sorted(set(item.split("/", 1)[0] for item in self.val_items)),
            )
            self.initialized["validate"] = True

        # load training set
        if stage == "fit":
            self.train_dataset = PhraseBoundaryDataset(
                self.train_items,
                deterministic=False,
                augmentations=self.augmentations,
                train_length=self.train_length,
                data_folder=self.data_dir,
                spect_fps=self.spect_fps,
                length_based_oversampling_factor=self.length_based_oversampling_factor,
            )
            print(
                "Training set:",
                len(self.train_dataset),
                "items from:",
                *sorted(set(item.split("/", 1)[0] for item in self.train_items)),
            )
            self.initialized["fit"] = True

        # load test set
        if stage == "test":
            test_annotations_dir = (
                annotation_dir / self.test_set_name / "annotations" / "phrase_changes"
            )
            self.test_items = sorted(
                f"{self.test_set_name}/{item.stem}"
                for item in test_annotations_dir.glob("*.txt")
            )
            self.test_dataset = PhraseBoundaryDataset(
                self.test_items,
                deterministic=True,
                augmentations={},
                train_length=None,
                data_folder=self.data_dir,
                spect_fps=self.spect_fps,
            )
            print(
                "Test set:", len(self.test_dataset), "items from:", self.test_set_name
            )
            self.initialized["test"] = True

        # load prediction set
        if stage == "predict":
            if self.predict_datasplit == "test":
                self.setup("test")
                # we can directly use the test dataset for predictions
                self.predict_dataset = self.test_dataset
            else:
                if self.predict_datasplit == "train":
                    self.setup("fit")
                    items = self.train_items
                elif self.predict_datasplit == "val":
                    self.setup("validate")
                    items = self.val_items
                # for prediction, we want to use full items (train_length=None)
                self.predict_dataset = PhraseBoundaryDataset(
                    items,
                    deterministic=True,
                    augmentations={},
                    train_length=None,
                    data_folder=self.data_dir,
                    spect_fps=self.spect_fps,
                )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        # Warning: for performances, this only runs on the middle excerpt of the long pieces
        # The paper results are computed after training in the predict script
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset, batch_size=1, num_workers=self.num_workers
        )

    def get_train_positive_weights(self, widen_target_mask=3):
        """
        Computes the relation of negative targets to positive targets.
        `widen_target_mask` reduces the number of negative targets by the given
        factor times the number of positive targets (for ignoring a number of
        frames around each positive label).
        For example a `widen_target_mask` of 3 will ignore 7 frames, 3 for each side plus the central.
        """
        dataset = self.train_dataset
        all_frames = 0
        for item in dataset.items:
            frames = len(dataset._get_spect(item))
            all_frames += frames
        boundary_frames = sum(len(item["boundary_time"]) for item in dataset.items)

        return {
            "boundary": int(
                np.round(
                    (all_frames - boundary_frames * (widen_target_mask * 2 + 1))
                    / boundary_frames
                )
            )
        }


def prepare_annotations(item, start_frame, end_frame, fps):
    boundary_time = item["boundary_time"]
    boundary_frame = (boundary_time * fps).round().astype(int)
    boundary_frame -= start_frame
    idx = np.searchsorted(boundary_frame, 0)
    boundary_frame = boundary_frame[idx:]
    idx = np.searchsorted(boundary_frame, end_frame - start_frame)
    boundary_frame = boundary_frame[:idx]
    framewise_truth_boundary = index_to_framewise(
        boundary_frame, end_frame - start_frame
    )
    truth_orig_boundary = boundary_time[
        (boundary_time >= start_frame / fps) & (boundary_time < end_frame / fps)
    ] - (start_frame / fps)
    truth_orig_boundary = truth_orig_boundary.tobytes()
    return framewise_truth_boundary, truth_orig_boundary

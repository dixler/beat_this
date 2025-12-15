import concurrent.futures
import itertools
import json
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from beat_this.dataset.augment import augment_mask_, augment_pitchtempo
from beat_this.utils import index_to_framewise


class PhraseBoundaryDataset(Dataset):
    """
    A PyTorch Dataset for phrase boundary detection using HarmonixSet annotations.
    """

    def __init__(
        self,
        item_names: list[str],
        data_folder,
        spect_fps,
        train_length=None,
        deterministic=False,
        augmentations={},
        length_based_oversampling_factor=0,
        mel_suffix="-mel.npy",
        segment_dir=None,
    ):
        self.data_folder = Path(data_folder)
        self.spect_basepath = self.data_folder
        self.segment_dir = (
            Path(segment_dir)
            if segment_dir is not None
            else self.data_folder / "harmonixset" / "dataset" / "segments"
        )
        self.mel_suffix = mel_suffix
        self.fps = spect_fps
        self.train_length = train_length
        self.deterministic = deterministic
        self.augmentations = augmentations
        self.length_based_oversampling_factor = length_based_oversampling_factor
        # load the annotations in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            items = executor.map(self._load_dataset_item, item_names)
        items = [item for item in items if item is not None]
        if self.length_based_oversampling_factor and self.train_length is not None:
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

    def _load_dataset_item(self, stem: str):
        spect_path = self.spect_basepath / f"{stem}{self.mel_suffix}"
        if not spect_path.exists():
            print(f"Skipping {stem} because mel spectrogram is missing at {spect_path}.")
            return

        annotation_path = self.segment_dir / f"{stem}.txt"
        if not annotation_path.exists():
            print(f"Skipping {stem} because segment annotation is missing at {annotation_path}.")
            return

        try:
            boundary_times = np.loadtxt(annotation_path, ndmin=1, usecols=[0])
        except Exception:
            print(f"Skipping {stem} because annotations could not be read from {annotation_path}.")
            return
        boundary_times = boundary_times[1:] if boundary_times.size else boundary_times

        return {
            "spect_path": spect_path,
            "boundary_time": boundary_times,
            "dataset": "harmonix",
        }

    def _get_spect(self, item):
        return np.load(item["spect_path"], mmap_mode="r")

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
    """Lightning DataModule wired for the local Harmonix mel/segment layout."""

    def __init__(
        self,
        data_dir=Path.home() / "Data" / "harmonix",
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
        spect_fps=None,
        length_based_oversampling_factor=0,
        fold=None,
        predict_datasplit="test",
        mel_suffix="-mel.npy",
        segment_dir=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.initialized = {}
        self.data_dir = Path(data_dir)
        self.segment_dir = (
            Path(segment_dir)
            if segment_dir is not None
            else self.data_dir / "harmonixset" / "dataset" / "segments"
        )
        self.batch_size = batch_size
        self.train_length = train_length
        self.num_workers = num_workers
        if not set(augmentations.keys()).issubset({"mask", "pitch", "tempo"}):
            raise ValueError(f"Unsupported augmentations: {augmentations.keys()}")
        self.augmentations = augmentations
        self.test_set_name = test_dataset
        self.hung_data = hung_data
        self.no_val = no_val
        self.length_based_oversampling_factor = length_based_oversampling_factor
        self.fold = fold
        self.predict_datasplit = predict_datasplit
        self.mel_suffix = mel_suffix
        suffix_ext = Path(mel_suffix).suffix
        self.mel_suffix_no_ext = (
            mel_suffix[: -len(suffix_ext)] if suffix_ext else mel_suffix
        )

        info_path = self.data_dir / "info.json"
        self.dataset_info = json.loads(info_path.read_text()) if info_path.exists() else {}
        derived_fps = None
        if "SR" in self.dataset_info and "HOP_LENGTH" in self.dataset_info:
            derived_fps = self.dataset_info["SR"] / self.dataset_info["HOP_LENGTH"]
        self.spect_fps = spect_fps if spect_fps is not None else derived_fps or 50
        self.spect_dim = self.dataset_info.get("N_MELS", 128)

    def setup(self, stage):
        if self.initialized.get(stage, False):
            return
        stems = self._available_stems()

        test_cut = max(1, int(len(stems) * 0.1)) if len(stems) > 1 else 0
        val_cut = 0 if self.no_val else (max(1, int(len(stems) * 0.1)) if len(stems) > 2 else 0)

        self.test_items = stems[-test_cut:] if test_cut else stems
        remaining = stems[:-test_cut] if test_cut else stems
        self.val_items = remaining[-val_cut:] if val_cut else []
        self.train_items = remaining if self.no_val else remaining[:-val_cut] or remaining

        if stage in ("fit", "validate"):
            self.val_dataset = PhraseBoundaryDataset(
                self.val_items,
                deterministic=True,
                augmentations={},
                train_length=self.train_length,
                data_folder=self.data_dir,
                spect_fps=self.spect_fps,
                segment_dir=self.segment_dir,
                mel_suffix=self.mel_suffix,
            )
            print("Validation set:", len(self.val_dataset), "items")
            self.initialized["validate"] = True

        if stage == "fit":
            self.train_dataset = PhraseBoundaryDataset(
                self.train_items,
                deterministic=False,
                augmentations=self.augmentations,
                train_length=self.train_length,
                data_folder=self.data_dir,
                spect_fps=self.spect_fps,
                length_based_oversampling_factor=self.length_based_oversampling_factor,
                segment_dir=self.segment_dir,
                mel_suffix=self.mel_suffix,
            )
            print("Training set:", len(self.train_dataset), "items")
            self.initialized["fit"] = True

        if stage == "test":
            self.test_dataset = PhraseBoundaryDataset(
                self.test_items,
                deterministic=True,
                augmentations={},
                train_length=None,
                data_folder=self.data_dir,
                spect_fps=self.spect_fps,
                segment_dir=self.segment_dir,
                mel_suffix=self.mel_suffix,
            )
            print("Test set:", len(self.test_dataset), "items")
            self.initialized["test"] = True

        if stage == "predict":
            if self.predict_datasplit == "test":
                self.setup("test")
                self.predict_dataset = self.test_dataset
            else:
                if self.predict_datasplit == "train":
                    self.setup("fit")
                    items = self.train_items
                else:
                    self.setup("validate")
                    items = self.val_items
                self.predict_dataset = PhraseBoundaryDataset(
                    items,
                    deterministic=True,
                    augmentations={},
                    train_length=None,
                    data_folder=self.data_dir,
                    spect_fps=self.spect_fps,
                    segment_dir=self.segment_dir,
                    mel_suffix=self.mel_suffix,
                )
            self.initialized["predict"] = True

    def _available_stems(self):
        if not self.segment_dir.exists():
            raise FileNotFoundError(
                f"Segment annotations not found at {self.segment_dir}. "
                "Pass an explicit segment_dir if your layout differs."
            )

        stems = []
        for mel in sorted(self.data_dir.glob(f"*{self.mel_suffix}")):
            stem = mel.stem
            if self.mel_suffix_no_ext:
                stem = stem.removesuffix(self.mel_suffix_no_ext)
            annotation_path = self.segment_dir / f"{stem}.txt"
            if annotation_path.exists():
                stems.append(stem)
            else:
                print(
                    f"Skipping {stem} because segment annotation is missing at {annotation_path}."
                )
        return stems

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            collate_fn=collate_phrase_batches,
        )

    def val_dataloader(self):
        # Warning: for performances, this only runs on the middle excerpt of the long pieces
        # The paper results are computed after training in the predict script
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_phrase_batches,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            collate_fn=collate_phrase_batches,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            collate_fn=collate_phrase_batches,
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


def collate_phrase_batches(batch):
    """Pad variable-length phrase excerpts so PyTorch can stack them."""

    max_len = max(item["spect"].shape[0] for item in batch)
    max_mel = max(item["spect"].shape[1] for item in batch)

    spect = torch.zeros((len(batch), max_len, max_mel), dtype=torch.float32)
    truth_boundary = torch.zeros((len(batch), max_len), dtype=torch.float32)
    padding_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)

    collated = {
        "spect": spect,
        "truth_boundary": truth_boundary,
        "padding_mask": padding_mask,
        "dataset": [],
        "spect_path": [],
        "start_frame": [],
        "truth_orig_boundary": [],
    }

    for idx, item in enumerate(batch):
        length = item["spect"].shape[0]
        mel_dim = item["spect"].shape[1]
        collated["spect"][idx, :length, :mel_dim] = torch.as_tensor(
            item["spect"], dtype=torch.float32
        )
        collated["truth_boundary"][idx, :length] = torch.as_tensor(
            item["truth_boundary"], dtype=torch.float32
        )
        collated["padding_mask"][idx, :length] = torch.as_tensor(
            item["padding_mask"], dtype=torch.bool
        )
        collated["dataset"].append(item["dataset"])
        collated["spect_path"].append(item["spect_path"])
        collated["start_frame"].append(item["start_frame"])
        collated["truth_orig_boundary"].append(item["truth_orig_boundary"])

    return collated

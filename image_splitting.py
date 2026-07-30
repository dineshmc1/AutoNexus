"""Automatic group discovery and leakage-safe image index splitting."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

_SUBJECT_PATTERN = re.compile(
    r"(?i)(?:^|[_\-/])((?:subject|person|actor|user|participant)[_-]?\d+)"
)
_FRAME_PATTERN = re.compile(
    r"(?i)^(.+?)[_-](?:frame|frm|f)[_-]?\d+$"
)


def infer_image_groups(
    files: Sequence[str],
    labels: Sequence[str],
    dataset_root: str | Path,
) -> tuple[list[str] | None, str]:
    """Infer repeated video/subject groups, rejecting weak heuristics."""
    if len(files) != len(labels):
        raise ValueError("Image paths and labels must have equal length.")
    root = Path(dataset_root).resolve()
    candidates: list[str | None] = []
    methods: list[str | None] = []

    for filename in files:
        path = Path(filename).resolve()
        try:
            relative = path.relative_to(root)
        except ValueError:
            relative = Path(path.name)
        normalized = relative.as_posix()
        subject = _SUBJECT_PATTERN.search(normalized)
        if subject:
            candidates.append(f"subject:{subject.group(1).lower()}")
            methods.append("subject-token")
            continue

        # A nested folder below the class usually represents a video/session.
        if len(relative.parts) >= 3:
            nested = "/".join(relative.parts[1:-1]).lower()
            if nested and nested != "images":
                candidates.append(f"folder:{relative.parts[0].lower()}:{nested}")
                methods.append("nested-folder")
                continue

        frame = _FRAME_PATTERN.match(path.stem)
        if frame:
            candidates.append(
                f"sequence:{relative.parent.as_posix().lower()}:"
                f"{frame.group(1).lower()}"
            )
            methods.append("frame-prefix")
            continue

        candidates.append(None)
        methods.append(None)

    if any(group is None for group in candidates):
        return None, "stratified-no-reliable-groups"

    groups = [str(group) for group in candidates]
    unique_groups = set(groups)
    if len(unique_groups) >= 0.9 * len(groups):
        return None, "stratified-groups-mostly-unique"

    labels_array = np.asarray(labels)
    groups_array = np.asarray(groups)
    per_class_groups = [
        len(set(groups_array[labels_array == label]))
        for label in np.unique(labels_array)
    ]
    if min(per_class_groups, default=0) < 2:
        return None, "stratified-insufficient-groups-per-class"

    counts = np.unique(groups_array, return_counts=True)[1]
    largest_group = int(counts.max()) if len(counts) else 0
    if largest_group > 0.5 * len(groups):
        return None, "stratified-dominant-group"

    method = max(
        (item for item in methods if item is not None),
        key=lambda item: methods.count(item),
    )
    return groups, f"stratified-group-{method}"


def split_labeled_indices(
    labels: Sequence[str],
    test_size: float,
    random_state: int,
    groups: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Split labels, preserving groups when a reliable grouping is available."""
    labels_array = np.asarray(labels)
    indices = np.arange(len(labels_array))
    if groups is not None:
        groups_array = np.asarray(groups)
        class_group_counts = [
            len(set(groups_array[labels_array == label]))
            for label in np.unique(labels_array)
        ]
        folds = min(
            max(2, int(round(1.0 / test_size))),
            min(class_group_counts, default=2),
        )
        if folds >= 2:
            splitter = StratifiedGroupKFold(
                n_splits=folds,
                shuffle=True,
                random_state=random_state,
            )
            candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
            expected_classes = set(np.unique(labels_array))
            for train_indices, test_indices in splitter.split(
                indices, labels_array, groups_array
            ):
                if (
                    set(labels_array[train_indices]) != expected_classes
                    or set(labels_array[test_indices]) != expected_classes
                ):
                    continue
                fraction_error = abs(len(test_indices) / len(indices) - test_size)
                candidates.append(
                    (fraction_error, train_indices, test_indices)
                )
            if candidates:
                _, train_indices, test_indices = min(
                    candidates, key=lambda item: item[0]
                )
                return train_indices, test_indices, "stratified-group"

    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=labels_array,
    )
    return (
        np.asarray(train_indices),
        np.asarray(test_indices),
        "stratified",
    )

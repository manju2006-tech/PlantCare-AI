"""Train/evaluate a separate rejection model without changing disease weights.

Run from the project root: .venv/bin/python models/train_support_detector.py
Features are extracted once; a numeric cache supports resuming an interrupted
run. Original images and model files are only read. No image folders are copied.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.support_detector import (
    FORMAT_VERSION, PREPROCESSING, file_sha256, kernel_scores,
    normalize_features, preprocess_image,
)

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
# Exclude broad related genera rather than accidentally calling a supported
# crop a negative. The inaturalist download contains raspberries/blueberries.
SUPPORTED_GENERA = {
    "malus", "musa", "vaccinium", "prunus", "zea", "vitis", "mangifera",
    "epipremnum", "scindapsus", "pothos", "citrus", "capsicum", "solanum",
    "rubus", "oryza", "glycine", "cucurbita", "fragaria",
}
SPLITS = ("train", "validation", "test")


def image_paths(folder):
    return sorted(p for p in folder.rglob("*")
                  if p.is_file() and p.suffix.lower() in EXTENSIONS)


def original_group(path):
    """Keep known augmented variants together, including rice originals."""
    stem = path.stem.lower()
    uuid = re.search(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", stem
    )
    if uuid:
        return uuid.group()
    if stem.startswith("rice_4class_"):
        return re.sub(r"_\d+$", "", re.sub(r"^rice_4class_\d+_", "", stem))
    return stem


def partition_groups(groups, seed):
    groups = sorted(groups)
    random.Random(seed).shuffle(groups)
    if len(groups) < 3:
        raise ValueError("At least three independent image groups are required.")
    validation = max(1, round(len(groups) * 0.15))
    test = max(1, round(len(groups) * 0.15))
    return {
        "test": groups[:test],
        "validation": groups[test:test + validation],
        "train": groups[test + validation:],
    }


def collect_samples(args, classes):
    samples = []
    for index, name in enumerate(classes):
        groups = defaultdict(list)
        for source_split in SPLITS:
            for path in image_paths(args.dataset / source_split / name):
                groups[original_group(path)].append(path)
        partitions = partition_groups(groups, args.seed + index)
        caps = dict(zip(SPLITS, (args.train_per_class,
                                args.validation_per_class, args.test_per_class)))
        for split, keys in partitions.items():
            # Round-robin sampling favors distinct originals over augmentations.
            pools = []
            for key in keys:
                pool = sorted(groups[key])
                random.Random(f"{args.seed}:{name}:{key}").shuffle(pool)
                pools.append((key, pool[:10]))
            selected = []
            for offset in range(10):
                for key, pool in pools:
                    if offset < len(pool):
                        selected.append((key, pool[offset]))
            for key, path in selected[:caps[split]]:
                samples.append(dict(path=str(path.relative_to(ROOT)), label=1,
                                    category=name, split=split,
                                    group=f"known:{name}:{key}"))

    negatives = []
    wheat = ROOT / "dataset/unknown_test"
    for category in ("wheat_healthy", "wheat_damaged/septoria",
                     "wheat_damaged/strip_rust"):
        for path in image_paths(wheat / category):
            negatives.append((path, "wheat:" + category, None))

    # These two species are never used for fitting the rejection classifier.
    held_out = {"Carica papaya": "validation", "Hibiscus": "test"}
    folder = ROOT / "dataset/unsupported_plant_test"
    with (folder / "metadata.csv").open(newline="") as source:
        for row in csv.DictReader(source):
            path = folder / "images" / row["filename"]
            if path.is_file():
                negatives.append((path, row["scientific_name"],
                                  held_out.get(row["scientific_name"])))

    # Group all photos of an unseen species in one split. Reserve species
    # rather than just photos to measure generalization beyond training plants.
    folder = ROOT / "dataset/plant_binary_source/inaturalist_download"
    with (folder / "metadata.csv").open(newline="") as source:
        for row in csv.DictReader(source):
            species = row["scientific_name"]
            if species.split()[0].lower() in SUPPORTED_GENERA:
                continue
            path = folder / "images" / row["filename"]
            if path.is_file():
                bucket = int(hashlib.sha256(
                    f"{args.seed}:{species}".encode()).hexdigest()[:8], 16) % 10
                split = "train" if bucket < 6 else "validation" if bucket < 8 else "test"
                negatives.append((path, "inat:" + species, split))

    # RGB leaf specimens only: omit BW masks, duplicate JPEG exports and the
    # strawberry genus, which is already a supported plant in this project.
    folder = ROOT / "dataset/unknown_test/other_plants/RGB"
    for species_folder in sorted(folder.iterdir()):
        if not species_folder.is_dir() or "fragaria" in species_folder.name.lower():
            continue
        species = re.sub(r"^\d+\.\s*", "", species_folder.name)
        bucket = int(hashlib.sha256(
            f"{args.seed}:{species}".encode()).hexdigest()[:8], 16) % 10
        split = "train" if bucket < 6 else "validation" if bucket < 8 else "test"
        for path in image_paths(species_folder):
            negatives.append((path, "leaf:" + species, split))

    # Hash raw files to prevent exact duplicates crossing split boundaries.
    by_category = defaultdict(dict)
    digest_split = {}
    for path, category, fixed_split in negatives:
        digest = file_sha256(path)
        if digest in digest_split:
            continue
        digest_split[digest] = fixed_split
        by_category[category][digest] = (path, fixed_split)
    for category, groups in sorted(by_category.items()):
        fixed = next(iter(groups.values()))[1]
        partitions = ({fixed: sorted(groups)} if fixed else
                      partition_groups(groups, f"{args.seed}:{category}"))
        for split, keys in partitions.items():
            for key in keys:
                path, _ = groups[key]
                samples.append(dict(path=str(path.relative_to(ROOT)), label=0,
                                    category=category, split=split,
                                    group=f"negative:{key}"))

    # Prefer the held-out copies if the same bytes also appear in training.
    unique = {}
    priority = {"train": 0, "validation": 1, "test": 2}
    for sample in samples:
        digest = file_sha256(ROOT / sample["path"])
        sample["sha256"] = digest
        previous = unique.get(digest)
        if previous and previous["label"] != sample["label"]:
            raise ValueError("An image has conflicting support labels.")
        if previous is None or priority[sample["split"]] > priority[previous["split"]]:
            unique[digest] = sample
    samples = sorted(unique.values(), key=lambda s: (s["split"], s["category"], s["path"]))
    memberships = defaultdict(set)
    for sample in samples:
        memberships[sample["group"]].add(sample["split"])
    if any(len(splits) != 1 for splits in memberships.values()):
        raise ValueError("Related source images cross detector splits.")
    return samples


def extract_features(args, samples, model_hash):
    cache_path = args.output / "feature_cache.npz"
    identity = hashlib.sha256(json.dumps(
        {"samples": samples, "model": model_hash, "preprocess": PREPROCESSING},
        sort_keys=True,
    ).encode()).hexdigest()
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as data:
            if str(data["identity"].item()) != identity:
                raise ValueError("Feature cache inputs changed. Choose a new output directory.")
            print("Reusing verified image feature cache.", flush=True)
            return data["features"], data["probabilities"]

    import tensorflow as tf
    from utils.unknown_detector import _build_extractor
    tf.keras.utils.set_random_seed(args.seed)
    model = tf.keras.models.load_model(args.model, compile=False)
    extractor = _build_extractor(model)
    features = []
    probabilities = []
    started = time.monotonic()
    for offset in range(0, len(samples), args.batch_size):
        batch_samples = samples[offset:offset + args.batch_size]
        batch = np.concatenate([preprocess_image(ROOT / s["path"]) for s in batch_samples])
        vectors, probs = extractor(batch, training=False)
        features.append(np.asarray(vectors, dtype=np.float32))
        probabilities.append(np.asarray(probs, dtype=np.float32))
        if offset == 0 or (offset // args.batch_size) % 20 == 0 or offset + args.batch_size >= len(samples):
            print(f"Features: {min(offset + args.batch_size, len(samples))}/{len(samples)}; "
                  f"elapsed {time.monotonic() - started:.0f}s", flush=True)
    features = np.concatenate(features)
    probabilities = np.concatenate(probabilities)
    np.savez_compressed(cache_path, identity=np.asarray(identity),
                        features=features, probabilities=probabilities)
    return features, probabilities


def metrics(samples, accepted, mask):
    selected = np.flatnonzero(mask)
    categories = defaultdict(lambda: [0, 0])
    known_accepted = known_total = unknown_accepted = unknown_total = 0
    for index in selected:
        s = samples[index]
        keep = bool(accepted[index])
        category = categories[s["category"]]
        category[0] += 1
        category[1] += int(keep)
        if s["label"]:
            known_total += 1
            known_accepted += int(keep)
        else:
            unknown_total += 1
            unknown_accepted += int(keep)
    wheat = [i for i in selected if samples[i]["category"].startswith("wheat:")]
    rice = [i for i in selected if samples[i]["category"].startswith("Rice___")]
    rate = lambda n, d: round(n / d, 6) if d else None
    return {
        "supported_total": known_total, "supported_accepted": known_accepted,
        "supported_acceptance": rate(known_accepted, known_total),
        "unsupported_total": unknown_total, "unsupported_accepted": unknown_accepted,
        "unsupported_rejection": rate(unknown_total - unknown_accepted, unknown_total),
        "wheat_total": len(wheat), "wheat_rejected": sum(not accepted[i] for i in wheat),
        "wheat_rejection": rate(sum(not accepted[i] for i in wheat), len(wheat)),
        "rice_total": len(rice), "rice_accepted": sum(bool(accepted[i]) for i in rice),
        "rice_acceptance": rate(sum(bool(accepted[i]) for i in rice), len(rice)),
        "categories": {k: {"total": n, "accepted": a, "acceptance": rate(a, n)}
                       for k, (n, a) in sorted(categories.items())},
    }


def select_threshold(scores, labels, confidence_ok, samples, mask):
    indices = np.flatnonzero(mask)
    positive = indices[labels[indices] == 1]
    negative = indices[labels[indices] == 0]
    wheat = np.asarray([i for i in negative if samples[i]["category"].startswith("wheat:")])
    rice = np.asarray([i for i in positive if samples[i]["category"].startswith("Rice___")])
    best = None
    # Explicit deployment targets; thresholds use validation only.
    for threshold in np.unique(np.r_[scores[indices], np.nextafter(scores[indices].max(), np.inf)]):
        accepted = (scores >= threshold) & confidence_ok
        known_rate = float(accepted[positive].mean())
        unknown_reject = float(1 - accepted[negative].mean())
        wheat_reject = float(1 - accepted[wheat].mean())
        rice_rate = float(accepted[rice].mean())
        feasible = known_rate >= 0.95 and rice_rate >= 0.90 and wheat_reject >= 0.95
        objective = unknown_reject + wheat_reject - 8 * max(0, 0.95-known_rate) - 4 * max(0, 0.90-rice_rate)
        key = (feasible, objective, known_rate)
        if best is None or key > best[0]:
            best = (key, float(threshold))
    return best[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset/plantcare_57class")
    parser.add_argument("--model", type=Path, default=ROOT / "trained_model_57class_run2/best_model.keras")
    parser.add_argument("--classes", type=Path, default=ROOT / "trained_model_57class_run2/class_names.json")
    parser.add_argument("--profile", type=Path, default=ROOT / "trained_model_57class_run2/unknown_profile.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "trained_model_support")
    parser.add_argument("--train-per-class", type=int, default=100)
    parser.add_argument("--validation-per-class", type=int, default=35)
    parser.add_argument("--test-per-class", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    artifact_path = args.output / "support_detector.npz"
    if artifact_path.exists():
        raise FileExistsError("A detector already exists here. Use a new --output directory.")
    classes = json.loads(args.classes.read_text())
    if not isinstance(classes, list) or len(classes) != 57:
        raise ValueError("Expected the active model's ordered 57-class label list.")
    model_hash = file_sha256(args.model)
    samples = collect_samples(args, classes)
    print("Split counts:", dict(Counter((s["split"], s["label"]) for s in samples)), flush=True)
    with (args.output / "sample_manifest.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)
    features, probabilities = extract_features(args, samples, model_hash)
    normalized = normalize_features(features)
    labels = np.asarray([s["label"] for s in samples])
    masks = {split: np.asarray([s["split"] == split for s in samples]) for split in SPLITS}
    confidence_ok = np.max(probabilities, axis=1) >= 0.5

    from sklearn.svm import SVC
    # Equal total weight for supported/unsupported. Balance the positive
    # disease classes and negative categories within their respective sides.
    counts = Counter((s["label"], s["category"]) for s in samples if s["split"] == "train")
    families = lambda label, category: "supported" if label else category.split(":")[0] if ":" in category else "garden"
    category_counts = Counter(families(label, category) for label, category in counts)
    negative_families = len(set(families(label, category) for label, category in counts if not label))
    weights = np.asarray([
        1 / ((1 if s["label"] else negative_families)
             * category_counts[families(s["label"], s["category"])]
             * counts[s["label"], s["category"]])
        for s in samples if s["split"] == "train"
    ])
    weights *= len(weights) / weights.sum()
    candidates = []
    best = None
    for c in (1.0, 10.0, 100.0):
        for gamma in (1.0, 4.0):
            print(f"Fitting support classifier: C={c}, gamma={gamma}", flush=True)
            classifier = SVC(C=c, gamma=gamma, cache_size=128)
            classifier.fit(normalized[masks["train"]], labels[masks["train"]], sample_weight=weights)
            scores = classifier.decision_function(normalized)
            threshold = select_threshold(scores, labels, confidence_ok, samples, masks["validation"])
            accepted = (scores >= threshold) & confidence_ok
            result = metrics(samples, accepted, masks["validation"])
            summary = {k: v for k, v in result.items() if k != "categories"}
            print("Validation:", json.dumps(summary), flush=True)
            candidates.append(dict(C=c, gamma=gamma, threshold=threshold, **summary))
            feasible = (result["supported_acceptance"] >= 0.95
                        and result["rice_acceptance"] >= 0.90
                        and result["wheat_rejection"] >= 0.95)
            objective = (result["unsupported_rejection"] + result["wheat_rejection"]
                         - 8 * max(0, 0.95-result["supported_acceptance"])
                         - 4 * max(0, 0.90-result["rice_acceptance"]))
            key = (feasible, objective, result["supported_acceptance"])
            if best is None or key > best[0]:
                best = (key, classifier, threshold, accepted, result)

    _, classifier, threshold, accepted, validation = best
    # Compare against the old centroid detector on the identical image split.
    with np.load(args.profile, allow_pickle=False) as profile:
        if profile["class_names"].tolist() != classes:
            raise ValueError("Old detector label order differs.")
        predicted = np.argmax(probabilities, axis=1)
        old_distance = np.linalg.norm(features - profile["centroids"][predicted], axis=1)
        baseline = confidence_ok & (old_distance <= profile["thresholds"][predicted])

    metadata = {
        "format_version": FORMAT_VERSION, "preprocessing": PREPROCESSING,
        "disease_model_sha256": model_hash, "class_names": classes,
        "image_size": [224, 224], "seed": args.seed,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "classifier": "RBF SVM on L2-normalized frozen MobileNetV2 features",
        "threshold_selection": "validation only; target 95% supported acceptance, 90% rice acceptance",
        "validation": validation,
        "limitations": [
            "Finite unsupported examples cannot guarantee rejection of every unseen plant.",
            "Known-source augmentation groups are disjoint for this detector; the frozen disease model was previously trained on this corpus.",
            "The old centroid baseline below uses bicubic preprocessing for a shared comparison; the legacy app used bilinear for centroid extraction.",
        ],
    }
    test = metrics(samples, accepted, masks["test"])
    report = {
        "metadata": metadata, "candidates_validation_only": candidates,
        "validation": validation, "test": test,
        "baseline_validation": metrics(samples, baseline, masks["validation"]),
        "baseline_test": metrics(samples, baseline, masks["test"]),
        "counts": {split: int(mask.sum()) for split, mask in masks.items()},
    }
    # Use plain numpy arrays at runtime and verify parity before exporting.
    portable_scores = np.concatenate([
        kernel_scores(features[start:start + 128], classifier.support_vectors_,
                      classifier.dual_coef_[0], classifier.intercept_[0], classifier._gamma)
        for start in range(0, len(features), 128)
    ])
    if not np.allclose(portable_scores, classifier.decision_function(normalized), atol=1e-7):
        raise ValueError("Portable detector does not match the fitted classifier.")
    report["deployment_checks_passed"] = bool(
        validation["supported_acceptance"] >= 0.95
        and validation["wheat_rejection"] >= 0.95
        and test["supported_acceptance"] >= 0.94
        and test["rice_acceptance"] >= 0.90
        and test["wheat_rejection"] >= 0.95
        and test["unsupported_rejection"] > report["baseline_test"]["unsupported_rejection"]
    )
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2))
    print("TEST:", json.dumps({k: v for k, v in test.items() if k != "categories"}), flush=True)
    print("BASELINE TEST:", json.dumps({k: v for k, v in report["baseline_test"].items() if k != "categories"}), flush=True)
    print("Deployment checks passed:", report["deployment_checks_passed"], flush=True)
    if not report["deployment_checks_passed"]:
        raise RuntimeError("Detector did not meet deployment criteria. Evaluation saved; live integration must stay unchanged.")
    np.savez_compressed(
        artifact_path, metadata=np.asarray(json.dumps(metadata)),
        support_vectors=classifier.support_vectors_, dual_coef=classifier.dual_coef_[0],
        intercept=np.asarray(classifier.intercept_[0]), gamma=np.asarray(classifier._gamma),
        threshold=np.asarray(threshold),
    )
    print("Detector saved:", artifact_path, flush=True)


if __name__ == "__main__":
    main()

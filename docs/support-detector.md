# Supported-plant rejection

The 57-class disease classifier always chooses one of its labels. Its softmax
confidence is not a probability that the plant is supported. The separate
`SupportedPlantDetector` learns supported versus unsupported examples from the
frozen MobileNetV2 features. The existing disease and plant/non-plant weights
are not retrained.

The detector is an RBF support vector classifier using L2-normalized features.
The deployed `.npz` contains numeric support vectors, coefficients, a calibrated
margin threshold, and JSON metadata. It requires no pickle or extra inference
dependency. A classifier margin is not a calibrated probability.

## Training and evaluation

From the project root, using the existing environment:

```bash
TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python models/train_support_detector.py
```

The script reads the existing datasets and writes only to its output folder,
`trained_model_support` by default. Use `--output` with a new directory for a
new experiment; it refuses to overwrite an exported detector. An intermediate
feature cache can resume fitting without extracting the images again. Cache
identity includes the selected files, their hashes, the preprocessing version,
and the disease model's SHA-256.

- Supported examples cover every disease class. Known UUID-based rotations and
  rice source-image augmentations stay in the same detector split.
- Wheat includes healthy, septoria, and stripe-rust images, divided into fitting,
  calibration, and test subsets. Exact duplicate files do not cross splits.
- Other unsupported species come from the existing iNaturalist and RGB leaf
  datasets. Related genera already covered by the supported crops are excluded
  from the negative labels. Unlabeled web downloads and black/white masks are
  excluded.
- Some unsupported species, including entire species groups, are reserved for
  calibration or testing. These images are not used to fit the classifier.
- Hyperparameters and the acceptance threshold use validation data. The test
  results are computed after selecting the candidate, with no test-driven
  threshold adjustment.

`sample_manifest.csv` records the exact inputs, their split, original-image
groups, and file hashes. `evaluation.json` records per-class and per-species
acceptance as well as wheat rejection, supported acceptance, and the previous
centroid method's results on the same images. The shared baseline uses the new
consistent bicubic preprocessing; legacy centroid inference used bilinear.

The training script exports a detector only when its deployment checks pass.
These require at least 95% wheat rejection, at least 94% supported acceptance on
the detector test split, at least 90% rice acceptance, and improved unsupported
rejection over the shared centroid baseline.

## Application behavior

Startup verifies the saved detector's version, preprocessing, class order,
feature dimensions, and disease-model fingerprint. `UNKNOWN_PLANT_CHECK` keeps
the rejection check enabled. Missing or incompatible artifacts stop scans with
an availability error rather than falling back to the weaker centroid method.

An unsupported or uncertain image returns HTTP 422 with `unknown_plant: true`.
The upload page displays an explanation and allows another photo. Rejected
uploads are removed and create no prediction row, recommendations, or report.

The original `utils/unknown_detector.py` and `unknown_profile.npz` remain
available for reproducing earlier experiments.

## Recorded evaluation (November 4, 2026)

The exported detector passed the deployment checks:

- Supported acceptance: 2,508 / 2,531 (99.1%).
- Rice acceptance: 39 / 40 (97.5%).
- Unsupported rejection: 382 / 382, including 31 / 31 wheat images.
- Of the unsupported test images, 311 came from negative categories absent
  during classifier fitting.

The three wheat photos that previously produced high-confidence rice diagnoses
were also rejected through the real Flask `/predict` endpoint. A supported rice
upload produced its result page, database record, and downloadable PDF. These
endpoint checks used a temporary database and temporary upload/report folders.

## Verification and limitations

```bash
.venv/bin/python -B -m unittest discover -s tests -v
node tests/upload_ui.test.cjs
TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 .venv/bin/python -B tests/smoke_support_detector.py
```

The new detector's fitting, calibration, and test source-image groups are
disjoint, but the feature extractor was already trained on the existing
supported corpus. These results evaluate the new rejection layer and do not
constitute an independent field benchmark of the disease classifier.

The available wheat photos have no plant/observation IDs, so exact-file
deduplication cannot establish independence at the individual-plant level.
Additional photos from different fields, cameras, and growth stages are needed
to measure that generalization. A finite negative dataset cannot guarantee
rejection of every unseen species, and some supported photos may be rejected.

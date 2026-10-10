# C3 Research Specification — Working Draft

- **Project:** AI-Powered Holistic Mental Health Assessment and Monitoring Framework
- **Component:** C3 — Facial Emotion and Micro-Expression Analysis
- **Status:** Research specification draft; not a validated clinical method
**Implementation branch:** `dev`

## 1. Purpose and scope

Define an experimental, reproducible pipeline for extracting facial-expression and micro-expression evidence from short video sequences. This component must produce research evidence for later integration; it must not produce a mental-health diagnosis or a final mental-health risk score until the relevant methodology has been justified and evaluated.

The research proposal remains the source of truth. The specified elements are webcam/video input, face detection and preprocessing, temporal frames, MobileNetV3-Large, temporal analysis, micro-expression recognition, facial asymmetry, emotional incongruence, and valence/arousal-related features. The proposal does not mandate a particular face detector or a final fusion formula.

## 2. Existing implementation baseline

The package currently contains:

- `video.py`: deterministic frame sampling and OpenCV video reading.
- `face.py`: replaceable OpenCV Haar Cascade reference detector and 224×224 RGB ROI extraction.
- `sequence.py`: fixed-length sliding windows, currently length 16 and stride 8.
- `model.py`: `FacialFeatureExtractor` abstraction and a MobileNetV3-Large interface; the deep model is not implemented yet.
- `pipeline.py`: frame/ROI/sequence reference pipeline.
- `contract.py`: shared score output with `score.value = null`, `confidence = null`, `risk_level = null`.

These defaults are engineering placeholders, not proposal-mandated parameters. The sliding-window utility is not yet a dataset-aware micro-expression sample pipeline: it may create multiple overlapping windows from one annotated event and drops a trailing partial window. The dataset training path must preserve event/sample IDs and subject identities.

## 3. Dataset and label handling

### Primary dataset

Use CASME II as the initial research dataset after obtaining the dataset through an authorised route and checking the applicable access/use terms. The original publication reports 247 spontaneous micro-expression samples from 26 participants, recorded at 200 frames per second, with onset/offset annotations, action-unit annotations, and five emotion categories. It also reports marked class imbalance (for example, 60 disgust samples and 7 sadness samples). The original baseline used LBP-TOP features, an SVM classifier, and leave-one-subject-out validation, reporting 63.41% accuracy for its five-class experiment.

### Required rules

1. Preserve each sample's official/native label and original subject ID in the manifest.
2. Do not automatically map the dataset to the six ordinary macro-emotion classes; the CASME II authors explicitly note that its labels are based on action units, participant self-report, and elicitation context, and are not simply the traditional six categories.
3. If a common three-class protocol is later used for comparison with other published work, implement it as a separately named, documented label-mapping configuration. Keep the native-label experiment distinct and record excluded/merged classes and sample counts.
4. Do not commit raw dataset videos, frames, or restricted annotations to Git. Keep paths configurable and outside tracked source directories.
5. Dataset-specific column names must be mapped explicitly into a canonical manifest; do not assume undocumented column headers.

### Proposed canonical sample manifest

Use one row per annotated micro-expression event, with at least these fields:

- `sample_id`: stable unique sample identifier
- `subject_id`: participant identifier used for group-based splits
- `video_path`: local path to the authorised video/sample
- `native_label`: unmodified source label
- `onset_frame`: onset frame index when available
- `apex_frame`: apex frame index when available
- `offset_frame`: offset frame index when available
- `fps`: recorded frame rate when known
- `action_units`: optional source AU annotations
- `source_dataset`: for example, `CASME_II`

Manifest validation must reject missing IDs/labels, invalid frame ordering, invalid FPS, and missing video paths at the point where video access is required. Do not silently invent missing annotation values.

## 4. Preprocessing and temporal representation

### Data path

Authorised video/sample → frame decode → dataset annotation/sample boundary → face alignment/ROI extraction → temporal resampling or a justified temporal representation → MobileNetV3-Large frame features → temporal aggregation/classification → facial evidence output.

### Decisions to evaluate

- Prefer official pre-cropped/aligned dataset inputs if supplied and their usage terms allow it; otherwise document the face detector, alignment strategy, and failure handling. The Haar Cascade currently in the repository is only a replaceable engineering reference.
- Preserve the source FPS and onset/apex/offset metadata. A fixed number of frames must not erase the distinction between source frame rate and elapsed time.
- Compare a documented fixed-length temporal resampling method with the current sliding-window approach before selecting one for event-level recognition. Temporal interpolation to a fixed sequence length is used in micro-expression literature, but the target length must be justified and reported rather than treated as a proposal requirement.
- Ensure that a single annotated event is associated with a single sample-level label. If multiple windows are generated, define how window predictions are aggregated into one event prediction before evaluation.
- Current ROI output is RGB `uint8` at 224×224. If ImageNet-pretrained TorchVision MobileNetV3-Large weights are used, apply their documented transform/normalisation exactly once. The default ImageNet V2 transform resizes to 232, center-crops to 224, scales pixel values to [0,1], and normalises using means `[0.485, 0.456, 0.406]` and standard deviations `[0.229, 0.224, 0.225]`. Record the selected weight version and preprocessing.

## 5. Model design to investigate

### Backbone

Use MobileNetV3-Large as required by the proposal. An ImageNet-pretrained checkpoint is a candidate initialisation, not evidence that the model already recognises micro-expressions. The model must be adapted and evaluated on the chosen micro-expression labels.

### Temporal recognition

Keep the temporal head as an experimental decision until supported by comparison. Candidate baselines include:

1. Frozen MobileNetV3-Large frame embeddings plus a simple, documented temporal pooling/classifier baseline.
2. A fine-tuned MobileNetV3-Large backbone plus a compact temporal model, selected only after an initial baseline and leakage-safe evaluation.

Report the exact backbone, weights, trainable layers, temporal head, optimiser, learning rate, batch size, epoch/early-stopping rule, random seeds, software versions, and checkpoint-selection rule. Given the dataset's small size and imbalance, monitor overfitting and keep model selection within the training subjects of each evaluation fold.

Facial asymmetry, emotional incongruence, and valence/arousal should be added as separately defined feature branches with their own measurement method, label/source justification, and ablation evaluation. Do not infer a clinical state directly from a facial emotion category.

## 6. Evaluation protocol

### Main split

Use leave-one-subject-out cross-validation (LOSO-CV) as the primary subject-independent evaluation protocol. Each fold holds out every sample from one participant for testing and uses the other participants for training/validation. Do not randomly split overlapping frames or windows from the same event/subject across train and test.

Model selection, normalisation fitting, feature selection, threshold selection, and any augmentation-policy tuning must use only the training subjects in each fold. Keep the held-out subject unseen until fold evaluation.

### Metrics

Report at minimum:

- Unweighted Average Recall (UAR / balanced recall)
- Unweighted F1-score (UF1) or macro F1, with the exact definition stated
- Overall accuracy as a secondary measure
- Per-class precision, recall, F1 and support
- Confusion matrix and results averaged across subject-held-out folds

Because classes are imbalanced, accuracy alone is insufficient. Clearly report the number of samples per class and per subject included in each experiment. Compare the reported protocol and label mapping with the cited baseline before claiming direct comparability.

### Baselines and ablations

1. Reproduce a clearly specified simple baseline before adding complexity.
2. Compare temporal representation choices under the same subject folds.
3. Compare pretrained/frozen versus fine-tuned backbone settings where compute permits.
4. Add the proposal's asymmetry, incongruence, and valence/arousal branches one at a time and report their incremental contribution.
5. Do not claim superiority unless the same dataset version, label mapping, split, and metrics are comparable.

## 7. Output contract and safety boundary

Until a valid, evaluated model and component methodology are implemented:

- `score.value = null`
- `score.normalized_value = null`
- `confidence = null`
- `risk_level = null`
- retain `unit = facial_emotional_evidence_pending` or another explicitly approved non-risk evidence unit

A recognised facial-expression class is not itself a mental-health diagnosis or calibrated mental-health risk. Do not create a final mental-health score or fusion weight inside C3. The shared contract and fusion engine remain separate integration boundaries.

## 8. Incremental implementation plan

1. Add this research specification and review it against the proposal.
2. Implement a canonical manifest schema/parser and tests without requiring the dataset or PyTorch.
3. Add a dataset adapter for the actual authorised CASME II annotation layout after confirming the supplied metadata format.
4. Implement sample-level temporal resampling with tests using synthetic frame arrays; preserve sample and subject IDs.
5. Add and pin a compatible PyTorch/TorchVision environment; smoke-test MobileNetV3-Large loading and documented preprocessing.
6. Implement feature extraction behind the existing `FacialFeatureExtractor` interface.
7. Add a reproducible LOSO training/evaluation entry point, baselines and metrics.
8. Only after experimental evaluation, define how model outputs map into the component's non-clinical evidence output and shared contract.

Every milestone must preserve the current shared API contract and pass the full project test suite before committing to `dev`.

## 9. References

1. Yan, W.-J. et al. (2014). *CASME II: An Improved Spontaneous Micro-Expression Database and the Baseline Evaluation.* PLOS ONE 9(1): e86041. https://doi.org/10.1371/journal.pone.0086041
2. Zhou, L., Shao, X. and Mao, Q. (2021). *A survey of micro-expression recognition.* Image and Vision Computing, 105, 104043. https://doi.org/10.1016/j.imavis.2020.104043
3. PyTorch/TorchVision documentation. *mobilenet_v3_large.* https://docs.pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v3_large.html

---

**Implementation note:** This document is a proposed engineering/research plan. It does not alter the proposal's mandated components, claim a clinically valid system, or declare any untested architecture to be the final method.

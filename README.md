# Component 3 — Facial Emotion and Micro-Expression Analysis

**Project:** AI-Powered Holistic Mental Health Assessment and Monitoring Framework  
**Project ID:** J26-IT-402  
**Component:** Component 3 — Facial Emotion and Micro-Expression Analysis

---

## 1. Introduction

The Facial Emotion and Micro-Expression Analysis component is designed to explore facial movements and temporal facial patterns as one source of non-verbal evidence within the AI-Powered Holistic Mental Health Assessment and Monitoring Framework.

The component focuses on analysing facial image sequences and video frames, extracting facial regions, constructing temporal sequences, and obtaining visual feature representations using deep learning techniques.

The research also explores micro-expression-related facial movement patterns using the CASME II dataset and optical-flow-based feature extraction.

This component is designed to contribute structured facial evidence to the wider framework through a shared backend interface.

**Research Consideration:** Facial expressions and micro-expressions are indirect, context-dependent signals. They cannot independently establish a person's emotional state or diagnose a mental-health condition. Dataset classification performance must not be interpreted as clinical mental-health assessment accuracy.

## 2. Objectives

The main objectives of this component are:

- To analyse facial image and video sequences for facial movement patterns.
- To detect and extract facial regions from individual frames.
- To preprocess facial regions into a consistent image representation.
- To construct temporal sequences while preserving frame order.
- To explore MobileNetV3-Large for facial visual feature extraction.
- To investigate micro-expression-related features using optical-flow techniques.
- To evaluate research models using appropriate subject-independent evaluation methods.
- To provide structured facial evidence through the shared backend score interface.

## 3. Processing Architecture

The general research workflow consists of the following stages:

1. **Input Acquisition:** Obtain facial image sequences or video frames.
2. **Face Detection:** Detect facial regions within individual frames.
3. **Face Selection:** Select an acceptable face candidate for each frame.
4. **ROI Extraction:** Crop and preprocess the selected facial region.
5. **Temporal Sequence Construction:** Organize processed frames into fixed-length temporal windows.
6. **Feature Extraction:** Extract visual representations using MobileNetV3-Large.
7. **Micro-Expression Research:** Explore temporal facial movement representations and optical-flow-based features.
8. **Evaluation:** Assess model performance using a documented subject-independent research protocol.
9. **Backend Integration:** Represent available component evidence using the shared score contract.

The processing stages are separated to support independent testing, research evaluation, and integration.

## 4. Technology Stack

The component uses the following technologies and libraries:

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| NumPy | Numerical processing and array operations |
| OpenCV | Image processing, video reading, face detection, and optical flow |
| PyTorch | Deep learning model execution |
| TorchVision | MobileNetV3-Large architecture and preprocessing |
| pytest | Automated testing |
| Git and GitHub | Version control and collaboration |

### Dependency Versions

The component dependencies are defined in `pyproject.toml` and `requirements.txt`.

| Dependency | Version |
|---|---|
| Python | `>=3.10,<3.14` |
| NumPy | `2.2.6` |
| OpenCV | `4.14.0.94` |
| PyTorch | `2.14.1` |
| TorchVision | `0.29.1` |

## 5. Project Structure

The facial-analysis component is organized into the following structure:

```text
facial-analysis/
│
├── src/
│   └── facial_analysis/
│       ├── __init__.py
│       ├── annotation_adapter.py
│       ├── contract.py
│       ├── dataset_manifest.py
│       ├── face.py
│       ├── image_sequence.py
│       ├── model.py
│       ├── pipeline.py
│       ├── sequence.py
│       └── video.py
│
├── tests/
│
├── pyproject.toml
└── requirements.txt
```

### Module Responsibilities

| Module | Description |
|---|---|
| `face.py` | Face detection, face selection, and facial ROI extraction |
| `image_sequence.py` | Reading and naturally sorting image-sequence frames |
| `video.py` | Reading video frames and deterministic frame sampling |
| `sequence.py` | Construction of fixed-length temporal facial sequences |
| `model.py` | MobileNetV3-Large preprocessing and visual feature extraction |
| `pipeline.py` | Facial sequence preparation and feature extraction pipeline |
| `dataset_manifest.py` | Canonical dataset metadata structure and validation |
| `annotation_adapter.py` | Mapping dataset-specific annotation tables to a canonical format |
| `contract.py` | Building the facial component score using the shared score interface |
| `__init__.py` | Exposing the component's public functions and classes |

## 6. Environment Setup and Installation

### Prerequisites

Before using the component, install:

- Python 3.10 or a compatible version within the project's supported range.
- Git.
- Required Python dependencies.
- PyTorch and TorchVision compatible with the selected execution environment.

### Step 1: Clone the Repository

```powershell
git clone https://github.com/Rashminavo/J26-IT-402.git

cd J26-IT-402
```

### Step 2: Create a Virtual Environment

```powershell
python -m venv .venv

.\.venv\Scripts\Activate.ps1
```

### Step 3: Upgrade pip

```powershell
python -m pip install --upgrade pip
```

### Step 4: Install the Facial Analysis Component

From the repository root:

```powershell
python -m pip install -e .\components\facial-analysis
```

Alternatively, install the dependencies directly:

```powershell
python -m pip install -r .\components\facial-analysis\requirements.txt
```

### Step 5: Verify the Installed Libraries

```powershell
python -c "import numpy, cv2, torch, torchvision; print('NumPy:', numpy.__version__); print('OpenCV:', cv2.__version__); print('PyTorch:', torch.__version__); print('TorchVision:', torchvision.__version__); print('CUDA Available:', torch.cuda.is_available())"
```

The MobileNetV3-Large feature extractor uses the TorchVision `IMAGENET1K_V2` pretrained weight definition. If the weights are not cached locally, network access may be required during the first model initialization.

The model supports CPU execution. CUDA execution can be selected when the installed PyTorch build and hardware support it.

## 7. Face Detection and Facial ROI Extraction

Face detection identifies facial regions within input frames.

The `face.py` module provides the following main functions:

| Function | Responsibility |
|---|---|
| `load_face_detector()` | Loads the OpenCV Haar Cascade detector |
| `detect_faces()` | Detects faces in an OpenCV BGR frame |
| `select_face_box()` | Selects an acceptable face bounding box |
| `extract_face_roi()` | Extracts and preprocesses the selected face region |

### Face Bounding Box Format

The component represents a detected face using:

```text
(x, y, width, height)
```

Where:

- `x` represents the horizontal starting coordinate.
- `y` represents the vertical starting coordinate.
- `width` represents the bounding-box width.
- `height` represents the bounding-box height.

### Facial ROI Preprocessing

The ROI extraction process performs the following operations:

1. Crop the selected facial region.
2. Convert the OpenCV BGR image into RGB format.
3. Resize the facial region to 224 × 224 pixels.

Expected ROI representation:

```text
Image Shape: (224, 224, 3)
Colour Format: RGB
Data Type: uint8
```

### Face Selection

The component includes a face-selection mechanism to handle multiple detected face candidates.

The selection logic uses a provisional largest-to-second-largest bounding-box area ratio. When multiple detections cannot be separated using the configured heuristic, an `AmbiguousFaceDetectionError` is raised.

The component also provides `FaceNotDetectedError` for frames where an acceptable face cannot be identified.

**Important:** Haar Cascade is used as a replaceable engineering detector in the reference pipeline. It is not presented as a final research detector prescribed by the project proposal. The face-area ratio is an engineering heuristic and has not been validated as a general face-tracking method.

## 8. Video and Image Sequence Processing

The component supports reading facial input through video files and ordered image sequences.

### Video Frame Reading

The `video.py` module provides:

- `calculate_sample_indices()`
- `read_video_frames()`

These functions support deterministic frame selection and video reading.

Example:

```python
from facial_analysis.video import read_video_frames

frames = read_video_frames(
    video_path="path/to/input_video.mp4",
    frame_stride=1,
    max_frames=None
)
```

Frames returned by OpenCV are in BGR format.

### Image Sequence Reading

The `image_sequence.py` module supports:

- JPG
- JPEG
- PNG
- BMP

It uses natural filename sorting to preserve the expected order of numbered image frames.

Example:

```python
from facial_analysis.image_sequence import read_image_sequence_frames

frames = read_image_sequence_frames(
    sequence_dir="path/to/image_sequence",
    frame_stride=1
)
```

The reader processes image files directly inside the specified sample directory rather than recursively searching subdirectories.

## 9. Temporal Sequence Construction

Micro-expressions involve facial movements that occur over time. Therefore, maintaining temporal order between frames is an important part of the research workflow.

The `sequence.py` module provides:

- `build_temporal_sequences()`
- `sequence_shape()`

The temporal sequence builder organizes facial ROIs into fixed-length windows.

### Default Temporal Parameters

| Parameter | Value |
|---|---:|
| Sequence Length | 16 frames |
| Sequence Stride | 8 frames |
| ROI Width | 224 pixels |
| ROI Height | 224 pixels |
| Colour Channels | 3 |
| Colour Format | RGB |

Expected temporal sequence shape:

```text
(16, 224, 224, 3)
```

Example:

```python
from facial_analysis.sequence import build_temporal_sequences

sequences = build_temporal_sequences(
    frames=face_rois,
    sequence_length=16,
    stride=8
)
```

The sequence builder advances using the selected stride. It also supports an end-aligned window when the regular windows do not include the final frame.

The pipeline includes `prepare_face_sequence()` for converting frame and bounding-box pairs into temporal facial ROI sequences.

It also provides `prepare_face_sequence_from_directory()` for preparing sequences directly from an image-sequence directory.

Missing or ambiguous face detections should not be silently removed because this could alter the temporal relationships between frames.

## 10. MobileNetV3-Large Feature Extraction

MobileNetV3-Large is used as a pretrained deep-learning feature extractor for facial image representations.

The implementation is provided in `model.py`.

The component uses:

```text
Model: MobileNetV3-Large
Pretrained Weights: IMAGENET1K_V2
Input Size: 224 × 224
Feature Dimension: 960
```

### Input Requirements

The feature extractor expects:

- A NumPy array.
- RGB colour format.
- `uint8` pixel values.
- Three colour channels.
- A spatial resolution of 224 × 224 pixels.
- A temporal input layout of `(T, 224, 224, 3)`.

### Preprocessing

The component uses the official TorchVision preprocessing transformation associated with the selected pretrained weights.

The transformation performs:

1. Image resizing.
2. Centre cropping.
3. Conversion to the normalized tensor representation.
4. ImageNet mean and standard-deviation normalization.

The transformed model input has the following shape:

```text
(T, 3, 224, 224)
```

The transformed output is `float32`. It should not be normalized a second time.

### Feature Extraction Output

The MobileNetV3-Large feature extractor obtains one 960-dimensional visual feature vector for each input frame.

```text
Input:
(T, 224, 224, 3)

Output:
(T, 960)

Data Type:
float32
```

Example:

```python
from facial_analysis.model import MobileNetV3LargeFeatureExtractor

extractor = MobileNetV3LargeFeatureExtractor(
    device="cpu"
)

features = extractor.extract(sequence)

print(features.shape)
print(features.dtype)
```

The `extract_facial_sequence_features()` function in `pipeline.py` can also process multiple temporal windows through the feature-extractor interface.

**Research distinction:** The output of MobileNetV3-Large is a visual feature representation. These features are not directly emotion labels, micro-expression probabilities, valence/arousal predictions, or mental-health risk scores. The feature extractor preserves frame order but does not itself perform temporal classification or temporal aggregation.

The `analyze_facial_sequence()` function prepares temporal facial sequences and returns processing information. It does not execute the pretrained model or generate emotion or risk predictions.

## 11. Dataset Management and Annotation Handling

The component includes a canonical dataset manifest structure to support consistent metadata handling across facial-expression datasets.

The relevant modules are:

- `dataset_manifest.py`
- `annotation_adapter.py`

### Canonical Dataset Fields

| Field | Description |
|---|---|
| `source_dataset` | Dataset name |
| `sample_id` | Sample identifier |
| `subject_id` | Subject identifier |
| `video_path` | Relative video path |
| `native_label` | Original dataset label |
| `onset_frame` | Onset frame annotation |
| `apex_frame` | Apex frame annotation |
| `offset_frame` | Offset frame annotation |
| `fps` | Frames per second |
| `action_units` | Action-unit information |

The manifest validator checks required metadata, relative video paths, frame values, and temporal annotation consistency.

The required temporal relationship is:

```text
onset_frame <= apex_frame <= offset_frame
```

The onset, apex, and offset frame values must either all be provided or all be blank.

### Annotation Adapter

The annotation adapter maps source-specific annotation table columns to the canonical manifest fields.

This approach avoids assuming that different datasets or workbooks use identical column names.

The adapter preserves native dataset labels. Any class mapping required for a particular research experiment should be documented separately as part of that experiment's evaluation protocol.

The manifest module validates metadata. It does not download datasets, decode videos, or produce model predictions.

## 12. CASME II Dataset and Research Evaluation

The component uses the CASME II dataset as a research resource for studying spontaneous facial micro-expressions.

The research workflow considers subject identities, sample identifiers, frame annotations, and facial movement information when preparing the dataset for evaluation.

### Dataset Preparation

The selected CASME II research subset used in the experiments contains:

| Dataset Information | Count |
|---|---:|
| Subjects | 26 |
| Selected Sequences | 255 |
| Image Frames | 17,124 |

The selected sequence frame counts range from 24 to 141 images.

Dataset annotations are associated with samples using the subject and sequence identifiers.

The original dataset labels are retained separately from any experiment-specific class mapping.

### Optical-Flow Feature Extraction

Farneback optical flow is explored to describe facial movement between consecutive frames.

The optical-flow experiment uses a 6 × 6 spatial grid to organize movement features.

The resulting feature representation contains:

```text
Feature Dimension: 432
```

This optical-flow representation is a separate research feature-extraction approach from the MobileNetV3-Large visual feature extractor.

### Subject-Independent Evaluation

A nested subject-independent class-weighted RBF Support Vector Machine (SVM) evaluation was conducted using the optical-flow feature representation.

The evaluated subset contains 243 samples.

The evaluation produced the following results:

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 46.50% |
| Unweighted Average Recall (UAR) | 45.19% |
| Macro F1-score | 47.57% |

### Evaluated Class Distribution

| Class | Number of Samples |
|---|---:|
| Disgust | 61 |
| Happiness | 32 |
| Others | 99 |
| Repression | 26 |
| Surprise | 25 |
| **Total** | **243** |

These results represent one specific dataset experiment under its subject-independent nested evaluation protocol.

They must not be interpreted as:

- Clinical mental-health assessment accuracy.
- A validated depression or anxiety detection score.
- Real-world emotion recognition performance across all populations.
- The performance of the backend facial analysis endpoint.

For reproducibility, evaluation reports should retain the data preparation rules, subject-independent splitting protocol, class mapping, preprocessing decisions, exclusions, and generated evaluation artifacts.

## 13. Backend Integration

The facial-analysis component communicates with the central backend using the shared component score interface.

The integration contract is implemented in:

```text
integration/score-interface/score-schema.json
```

The facial score builder is provided through:

```text
facial_analysis.contract.build_facial_score()
```

### Component Identifier

```text
facial_analysis
```

### Backend API Endpoint

```http
POST /api/v1/facial/analyze
Content-Type: application/json
```

### Expected Input Fields

The facial endpoint accepts JSON containing:

| Field | Description |
|---|---|
| `frames` | Facial frame information |
| `bounding_boxes` | Face bounding boxes associated with frames |
| `observation_id` | Observation identifier |
| `user_id` | User identifier |
| `session_id` | Optional session identifier |

The frame and bounding-box collections must correspond to each other. The exact frame representation must follow the backend request contract.

The backend facial endpoint expects frame and bounding-box information in JSON. It is not a raw multipart image or video upload endpoint. A separate upload interface must prepare the frames and bounding boxes before submitting the API request.

### Shared Score Structure

The shared score contract provides fields for:

- Schema version.
- Observation ID.
- User ID.
- Session ID.
- Component identifier.
- Timestamp.
- Data status.
- Score information.
- Confidence.
- Risk information.
- Data-quality information.
- Model metadata.
- Processing metadata.

The facial evidence unit is:

```text
facial_emotional_evidence_pending
```

A score, normalized score, confidence, or risk level must remain null when the corresponding validated output is unavailable.

The availability of facial frames or temporal sequences does not automatically represent a validated emotion or mental-health prediction.

## 14. Testing

The component includes automated tests for verifying its processing functions, data contracts, and input validation rules.

### Run Facial Component Tests

From the project root:

```powershell
python -m pytest components/facial-analysis/tests -q
```

### Run the Complete Project Test Suite

```powershell
python -m pytest -q
```

The test suite covers software behaviour represented by the test cases, including:

- Face detection input validation.
- Face bounding-box selection.
- Facial ROI extraction.
- Temporal sequence construction.
- Video frame reading.
- Image-sequence processing.
- MobileNetV3 preprocessing.
- Feature output validation.
- Dataset manifest validation.
- Annotation adapter behaviour.
- Facial score contract validation.

Passing automated tests verifies the tested software behaviour. It does not establish clinical validity or guarantee mental-health prediction accuracy.

## 15. Research Limitations and Ethical Considerations

Facial emotion and micro-expression analysis involves several research and ethical limitations.

### Dataset Limitations

Research datasets may have limitations related to:

- Dataset size and class distribution.
- Subject diversity.
- Recording conditions.
- Lighting and camera quality.
- Annotation consistency.
- Generalization to real-world environments.

### Facial Analysis Limitations

Facial movements are influenced by individual differences, culture, context, environment, and recording conditions.

A detected facial expression does not necessarily represent a person's actual emotional state.

### Privacy and Data Protection

Facial images and videos may contain sensitive personal information.

The following principles should be considered:

- Obtain appropriate permission before collecting or processing facial data.
- Protect participant information and research records.
- Do not commit raw participant images or videos to the public repository.
- Follow CASME II research-use and non-commercial restrictions.
- Do not publicly redistribute restricted subject images.
- Keep datasets and generated experiment outputs separate from the source-code repository.
- Preserve historical research results and evaluation artifacts.

### Clinical Limitations

This component is intended for academic research and software development.

It is not a medical diagnostic tool and should not independently be used to identify or diagnose mental-health conditions.

CASME II classification results should not be presented as clinical mental-health assessment accuracy.

## 16. Future Research Directions

Possible research directions for this component include:

- Exploring more suitable face-detection and tracking approaches.
- Investigating temporal deep-learning architectures.
- Evaluating additional micro-expression feature representations.
- Studying facial movement asymmetry and temporal incongruence.
- Investigating valence and arousal evidence with appropriate validation.
- Improving robustness under different recording conditions.
- Conducting additional subject-independent evaluation experiments.
- Studying the reliability of facial evidence before broader system integration.

Any new model, threshold, score, or evaluation approach should be supported by an appropriate research protocol and clearly documented experimental evidence.

## 17. Related Project Resources

**Main Repository:**  
https://github.com/Rashminavo/J26-IT-402

**Shared Score Schema:**  
`integration/score-interface/score-schema.json`

**Backend Facial Endpoint:**  
`POST /api/v1/facial/analyze`

**Component Directory:**  
`components/facial-analysis/`

---

*Component 3 — Facial Emotion and Micro-Expression Analysis | J26-IT-402*

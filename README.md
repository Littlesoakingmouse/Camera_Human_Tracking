# Multi-Camera Person Tracking Baseline

A complete offline baseline for tracking people within each camera and assigning the
same **global ID** to visually matching tracks across cameras. The project is intended
for an understandable university Computer Vision demo, not for model training or
real-time deployment.

No command in this repository downloads weights, videos, or datasets. You supply all
three required input files locally.

## Pipeline

```text
configured camera videos
        |
        v
YOLO person detection
        |
        v
ByteTrack local tracking
        |
        v
sampled person crops -> OSNet embeddings
        |
        v
mean + L2 normalization per local track
        |
        v
same-camera tracklet stitching -> stable camera IDs (CID)
        |
        v
cross-camera cosine similarity + time constraint
        |
        v
Hungarian assignment -> conflict-safe global IDs
        |
        v
CSV reports + replayed annotated videos
```

This is deliberately a three-pass design:

1. Process each camera independently and build raw ByteTrack local tracks.
2. Stitch non-overlapping same-camera tracklets into stable camera identities (CID),
   then match those identities across cameras and assign global IDs.
3. Replay the videos and render the stored boxes using the CID/GID mappings.

## Project structure

```text
.
|-- main.py
|-- README.md
|-- requirements.txt
|-- configs/
|   |-- config.yaml
|   `-- bytetrack.yaml
|-- data/
|   |-- raw/                 # put configured camera videos here
|   `-- crops/
|-- models/
|   |-- detector/            # put yolo26m.pt here
|   `-- reid/                # put osnet_ain_x1_0.pth here
|-- scripts/
|   |-- test_detector.py
|   |-- test_tracker.py
|   |-- test_reid.py
|   `-- test_matching.py
|-- src/
|   |-- detector_tracker.py
|   |-- reid.py
|   |-- osnet.py
|   |-- osnet_ain.py
|   |-- track_database.py
|   |-- intra_camera_matcher.py
|   |-- matcher.py
|   |-- global_tracker.py
|   |-- visualizer.py
|   |-- pipeline.py
|   `-- utils/
|       |-- common.py
|       `-- video.py
`-- outputs/
    |-- videos/
    |-- crops/
    `-- logs/
```

CSV files are created directly under `outputs/` when the pipeline runs. Empty model,
video, and output directories are already created locally; large inputs and generated
outputs are excluded by `.gitignore`.

## Requirements and installation

Use Python 3.10 or newer. Python 3.10 or 3.11 is a conservative choice for broad
PyTorch/Ultralytics compatibility.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```bat
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For a particular CUDA version, install the matching PyTorch wheel using the command
from the official PyTorch installer, then install this requirements file. With
`device: auto`, the demo selects CUDA when `torch.cuda.is_available()` is true and CPU
otherwise.

The OSNet-x1.0 and OSNet-AIN-x1.0 network definitions are included in `src/osnet.py`
and `src/osnet_ain.py`. Both follow the state-dictionary layouts used by
torchreid/deep-person-reid. The default configuration uses OSNet-AIN-x1.0, whose
InstanceNorm layers generally improve cross-domain robustness. This local
implementation avoids an extra package and any pretrained-model download behavior.
The checkpoint may be a raw state dictionary or contain `state_dict`,
`model_state_dict`, or `model`; `module.` and `model.` prefixes are accepted.
Classifier weights of a different size are safely ignored.

## Add your files

Place files at these exact default paths:

```text
models/detector/yolo26m.pt
models/reid/osnet_ain_x1_0.pth
data/raw/video3_1.avi
data/raw/video3_2.avi
```

- `yolo26m.pt` must be a local Ultralytics-compatible detection checkpoint.
- `osnet_ain_x1_0.pth` must be a torchreid/deep-person-reid-compatible
  OSNet-AIN-x1.0 checkpoint. A regular OSNet-x1.0 checkpoint is not interchangeable.
- Videos must be readable by the codecs available to your OpenCV build.

You may use different names or locations by changing `detector.model_path`,
`reid.model_path`, and the camera `path` values in `configs/config.yaml`. Paths are
resolved relative to the project root, so the command may be launched from a different
working directory.

## Configuration

Important settings in `configs/config.yaml`:

- `detector.confidence_threshold`: minimum YOLO confidence.
- `detector.person_class_id`: `0` for person in COCO-trained YOLO models.
- `detector.device` and `reid.device`: `auto`, `cpu`, `cuda`, or for example `cuda:0`.
- `tracker.config_path`: local Ultralytics ByteTrack YAML.
- `reid.sample_interval`: extract one crop/embedding every N video frames.
- `reid.min_crop_width` / `min_crop_height`: reject tiny or invalid crops.
- `intra_camera.enabled`: enable or disable same-camera tracklet stitching.
- `intra_camera.similarity_threshold`: minimum appearance similarity for re-entry.
- `intra_camera.max_time_gap_seconds`: maximum time between same-camera segments.
- `intra_camera.time_penalty_weight`: prefer closer re-entries when appearance ties.
- `intra_camera.min_embeddings`: minimum descriptors required before stitching a track.
- `matching.similarity_threshold`: minimum cosine similarity for a candidate match.
- `matching.use_time_constraint`: enable or disable temporal filtering.
- `matching.max_time_gap_seconds`: maximum gap between two non-overlapping tracks.
- `video.*.time_offset_seconds`: added to `frame_index / FPS` for synchronization.
- `output.save_crops`: write sampled crops under `outputs/crops/`.
- `output.save_local_tracking_video`: also produce videos labeled only with local IDs.
- `output.save_global_tracking_video`: produce the primary global-ID videos.

For more cameras, add more entries under `video` with unique `id` values. Matching is
run for every camera pair. Global merging prevents one identity from containing two
tracks from the same camera.

### Choosing the similarity threshold

OSNet produces a feature vector for each sampled crop. Each vector is L2-normalized;
all vectors for a local track are averaged and normalized again. Cosine similarity is
then the dot product of two normalized track vectors. Values near 1 indicate more
similar appearance and lower values indicate less similar appearance.

The configured cross-camera threshold is only a starting point. Run `test_reid.py` on known
same-person and different-person examples, inspect `similarity_matrix.csv`, and choose
a threshold that separates the two distributions for your cameras. Raising the
threshold reduces false matches but creates more unmatched identities; lowering it
does the opposite.

The time constraint uses global timestamps:

```text
global timestamp = frame index / video FPS + camera time offset
```

Overlapping track intervals have a gap of zero. Otherwise the closest interval-edge
distance must be no greater than `max_time_gap_seconds`.

## Run the validation utilities

The matching test needs no models or videos and should be run first:

```bash
python scripts/test_matching.py
```

Test YOLO on the first ten frames:

```bash
python scripts/test_detector.py --camera cam1 --frames 10
```

Check how often each ByteTrack local ID persists:

```bash
python scripts/test_tracker.py --camera cam1 --frames 100
```

Test same-camera re-entry stitching without loading models or videos:

```bash
python scripts/test_intra_camera.py
```

Test Re-ID using three person crop images that you supply:

```bash
python scripts/test_reid.py person_A_cam1.jpg person_A_cam2.jpg person_B.jpg
```

The same-person score should generally be higher than the different-person score.
Real values depend heavily on checkpoint quality and camera domain.

All scripts accept `--config PATH`; the detector/tracker scripts also accept a camera
ID configured in the YAML.

## Run the complete pipeline

From the project directory:

```bash
python main.py --config configs/config.yaml
```

The application validates configured files before inference and never substitutes or
downloads a missing model. It prints camera progress, device selection, track counts,
global-person count, and accepted cross-camera match count.

## Local IDs, camera IDs, and global IDs

ByteTrack assigns a **local ID** independently inside each camera. Therefore `cam1` ID
3 and a later `cam1` ID 12 may be two visits by the same person. Same-camera stitching
can map both tracklets to one stable **camera person ID (CID)**. OSNet and cross-camera
matching then merge CIDs from different cameras into one **global ID (GID)**. A CID
without an accepted cross-camera match receives its own GID.

## Outputs

After a successful default run:

```text
outputs/
|-- videos/
|   |-- cam1_global_tracking.mp4
|   `-- cam2_global_tracking.mp4
|-- crops/
|   `-- <camera>/track_<id>/frame_<index>.jpg
|-- local_tracks.csv
|-- camera_tracks.csv
|-- intra_camera_matches.csv
|-- global_tracks.csv
|-- local_similarity_matrix.csv
`-- similarity_matrix.csv
```

`local_tracks.csv` summarizes each raw LID, its assigned CID, embedding count, and
same-camera stitch similarity. `camera_tracks.csv` summarizes each stable CID and lists
the source LIDs joined into it. `intra_camera_matches.csv` records every accepted
same-camera link with its appearance similarity, time gap, and final score.
`global_tracks.csv` maps every raw LID through its CID to a GID; unmatched CIDs have
a blank `similarity_to_match`.
`local_similarity_matrix.csv` contains diagnostic similarities between raw LIDs,
including same-camera pairs used by the stitching stage.
`similarity_matrix.csv` contains the CID similarities actually used for cross-camera
matching. Same-camera cells and identities without embeddings are intentionally blank.

The video writer preserves each source video's FPS and frame dimensions. Colors are a
deterministic function of identity and remain stable across frames.

## Baseline limitations

This baseline may fail when people wear very similar clothes, undergo severe
occlusion, appear under drastically different viewpoints or lighting, or disappear
for long periods. Poor synchronization and a badly chosen similarity threshold also
produce incorrect assignments. Appearance mean-pooling ignores crop quality, and
pairwise Hungarian matching is not a learned multi-camera association model.

ByteTrack IDs can still switch after occlusion. The stitching stage repairs a switch
only when the segments do not overlap, fall within the configured time gap, contain
enough embeddings, and exceed the same-camera similarity threshold. Similar clothing
can still cause a false stitch; calibrated entry/exit zones are a useful future signal.

## Future work

Natural extensions include camera-topology or NetworkX graphs, learned temporal
transition constraints, quality-weighted embedding aggregation, fine-tuned Re-ID,
three or more calibrated cameras, HOTA/IDF1 evaluation, real-time streams, and a
FastAPI/Streamlit dashboard. Those additions are intentionally outside this clean
offline baseline.

## Troubleshooting

- A missing-file error is expected until all weights/videos are placed at configured
  paths.
- If OpenCV cannot create MP4 output, install a build with MP4 support or change the
  four-character `output.codec` setting to a codec supported on your machine.
- If GPU inference fails, set both device values to `cpu` to verify the pipeline.
- If there are no Re-ID embeddings, reduce crop size limits and confirm that ByteTrack
  is producing IDs. The CSV `num_embeddings` column makes this visible.
- If OSNet reports incompatible parameters, ensure the checkpoint architecture exactly
  matches `reid.architecture`. The supported values are `osnet_ain_x1_0` and
  `osnet_x1_0`; checkpoints from different widths are not interchangeable.

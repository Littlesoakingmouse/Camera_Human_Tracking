# Multi Camera Person Tracking

Tài liệu kỹ thuật cho hệ thống theo dõi người đa camera chạy offline. Dự án nhận video từ nhiều camera, phát hiện người bằng YOLO, theo dõi cục bộ bằng BoT-SORT kết hợp ngoại hình OSNet AIN, sửa các lỗi ID phổ biến, rồi gán một Global ID (GID) cho cùng một người giữa các camera.

Mục tiêu chính không phải là huấn luyện model hay realtime streaming. Đây là pipeline có thể kiểm tra được bằng video và CSV: mỗi quyết định split, merge hoặc từ chối merge đều có dữ liệu để truy vết.

**Phiên bản hiện tại: v3.0.** Độ chính xác và hiệu năng của từng phiên bản được đo trên benchmark HDA có ground truth: xem [docs/EVALUATION.md](docs/EVALUATION.md) (phương pháp), [docs/BENCHMARK_RESULTS.md](docs/BENCHMARK_RESULTS.md) (số liệu), [docs/ABLATION.md](docs/ABLATION.md) (đóng góp từng thay đổi) và [docs/CHANGELOG.md](docs/CHANGELOG.md) (lịch sử).

## Mục lục

1. [Kiến trúc và luồng xử lý](#kiến-trúc-và-luồng-xử-lý)
2. [Dữ liệu và các loại ID](#dữ-liệu-và-các-loại-id)
3. [YOLO và NMS](#yolo-và-nms)
4. [ByteTrack](#bytetrack)
5. [OSNet AIN Re Identification](#osnet-ain-re-identification)
6. [Embedding và cosine similarity](#embedding-và-cosine-similarity)
7. [Tách Local ID bị lẫn người](#tách-local-id-bị-lẫn-người)
8. [Gộp track trong cùng camera](#gộp-track-trong-cùng-camera)
9. [Ghép liên camera và Global ID](#ghép-liên-camera-và-global-id)
10. [Output và cách kiểm tra](#output-và-cách-kiểm-tra)
11. [Cấu hình và cách chạy](#cấu-hình-và-cách-chạy)
12. [Kiểm thử hiệu chỉnh và giới hạn](#kiểm-thử-hiệu-chỉnh-và-giới-hạn)
13. [Đánh giá và phiên bản](#đánh-giá-và-phiên-bản)

## Kiến trúc và luồng xử lý

```text
Video từng camera
    |
    v
YOLO person detection + NMS
    |
    v
OSNet AIN cho mọi detection (một forward mỗi batch frame)
    |
    v
BoT-SORT: IoU + appearance trong từng camera -> LID thô (+ ghi bù frame đầu)
    |
    v
Embedding của track được lấy mẫu theo giây -> EmbeddingSample
    |
    v
Tách LID có appearance thay đổi -> TrackSegment
    |
    v
Gộp re-entry / duplicate overlap / motion link trong cùng camera -> CID
    |
    v
Lọc CID quá ngắn (nhiễu) -> GID 0
    |
    v
Phân cụm agglomerative có ràng buộc, so khớp tập-với-tập -> GID
    |
    v
Replay video, vẽ GID | CID | LID và xuất CSV diagnostics
```

Pipeline chia thành ba pha lớn:

1. **Inference cục bộ:** đọc từng video độc lập, chạy YOLO và OSNet theo batch, cập nhật BoT-SORT theo thứ tự frame và lưu mọi bbox/embedding. Kết quả pha này được cache (`cache.enabled`), nên chỉnh ngưỡng ở pha 2 không cần chạy lại model.
2. **Repair và matching offline:** tách LID bị switch, stitch segment thành CID, ghép CID giữa camera thành GID.
3. **Replay:** đọc lại video gốc, tra CID/GID theo `(camera, LID, frame)` rồi render song song nhiều camera. Pha này không chạy lại model.

### Tối ưu tốc độ

| Kỹ thuật | Vị trí | Lý do |
|---|---|---|
| YOLO batch (`detector.batch_size: 8`) + FP16 | `DetectorTracker.track_batch` | Gọi `model.track()` từng frame bị overhead Python/preprocess chi phối; batch tăng throughput YOLO khoảng 3 lần trên RTX 3050 |
| ByteTrack độc lập | `DetectorTracker.track_batch` | Detection không có state nên chạy batch được; ByteTrack vẫn nhận detection từng frame theo đúng thứ tự, cùng thuật toán với `model.track()` |
| OSNet batch + FP16 | `ReIDModel.extract_embeddings` | Mọi crop của một batch frame đi qua OSNet trong một forward; ảnh uint8 được upload rồi chuẩn hóa trên GPU |
| Prefetch decode | `utils/video.prefetch` | Thread nền đọc video trong khi GPU đang infer |
| Render song song | `output.render_workers` | Replay là decode/draw/encode trên CPU, OpenCV nhả GIL nên thread chạy song song thật |
| Similarity vector hóa | `matcher.similarity_matrix` | Một phép nhân ma trận thay cho vòng lặp cosine từng cặp |

Với FP32 và `batch_size: 1`, track sinh ra giống hệt `model.track()` của Ultralytics (cùng LID, cùng tọa độ). FP16 có thể làm vài box confidence sát ngưỡng thay đổi; đặt `detector.half: false` nếu cần tái lập đúng kết quả FP32.

Mỗi lần chạy in thời gian từng pha (`detect_track_reid`, `repair_and_matching`, `render`) để so sánh.

| Module | Trách nhiệm |
|---|---|
| `src/detector_tracker.py` | Bọc Ultralytics YOLO, NMS, BoT-SORT/ByteTrack, feature OSNet cho tracker, ghi bù frame đầu, tạo `TrackObservation` |
| `src/reid.py` | Tải OSNet/OSNet-AIN, tiền xử lý crop và tạo embedding |
| `src/track_database.py` | Lưu LocalTrack, bbox, EmbeddingSample, descriptor bền vững |
| `src/track_segmenter.py` | Tìm điểm đổi appearance để tách track |
| `src/intra_camera_matcher.py` | Gộp re-entry và duplicate track trong cùng camera |
| `src/matcher.py` | Cosine similarity, ràng buộc thời gian, Hungarian liên camera |
| `src/global_association.py` | v3.0: phân cụm agglomerative có ràng buộc, similarity tập-với-tập |
| `src/global_tracker.py` | v2.x: gán GID bằng union tham lam (dùng khi `matching.method: hungarian`) |
| `src/pipeline.py` | Điều phối toàn bộ pipeline và ghi output |
| `src/visualizer.py` | Vẽ bbox, GID, CID, LID lên video |

## Dữ liệu và các loại ID

| Khái niệm | Khóa | Ý nghĩa | Độ tin cậy |
|---|---|---|---|
| Observation | `(camera, LID, frame)` | Một bbox, confidence, timestamp tại một frame | Chỉ có ý nghĩa tại frame đó |
| LID | `(camera, local_track_id)` | Local ID do ByteTrack cấp | Có thể switch hoặc bị duplicate |
| TrackSegment | `(camera, LID, segment_index)` | Một đoạn thời gian appearance nhất quán của LID | Tốt hơn LID thô |
| CID | `(camera, camera_person_id)` | Danh tính đã được stitch trong một camera | Có thể gồm nhiều segment |
| GID | `global_id` | Danh tính toàn cục giữa các camera | Mỗi GID tối đa một CID mỗi camera |

`EmbeddingSample` không chỉ giữ vector ReID mà còn giữ `frame_index`, `timestamp`, `bbox`, `confidence`, `occlusion` và cờ `reliable`. Thông tin thời gian là bắt buộc để biết embedding nào thuộc phần đầu/phần cuối của LID khi phát hiện ID switch.

```text
timestamp = frame_index / FPS + time_offset_seconds
```

`time_offset_seconds` cho phép đồng bộ tương đối giữa các camera. Nếu offset sai, điều kiện thời gian có thể loại một match đúng hoặc giữ một match sai.

## YOLO và NMS

`DetectorTracker` tải checkpoint local `models/detector/yolo26m.pt` qua API Ultralytics YOLO. Code chỉ phụ thuộc `predict()` và `BYTETracker` của Ultralytics; kiến trúc chính xác nằm trong checkpoint YOLO local. Chỉ class person được xử lý, mặc định class `0` của COCO.

### Confidence threshold

`detector.confidence_threshold` hiện là `0.10`. Ngưỡng thấp này có chủ đích: ByteTrack có association bậc hai với detection confidence thấp. Tăng thẳng lên `0.50` có thể làm mất người thật nếu YOLO chỉ dự đoán confidence khoảng `0.47`.

### IoU và NMS

Mỗi bbox có dạng `B = (x1, y1, x2, y2)`. Intersection over Union là:

```text
IoU(A, B) = area(A intersect B) / area(A union B)
```

NMS sắp box theo confidence, giữ box mạnh nhất, rồi loại box yếu hơn nếu IoU với box đã giữ lớn hơn ngưỡng. Code dùng `detector.iou_threshold`; nếu chưa khai báo trong YAML, default là `0.60`.

Ngưỡng `0.60` được chọn để loại trường hợp hai box cùng người có IoU khoảng `0.6808`, trong khi vẫn giữ các người thật ở vị trí khác nhau. Regression `scripts/test_duplicate_detection.py` kiểm tra frame lỗi `cam5` 3255 phải còn đúng hai detection.

Box sau YOLO được clip vào biên ảnh và chỉ giữ khi `x2 > x1` và `y2 > y1`, tránh crop rỗng hoặc tọa độ không hợp lệ.

## ByteTrack

ByteTrack là tracker tracking-by-detection. Mỗi camera reset tracker state trước khi bắt đầu để LID tuyệt đối không đi từ camera này sang camera khác.

ByteTrack thường liên kết track đang tồn tại với detection confidence cao trước, sau đó dùng detection confidence thấp còn lại để cứu các track khó. Trong project này, ByteTrack **không dùng embedding OSNet trong vòng lặp tracking**; embedding chỉ được dùng ở phần repair offline.

| Tham số | Giá trị | Ý nghĩa |
|---|---:|---|
| `track_high_thresh` | 0.25 | Detection mạnh cho association chính |
| `track_low_thresh` | 0.10 | Detection yếu vẫn được xét |
| `new_track_thresh` | 0.25 | Điều kiện tạo LID mới |
| `track_buffer` | 45 frame | Giữ track ngắn khi mất detection; tránh nối lại quá lâu |
| `match_thresh` | 0.70 | Association motion/IoU chặt hơn |
| `fuse_score` | true | Kết hợp score detection trong association |

ByteTrack tốt cho chuyển động liên tục nhưng không phải định danh lâu dài. Occlusion, hai người đứng gần nhau, duplicate detection và người rời rồi quay lại đều có thể làm LID switch/duplicate. Do đó LID là dữ liệu thô, không phải identity cuối cùng.

### BoT-SORT với OSNet (v3.0, mặc định)

ByteTrack chỉ dùng IoU giữa box dự đoán và detection. Khi người di chuyển xa giữa hai frame (frame rate thấp, đi nhanh, camera gần), IoU về 0 và track bị đứt hoặc đổi ID. Trên benchmark HDA (khoảng 2 FPS), YOLO tìm thấy 90–98% người nhưng chỉ 31–80% còn lại sau ByteTrack.

`configs/botsort.yaml` dùng BoT-SORT của Ultralytics và nạp feature OSNet của chính project:

```text
dist_iou(t, d)  = 1 - IoU(t, d)
dist_app(t, d)  = (1 - cos(f_t, f_d)) / 2      chỉ xét khi IoU >= proximity_thresh
                  bị loại nếu cos < 2 * appearance_thresh - 1
cost(t, d)      = min(dist_iou, dist_app)
```

`DetectorTracker` crop mọi detection của một batch frame, chạy OSNet một lần rồi đưa feature vào `tracker.update(..., feats=...)`. Feature của detection được gán vào `TrackObservation.embedding` và dùng lại làm `EmbeddingSample`, nên OSNet chỉ chạy một lần cho mỗi detection; pipeline không crop lại.

| Tham số (`botsort.yaml`) | Giá trị | Ý nghĩa |
|---|---:|---|
| `proximity_thresh` | 0.10 | Chỉ dùng appearance cho cặp track–detection còn chồng nhau ít nhất 10% IoU |
| `appearance_thresh` | 0.75 | Cosine tối thiểu 0.5 để appearance được tính |
| `match_thresh` | 0.80 | Cost tối đa để chấp nhận một cặp |
| `gmc_method` | none | Camera tĩnh, không cần bù chuyển động camera |

`tracker.config_path: configs/bytetrack.yaml` quay lại hành vi v2.x.

### Ghi bù frame đầu của track

Ultralytics tạo track mới ở trạng thái *chưa xác nhận* và chỉ trả về từ frame nó được khớp lần thứ hai, nên **detection đầu tiên của mỗi người luôn bị bỏ**. Với `tracker.backfill_first_frame: true`, `DetectorTracker` nhớ box chưa xác nhận và trả lại nó (qua `drain_backfilled()`) khi track được xác nhận; pipeline chèn box vào đúng frame của nó. Track không bao giờ được xác nhận vẫn bị bỏ như cũ, nên không thêm false positive.

## OSNet AIN Re Identification

`ReIDModel` dùng `osnet_ain_x1_0` với embedding 512 chiều. Input crop BGR của OpenCV được đổi sang RGB, resize về `128 x 256`, đổi sang float `[0, 1]`, chuẩn hóa ImageNet và infer trong `torch.inference_mode()`.

```text
x_rgb = resize(BGR_to_RGB(crop), 128, 256) / 255
x_norm[c, h, w] = (x_rgb[c, h, w] - mean[c]) / std[c]

mean = (0.485, 0.456, 0.406)
std  = (0.229, 0.224, 0.225)
```

Crop được gom thành batch (`reid.batch_size`), infer FP16 trên CUDA (`reid.half`). Với `reid.flip_augmentation: true`, mỗi crop còn được infer bản lật ngang và cộng hai feature trước khi L2 normalize:

```text
e = L2(f(x) + f(flip(x)))
```

Người đi trái sang phải và phải sang trái cho embedding gần nhau hơn, giúp ổn định so khớp giữa camera có hướng di chuyển ngược nhau. Embedding FP16 so với FP32 có cosine tối thiểu khoảng `0.9999`.

### Chất lượng crop

Crop bị người khác che hoặc bị cắt bởi biên ảnh chứa nền/người khác và làm embedding sai lệch. Tại mỗi frame được lấy mẫu, `crop_quality` tính:

```text
occlusion(A) = max_B area(A intersect B) / area(A)   với B có y2_B >= y2_A
truncated(A) = A chạm biên ảnh trong khoảng border_margin pixel
reliable(A)  = occlusion(A) <= max_occlusion và không truncated(A)
```

Với camera đặt cao nhìn xuống sàn, người có cạnh dưới bbox thấp hơn là người đứng phía trước, nên chỉ người phía sau bị tính là bị che. Sample không reliable vẫn được lưu (và vẫn thuộc segment theo frame), nhưng descriptor và phát hiện ID switch chỉ dùng sample reliable khi có đủ; nếu track quá ít sample sạch thì fallback dùng tất cả.

### OSNet là gì

OSNet là Omni Scale Network cho Person ReID. Một `OSBlock` dùng nhiều nhánh LightConv 3x3 với độ sâu khác nhau để nhìn đặc trưng ở nhiều receptive field: chi tiết nhỏ như màu áo, và cấu trúc lớn như dáng người. `ChannelGate` học trọng số theo kênh để ưu tiên thang đo hữu ích. Nhánh residual giúp giữ thông tin đầu vào và tối ưu ổn định.

### Vì sao dùng AIN

AIN là OSNet có thêm Instance Normalization tại các vị trí chọn lọc. Với feature map `x`, InstanceNorm chuẩn hóa riêng từng ảnh/từng kênh:

```text
y = gamma * (x - mu) / sqrt(sigma^2 + epsilon) + beta
```

Điều này giảm ảnh hưởng style riêng của camera như ánh sáng, màu nền và độ tương phản, nên thường phù hợp hơn khi so người giữa nhiều camera.

### Tải checkpoint an toàn

Code hỗ trợ checkpoint raw hoặc bọc trong `state_dict`, `model_state_dict`, `model`; tự bỏ prefix `module.` và `model.`. Tensor chỉ được load nếu trùng tên và shape. Layer classifier sai kích thước được bỏ qua, nhưng tối thiểu 80% tensor backbone phải tương thích. Không thể thay checkpoint OSNet thường bằng chỉ sửa path trong khi vẫn để `architecture: osnet_ain_x1_0`; cần đổi architecture khớp checkpoint.

## Embedding và cosine similarity

Mỗi embedding được L2 normalize:

```text
L2(v) = v / max(||v||_2, epsilon)
```

Cosine similarity của hai vector là:

```text
cos(a, b) = (a dot b) / (||a||_2 * ||b||_2)
```

Sau L2 normalization, cosine similarity bằng dot product. Giá trị gần `1` nghĩa là appearance gần nhau; giá trị thấp không tự động chứng minh khác người vì góc nhìn, crop, occlusion và ánh sáng cũng làm score giảm.

### Robust descriptor

Descriptor LocalTrack thô là trung bình embedding rồi L2 normalize. Descriptor TrackSegment/CID dùng medoid để giảm ảnh hưởng crop lỗi hoặc sample bị lẫn người. Cả hai chỉ dùng sample `reliable` nếu có ít nhất 2 sample như vậy:

```text
j* = argmax_j mean_i cos(e_j, e_i)
giữ e_i nếu cos(e_j*, e_i) >= tau_outlier
d = L2(mean các e_i được giữ)
quality = median_i cos(e_i, d)
```

`tau_outlier` mặc định là `0.55`. `retained_embeddings` là số sample được giữ, còn `descriptor_quality` là median similarity giữa sample được giữ và descriptor cuối. Quality là chỉ số chẩn đoán, không phải xác suất đúng/sai.

## Tách Local ID bị lẫn người

`TrackSegmenter` giải quyết trường hợp một LID ByteTrack đổi từ người A sang người B. Nó sắp embedding theo frame, so sánh cửa sổ trước/sau thay vì lấy mean của cả LID.

Default của code khi YAML chưa khai báo `tracklet_split`:

| Tham số | Giá trị | Vai trò |
|---|---:|---|
| `window_size` | 4 embedding | Số sample mỗi phía của một điểm kiểm tra |
| `change_similarity_threshold` | 0.65 | Cắt khi hai cửa sổ có cosine thấp hơn ngưỡng |
| `min_separation_frames` | 20 frame | Hai boundary phải cách nhau đủ xa |
| `min_segment_embeddings` | 4 | Hai bên boundary cần đủ embedding |
| `outlier_similarity_threshold` | 0.55 | Lọc sample xa medoid khi tạo descriptor segment |

```text
L_k = L2(mean(e[k-w], ..., e[k-1]))
R_k = L2(mean(e[k], ..., e[k+w-1]))
s_k = L_k dot R_k

Tạo candidate boundary khi s_k < 0.65
```

Các vị trí low-similarity liên tiếp là một transition, nên code chỉ giữ vị trí có score nhỏ nhất. Candidate bị loại nếu thiếu embedding ở một trong hai phía hoặc gần boundary trước hơn 20 frame. Boundary frame là trung điểm giữa hai embedding kề nhau. Từ đó cùng LID có thể nhận CID/GID khác nhau trước và sau thời điểm switch.

Chuỗi `e` dùng để tìm boundary chỉ gồm sample reliable khi track có ít nhất `2 * max(window_size, min_segment_embeddings)` sample sạch. Lý do: khi người A bị người B che vài giây, crop của A giống B và tạo ra một "switch" giả A -> B -> A, cắt track thành 3 đoạn. Sau khi tìm boundary, mọi sample (kể cả sample bị che) được gán vào segment theo frame.

## Gộp track trong cùng camera

`IntraCameraMatcher` tạo Camera ID (CID) từ TrackSegment. Matcher xử lý từng camera riêng, chọn cạnh hợp lệ bằng Hungarian, sau đó union-find gộp các segment thuộc cùng CID.

### Re entry không chồng thời gian

Đây là trường hợp một người rời camera rồi quay lại với LID khác. Điều kiện default:

- Có embedding và ít nhất 2 embedding mỗi segment.
- Appearance similarity >= `0.78`.
- Gap thời gian <= `120` giây.
- Hai segment không overlap.

```text
gap = start_time_late - end_time_early
score_reentry = similarity - 0.05 * gap / 120
```

Time penalty chỉ ưu tiên re-entry gần hơn khi appearance tương tự, không biến một pair dưới ngưỡng thành match.

### Motion link (v3.0)

Khi tracker mất người vài frame rồi tạo LID mới, hai đoạn nằm sát nhau về thời gian và vị trí. Nhiều đoạn quá ngắn để có embedding nên bước re-entry bỏ qua chúng. Motion link nối các đoạn này dựa trên chuyển động:

```text
v          = (center_cuối - center_đầu) / số frame, trên tối đa 6 box cuối của đoạn trước
predicted  = center_cuối + v * gap
distance   = ||center_đầu_đoạn_sau - predicted|| / đường chéo box
link nếu   1 <= gap <= motion_max_gap_frames (30)
           và distance <= motion_max_center_ratio (0.75)
           và chiều cao hai box chênh không quá 1.5 lần
           và (thiếu embedding hoặc appearance >= motion_min_similarity (0.50))
```

Hai segment của **cùng một LID** không bao giờ được motion link, vì chúng bị tách có chủ đích do đổi người. Score của motion link cao hơn mọi score re-entry, nên Hungarian ưu tiên nối liền mạch hơn nối theo ngoại hình.

### Duplicate overlap

Đây là trường hợp cùng một người bị tạo hai LID chạy song song, thường do duplicate detection. Khác với re-entry, overlap được phép nhưng phải thỏa đồng thời tất cả điều kiện dưới đây để không merge hai người thật đứng cạnh nhau.

| Điều kiện | Default | Ý nghĩa |
|---|---:|---|
| Appearance similarity | >= 0.82 | Descriptor phải rất gần nhau |
| Shared frames | >= 2 | Không quyết định từ một frame đơn lẻ |
| Overlap frames | <= 45 | Chỉ sửa duplicate ngắn |
| Overlap / track ngắn hơn | <= 0.30 | Không gộp hai track dài cùng tồn tại |
| Median bbox IoU | >= 0.60 | Hai bbox phải gần trùng |
| Tỷ lệ shared frame có IoU >= 0.60 | >= 0.70 | Đa số overlap phải là duplicate |
| Median center distance / diagonal nhỏ | <= 0.25 | Loại hai người ở hai vị trí khác nhau |

```text
overlap_ratio = shared_frames / min(num_frames_A, num_frames_B)
center_ratio = ||center_A - center_B||_2 / min(diagonal_A, diagonal_B)
score_duplicate = 0.70 * appearance_similarity + 0.30 * median_IoU
```

Khi merge duplicate, cùng một frame chỉ giữ observation có confidence cao hơn. Embedding cùng frame cũng chỉ góp một lần. Vì vậy `num_frames` là hợp của các frame, không bị cộng trùng và video chỉ vẽ một bbox cho cùng CID.

## Ghép liên camera và Global ID

Cross-camera matching làm việc trên CID đã aggregate, không dùng raw LID. Với từng cặp camera, `CrossCameraMatcher` tạo ma trận cost cho các pair đủ điều kiện rồi dùng Hungarian assignment để chọn matching một-một.

```text
eligible(A, B) = similarity(A, B) >= 0.70
                 and time_gap(A, B) <= 30 giây

cost(A, B) = 1 - similarity(A, B)
```

`time_gap` bằng `0` khi hai interval overlap; nếu không, là khoảng cách nhỏ nhất giữa hai đầu interval. Nếu `matching.use_time_constraint: false`, điều kiện gap bị bỏ qua.

Sau đó `GlobalTracker` sắp candidate match giảm dần theo similarity và merge bằng union-find. Trước khi merge, nó bắt buộc tập camera của hai group không giao nhau:

```text
Nếu cameras(group_A) intersect cameras(group_B) != empty: reject
Nếu không: union hai group
```

Bất biến này ngăn một GID chứa hai CID của cùng camera, ngay cả khi pairwise matching tạo ra cạnh bắc cầu mâu thuẫn.

Đây là hành vi v2.x (`matching.method: hungarian`). Điểm yếu: mỗi cặp camera được quyết định riêng, nên A~B và B~C có thể kéo A vào cùng GID với một C rất khác.

### Phân cụm agglomerative có ràng buộc (v3.0, mặc định)

`AgglomerativeGlobalTracker` bắt đầu với mỗi CID là một cụm, rồi lặp lại việc gộp hai cụm có **average linkage** cao nhất:

```text
linkage(X, Y) = mean_{a in X, b in Y} sim(a, b)
gộp nếu        linkage >= similarity_threshold (0.70)
               và không cặp (a, b) nào bị cấm:
                   cùng camera, hoặc time_gap > max_time_gap_seconds (khi bật ràng buộc thời gian)
```

Average linkage chấm điểm theo mọi cặp thành viên, nên một cạnh bắc cầu mạnh không kéo được hai nhóm khác nhau vào nhau. Linkage được cập nhật theo công thức Lance–Williams nên mỗi bước là O(n²).

`sim(a, b)` mặc định là **so khớp tập-với-tập** (`similarity_mode: set_topk`):

```text
S       = { cos(e_i, e_j) : e_i thuộc a, e_j thuộc b }   (tối đa 64 mẫu mỗi CID)
sim     = mean của top_fraction (25%) giá trị lớn nhất của S
```

Hai camera hiếm khi thấy cùng một tư thế. So các góc nhìn gần nhau nhất của hai CID ổn định hơn so hai vector trung bình của các tư thế khác nhau. `similarity_mode: descriptor` quay lại cách v2.x.

### Lọc tracklet ngắn (v3.0)

CID có ít hơn `postprocess.min_camera_track_frames` (5) frame thường là false detection hoặc mảnh vỡ. Chúng nhận GID 0, không được vẽ và không được xuất. Ngưỡng tính bằng frame: ở 25 FPS là 0.2 giây, ở 2 FPS là 2.5 giây. Benchmark cho thấy 5 cân bằng tốt nhất giữa IDF1 liên camera và MOTA từng camera (xem [docs/ABLATION.md](docs/ABLATION.md)).

## Output và cách kiểm tra

| File | Nội dung | Dùng để kiểm tra |
|---|---|---|
| `local_tracks.csv` | LID ByteTrack thô, frame/thời gian, số frame, số embedding | Tracker tạo/sống LID ra sao |
| `track_segments.csv` | Segment, CID, quality, retained embedding, lý do split | ID switch và descriptor |
| `camera_tracks.csv` | Mỗi CID và source segment `Lx:Sy` | Re-entry/duplicate đã gộp |
| `intra_camera_matches.csv` | Match type, IoU, overlap, appearance, score | Vì sao stitch trong camera |
| `global_tracks.csv` | GID cho từng segment theo thời gian | Mapping CID/GID chính xác |
| `tracking_diagnostics.csv` | Post-NMS overlap, split event, decision accept/reject | Hiệu chỉnh threshold dựa trên bằng chứng |
| `local_similarity_matrix.csv` | Similarity segment, có cả same-camera để diagnosis | Quan sát lỗi split/stitch |
| `similarity_matrix.csv` | Similarity CID giữa các camera | Chọn ngưỡng cross-camera |
| `frame_tracks.csv` | v3.0: mỗi box được vẽ, với GID/CID/LID và bbox `x, y, w, h` | Chấm điểm, phân tích ngoài pipeline |
| `run_info.json` | v3.0: phiên bản code (`git describe`), config đầy đủ, thời gian từng pha | Tái lập một lần chạy |
| `videos/camX_global_tracking.mp4` | Video replay với nhãn | Kiểm tra trực quan |

Nhãn video mặc định có dạng `GID: 7 | CID: 3 | LID: 28`. Segment index không hiện trên video để nhãn gọn, nhưng có đầy đủ trong CSV. Trước khi vẽ, pipeline tra mapping bằng `(camera, LID, frame)`, vì LID bị split có thể đổi CID/GID theo thời gian.

Màu box ổn định theo identity:

```text
hue = (identity * 0.61803398875) mod 1
```

Hue sau đó đổi từ HSV sang BGR. Hằng số golden ratio giúp các ID gần nhau có màu khác biệt tương đối.

## Cấu hình và cách chạy

Các path trong `configs/config.yaml` được resolve theo project root. Có thể thêm camera bằng một entry mới trong `video`, miễn `id` duy nhất; pipeline yêu cầu ít nhất hai camera.

```yaml
video:
  camera_6:
    id: cam6
    path: data/raw/6.avi
    time_offset_seconds: 0
```

`device: auto` chọn CUDA nếu `torch.cuda.is_available()` là `True`, nếu không chọn CPU. Có thể ép bằng `device: cuda`, `device: cuda:0` hoặc `device: cpu` riêng cho detector và ReID.

```powershell
# Cài môi trường
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Chạy mặc định
python main.py --config configs/config.yaml

# Ghi một lần chạy so sánh vào output mới, không đụng baseline
python main.py --config configs/config.yaml --output-directory outputs_fix_validation
```

| Nhóm | Thiết lập | Khi điều chỉnh |
|---|---|---|
| Detector | confidence, NMS IoU | Bỏ sót người hoặc nhiều box duplicate |
| Detector tốc độ | `batch_size`, `half`, `image_size` | Hết VRAM thì giảm `batch_size`; cần kết quả FP32 thì `half: false` |
| Tracker | `tracker.config_path` (BoT-SORT/ByteTrack), buffer, `proximity_thresh`, `appearance_thresh`, `backfill_first_frame` | Occlusion, LID đứt/nhảy |
| ReID | checkpoint, architecture, `samples_per_second` (hoặc `sample_interval`), crop size | Similarity kém hoặc chậm |
| ReID chất lượng | `flip_augmentation`, `max_occlusion`, `border_margin` | Split giả khi người bị che, descriptor lẫn người |
| Render | `output.render_workers` | Số camera được render song song |
| Track split | window, change threshold, separation | Một LID lẫn nhiều người hoặc split quá nhiều |
| Intra camera | re-entry/duplicate thresholds, `motion_link_*` | CID tách sai/gộp nhầm |
| Hậu xử lý | `postprocess.min_camera_track_frames` | Quá nhiều ID rác / mất người xuất hiện ngắn |
| Cross camera | `method`, `similarity_mode`, `top_fraction`, cosine threshold, time gap | GID false positive/false negative |
| Cache | `cache.enabled`, `cache.directory` | Tắt nếu muốn mỗi lần chạy đều chạy lại model |

## Kiểm thử hiệu chỉnh và giới hạn

| Script | Mục đích |
|---|---|
| `scripts/test_detector.py` | Kiểm tra YOLO trên một số frame |
| `scripts/test_tracker.py` | Kiểm tra độ liên tục Local ID |
| `scripts/test_reid.py` | So similarity cùng người/khác người |
| `scripts/test_matching.py` | Test Hungarian và GID không cần model |
| `scripts/test_global_association.py` | Test phân cụm agglomerative: cùng người nhiều camera, không gộp cùng camera, không bắc cầu, ràng buộc thời gian |
| `scripts/test_backfill.py` | Test frame đầu bị Ultralytics bỏ và được ghi bù; BoT-SORT với feature OSNet |
| `benchmarks/test_metrics.py` | Test MOTA/IDF1/HOTA trên các ca tính tay |
| `scripts/test_track_segmenter.py` | Test ID switch phải split, thay góc ngắn không split, bị che ngắn không split, `crop_quality` |
| `scripts/test_intra_camera.py` | Test re-entry, duplicate overlap, người thật đồng thời, motion link |
| `scripts/test_duplicate_detection.py` | Regression NMS tại cam5 frame 3255 |

Quy trình tuning nên dựa trên dữ liệu:

1. Chạy ra một output directory mới.
2. Lọc `tracking_diagnostics.csv` theo `event_type` hoặc `accepted`.
3. Mở video đúng frame liên quan.
4. Chỉ thay một ngưỡng mỗi lần chạy, rồi so CSV/video với baseline.

Những giới hạn cần biết:

- ReID dựa appearance nên có thể nhầm người mặc giống nhau, bị che hoặc ở ánh sáng rất khác.
- BoT-SORT chỉ dùng appearance cho cặp còn chồng nhau (`proximity_thresh`); người nhảy xa hoàn toàn giữa hai frame vẫn được nối ở bước motion link/re-entry offline.
- Phân cụm agglomerative vẫn là heuristic dựa trên ngưỡng, không phải multi-camera association học end-to-end.
- Ngưỡng không có giá trị đúng cho mọi camera; cần calibration theo checkpoint và môi trường.
- Số thứ tự GID có thể thay đổi giữa các lần chạy nếu tập track/candidate match thay đổi. Quan hệ bên trong một lần chạy mới là ý nghĩa quan trọng.

## Đánh giá và phiên bản

Mỗi phiên bản là một git tag (`v1.0`, `v2.0`, `v2.1`, `v3.0`), phát triển trên nhánh `version/vX.Y`. Benchmark chấm mọi phiên bản trên cùng dữ liệu HDA có ground truth, bằng cùng code chấm điểm:

```powershell
python benchmarks/prepare_hda.py --hda-root ../HDA_Dataset_V1.3   # một lần
python -m benchmarks.run_benchmark --name v3.0                    # chấm checkout hiện tại
python -m benchmarks.report                                       # sinh docs/BENCHMARK_RESULTS.md
python -m benchmarks.ablation                                     # sinh docs/ABLATION.md
```

Chi tiết phương pháp, chỉ số và quy trình phát hành phiên bản mới: [docs/EVALUATION.md](docs/EVALUATION.md).

## Cấu trúc project

```text
.
|-- main.py
|-- configs/
|   |-- config.yaml
|   |-- botsort.yaml          # v3.0 (mặc định)
|   `-- bytetrack.yaml        # v2.x
|-- benchmarks/               # Benchmark HDA, chỉ số, runner, báo cáo, ablation
|-- docs/                     # Đánh giá, kết quả, ablation, changelog
|-- data/raw/                 # Video đầu vào
|-- data/benchmark/           # Dữ liệu HDA đã chuẩn bị (không commit)
|-- models/detector/          # YOLO checkpoint local
|-- models/reid/              # OSNet checkpoint local
|-- scripts/                  # Các regression/unit test
|-- src/                      # Pipeline và thuật toán
`-- outputs/                  # Video/CSV sinh ra sau khi chạy
```

Tất cả model và video phải tồn tại local. Project không tự tải weights hoặc dataset.

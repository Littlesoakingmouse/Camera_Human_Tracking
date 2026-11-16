# Lịch sử phiên bản

Mọi số liệu đo trên benchmark HDA (5 camera, 16 697 frame, RTX 3050 6 GB), cùng code chấm điểm cho mọi phiên bản. Bảng đầy đủ ở [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md), phương pháp ở [EVALUATION.md](EVALUATION.md).

| Phiên bản | MTMC IDF1 | MTMC HOTA | HOTA/camera | MOTA/camera | IDSW | FPS |
|---|---:|---:|---:|---:|---:|---:|
| v1.0 | 45.8 | 39.4 | 41.2 | 48.5 | 327 | 27.9 |
| v2.0 | 41.1 | 37.7 | 40.2 | 49.2 | 466 | 40.9 |
| v2.1 | 39.3 | 37.2 | 40.3 | 48.4 | 469 | 89.0 |
| **v3.0** | **59.0** | **53.7** | **55.4** | **72.7** | **229** | 82.0 |

## v3.0 (nhánh `version/v3.0`)

So với v2.1: **MTMC IDF1 +19.7 điểm**, HOTA đa camera +16.5, MOTA từng camera +24.3, số lần đổi ID giảm 51%, số GID dự đoán giảm từ 306 xuống 130 (thực tế có 66 người). Tốc độ giảm 8% (89.0 → 82.0 FPS).

### Chẩn đoán dẫn tới các thay đổi

Benchmark cho thấy lỗi lớn nhất của v2.1 **không nằm ở detector**. Trên các frame có nhãn, YOLO tìm thấy 90–98% người, nhưng sau ByteTrack recall chỉ còn 31% (cam56) đến 80% (cam57). Ba nguyên nhân:

1. HDA ghi khoảng 2 FPS (đọc từ đồng hồ in trên frame). Giữa hai frame người đi xa, IoU giữa box dự đoán và detection về 0, ByteTrack không nối được.
2. Ultralytics không trả về frame đầu của track mới. Ở 2 FPS một người chỉ xuất hiện vài frame, nên mất một frame là mất một phần lớn track.
3. Lấy mẫu ReID mỗi 5 frame nghĩa là mỗi 2.5 giây ở 2 FPS. Nhiều track không có mẫu nào, nên không ghép được trong camera hay giữa camera.

### Thay đổi

Cột cuối là thay đổi khi **bỏ** riêng thay đổi đó khỏi v3.0 (âm = thay đổi có ích), số đầy đủ ở [ABLATION.md](ABLATION.md).

| Thay đổi | Cờ cấu hình | Bỏ đi thì MTMC IDF1 / MOTA từng camera |
|---|---|---|
| BoT-SORT nhận feature OSNet của từng detection; feature được dùng lại làm mẫu ReID | `tracker.config_path: configs/botsort.yaml` | −13.7 / −18.6 |
| Lấy mẫu ReID theo giây | `reid.samples_per_second` | −5.2 / −0.4 |
| So khớp tập-với-tập (top 25% cặp mẫu) | `matching.similarity_mode: set_topk` | −4.0 / 0 |
| Phân cụm agglomerative, average linkage, cannot-link | `matching.method: agglomerative` | −1.0 / 0 (HOTA đa camera −2.3) |
| Lọc CID ngắn hơn 5 frame | `postprocess.min_camera_track_frames` | −0.9 / +1.0 |
| Ghi bù frame đầu của track | `tracker.backfill_first_frame` | +0.9 / −5.0 |
| Motion link trong camera | `intra_camera.motion_link_enabled` | +0.7 / −0.5 (IDSW +12) |
| Xuất `frame_tracks.csv`, `run_info.json` | luôn bật | Chấm điểm, tái lập |
| Cache kết quả detection/ReID | `cache.enabled` | Chỉnh ngưỡng trong vài giây |
| Tự chọn `half`/`quantize` theo phiên bản Ultralytics | luôn bật | Chạy được với 8.3 lẫn 8.4 |

Hai thay đổi cuối của nhóm thuật toán cho kết quả lẫn lộn trên benchmark này:

- **Ghi bù frame đầu** thêm đúng các box người thật (MOTA từng camera +5.0, HOTA từng camera +2.3), nhưng các box mới đó thuộc các track ngắn khó ghép liên camera, nên MTMC IDF1 giảm 0.9. Giữ bật vì nó sửa một lỗi có thật của Ultralytics và cải thiện tracking trong từng camera.
- **Motion link** giảm số lần đổi ID (241 → 229) nhưng MTMC IDF1 thấp hơn 0.7 điểm. Mức chênh này nhỏ hơn độ dao động khi đổi ngưỡng ghép (±1–2 điểm), nên chưa kết luận được trên một benchmark 2 FPS. Giữ bật vì nó nhắm vào lỗi phân mảnh của video FPS cao, nơi khoảng trống vài frame là phổ biến; cần đo lại khi có benchmark FPS cao.

Mọi thay đổi có cờ riêng; đặt cờ về giá trị cũ là quay lại hành vi v2.1 cho phần đó.

Vì sao v3.0 không chậm hơn nhiều dù OSNet chạy cho **mọi** detection: v2.1 crop lại và chạy OSNet riêng cho các frame được lấy mẫu. v3.0 chạy OSNet đúng một lần mỗi detection, gom cả batch frame vào một forward, và dùng chung kết quả cho tracker lẫn ReID.

### Chọn ngưỡng

Các ngưỡng ghép (0.70, `top_fraction` 0.25, ngưỡng re-entry 0.78) được giữ nguyên từ v2.1 hoặc đặt theo lý do thiết kế. Thử nghiệm cho thấy 0.68 nhỉnh hơn 0.70 khoảng 1 điểm IDF1, nhưng mức chênh này dễ là overfit với 5 camera, nên không đổi. Ngưỡng lọc tracklet chọn 5 thay vì 10: hai giá trị cho HOTA đa camera ngang nhau, nhưng 10 làm MOTA từng camera giảm khoảng 9 điểm vì bỏ cả người thật xuất hiện ngắn.

### Trên video demo (không có ground truth)

Chạy `python main.py` trên 5 video `data/raw/` (RTX 3050):

| | v2.1 (`outputs/`) | v3.0 (`outputs_v3/`) |
|---|---:|---:|
| LID thô | 65 | 37 |
| GID | 35 (32 GID chỉ ở 1 camera) | 12 (1 GID đi qua cả 5 camera, 3 GID qua 2 camera) |
| Detect+Track+ReID | không ghi trong output | 200 s |
| Render 5 video | không ghi trong output | 48 s |

Video demo không có nhãn nên bảng này chỉ cho thấy mức phân mảnh giảm, không chứng minh GID đúng. Kiểm tra bằng mắt vài frame cho thấy nhãn GID/CID/LID được vẽ đúng chỗ.

### Giới hạn còn lại

- cam56 vẫn yếu nhất (IDF1 46%): camera đặt rất gần cửa, người bị cắt ở mép ảnh và che nhau dày đặc.
- Chưa đo được tác dụng của ràng buộc thời gian giữa camera, vì HDA không đồng bộ thời gian.
- Benchmark chỉ có 2 FPS. Trên video demo FPS cao hơn, BoT-SORT và motion link vẫn đúng về nguyên lý nhưng chưa có ground truth để đo.

## v2.1 (tag `v2.1`, commit `7883e20`)

Tối ưu tốc độ: YOLO và OSNet theo batch với FP16, ByteTrack chạy riêng nhận detection từng frame, đọc video trên thread nền, render song song. Đánh dấu crop bị che hoặc bị cắt mép là không tin cậy.

Ngưỡng ghép liên camera đổi từ 0.6654 lên 0.70.

Kết quả benchmark: **nhanh gấp 2.2 lần v2.0** (40.9 → 89.0 FPS). Độ chính xác trong từng camera không đổi (HOTA 40.2 → 40.3). MTMC IDF1 giảm 1.8 điểm (41.1 → 39.3), phù hợp với ngưỡng ghép liên camera chặt hơn.

## v2.0 (tag `v2.0`, commit `75a7302`)

Thêm tách LID khi ngoại hình đổi đột ngột (ID switch), ghép re-entry và track trùng trong cùng camera thành CID, rồi ghép CID giữa camera.

Kết quả benchmark: **MTMC IDF1 giảm 4.7 điểm so với v1.0** (45.8 → 41.1), số lần đổi ID tăng từ 327 lên 466, số GID dự đoán tăng từ 193 lên 285. v2.0 đổi đồng thời ba thiết lập ảnh hưởng tới frame rate thấp:

- ByteTrack `match_thresh` 0.80 → 0.70: cần IoU ≥ 0.3 thay vì ≥ 0.2 mới nối track. Ở 2 FPS người di chuyển xa giữa hai frame, nên đổi ID tăng.
- Confidence detector 0.5 → 0.1: thêm detection yếu, sinh nhiều track ngắn.
- Lấy mẫu ReID vẫn mỗi 5 frame, nên các track ngắn đó không có mẫu, không ghép được và thành GID riêng.

Benchmark không tách riêng được thay đổi nào gây ra phần lớn mức giảm. Trên video demo FPS cao hơn, các thay đổi này có thể có lợi; benchmark chỉ cho thấy chúng không tổng quát sang frame rate thấp. v3.0 xử lý gốc rễ bằng association có ngoại hình, lấy mẫu theo giây và lọc tracklet ngắn.

## v1.0 (tag `v1.0`, commit `fa97044`)

Pipeline đầy đủ đầu tiên: YOLO (`yolo26m`, confidence 0.5) + ByteTrack (`match_thresh` 0.80) qua `model.track()` từng frame, OSNet-AIN lấy mẫu mỗi 5 frame, descriptor trung bình, ghép liên camera bằng Hungarian từng cặp camera.

Kết quả benchmark: MTMC IDF1 45.8, 27.9 FPS.

# Đánh giá dự án qua từng phiên bản

Tài liệu này mô tả cách đo độ chính xác và hiệu năng của pipeline, sao cho mọi phiên bản được so sánh trên cùng dữ liệu, cùng code chấm điểm và cùng phần cứng. Bảng số liệu nằm ở [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md), đóng góp của từng thay đổi ở [ABLATION.md](ABLATION.md), lịch sử và phân tích ở [CHANGELOG.md](CHANGELOG.md).

## 1. Vì sao cần benchmark có ground truth

Video demo trong `data/raw/` không có nhãn. Trước đây ngưỡng được chỉnh bằng cách xem video, nên không thể biết một thay đổi làm hệ thống tốt lên hay chỉ khác đi. Benchmark này dùng **HDA Person Dataset** (Instituto Superior Técnico, Lisbon), có nhãn box từng frame và **nhãn danh tính toàn cục**: `person045` ở camera 56 và camera 58 là cùng một người. Nhờ vậy đo được cả tracking trong từng camera lẫn ghép người giữa các camera.

## 2. Dữ liệu benchmark

| Thuộc tính | Giá trị |
|---|---|
| Camera | `cam56`, `cam57`, `cam58`, `cam59`, `cam60` (tầng 8 của HDA) |
| Số frame | 16 697 (3798 + 3780 + 3721 + 3670 + 1728) |
| Độ phân giải | 1280×800 (cam60 gốc 2560×1600 được thu nhỏ 0.5, box GT thu nhỏ tương ứng) |
| Box người có nhãn | 8 082 |
| Danh tính thật | 33 / 41 / 33 / 32 / 20 theo camera, nhiều người xuất hiện ở nhiều camera |
| Vùng bỏ qua | box nhãn `crowd` và `personUnk` (34 box) |

Lý do chọn 5 camera này: cùng một tầng, số người chung giữa các cặp camera cao nhất (cam57–cam58 có 32 người chung), và một lần chạy v3.0 chỉ mất khoảng 3.5 phút trên RTX 3050 6 GB (bản chậm nhất, v1.0, mất 10 phút).

Hai đặc điểm của HDA cần nhớ khi đọc kết quả:

- **Các camera không quay cùng lúc** (dấu thời gian cách nhau nhiều ngày). Vì vậy benchmark tắt `matching.use_time_constraint`; ghép liên camera chỉ dựa vào ngoại hình. Đây là điều kiện khó hơn video demo.
- **HDA ghi khoảng 2 FPS.** File `.seq` không lưu frame rate; con số 2 FPS được đọc từ đồng hồ in trên frame (57 frame trong 28 giây ở cam56). Video được đóng gói ở 2 FPS để mọi ngưỡng tính bằng giây có nghĩa thật. Frame rate thấp khiến người di chuyển xa giữa hai frame, nên đây là điều kiện khó cho tracker chỉ dựa trên IoU.

Chuẩn bị dữ liệu (một lần, khoảng 3 phút):

```powershell
python benchmarks/prepare_hda.py --hda-root ../HDA_Dataset_V1.3
```

Script đọc trực tiếp file `.seq` (JPEG trong container Norpix) và nhãn VBB, ghi ra `data/benchmark/hda/camXX/{frames/, video.avi, gt.csv}`. `video.avi` là các JPEG gốc trong container MJPEG (ffmpeg stream copy, không nén lại; nếu không có ffmpeg thì OpenCV nén lại MJPEG), để mọi phiên bản cũ đọc được bằng `cv2.VideoCapture` mà không cần sửa code.

## 3. Các chỉ số

Tất cả cài trong [benchmarks/metrics.py](../benchmarks/metrics.py) bằng numpy/scipy, có test trên các ca tính tay ([benchmarks/test_metrics.py](../benchmarks/test_metrics.py)). Một dự đoán khớp một GT khi IoU ≥ 0.5.

### 3.1. Độ chính xác đa camera (MTMC), chỉ số chính

Ghép tất cả camera thành một chuỗi duy nhất (khóa frame = `camera, frame`). Global ID dự đoán được so với danh tính thật toàn cục.

| Chỉ số | Ý nghĩa | Bắt lỗi gì |
|---|---|---|
| **IDF1** | `2·IDTP / (2·IDTP + IDFP + IDFN)`, với IDTP tính sau một phép ghép song ánh toàn cục giữa ID thật và ID dự đoán | Một người bị tách thành nhiều GID (giảm IDR), hoặc nhiều người bị gộp vào một GID (giảm IDP) |
| IDP / IDR | Precision / recall ở mức danh tính | Phân biệt lỗi gộp nhầm và lỗi tách nhầm |
| **HOTA** | `sqrt(DetA · AssA)`, trung bình trên ngưỡng IoU 0.05…0.95 | Cân bằng giữa phát hiện và liên kết |
| AssA | Độ chính xác liên kết | Chất lượng giữ ID theo thời gian và giữa camera |

### 3.2. Độ chính xác trong từng camera

Chấm riêng từng camera rồi lấy trung bình (các số đếm FP/FN/IDSW/Frag là tổng 5 camera).

| Chỉ số | Ý nghĩa |
|---|---|
| HOTA, DetA, AssA | Như trên, trong một camera |
| MOTA | `1 − (FN + FP + IDSW) / GT`; bị chi phối bởi chất lượng phát hiện |
| IDF1 | Giữ ID trong một camera |
| Precision / Recall | Ở mức box |
| IDSW | Số lần một người thật bị đổi ID |
| Frag | Số lần một người thật đang được track thì mất rồi được track lại |

Trước khi chấm, các dự đoán khớp với vùng `crowd`/`personUnk` bị loại (giống MOT17), để không phạt hệ thống vì phát hiện người trong đám đông không được gán nhãn.

### 3.3. Hiệu năng

Đo trong cùng tiến trình chạy pipeline:

| Chỉ số | Cách đo |
|---|---|
| Tổng thời gian | Từ lúc gọi `pipeline.run()` đến lúc xong (không tính nạp model, không render video) |
| Detect+Track+ReID | Tổng thời gian `_process_camera` của mọi camera |
| Ghép ID | Phần còn lại: tách segment, ghép trong camera, ghép liên camera, ghi CSV |
| FPS end-to-end | Tổng số frame / tổng thời gian |
| GPU peak | `torch.cuda.max_memory_allocated()` |
| RAM peak | RSS lớn nhất của tiến trình, lấy mẫu mỗi 0.5 s |

Render video bị tắt trong benchmark vì nó không ảnh hưởng độ chính xác và phụ thuộc codec.

## 4. So sánh các phiên bản

Mỗi phiên bản là một git tag. Runner tạo git worktree của tag đó và **import pipeline của chính phiên bản đó**, nên code cũ chạy đúng như lúc phát hành; chỉ dữ liệu, config đường dẫn và code chấm điểm là chung.

| Tag | Commit | Nội dung |
|---|---|---|
| `v1.0` | `fa97044` | Bản đầy đủ đầu tiên: YOLO + ByteTrack + OSNet, ghép liên camera bằng Hungarian |
| `v2.0` | `75a7302` | Thêm tách LID bị đổi người, ghép re-entry và track trùng trong cùng camera |
| `v2.1` | `7883e20` | Tối ưu tốc độ (batch, FP16, prefetch), đánh dấu crop bị che |
| `v3.0` | `5423e3b` (nhánh `version/v3.0`) | BoT-SORT + OSNet, ghi bù frame đầu, motion link, phân cụm có ràng buộc. Xem [CHANGELOG.md](CHANGELOG.md) |

Benchmark ghi đè 4 thiết lập trong config của mọi phiên bản, để so sánh công bằng: đường dẫn video và model, tắt render video và lưu crop, và tắt ràng buộc thời gian giữa camera (lý do ở mục 2). Các ngưỡng khác giữ nguyên như phiên bản đó phát hành.

Chạy lại toàn bộ:

```powershell
foreach ($v in "v1.0", "v2.0", "v2.1", "v3.0") {
    git worktree add .versions/$v $v
    python -m benchmarks.run_benchmark --name $v --code-dir .versions/$v
}
python -m benchmarks.report      # docs/BENCHMARK_RESULTS.md
python -m benchmarks.ablation    # docs/ABLATION.md
```

Kết quả thô (dự đoán từng frame, config đã dùng, output pipeline) nằm ở `benchmarks/runs/<tên>/` (không commit). Bản tóm tắt nằm ở `benchmarks/results/<tên>.json` (có commit), và `benchmarks.report` sinh lại bảng trong [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md).

## 5. Quy trình khi phát triển phiên bản mới

1. Tạo nhánh `version/vX.Y` từ tag gần nhất.
2. Mỗi thay đổi thuật toán phải có cờ trong config, để đo riêng tác dụng của nó (ablation).
3. Khi chỉ chỉnh ngưỡng ở các bước sau detection, dùng cache để mỗi lần thử chỉ mất vài giây:

   ```powershell
   python -m benchmarks.run_benchmark --name tune-a --use-cache --set matching.similarity_threshold=0.75
   ```

   Lần chạy dùng cache hoặc có `--set` không được đưa vào bảng chính thức (thời gian không so sánh được).
4. Không chọn ngưỡng chỉ vì nó tốt nhất trên benchmark nếu không giải thích được vì sao; benchmark chỉ có 5 camera, dễ overfit.
5. Chạy chính thức không cache, tag phiên bản, chạy `benchmarks.report`, ghi phân tích vào [CHANGELOG.md](CHANGELOG.md).

## 6. Giới hạn của benchmark

- Một tầng nhà, 5 camera trong nhà, người đi bộ; chưa đại diện cho ngoài trời hay đám đông dày.
- Camera không đồng bộ thời gian, nên chưa đo được tác dụng của ràng buộc thời gian và topology camera. Bộ CITIC-MTMC (đồng bộ, có hiệu chỉnh camera) là ứng viên cho benchmark thứ hai, nhưng ground truth của nó là tọa độ sàn chứ không phải box, cần chiếu qua hiệu chỉnh camera trước khi dùng.
- HOTA và IDF1 tự cài đặt theo định nghĩa gốc; đã kiểm tra bằng ca tính tay nhưng chưa đối chiếu số với TrackEval chính thức.

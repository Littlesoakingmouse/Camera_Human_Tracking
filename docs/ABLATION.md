# Ablation v3.0

> Sinh bởi `python -m benchmarks.ablation`. Mỗi dòng bỏ **một** thay đổi của v3.0 khỏi config phát hành; mọi thứ khác giữ nguyên. Dữ liệu: HDA cam56–cam60.

| Biến thể | MTMC IDF1 | MTMC HOTA | HOTA/camera | MOTA/camera | IDSW | GID dự đoán | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| v3.0 đầy đủ | 59.0 | 53.7 | 55.4 | 72.7 | 229 | 130 | 81.6 |
| ByteTrack thay BoT-SORT + OSNet | 45.3 (-13.7) | 41.5 | 43.6 | 54.1 | 322 | 145 | 83.5 |
| Không ghi bù frame đầu của track | 59.9 (+0.9) | 53.3 | 53.1 | 67.7 | 211 | 118 | 81.3 |
| Lấy mẫu ReID mỗi 5 frame (như v2.x) | 53.7 (-5.2) | 50.9 | 54.4 | 72.3 | 244 | 157 | 82.0 |
| Không motion link trong camera | 59.7 (+0.7) | 54.0 | 55.3 | 72.2 | 241 | 136 | cache |
| Không lọc tracklet ngắn | 58.1 (-0.9) | 53.4 | 55.5 | 73.7 | 275 | 161 | cache |
| Lọc tracklet < 10 frame (thay vì 5) | 60.6 (+1.6) | 53.5 | 52.4 | 64.0 | 163 | 89 | cache |
| Hungarian từng cặp camera thay phân cụm | 58.0 (-1.0) | 51.4 | 55.4 | 72.7 | 229 | 71 | cache |
| Descriptor trung bình thay so khớp tập-với-tập | 55.0 (-4.0) | 50.4 | 55.4 | 72.7 | 229 | 89 | cache |

Cột FPS chỉ có ý nghĩa với biến thể phải chạy lại model (biến thể đầu tiên của mỗi cấu hình detection/ReID). Biến thể ghi `cache` dùng lại kết quả detection, ReID của lần chạy trước nên thời gian không so sánh được.

Số trong ngoặc là chênh lệch MTMC IDF1 so với v3.0 đầy đủ: âm nghĩa là thay đổi bị bỏ có ích. Phân tích từng dòng nằm ở [CHANGELOG.md](CHANGELOG.md#v30-nhánh-versionv30).

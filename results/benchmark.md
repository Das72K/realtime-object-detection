| Model | Backend | Device | p50 ms | p95 ms | FPS | pre / inf / post (p50 ms) | Parity F1 | mAP50-95 |
|---|---|---|--:|--:|--:|---|--:|--:|
| yolo26n | PyTorch (Ultralytics) | 0 | **14.4** | 23.5 | 62 | 0.9 / 11.4 / 1.5 | – | 0.487 |
| yolo26n | ONNX Runtime (Ultralytics) | 0 | **9.9** | 14.9 | 92 | 0.8 / 6.7 / 1.7 | 1.000 | 0.487 |
| yolo26n | ONNX Runtime (raw) | 0 | **10.3** | 14.7 | 87 | 1.5 / 7.4 / 1.2 | 1.000 | 0.487 |
| yolo26s | PyTorch (Ultralytics) | 0 | **14.8** | 19.2 | 64 | 0.9 / 11.6 / 1.5 | – | 0.568 |
| yolo26s | ONNX Runtime (Ultralytics) | 0 | **10.0** | 12.6 | 95 | 0.8 / 7.0 / 1.7 | 1.000 | 0.568 |
| yolo26s | ONNX Runtime (raw) | 0 | **11.4** | 16.5 | 83 | 1.5 / 8.5 / 1.2 | 1.000 | 0.568 |
| yolo26m | PyTorch (Ultralytics) | 0 | **16.0** | 22.4 | 57 | 0.9 / 12.7 / 1.6 | – | 0.616 |
| yolo26m | ONNX Runtime (Ultralytics) | 0 | **12.8** | 16.1 | 75 | 0.9 / 9.2 / 1.9 | 1.000 | 0.616 |
| yolo26m | ONNX Runtime (raw) | 0 | **13.2** | 17.5 | 70 | 1.6 / 10.0 / 1.3 | 1.000 | 0.616 |
| yolo26n | PyTorch (Ultralytics) | cpu | **42.3** | 50.8 | 23 | 1.4 / 39.4 / 0.8 | – | 0.487 |
| yolo26n | ONNX Runtime (Ultralytics) | cpu | **20.8** | 58.6 | 35 | 2.5 / 16.1 / 1.1 | 1.000 | 0.487 |
| yolo26n | ONNX Runtime (raw) | cpu | **20.4** | 23.9 | 48 | 1.6 / 17.2 / 1.4 | 1.000 | 0.487 |
| yolo26s | PyTorch (Ultralytics) | cpu | **86.4** | 94.7 | 12 | 1.4 / 83.8 / 0.8 | – | 0.568 |
| yolo26s | ONNX Runtime (Ultralytics) | cpu | **57.8** | 84.2 | 16 | 2.6 / 51.1 / 1.6 | 1.000 | 0.568 |
| yolo26s | ONNX Runtime (raw) | cpu | **52.0** | 55.3 | 19 | 1.6 / 48.3 / 1.4 | 1.000 | 0.568 |
| yolo26m | PyTorch (Ultralytics) | cpu | **225.9** | 237.8 | 4 | 1.4 / 223.1 / 0.9 | – | 0.616 |
| yolo26m | ONNX Runtime (Ultralytics) | cpu | **138.5** | 156.3 | 7 | 2.4 / 133.9 / 1.5 | 1.000 | 0.616 |
| yolo26m | ONNX Runtime (raw) | cpu | **142.8** | 148.4 | 7 | 1.6 / 139.1 / 1.4 | 1.000 | 0.616 |

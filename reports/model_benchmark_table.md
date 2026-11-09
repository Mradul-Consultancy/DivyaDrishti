# Model Complexity and Inference Speed Benchmark

- **Hardware Platform**: `CPU` (`cpu`)
- **Input Resolution**: `224 × 224 × 3`
- **Latency Measurement Protocol**: 20 warmup cycles + 100 synchronized benchmark timing iterations

| Architecture | Parameters (M) | Model Size (MB) | Compute (GFLOPs) | Latency (Batch=1, ms) | P95 Latency (ms) | Throughput (FPS, B=32) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| ConvNeXt-Tiny (timm) | 27.82 M | 334.1 MB | 8.93 | 101.36 ms | 128.47 ms | 13.9 fps |
| EfficientNet-B3 (timm) | 10.70 M | 129.2 MB | 2.03 | 62.93 ms | 74.69 ms | 17.7 fps |
| YOLO26n-cls (Ultralytics) | 1.53 M | 3.2 MB | 0.40 | 14.05 ms | 18.12 ms | 164.0 fps |
| **Soft-Voting Ensemble (All 3)** | **40.05 M** | **466.5 MB** | **11.36** | **178.34 ms** | **221.28 ms** | **14.0 fps** |

> [!NOTE]
> - **YOLO26n-cls**: Minimal parameter count and highest batch throughput, ideal for real-time edge screening.
> - **EfficientNet-B3**: High parameter efficiency with compound scaling; strong feature fidelity for subtle compression artifacts.
> - **ConvNeXt-Tiny**: Pure modern ConvNet inductive bias with inverted bottleneck design.
> - **Soft-Voting Ensemble**: Delivers maximum classification margin and robustness by fusing three architecturally diverse inductive biases.
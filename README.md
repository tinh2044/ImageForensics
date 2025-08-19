## Image Forgery Detection
## Cấu trúc dự án
```
.
├── augmentations.py              # (chưa dùng; tăng cường dữ liệu)
├── configs/
│   └── sample.yaml               # cấu hình mẫu cho mô hình
├── data/                         # thư mục dữ liệu mong đợi (xem bên dưới)
├── dataset.py                    # dataset và dataloader
├── images/                       
├── LICENSE
├── main.py                       # script chính (train/test)
├── models/
│   └── __init__.py               # định nghĩa model ImageForgeryDetection
├── outputs/
│   └── ImageForgeryDetection/
│       └── image_forgery_detection.log  # log và artifact
├── pretrained_models/            # weight của mô hình
├── scripts/
│   └── train.sh          # script dùng để train giảm thời gian nhập cmd(gọi đến main.py)
├── utils.py                      # vòng lặp train/eval, metrics, logging, checkpoint
└── README.md
```

## Cài đặt
- Python 3.9+
- PyTorch, TorchVision (GPU tùy chọn)
- OpenCV, NumPy, PyYAML, scikit-learn, matplotlib, seaborn, tqdm

Cài đặt ví dụ (CPU-only; nếu dùng CUDA, cài bản phù hợp hệ thống):
```bash
python -m venv .venv
. .venv/Scripts/activate        # Windows Git Bash/PowerShell: .venv\Scripts\activate
pip install -r requirements.txt
```

## Dữ liệu
Đặt dữ liệu dưới `data/` với cấu trúc:
```
data/
├── train/
│   ├── Authentic/*.jpg|*.png|...
│   ├── AI/*.jpg|*.png|...
│   └── Splicing/*.jpg|*.png|...
└── test/
    ├── Authentic/*.jpg|*.png|...
    ├── AI/*.jpg|*.png|...
    └── Splicing/*.jpg|*.png|...
```
- Tên thư mục lớp phải trùng `data.class_names` trong file cấu hình.
- Ảnh được resize về `data.input_size` (mặc định 224).

## Cấu hình
Tham khảo `configs/sample.yaml`:
```
data:
  root: "./data"
  input_size: 224
  num_classes: 3
  class_names: ["Authentic", "AI", "Splicing"]

device: "cuda"  # hoặc "cpu"

training:
  model_dir: "./outputs/ImageForgeryDetection"
  experiment_name: "image_forgery_detection"
  total_epochs: 200
  batch_size: 4
  optimization:
    optimizer: "AdamW"
    learning_rate: 2e-5
    weight_decay: 1e-4
    scheduler: "ReduceLROnPlateau"
    scheduler_factor: 0.1
    scheduler_patience: 5

model:
  name: "ImageForgeryDetection"
  backbone: "resnet18"  # resnet18|resnet34|resnet50
  pretrained: true
  input_channels: 4       # RGB + ELA
  dropout: 0.2
  ela:
    quality: 75
    enabled: true
  fft:
    enabled: true
    epsilon: 1e-10
  fusion:
    cnn_features: 1000
    fft_features: 65536
    hidden_dim: 512
    output_dim: 3

evaluation:
  save_predictions: true
  save_confusion_matrix: true

checkpoint:
  save_dir: "./checkpoints"
  save_best: true
  save_last: true
  resume: false
  resume_path: ""
```

## Cách dùng
### train
```bash
python main.py --config_path configs/sample.yaml --task train
```
Tiếp tục từ checkpoint mới nhất trong `checkpoint.save_dir`:
```bash
python main.py --config_path configs/sample.yaml --task train --resume
```
Khôi phục từ checkpoint chỉ định:
```bash
python main.py --config_path configs/sample.yaml --task train --checkpoint_path checkpoints/best.pth
```

### Kiểm tra nhanh
Chạy forward một batch từ tập `test` để kiểm tra setup:
```bash
python main.py --config_path configs/sample.yaml --task test
```
Lưu ý: `--task eval` chưa được triển khai trong `main.py`.

## Kết quả/Outputs
- Log và artifact: `training.model_dir` (mặc định `./outputs/ImageForgeryDetection`)
  - File log: `<experiment_name>.log`
  - Ma trận nhầm lẫn: `confusion_matrix_epoch_*.png` (nếu bật)
- Checkpoints: `checkpoint.save_dir` (mặc định `./checkpoints`)
  - `latest.pth`, `best.pth`, cùng checkpoint theo epoch

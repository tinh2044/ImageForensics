import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class DWConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.dw_bn = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dw(x)
        x = self.dw_bn(x)
        x = self.act(x)
        x = self.pw(x)
        x = self.pw_bn(x)
        x = self.act(x)
        return x


class LightBackbone(nn.Module):
    def __init__(self, in_channels=4, d_cnn=256):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)  
        self.bn0 = nn.BatchNorm2d(32)
        self.act = nn.ReLU(inplace=True)

        self.ds1 = DWConv(32, 64, stride=1)  
        self.ds2 = DWConv(64, 128, stride=2) 
        self.ds3 = DWConv(128, 128, stride=1)

        self.conv_proj = nn.Conv2d(128, d_cnn, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_proj = nn.BatchNorm2d(d_cnn)

    def forward(self, x):
        # x: N x C_in x H x W
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.act(x)

        x = self.ds1(x)
        x = self.ds2(x)
        x = self.ds3(x)

        x = self.conv_proj(x)
        x = self.bn_proj(x)
        x = self.act(x)

        x = x.mean(dim=(2, 3))
        return x


class FFTExtractor(nn.Module):
    def __init__(self, K=16, d_fft=None):
        super().__init__()
        self.K = K
        self.d_fft = K * K if d_fft is None else d_fft
        self.proj = nn.Linear(self.d_fft, self.d_fft)  # identity-size projector (learnable)
        self.norm = nn.LayerNorm(self.d_fft)

    def forward(self, gray):
        N, H, W = gray.shape
        f = torch.fft.fft2(gray)
        fshift = torch.roll(f, shifts=(H // 2, W // 2), dims=(1, 2))
        mag = torch.abs(fshift)
        log_mag = torch.log1p(mag)

        h0 = (H - self.K) // 2
        w0 = (W - self.K) // 2
        crop = log_mag[:, h0 : h0 + self.K, w0 : w0 + self.K]  
        flat = crop.reshape(N, -1) 
        flat = self.norm(flat)
        out = self.proj(flat)
        return out


def _tensor_to_uint8_rgb(img_tensor):
    arr = (img_tensor.mul(255.0).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy())
    return arr


def generate_ela_batch(images, quality=75):
    N, C, H, W = images.shape
    elas = []
    for i in range(N):
        img = images[i, :3] 
        img_uint8 = _tensor_to_uint8_rgb(img) 

        bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        try:
            ret, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
            if not ret:
                ela_gray = np.zeros((H, W), dtype=np.uint8)
            else:
                dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # BGR
                if dec is None:
                    ela_gray = np.zeros((H, W), dtype=np.uint8)
                else:
                    dec_rgb = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
                    diff = cv2.absdiff(img_uint8, dec_rgb)
                    ela_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        except Exception:
            ela_gray = np.zeros((H, W), dtype=np.uint8)

        ela_f = ela_gray.astype(np.float32) / 255.0
        elas.append(torch.from_numpy(ela_f).unsqueeze(0))

    elas = torch.stack(elas, dim=0) 
    return elas


class ImageForgeryDetection(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        cfg = config or {}
        model_cfg = cfg.get("model", {})

        # sizes
        self.d_cnn = model_cfg.get("cnn_features", 256)
        self.K = model_cfg.get("fft_K", 16)
        self.d_fft = self.K * self.K

    
        self.backbone = LightBackbone(in_channels=4, d_cnn=self.d_cnn)
        self.fft_extractor = FFTExtractor(K=self.K, d_fft=self.d_fft)

        fusion_dim = self.d_cnn + self.d_fft
        hidden = model_cfg.get("hidden_dim", 128)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(model_cfg.get("dropout", 0.2)),
            nn.Linear(hidden, model_cfg.get("output_dim", 3)),
        )

        self.loss_fn = nn.CrossEntropyLoss()
        # ELA config
        self.ela_enabled = model_cfg.get("ela", {}).get("enabled", True)
        self.ela_quality = model_cfg.get("ela", {}).get("quality", 75)

    def forward(self, src_inputs):
        images = src_inputs.get("images", None)
        labels = src_inputs.get("labels", None)

        if images is None:
            raise ValueError("images must be provided in src_inputs and be a tensor N x 3 x H x W")

        if not torch.is_tensor(images):
            raise ValueError("images must be a torch tensor")

        device = images.device
        N, C, H, W = images.shape
        if C < 3:
            raise ValueError("images must have at least 3 channels (RGB)")

        if self.ela_enabled:
            elas_cpu = generate_ela_batch(images[:, :3, :, :], quality=self.ela_quality)  # N x 1 x H x W (cpu)
            elas = elas_cpu.to(device=device, dtype=images.dtype)
        else:
            elas = torch.zeros((N, 1, H, W), dtype=images.dtype, device=device)

        rgb = images[:, :3, :, :]

        combined = torch.cat([rgb, elas], dim=1)

        v_cnn = self.backbone(combined)  

        r = rgb[:, 0, :, :]
        g = rgb[:, 1, :, :]
        b = rgb[:, 2, :, :]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b  
        gray = gray * 255.0 

        v_fft = self.fft_extractor(gray) 
        
        z = torch.cat([v_cnn, v_fft], dim=1) 
        logits = self.classifier(z)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss = self.loss_fn(logits, labels)

        return loss, logits


def create_model(config=None):
    cfg = config or {"model": {}}
    return ImageForgeryDetection(cfg)


# Optional helper to count params
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = create_model()
    model.eval()

    x = torch.rand(2, 3, 224, 224)
    inputs = {"images": x}
    with torch.no_grad():
        loss, logits = model(inputs)
    print("Logits shape:", logits.shape)
    print("Param count:", count_parameters(model))

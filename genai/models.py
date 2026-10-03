"""Network definitions for all four tasks."""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================ Tasks 1-3: autoencoder
def conv_block(cin, cout, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.LeakyReLU(0.2, inplace=True),
    )


class Encoder(nn.Module):
    """128x128x3 -> (128/2^depth)^2 x latent_ch. Spatial size halves, channels grow at every stage."""

    def __init__(self, base=32, depth=3, latent_ch=16, dropout=0.0):
        super().__init__()
        chans = [base * 2 ** i for i in range(depth + 1)]           # e.g. 32, 64, 128, 256
        layers = [conv_block(3, chans[0])]
        for i in range(depth):
            layers.append(conv_block(chans[i], chans[i + 1], stride=2))
        self.features = nn.Sequential(*layers)
        self.drop = nn.Dropout2d(dropout)
        self.to_latent = nn.Conv2d(chans[-1], latent_ch, 1)         # the bottleneck: few channels, no skips
        self.out_ch = chans[-1]

    def forward(self, x):
        return self.to_latent(self.drop(self.features(x)))


class Decoder(nn.Module):
    def __init__(self, base=32, depth=3, latent_ch=16, dropout=0.0):
        super().__init__()
        chans = [base * 2 ** i for i in range(depth + 1)][::-1]     # 256, 128, 64, 32
        self.from_latent = nn.Sequential(nn.Conv2d(latent_ch, chans[0], 3, 1, 1), nn.LeakyReLU(0.2, True))
        self.drop = nn.Dropout2d(dropout)
        ups = []
        for i in range(depth):   # bilinear upsampling + conv avoids checkerboard artefacts of transposed convs
            ups.append(nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                     conv_block(chans[i], chans[i + 1])))
        self.ups = nn.Sequential(*ups)
        self.head = nn.Conv2d(chans[-1], 3, 3, 1, 1)

    def forward(self, z):
        return torch.sigmoid(self.head(self.ups(self.drop(self.from_latent(z)))))


class DenoisingAE(nn.Module):
    """x_hat = D(E(x_tilde)). No skip connections, so all information passes through the latent."""

    def __init__(self, base=32, depth=3, latent_ch=16, dropout=0.0):
        super().__init__()
        self.cfg = dict(base=base, depth=depth, latent_ch=latent_ch, dropout=dropout)
        self.encoder = Encoder(base, depth, latent_ch, dropout)
        self.decoder = Decoder(base, depth, latent_ch, dropout)

    def latent_dim(self, size=128):
        s = size // 2 ** self.cfg["depth"]
        return s * s * self.cfg["latent_ch"]

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ============================================================ Task 2: corruption classifier
CLS_CHANNELS = {
    "small": (16, 32, 64, 128),
    "medium": (32, 64, 128, 256),
    "large": (32, 64, 128, 256, 256),
}


class CorruptionClassifier(nn.Module):
    """Predicts clean / salt-pepper / blur / occlusion logits.

    The first block keeps full resolution because blur and sparse noise are
    high-frequency cues that early downsampling would destroy.
    """

    def __init__(self, channels="medium", dropout=0.3, n_classes=4):
        super().__init__()
        self.cfg = dict(channels=channels, dropout=dropout)
        ch = CLS_CHANNELS[channels]
        layers, cin = [], 3
        for c in ch:
            layers += [nn.Conv2d(cin, c, 3, 1, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(True),
                       nn.Conv2d(c, c, 3, 1, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(True),
                       nn.MaxPool2d(2)]
            cin = c
        self.features = nn.Sequential(*layers)
        # global avg + max pooling: max-pool keeps evidence of a single occluder / impulse anywhere
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(2 * cin, n_classes))

    def forward(self, x):
        f = self.features(x)
        f = torch.cat([F.adaptive_avg_pool2d(f, 1), F.adaptive_max_pool2d(f, 1)], 1)
        return self.head(f)


# ============================================================ Task 2: hard routing (inference only)
class HardRouter(nn.Module):
    """Classifier + specialists. Clean predictions use the identity bypass."""

    def __init__(self, classifier, experts):
        super().__init__()
        self.classifier = classifier
        self.experts = nn.ModuleList(experts)   # salt, blur, occlusion

    def forward(self, x, route=None):
        logits = self.classifier(x)
        probs = logits.softmax(1)
        r = probs.argmax(1) if route is None else route
        out = x.clone()
        for k, expert in enumerate(self.experts, start=1):
            m = r == k
            if m.any():                       # only the selected expert is executed
                out[m] = expert(x[m])
        return out, probs, r


# ============================================================ Task 3: soft mixture of experts
class SoftMoE(nn.Module):
    """x_hat = w0 * x + w1 A_salt(x) + w2 A_blur(x) + w3 A_occ(x),  w = softmax(G(x) / tau)."""

    def __init__(self, gate, experts, tau=1.0):
        super().__init__()
        self.gate = gate
        self.experts = nn.ModuleList(experts)
        self.register_buffer("tau", torch.tensor(float(tau)))

    def forward(self, x):
        logits = self.gate(x)
        w = torch.softmax(logits / self.tau, dim=1)                           # B x 4
        branches = torch.stack([x] + [e(x) for e in self.experts], dim=1)    # B x 4 x 3 x H x W
        out = (w[:, :, None, None, None] * branches).sum(1)
        return out, w, logits


# ============================================================ Task 4: conditional pix2pix
class Down(nn.Module):
    def __init__(self, cin, cout, norm=True):
        super().__init__()
        layers = [nn.Conv2d(cin, cout, 4, 2, 1, bias=not norm)]
        if norm:
            layers.append(nn.BatchNorm2d(cout))
        layers.append(nn.LeakyReLU(0.2, True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, cin, cout, dropout=0.0):
        super().__init__()
        layers = [nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True)]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x, skip):
        return torch.cat([self.net(x), skip], 1)


class StyleUNetGenerator(nn.Module):
    """U-Net generator G(x, s). The learned style embedding is broadcast as extra input
    channels and injected again at the bottleneck, so the style controls both local
    texture (early layers) and global appearance (bottleneck)."""

    def __init__(self, base=64, emb_dim=8, dropout=0.5, n_styles=3):
        super().__init__()
        self.cfg = dict(base=base, emb_dim=emb_dim, dropout=dropout)
        self.emb = nn.Embedding(n_styles, emb_dim)
        b = base
        self.d1 = Down(3 + emb_dim, b, norm=False)   # 64
        self.d2 = Down(b, 2 * b)                      # 32
        self.d3 = Down(2 * b, 4 * b)                  # 16
        self.d4 = Down(4 * b, 8 * b)                  # 8
        self.d5 = Down(8 * b, 8 * b)                  # 4
        self.d6 = Down(8 * b, 8 * b)                  # 2
        self.bott = nn.Sequential(nn.Conv2d(8 * b + emb_dim, 8 * b, 4, 2, 1), nn.ReLU(True))   # 1
        self.u1 = Up(8 * b, 8 * b, dropout)           # 2
        self.u2 = Up(16 * b, 8 * b, dropout)          # 4
        self.u3 = Up(16 * b, 8 * b, dropout)          # 8
        self.u4 = Up(16 * b, 4 * b)                   # 16
        self.u5 = Up(8 * b, 2 * b)                    # 32
        self.u6 = Up(4 * b, b)                        # 64
        self.out = nn.Sequential(nn.ConvTranspose2d(2 * b, 3, 4, 2, 1), nn.Tanh())

    def forward(self, x, style):
        e = self.emb(style)[:, :, None, None]
        h1 = self.d1(torch.cat([x, e.expand(-1, -1, x.shape[2], x.shape[3])], 1))
        h2 = self.d2(h1)
        h3 = self.d3(h2)
        h4 = self.d4(h3)
        h5 = self.d5(h4)
        h6 = self.d6(h5)
        z = self.bott(torch.cat([h6, e.expand(-1, -1, h6.shape[2], h6.shape[3])], 1))
        y = self.u1(z, h6)
        y = self.u2(y, h5)
        y = self.u3(y, h4)
        y = self.u4(y, h3)
        y = self.u5(y, h2)
        y = self.u6(y, h1)
        return self.out(y)


class StylePatchDiscriminator(nn.Module):
    """70x70-style PatchGAN on (photo, sketch, style map). Output: 14x14 map of real/fake logits for 128px input."""

    def __init__(self, base=64, emb_dim=8, n_styles=3):
        super().__init__()
        self.emb = nn.Embedding(n_styles, emb_dim)
        b = base
        self.net = nn.Sequential(
            nn.Conv2d(6 + emb_dim, b, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(b, 2 * b, 4, 2, 1, bias=False), nn.BatchNorm2d(2 * b), nn.LeakyReLU(0.2, True),
            nn.Conv2d(2 * b, 4 * b, 4, 2, 1, bias=False), nn.BatchNorm2d(4 * b), nn.LeakyReLU(0.2, True),
            nn.Conv2d(4 * b, 8 * b, 4, 1, 1, bias=False), nn.BatchNorm2d(8 * b), nn.LeakyReLU(0.2, True),
            nn.Conv2d(8 * b, 1, 4, 1, 1),
        )

    def forward(self, photo, sketch, style):
        e = self.emb(style)[:, :, None, None].expand(-1, -1, photo.shape[2], photo.shape[3])
        return self.net(torch.cat([photo, sketch, e], 1))


def init_gan_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)

import torch
import torch.nn as nn
import timm
import open_clip
from timm.models.vision_transformer import Block
from models.swin import SwinTransformer
from einops import rearrange


# === 复制 TABlock 和 SaveOutput，保证文件独立性 ===
class TABlock(nn.Module):
    def __init__(self, dim, drop=0.1):
        super().__init__()
        self.c_q = nn.Linear(dim, dim)
        self.c_k = nn.Linear(dim, dim)
        self.c_v = nn.Linear(dim, dim)
        self.norm_fact = dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x):
        _x = x
        B, C, N = x.shape
        q = self.c_q(x)
        k = self.c_k(x)
        v = self.c_v(x)

        attn = q @ k.transpose(-2, -1) * self.norm_fact
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, C, N)
        x = self.proj_drop(x)
        x = x + _x
        return x


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []


# === 模块 1: 语义映射器 (Semantic Mapper) ===
class SemanticMapper(nn.Module):
    def __init__(self, clip_dim, maniqa_dim):
        super().__init__()
        # 这是一个简单的 MLP: CLIP特征 -> 缩放因子(Gamma) 和 偏移因子(Beta)
        self.net = nn.Sequential(
            nn.Linear(clip_dim, maniqa_dim // 4),
            nn.ReLU(),
            nn.Linear(maniqa_dim // 4, maniqa_dim * 2)  # 输出 gamma 和 beta
        )

        # [核心技巧] 零初始化 (Zero-Initialization)
        # 最后一层初始化为0，确保训练初始阶段 F_new = F_old * (1+0) + 0，即保留原模型能力
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, clip_embedding):
        # clip_embedding: [B, 512]
        # output: [B, maniqa_dim * 2]
        out = self.net(clip_embedding)
        gamma, beta = out.chunk(2, dim=1)  # 分割成两份
        return gamma, beta


# === 模块 2: 畸变诊断头 (Diagnosis Head) ===
class DiagnosisHead(nn.Module):
    def __init__(self, in_dim, num_classes=25):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.pool(x)  # [B, C, 1, 1]
        x = x.flatten(1)  # [B, C]
        logits = self.classifier(x)  # [B, num_classes]
        return logits


# === 主模型: MANIQA + CLIP + Diagnosis ===
class MANIQA_NEW(nn.Module):
    # [修改] 增加了 num_classes 参数，默认为 25
    def __init__(self, embed_dim=768, num_outputs=1, patch_size=8, drop=0.1,
                 depths=[2, 2], window_size=4, dim_mlp=768, num_heads=[4, 4],
                 img_size=224, num_tab=2, scale=0.8, num_classes=25, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.input_size = img_size // patch_size
        self.patches_resolution = (img_size // patch_size, img_size // patch_size)

        # 1. 基线部分: ViT
        self.vit = timm.create_model('vit_base_patch8_224', pretrained=True)
        self.save_output = SaveOutput()
        hook_handles = []
        for layer in self.vit.modules():
            if isinstance(layer, Block):
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)

        # 2. 新增部分: CLIP (冻结参数)
        print("Loading CLIP model (ViT-B-16)...")
        self.clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')
        for param in self.clip_model.parameters():
            param.requires_grad = False  # 彻底冻结

        # 计算特征维度: MANIQA extract_feature 拼接了4层 (6,7,8,9)，每层768
        self.maniqa_feat_dim = 768 * 4

        # 3. 新增部分: 语义映射器
        self.semantic_mapper = SemanticMapper(clip_dim=512, maniqa_dim=self.maniqa_feat_dim)

        # 继续基线部分
        self.tablock1 = nn.ModuleList()
        for i in range(num_tab):
            tab = TABlock(self.input_size ** 2)
            self.tablock1.append(tab)

        self.conv1 = nn.Conv2d(embed_dim * 4, embed_dim, 1, 1, 0)
        self.swintransformer1 = SwinTransformer(
            patches_resolution=self.patches_resolution,
            depths=depths,
            num_heads=num_heads,
            embed_dim=embed_dim,
            window_size=window_size,
            dim_mlp=dim_mlp,
            scale=scale
        )

        self.tablock2 = nn.ModuleList()
        for i in range(num_tab):
            tab = TABlock(self.input_size ** 2)
            self.tablock2.append(tab)

        self.conv2 = nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0)
        self.swintransformer2 = SwinTransformer(
            patches_resolution=self.patches_resolution,
            depths=depths,
            num_heads=num_heads,
            embed_dim=embed_dim // 2,
            window_size=window_size,
            dim_mlp=dim_mlp,
            scale=scale
        )

        # 4. 新增部分: 畸变诊断头 (使用传入的 num_classes)
        # [修改] 这里的 25 变成了 num_classes 变量
        self.diagnosis_head = DiagnosisHead(in_dim=embed_dim // 2, num_classes=num_classes)

        self.fc_score = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim // 2, num_outputs),
            nn.ReLU()
        )
        self.fc_weight = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim // 2, num_outputs),
            nn.Sigmoid()
        )

        # CLIP 归一化参数 (Mean, Std)
        self.register_buffer('clip_mean', torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer('clip_std', torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))

    def extract_feature(self, save_output):
        x6 = save_output.outputs[6][:, 1:]
        x7 = save_output.outputs[7][:, 1:]
        x8 = save_output.outputs[8][:, 1:]
        x9 = save_output.outputs[9][:, 1:]
        x = torch.cat((x6, x7, x8, x9), dim=2)
        return x

    def process_for_clip(self, x):
        # x 是 MANIQA 的输入，已经做过 (x - 0.5)/0.5 的归一化
        # 我们先把它还原回 [0, 1]
        x_unnorm = x * 0.5 + 0.5
        # 然后应用 CLIP 的归一化
        x_clip = (x_unnorm - self.clip_mean) / self.clip_std
        return x_clip

    def forward(self, x):
        # 1. 获取 CLIP 语义向量
        with torch.no_grad():
            x_clip_in = self.process_for_clip(x)
            # encode_image 返回 [B, 512]
            semantic_vec = self.clip_model.encode_image(x_clip_in)
            semantic_vec = semantic_vec.float()  # 确保精度一致

        # 2. 计算 FiLM 参数 (Gamma, Beta)
        gamma, beta = self.semantic_mapper(semantic_vec)
        # 调整形状以便广播: [B, C] -> [B, 1, C]
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)

        # 3. MANIQA 原始流程
        _x = self.vit(x)
        x = self.extract_feature(self.save_output)  # x shape: [B, N, 3072]
        self.save_output.outputs.clear()

        # === 关键点: 语义注入 (Semantic Injection) ===
        # Formula: F_new = F_old * (1 + gamma) + beta
        x = x * (1 + gamma) + beta

        # stage 1
        x = rearrange(x, 'b (h w) c -> b c (h w)', h=self.input_size, w=self.input_size)
        for tab in self.tablock1:
            x = tab(x)
        x = rearrange(x, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
        x = self.conv1(x)
        x = self.swintransformer1(x)

        # stage 2
        x = rearrange(x, 'b c h w -> b c (h w)', h=self.input_size, w=self.input_size)
        for tab in self.tablock2:
            x = tab(x)
        x = rearrange(x, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
        x = self.conv2(x)
        x = self.swintransformer2(x)  # x shape: [B, 384, H, W]

        # === 关键点: 畸变诊断分支 (Distortion Diagnosis) ===
        # 这里 x 还在 [B, C, H, W] 状态，非常适合做 GlobalAvgPool
        dist_logits = self.diagnosis_head(x)

        # 继续回归分支 (Scoring)
        x = rearrange(x, 'b c h w -> b (h w) c', h=self.input_size, w=self.input_size)
        score = torch.tensor([]).cuda()
        for i in range(x.shape[0]):
            f = self.fc_score(x[i])
            w = self.fc_weight(x[i])
            _s = torch.sum(f * w) / torch.sum(w)
            score = torch.cat((score, _s.unsqueeze(0)), 0)

        # 返回两个结果: 分数 和 畸变分类logits
        return score, dist_logits
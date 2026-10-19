# EVA3 —— EVA（Decoupling Primitive with Experts, ICLR 2026）的 Troika 框架复现实现

> 基于 [Troika](https://github.com/bighuang624/Troika)（CVPR 2024, arXiv:2303.15230）
> MIT 代码框架复现的 **EVA**（Mixture-of-Experts Framework for Semantic Variant
> Alignment）完整实现。本仓库包含**完整可运行代码**：Troika 原有文件原样保留，
> 新增 MoE Adapter 与语义变体对齐模块。

## 1. 方法概要

EVA 在冻结 CLIP ViT-L/14 上做两个核心改动（方法归属原 EVA 论文，本仓库为其
基于 Troika 代码框架的复现/扩展实现）：

1. **Domain-Expert Adaption（MoE Adapter）**：在图像与文本 encoder 的**每一层
   FFN 旁并行注入** MoE Adapter——1 个 shared expert + 8 个 routed expert，
   TopK=2，每个 expert 为 LoRA 结构（r=64，B 矩阵零初始化使训练从冻结 CLIP
   平滑启动），Router 为单线性层的 token→expert 亲和度打分（图像 24 层 + 文本
   12 层全覆盖）；
2. **Semantic Variant Alignment（SVA）**：
   - text-to-image：partner-max 损失（带可学习温度 τs/τo，scatter-max 实现）；
   - image-to-text：把最后一层各 expert 输出视为图像变体 CLS，做 inter/intra
     affinity 后 argmax 选择变体做对齐；
3. **训练目标**：`L = Lc + λ1·(Ls + Lo) + λ2·(Lvs + Lvo)`，组合损失为
   `softmax(fc·tc/τ)` 上的 CE（τ 为 CLIP 原生温度）；三套 soft prompt
   （Pc/Ps/Po，"a photo of x x" 风格，Troika 式）；
4. **推理打分**：`score = pc + β·(ps + po)`。

关键超参（论文 Sec 4.1 / 附录）：Ne=8, K=2, r=64, α=0.5, λ1=0.5, λ2=0.1,
β=0.5, lr=1e-4, 20 epoch, Adam；weight decay：MIT 1e-4 / UT 5e-4 / CGQA 1e-4。
可训练参数约 39.2M。

## 2. 仓库结构

```
EVA3/
├── README.md                 # 本文件
└── code/                     # 代码主体
    ├── README.md             # 详细使用说明（训练/评估命令、论文对应表、实现选择）
    ├── REPRODUCTION_REPORT.md # 复现报告（复现值 vs 论文值、分歧点分析）
    ├── LICENSE               # MIT（继承自 Troika，版权行保持原样）
    ├── model/
    │   ├── eva.py            # ★新增★ EVA 主模型（MoE Adapter 注入 + 对齐损失 + 推理打分）
    │   ├── moe_adapter.py    # ★新增★ LoRAExpert / Router / MoEAdapter
    │   └── model_factory.py  # 注册 eva
    ├── config/eva/           # 三个数据集 × closed/open-world 配置
    │   ├── mit-states.yml / ut-zappos.yml / cgqa.yml（及 *-ow.yml）
    │   └── ut-zappos-{base,exp,exp2,lr,nosva,notxt,scale,stab}.yml  # 消融配置
    ├── train.py / test.py    # 训练与评估入口（train 复用 Troika，test 改造为 EVA 打分）
    ├── parameters.py         # 超参（含 EVA 专属参数）
    ├── dataset.py / utils.py # CompositionDataset 等（复用）
    ├── clip_modules/         # CLIP 模型与 tokenizer（复用）
    ├── tools/                # 优化器 / mixup（复用）
    └── download_data/        # 数据集下载与整理脚本（含 open-world feasibility）
```

## 3. 复现步骤

### 3.1 环境

Troika 同款环境即可（存档使用 torch 1.13.1+cu117）：

```bash
conda create -n troika python=3.8
pip install -r code/requirements.txt
```

另需 OpenAI CLIP `ViT-L-14.pt` 权重，通过 `--clip_arch` 传入。

### 3.2 数据集

MIT-States、UT-Zappos、C-GQA，使用仓库内脚本下载整理：

```bash
cd code/download_data
bash download_data.sh
# open-world 评估还需 data/feasibility_<dataset>.pt，
# 由 download_data/feasibility.py 生成（配置自动加载）
```

### 3.3 训练（Closed-World）

```bash
cd code
CUDA_VISIBLE_DEVICES=0 python -u train.py \
    --yml_path config/eva/ut-zappos.yml \
    --dataset_path data/ut-zappos \
    --clip_arch /path/to/ViT-L-14.pt \
    --save_path output/ut --num_workers 10 --seed 0
# MIT-States / C-GQA 替换 config 与数据路径即可
```

每个 epoch 自动在 val 上评估，按 `val_metric: AUC` 保存 `val_best.pt`。

### 3.4 评估

```bash
# Closed-World
python -u test.py --yml_path config/eva/ut-zappos.yml \
    --dataset_path data/ut-zappos --clip_arch /path/to/ViT-L-14.pt \
    --save_path output/ut --load_model output/ut/val_best.pt

# Open-World（使用 *-ow.yml，自动加载 feasibility_*.pt）
python -u test.py --yml_path config/eva/ut-zappos-ow.yml \
    --dataset_path data/ut-zappos --clip_arch /path/to/ViT-L-14.pt \
    --save_path output/ut --load_model output/ut/val_best.pt
```

注意：MoE 显存开销高于 Troika，A100-40G 实测 batch 需降为 UT 32（梯度累积 2）、
MIT 32、CGQA 24。更多实现细节与论文未明确的实现选择见
[`code/README.md`](code/README.md)。

## 4. 复现结果

以下为 [`code/REPRODUCTION_REPORT.md`](code/REPRODUCTION_REPORT.md) 中记录的
**本仓库实现的复现值**（括号内为 EVA 论文报告值，供对照；复现值低于论文值，
差距来源分析见该报告）：

| 设置 | 数据集 | 复现 AUC | 论文 AUC |
|---|---|---|---|
| Closed-World | UT-Zappos | 37.3 | 50.2 |
| Closed-World | MIT-States | 20.8 | 24.0 |
| Open-World | UT-Zappos | 24.9 | 40.2 |
| Open-World | MIT-States | 6.5 | 9.4 |

横向参照（UT-Zappos closed AUC）：CSP 33.0、无 MoE/SVA 的 CLIP+软提示基线 29.7、
Troika 41.7、本复现 37.3（论文 EVA 50.2）。

## 5. 引用

本实现基于 Troika 代码框架，请引用：

```bibtex
@inproceedings{Huang2024Troika,
  title={Troika: Multi-Path Cross-Modal Traction for Compositional Zero-Shot Learning},
  author={Siteng Huang and Biao Gong and Yutong Feng and Min Zhang and Yiliang Lv and Donglin Wang},
  booktitle={CVPR},
  year={2024}
}
```

同时建议引用 EVA 原论文：*EVA: Decoupling Primitive with Experts*（ICLR 2026，
Mixture-of-Experts Framework for Semantic Variant Alignment）；本仓库为其
独立复现实现，方法权利归原论文作者。

## 6. License

本仓库基于 **MIT 协议的 Troika** 发布：随代码保留的 MIT LICENSE 文本（
[code/LICENSE](code/LICENSE)，版权行 `Copyright (c) 2022 Carman Lu` 保持原样）
继续覆盖 Troika 原有文件；本仓库新增模块（`model/eva.py`、`model/moe_adapter.py`
及 EVA 相关配置/脚本）同样以 MIT 协议发布。

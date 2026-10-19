# EVA: Decoupling Primitive with Experts (ICLR 2026) — 复现实现

基于 Troika (CVPR 2024) 代码框架复现的 **EVA**（Mixture-of-Experts Framework for
Semantic Variant Alignment）。冻结 CLIP ViT-L/14，在图像与文本 encoder 的**每一层
FFN 旁并行注入 MoE Adapter**（domain-expert adaption），并用 **Semantic Variant
Alignment (SVA)** 做细粒度的 image-primitive 对齐。

## 1. 目录结构

```
code/
├── train.py                # 训练入口（复用自 Troika，无需改）
├── test.py                 # 评估入口（已改造为 EVA 的文本缓存 + Eq.16 打分）
├── dataset.py              # CompositionDataset（复用）
├── parameters.py           # 所有超参（已加入 EVA 专属参数）
├── utils.py                # 优化器/调度器/yml 加载（复用）
├── config/eva/             # 各数据集配置
│   ├── mit-states.yml / ut-zappos.yml / cgqa.yml
│   └── *-ow.yml            # open-world 版本
├── model/
│   ├── model_factory.py    # 注册 eva
│   ├── moe_adapter.py      # LoRAExpert / Router / MoEAdapter
│   └── eva.py              # ★EVA 主模型★
├── clip_modules/           # CLIP 模型与 tokenizer（复用）
├── tools/                  # 优化器/warmup（复用）
└── data/                   # feasibility_*.pt（open-world 用，复用）
```

## 2. 环境

直接复用服务器上的 `troika` conda 环境（torch 1.13.1+cu117）：

```bash
source ./anaconda3/etc/profile.d/conda.sh
conda activate troika
```

关键路径（本机已就位）：
- CLIP 权重：`./lyt/czsl/CLIP-main/ViT-L-14.pt`（加载后为 fp32）
- 数据集：`./8222/all_data/{mit-states,ut-zappos,cgqa}`

## 3. 训练（Closed-World）

```bash
# UT-Zappos
CUDA_VISIBLE_DEVICES=0 python -u train.py \
    --yml_path config/eva/ut-zappos.yml \
    --dataset_path data/ut-zappos \
    --clip_arch ./lyt/czsl/CLIP-main/ViT-L-14.pt \
    --save_path output/ut --num_workers 10 --seed 0

# MIT-States
CUDA_VISIBLE_DEVICES=0 python -u train.py \
    --yml_path config/eva/mit-states.yml \
    --dataset_path data/mit-states \
    --clip_arch ./lyt/czsl/CLIP-main/ViT-L-14.pt \
    --save_path output/mit --num_workers 10 --seed 0

# C-GQA
CUDA_VISIBLE_DEVICES=0 python -u train.py \
    --yml_path config/eva/cgqa.yml \
    --dataset_path data/cgqa \
    --clip_arch ./lyt/czsl/CLIP-main/ViT-L-14.pt \
    --save_path output/cgqa --num_workers 10 --seed 0
```

- 训练每个 epoch 自动在 val 上评估，按 `val_metric: AUC` 保存最优权重到
  `<save_path>/val_best.pt`；最后用 val_best 在 test 上做闭世界评估。
- 后台跑：`nohup env CUDA_VISIBLE_DEVICES=0 python -u train.py ... > logs/xxx.log 2>&1 &`

## 4. 评估

### Closed-World
```bash
CUDA_VISIBLE_DEVICES=0 python -u test.py \
    --yml_path config/eva/ut-zappos.yml \
    --dataset_path data/ut-zappos \
    --clip_arch ./lyt/czsl/CLIP-main/ViT-L-14.pt \
    --save_path output/ut --num_workers 10 --seed 0 \
    --load_model output/ut/val_best.pt
```

### Open-World
用 `*-ow.yml` 配置（会自动加载 `data/feasibility_<dataset>.pt` 做可行性校准）：
```bash
CUDA_VISIBLE_DEVICES=0 python -u test.py \
    --yml_path config/eva/ut-zappos-ow.yml \
    --dataset_path data/ut-zappos \
    --clip_arch ./lyt/czsl/CLIP-main/ViT-L-14.pt \
    --save_path output/ut --num_workers 10 --seed 0 \
    --load_model output/ut/val_best.pt
```

## 5. 方法实现要点（与论文对应）

| 模块 | 论文 | 代码位置 |
|---|---|---|
| MoE Adapter（1 shared + 8 routed, K=2, r=64, 并行 FFN） | Eq.1-3, Sec 3.3 | `model/moe_adapter.py`, `eva.encode_image/encode_text` |
| 三套 soft prompt Pc/Ps/Po | Eq.4,9 | `eva.construct_soft_prompt / construct_token_tensors` |
| 组合损失 Lc | Eq.5-6 | `eva.loss_calu` |
| Text-to-image（partner-max + 可学习 τs/τo） | Eq.7-8 | `eva._partner_max`, `loss_calu` |
| Image-to-text（per-expert CLS 变体 + inter/intra affinity + argmax） | Eq.9-14 | `eva.encode_image`(变体), `loss_calu` |
| 总损失 L = Lc + λ1(Ls+Lo) + λ2(Lvs+Lvo) | Eq.15 | `loss_calu` |
| 推理 score = pc + β(ps+po) | Eq.16 | `eva.logit_infer` |

超参（论文 Sec 4.1 / 附录）：Ne=8, K=2, r=64, α=0.5, λ1=0.5, λ2=0.1, β=0.5,
lr=1e-4, 20 epoch, Adam；weight decay：MIT 1e-4 / UT 5e-4 / CGQA 1e-4。

## 6. 论文未明确、实现中做的选择（待验证，详见与作者沟通记录）

1. **partner-max 的组合空间**：训练用 seen 集 Cs，推理用 C_test（closed=Cs∪Cu，
   open=全空间 C）。由论文 Eq.16 `argmax over C_test` 与"训练只可见 Cs"推得。
2. **ps 的尺度**：严格按 Eq.7 用 `softmax(pc)` 后取 max 再乘 τs，结果送 CrossEntropy。
3. **图像变体 v_i 的构造**：取最后一层 CLS 的「残差+FFN」基底，分别加上单个 expert
   （shared 为 v0，各 routed 为 v1..vNe）的输出，再过 `ln_post + proj` 投影到 768 维。
   即变体是"完整 CLS token，只换单个 expert"。
4. **推理 Eq.16 中的 ps/po 是否带 τ**：当前实现带 τs/τo（与 Eq.7 定义一致）。
5. **学习率调度**：论文未提，沿用 Troika 的 StepLR(step=5, gamma=0.5)；可在 yml 关闭。
6. **batch size**：论文未给具体值。EVA 的 MoE 比 Troika 显存重，单卡 A100-40G 下
   训练 batch 实测需降到：UT 32（梯度累积 2，等效 64）/ MIT 32 / CGQA 24。可按显存调整。
7. **MoE 显存/实现**：采用稀疏 dispatch（每 token 只算 top-K 个 routed expert + shared）。
   论文 Table 8 报 UT 训练仅 3min30s，疑似用了更激进的批处理或特征缓存；本实现为忠实
   端到端版本，UT 约 15min/epoch，不影响精度复现。
8. **梯度检查点**：C-GQA 的 seen 组合多达 5592，每步全量编码文本的反向图很大，单卡 40G
   会 OOM。已实现 `--grad_checkpoint`（或 yml 里 `grad_checkpoint: True`），反向时重算激活，
   C-GQA bs24 峰值显存仅 ~7.3G。MIT/UT 显存足够，默认不开。

### C-GQA 启动命令（已就绪，需要时直接跑）
```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python -u train.py \
    --yml_path config/eva/cgqa.yml \
    --dataset_path data/cgqa \
    --clip_arch ./lyt/czsl/CLIP-main/ViT-L-14.pt \
    --save_path output/cgqa --num_workers 10 --seed 0
```

以上 1-6 都集中在配置/`eva.py` 中，便于按作者答复调整。

## 7. 论文目标指标（用于 sanity check）

Closed-World：MIT AUC 24.0 / UT AUC 50.2 / CGQA AUC 18.8
Open-World：MIT AUC 9.4 / UT AUC 40.2 / CGQA AUC 5.6
（消融定位：仅 prompt baseline ≈ MIT AUC 20.2；+MoE ≈ 23.0）

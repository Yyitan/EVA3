# EVA (ICLR 2026) 复现报告

## 1. 复现结果总表（vs 论文）

### Closed-World
| 数据集 | 指标 | 复现 | EVA |
|---|---|---|---|
| UT-Zappos | AUC | **37.3** | 50.2 |
| | Seen / Unseen / HM | 64.3 / 75.6 / 49.3 | 71.2 / 79.6 / 60.2 |
| MIT-States | AUC | **20.8** | 24.0 | 
| | Seen / Unseen / HM | 47.5 / 51.8 / 37.7 | 51.2 / 55.0 / 41.0 | 


### Open-World
| 数据集 | AUC（复现 / 论文） | Seen / Unseen / HM |
|---|---|---|
| UT-Zappos | **24.9** / 40.2 | 64.4 / 50.2 / 39.2 |
| MIT-States | **6.5** / 9.4 | 47.4 / 18.1 / 18.6 |

### 横向参照（UT-Zappos closed AUC，越能说明差距来源）
| 方法 | test AUC |
|---|---|
| CSP (ICLR'23) | 33.0 |
|（CLIP+软提示，无MoE/SVA） | **29.7** |
| Troika (CVPR'24) | 41.7 |
| **完整 EVA** | **37.3** |
| **论文 EVA** | **50.2** |

---

## 2. 实现要点

基于 Troika 代码框架，新增 `model/eva.py`、`model/moe_adapter.py`。

| 模块 | 论文 | 实现 | 核对 |
|---|---|---|---|
| Backbone | 冻结 CLIP ViT-L/14 | 同（fp32） | ✅ |
| MoE Adapter | 1 shared + 8 routed, TopK=2, LoRA r=64, 并行 FFN, 图24+文12层 | 同 | ✅ Eq.1-3 |
| 组合损失 Lc | `softmax(fc·tc/τ)` → CE，τ=CLIP原生(=100) | 同 | ✅ Eq.5-6 |
| 三套软提示 Pc/Ps/Po | "a photo of state object" 等 | Troika式 soft prompt | ✅ Eq.4,9 |
| t2i 损失 Ls/Lo | `max_partner pc · τs` → CE | 同（scatter-max） | ✅ Eq.7-8 |
| i2t 变体对齐 | 变体=最后层各expert；inter/intra affinity；argmax选变体 | 同 | ✅ Eq.9-14 |
| 总损失 | `Lc + λ1(Ls+Lo) + λ2(Lvs+Lvo)` | 同 | ✅ Eq.15 |
| 推理 | `pc + β(ps+po)` | 同 | ✅ Eq.16 |

超参（论文 4.1 + 附录）：Ne=8, K=2, r=64, α=0.5, λ1=0.5, λ2=0.1, β=0.5, τ初值1.0, lr=1e-4, Adam, 20 epoch；weight decay：MIT 1e-4 / UT 5e-4 / CGQA 1e-4。可训练参数 39.2M（论文 36.7M）。

---

## 3. 论文未明确、由我们自行决定的点（潜在分歧来源）

| # | 项目 | 论文怎么说 | 我们的选择 |
|---|---|---|---|
| 1 | 图像变体 vi 构造 | "CLS tokens from various experts"| base(残差+FFN)+单expert；另：EVA2 用纯expert输出，结果相同 |
| 2 | partner-max 组合空间 | Eq.7 只写 Ctarget | 训练用 seen 集 Cs，推理用 Ctest（按原文推断） |
| 3 | ps 的尺度 | Eq.7 `max(pc)·τs` | softmax(pc)后取max再×τs，送CE |
| 4 | 学习率调度 | 未提 | 默认无（恒定 1e-4）；Troika 式 StepLR 也试过 |
| 5 | batch size | 未提 | UT 等效64 / MIT 32 / CGQA 24 |
| 6 | 上下文长度 / dropout | 未提 | context_length=8；attr_dropout=0.3 |
| 7 | open-world 阈值搜索数 | 未提 | 50 |

---

## 4. 全部实验记录（均在 UT-Zappos 上，对照原版 ep2 峰值 0.445 后崩盘）

每个实验通过 `config/eva/ut-zappos-*.yml` + 对应代码开关（默认全关=原始实现）控制。

| # | 实验 | 改动 | 峰值 AUC(轮次) | 结论 |
|---|---|---|---|---|
| 0 | 原版 | eff batch64, StepLR, lr1e-4 | 0.445 (ep2) | 基准，ep2后崩到0.25 |
| 1 | 大 batch | eff batch128 | 0.399 (ep2) | ❌ 更差 |
| 2 | t2i 用 logits | `t2i_use_logits` | 0.455 (ep2) | ❌ 仅微涨，照崩 |
| 3 | 关文本 MoE | `disable_text_moe` | 0.429 (ep2) | ❌ 下跌略缓，照崩 |
| 4 | expert 缩放 | `expert_scale=0.1` | 0.433 (ep2) | ❌ 照崩（scale被权重吸收）；test 0.401 略升 |
| 5 | 无 SVA | λ1=λ2=0（纯MoE+Lc） | 0.453 (ep2) | MoE有效(+0.10)，但照崩 |
| 6 | 纯 base | +`expert_scale=0`（无MoE无SVA） | 0.355 (ep20) | ✅ **稳定不崩，≈CSP** |
| 7 | 低 lr | lr=2e-5 恒定 | 0.410 (ep2) | ❌ 更差，照崩 |
| 8 | 稳定化组合 | dropout0.1 + wd×10 + warmup+cosine | 0.425 (ep3) | ❌ 照崩，峰值更低 |

**核心观察**：纯 base（无 MoE）稳定训练、爬升到 ep20 不崩；**一旦加上 MoE，第 2-3 轮就过拟合崩盘，且此崩盘对所有正则化/优化手段免疫**。

---

## 5. 诊断实验（定位问题，非调参）

| 诊断 | 方法 | 结果 | 含义 |
|---|---|---|---|
| Troika 基线曲线 | 在 UT 上跑 Troika | ep3 峰 0.502 后跌到 0.40 | **早见顶是正常的**（Troika也这样） |
| CAMS（开源）曲线 | 读其 UT 训练日志 | ep3 峰 0.49，之后**稳在 0.45** | 早见顶正常；但好方法见顶后**稳住**，我们却崩 |
| MoE router 负载 | hook checkpoint 路由 | 各层专家分布多样、**无死专家** | MoE 路由健康，未塌缩 |
| 变体差异度 | 量 checkpoint 变体余弦 | 变体间 0.89，与f_c 0.94 | 变体有差异、未塌缩（但被base主导） |
| 纯 base vs CSP | 跑纯软提示 | test 0.297 vs CSP 0.330 | base 已比 CSP 低 ~3 点 |

---

## 6. 关键结论与分析

 **差距主要在"方法增益不足"**：
   - 纯 base ≈ CSP（差 3 点）。
   - MoE+SVA 加 **+7.6** 点（0.297→0.373）；论文方法加 **+17** 点（0.330→0.502）。
   - **我们的方法增益只有论文的一半左右。**




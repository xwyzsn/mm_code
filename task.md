HyperLoRA 事件条件动态超网络模型

> **核心建模思想**：将 **Qwen3.8-27B** 作为“事件条件超网络（Event-Conditioned HyperNetwork）”。Qwen 读取历史弹性与当前事件组合后，实时生成 **TimesFM 3.0** 末 4 层线性模块的**专属动态奇异值矩阵 $\Sigma(E)$（Hyper-DoRA）**以及**输出头的动态低秩投影权重 $\Delta W_{\text{head}}(E, t)$**，实现“一事一算子、无事件零漂移”。

```markdown
# Role & Mission
你是一位顶尖的多模态时序大模型架构师与 PyTorch 工程专家。当前目录是一个全新的空项目（无任何历史遗留代码）。
请从 0 到 1 构建一套通用的多模态事件时序预测模型：**Scheme 2 — Event-Conditioned HyperLoRA & Dynamic Quantile Head（事件条件动态超网络架构）**，利用 **Qwen3.8-27B** 作为超网络动态生成 **TimesFM 3.0** 顶层与输出头的调制权重。

## 核心设计原则与环境约束
1. **通用动态算子生成（不绑定任何特定评测指标）**：
   - 解决静态微调（Static LoRA/DoRA）在无事件窗口漂移、且冻结预测头无法表达大幅度脉冲的根本矛盾。
   - 模型支持任意预测步长 $H$、任意采样频率与分位数集合 $Q$。不编写任何下游业务评测代码，训练完全由通用的条件分布似然与时序差分动力学损失驱动。
2. **本地 Mac 开发与生产环境双模底座（Dual-Mode Backbone Factory）**：
   - 当前在 **本地 Mac（Apple Silicon MPS / CPU）** 上开发，无法在本地加载 27B 真实权重。
   - 请实现 `mode="prod"`（对接 HuggingFace `Qwen3.8-27B` + PEFT LoRA 与官方 `TimesFM 3.0`）与 `mode="mac_debug"`（默认激活，使用纯 PyTorch 实现结构同构的轻量级 `MockQwenCausalLM` 与可挂载动态层的 `ModularTimesFM3Backbone`），确保所有动态权重合成、Batch 级向量化矩阵乘法与训练闭环在 Mac 上秒级跑通。

---

# 1. 项目目录结构规范
请构建以下纯模型研发目录结构：
```text
.
├── configs/
│   └── model_config.py          # 模型超参数配置（含 mac_debug / prod 模式切换、HyperLoRA 秩配置）
├── data/
│   ├── schema.py                # 通用多模态时序与事件输入张量契约
│   └── synthetic_generator.py   # 合成多弹性、多阶段脉冲时序数据生成器
├── backbones/
│   ├── qwen_hyper_backbone.py   # Qwen3.8-27B 超网络编码器（含 EventContextCache 上下文缓存复用）
│   └── timesfm_modular.py       # 支持末 N 层线性模块替换与动态头注入的 TimesFM 3.0 封装
├── models/
│   ├── hyper_dora_linear.py     # 核心：样本级/时间步级动态奇异值调制的 HyperDoRALinear 层
│   ├── dynamic_quantile_head.py # 核心：由超网络实时生成低秩增益权重的动态分位数头
│   └── hyper_timesfm_model.py   # 方案二端到端顶层模型封装
├── losses/
│   └── dynamics_loss.py         # 通用物理动力学损失（线性分位数似然 + 一阶/二阶时序差分 + 超网络正则）
├── tests/
│   └── test_hyper_mechanics.py  # 动态算子恒等性、Batch向量化一致性、缓存与梯度流单元测试
└── train_demo.py                # 最小化通用超网络模型训练闭环演示脚本

```

---

# 2. 核心模型架构实现规范 (`models/` & `backbones/`)

## 2.1 Qwen3.8-27B 事件与内生弹性超网络编码器 (`qwen_hyper_backbone.py`)

1. **输入构造（事件上下文 + 历史局部响应统计）**：

* 自动从输入历史序列 `history_x` 中提取过去已发生事件 $E_{\text{hist}}$ 期间相对于滑动周期基线的局部响应特征（蓄水深度、峰值倍率、持续期积分增益、恢复斜率）。
* 将 `[序列基础统计量 + 历史事件语义与真实响应特征对 + 未来耦合事件语义与时间拓扑]` 编码为嵌入序列输入 `QwenHyperBackbone`（挂载 LoRA）。

2. **输出双层级超网络控制潜码**：

* **序列级事件机制潜码** $C_{\text{event}} \in \mathbb{R}^{B \times d_{\text{llm}}}$：用于生成 TimesFM 3.0 顶层 Transformer Block 的动态低秩奇异值与行范数调制量。
* **时间步级相位潜码** $P_{\text{step}}(t) \in \mathbb{R}^{B \times H \times d_{\text{llm}}}$：结合未来每一步 $t \in [0, H)$ 相对于未来事件的连续生命周期位置（预热/起爆/持续/尾盘/透支相位），用于生成输出头的逐时动态权重。

3. **上下文哈希缓存 (`EventContextCache`)**：

* 提供按 `(series_id, event_context_hash)` 索引的特征缓存机制，在推理或冻结特征阶段避免对相同事件上下文重复执行 27B 前向计算。



## 2.2 动态奇异值调制线性层 `HyperDoRALinear` (`hyper_dora_linear.py`)

编写通用的模块替换函数，将 TimesFM 3.0 最后 $N_{\text{last}}$ 层（如末 4 层）中的 `nn.Linear` 替换为 `HyperDoRALinear`：

1. **参数结构**：

* 冻结原始权重 $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$、偏置 $b_0$ 及原始行范数 $m_0 = \Vert{}W_0\Vert{}_{\text{row}} \in \mathbb{R}^{d_{\text{out}} \times 1}$。
* 可学习共享低秩基底：$A \in \mathbb{R}^{r \times d_{\text{in}}}$（Kaiming 初始化）与 $B \in \mathbb{R}^{d_{\text{out}} \times r}$（使用 $\mathcal{N}(0, 0.02)$ 小随机数初始化，**禁止初始化为全 0**，以确保第 1 步反向传播时超网络能收到梯度）。
* 超网络映射头 `HyperParamProjector`：将 $C_{\text{event}}$ 映射为每个样本专属的低秩奇异值对角向量 $s(E) \in \mathbb{R}^{B \times r}$（权重 $\mathcal{N}(0, 10^{-3})$、偏置为 0 初始化）与行幅值相对增益 $\Delta m(E) \in \mathbb{R}^{B \times d_{\text{out}}}$。

2. **高效 Batch 向量化前向计算**：
对于输入隐状态 $h \in \mathbb{R}^{B \times T \times d_{\text{in}}}$ 与事件激活包络 $m_{\text{act}} \in [0, 1]^{B \times T \times 1}$：

* 原生输出：$y_0 = h W_0^\top + b_0$；
* 动态低秩算子作用：通过张量广播与矩阵乘法计算 $\Delta y = \big((h A^\top) \odot \text{unsqueeze}(s(E), 1)\big) B^\top \in \mathbb{R}^{B \times T \times d_{\text{out}}}$；
* 动态幅值与门控合成：

$$y = \big(1 + m_{\text{act}} \odot \tanh(\Delta m(E))\big) \odot y_0 + m_{\text{act}} \odot \Delta y$$

* 当 $m_{\text{act}} = 0$ 时，$y \equiv y_0$，严格保持原生基模在无事件区间的零漂移特性。



## 2.3 动态低秩分位数解码头 (`dynamic_quantile_head.py`)

为了彻底解除冻结输出头对大促脉冲幅度的限制，由逐步相位潜码 $P_{\text{step}}(t) \in \mathbb{R}^{B \times H \times d_{\text{llm}}}$ 动态生成输出层算子：

1. **逐时间步动态低秩算子生成**：

* 将 $P_{\text{step}}(t)$ 通过 `HeadHyperMLP` 映射为逐时间步低秩矩阵 $U_{\text{head}}(t) \in \mathbb{R}^{B \times H \times d_{\text{ts}} \times r_h}$ 与 $V_{\text{head}}(t) \in \mathbb{R}^{B \times H \times r_h \times 3}$。
* 作用于 TimesFM 3.0 顶层输出隐状态 $H_{\text{top}}(t) \in \mathbb{R}^{B \times H \times d_{\text{ts}}}$：

$$[\tilde{u}_{0.5}(t), \; \beta_{0.5}(t), \; \tilde{s}_{\text{spread}}(t)] = \text{bmm}\Big(\text{bmm}\big(H_{\text{top}}(t), U_{\text{head}}(t)\big), V_{\text{head}}(t)\Big)$$




2. **保序分位数物理合成**：

* 令 $u_{0.5}(t) = 2.5 \cdot \tanh(\tilde{u}_{0.5}(t))$，$s_{\text{spread}}(t) = \text{Softplus}(\tilde{s}_{\text{spread}}(t)) + 0.1$，结合连续事件激活包络 $m(t) \in [0, 1]$：

$$\hat{Y}_{0.5}(t) = \hat{Y}_{0.5, \text{base}}(t) \cdot \exp\big(m(t) \cdot u_{0.5}(t)\big) + m(t) \cdot \beta_{0.5}(t) \cdot \sigma_X$$

$$\hat{Y}_{q}(t) = \hat{Y}_{0.5}(t) + \big(\hat{Y}_{q, \text{base}}(t) - \hat{Y}_{0.5, \text{base}}(t)\big) \cdot \exp\big(m(t) \cdot u_{0.5}(t)\big) \cdot \Big(1 + m(t)\big(s_{\text{spread}}(t) - 1\big)\Big)$$





---

# 3. 通用物理动力学损失函数 (`losses/dynamics_loss.py`)

不引入任何下游特定窗口评测指标，仅使用通用时序分布与导数损失：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{linear-pinball}}(\tilde{Y}, \hat{\tilde{Y}}) + \lambda_1 \mathcal{L}_{\text{diff-1st}}(\nabla_t \tilde{Y}, \nabla_t \hat{\tilde{Y}}) + \lambda_2 \mathcal{L}_{\text{diff-2nd}}(\nabla^2_t \tilde{Y}, \nabla^2_t \hat{\tilde{Y}}) + \lambda_3 \mathcal{L}_{\text{cf}}(u_{0.5}, u^*) + \lambda_{\text{reg}} \big(\Vert{}s(E)\Vert{}_2^2 + \Vert{}\Delta m(E)\Vert{}_2^2\big)$$

* 移除 `arcsinh` 变换，在线性标准化空间 $\tilde{Y} = (Y - \mu_X)/\sigma_X$ 计算全分位数 Pinball Loss；
* 通过一阶差分 $\nabla_t Y$ 与二阶差分 $\nabla^2_t Y$ 监督脉冲的阶跃斜率与波峰拐点；
* 对超网络生成的动态奇异值施加轻量 L2 正则以保证算子平稳性。

---

# 4. 模型机制单元测试与演示 (`tests/test_hyper_mechanics.py` & `train_demo.py`)

请编写并运行 `pytest`，确保以下 5 项纯模型单元测试全部通过：

1. `test_zero_event_strict_weight_identity`：当 $m(t)=0$ 时，验证替换了 `HyperDoRALinear` 和 `DynamicQuantileHead` 的模型输出与未修改的原生 `TimesFM 3.0` 输出误差严格为 `0.0`。
2. `test_batched_hyper_dora_equivalence`：验证 `HyperDoRALinear` 的 Batch 向量化前向计算结果，与逐个样本显式构造权重矩阵 $W_{\text{eff}}^{(b)} = W_0 + B \cdot \text{Diag}(s_b) \cdot A$ 再与样本相乘的结果在数值上完全一致（误差 $< 10^{-6}$）。
3. `test_hypernetwork_gradient_flow`：执行一次 backward，断言 `QwenHyperBackbone` 的 LoRA 参数、`HyperDoRALinear` 的共享基底 $A, B$ 与超网络投影头、以及 `DynamicQuantileHead` 的梯度均大于 0 且无 NaN。
4. `test_context_cache_and_quantile_monotonicity`：验证 `EventContextCache` 命中前后的输出完全一致，且在任意正负脉冲下所有分位数通道严格递增不交叉。
5. `test_dynamic_operator_impulse_learning`：在 `train_demo.py` 中构造包含不同实体弹性与复杂脉冲形态（预热下拉、起爆尖峰、指数衰减、期后透支）的合成序列，在本地 Mac 上运行 50 步训练演示，验证损失快速收敛，且动态生成的算子能对同一事件在不同弹性序列上产生截然不同的精确脉冲轨迹！

```

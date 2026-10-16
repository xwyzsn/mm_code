LLaVA-TS 交错式多模态上下文联合微调模型

> **核心建模思想**：将“历史事件发生前后的真实时序跳变切片”通过时序编码器映射为连续软词元 `<TS_PATCH>`，与事件语义在 **Qwen3.8-27B** 的上下文中交错排列（实现跨模态少样本内生弹性归纳）；随后利用 $H$ 个可学习查询词元 `<HORIZON_Q_1..H>` 通过**单次 Prefill 前向传播**提取逐时控制潜向量，驱动 **TimesFM 3.0** 顶层适配器与分布级 FiLM 仿射变换算子。

请直接复制以下代码块交由 Claude Code 执行：

```markdown
# Role & Mission
你是一位顶尖的多模态时序大模型架构师与 PyTorch 工程专家。当前目录是一个全新的空项目（无任何历史遗留代码）。
请从 0 到 1 构建一套通用的多模态事件时序预测模型：**Scheme 1 — LLaVA-TS Interleaved Multimodal Time-Series Model（交错式多模态上下文联合微调架构）**，将 **Qwen3.8-27B** 的因果与上下文推理能力与 **TimesFM 3.0** 的长周期时序建模能力深度融合。

## 核心设计原则与环境约束
1. **追求纯粹的通用动态建模能力**：
   - 模型必须支持任意历史长度 $L$、任意预测步长 $H$ 以及任意分位数集合 $Q$。
   - 训练目标严格基于**条件概率分布生成似然**与**时序导数动力学**，严禁绑定任何特定下游业务评测指标，也不编写任何业务评测模块。
2. **本地 Mac 开发与生产环境双模底座（Dual-Mode Backbone Factory）**：
   - 当前在 **本地 Mac（Apple Silicon MPS / CPU）** 上开发，无法在本地加载 54GB 的 Qwen3.8-27B 真实权重。
   - 请实现 `mode="prod"` 与 `mode="mac_debug"`（默认激活）双模式：
     - `prod` 模式：使用 `transformers` (`AutoModelForCausalLM`) + `peft` (`LoraConfig`) 加载真实的 `Qwen3.8-27B` 与 `TimesFM 3.0` 权重，支持 BF16 与 Gradient Checkpointing。
     - `mac_debug` 模式：使用纯 PyTorch 实现结构与接口 100% 同构的轻量级替代主干 `MockQwenCausalLM`（如 2 层 Transformer，`d_llm=128`）与 `MockTimesFM3Backbone`（如 2 层 Patch Transformer，`d_ts=64`），确保在本地 Mac 上可秒级跑通前向传播、反向梯度流与单元测试。

---

# 1. 项目目录结构规范
请仅围绕模型研发构建以下目录结构（无需编写任何业务评测模块）：
```text
.
├── configs/
│   └── model_config.py          # 模型超参数配置（支持 mac_debug 与 prod 模式切换，任意预测步长 H）
├── data/
│   ├── schema.py                # 通用多模态时序与事件输入张量契约（Dataclass）
│   └── synthetic_generator.py   # 用于验证通用动力学能力的合成时序脉冲数据生成器
├── backbones/
│   ├── qwen_backbone.py         # Qwen3.8-27B 封装（支持 LoRA 与 inputs_embeds 单次 Prefill）
│   └── timesfm_backbone.py      # TimesFM 3.0 封装（提取编码器隐状态、未来高层表征与基线分位数分布）
├── models/
│   ├── time_patch_tokenizer.py  # 浅层局部时序跳变提取器与 Time-to-LLM 软词元投影器
│   ├── interleaved_context.py   # 交错式多模态上下文构造器（含 H 个 <HORIZON_Q> 连续查询槽）
│   ├── distributional_film.py   # 跨模态注意力与保序分位数 FiLM 仿射调制解码器
│   └── llava_ts_model.py        # 方案一端到端顶层模型封装
├── losses/
│   └── dynamics_loss.py         # 通用物理动力学损失（线性尺度分位数似然 + 一阶/二阶时序差分损失）
├── tests/
│   └── test_model_mechanics.py  # 模型结构、因果性、零事件恒等性、梯度流与动力学拟合单元测试
└── train_demo.py                # 最小化通用模型训练闭环演示脚本

```


使用lighting进行开发python >=3.10

---

# 2. 输入张量契约 (`data/schema.py`)

定义通用的 `MultimodalTSBatch` 数据结构，支持任意步长 $H$：

* `history_x`: `[B, L_hist]` 历史观测序列，配合 `history_mask: [B, L_hist]`。
* `future_y`: `Optional[[B, H]]` 未来真实序列（仅训练阶段提供），配合 `future_mask: [B, H]`。
* **事件模态（同时支持原始文本 Token IDs 或预提取的多路语义向量 + 离散标签）**：
* 每个事件包含相对当前预测起点 $t_0=0$ 的连续/离散时间坐标：`start_offset`、`end_offset`、`publish_offset`（严格执行因果过滤：仅保留 `publish_offset < 0` 的已发布事件）。
* 模型内部自动将合法事件划分为：
1. **历史已发生事件 $E_{\text{hist}}$**（发生在 $[-L_{\text{hist}}, 0)$ 内，最多取最近 $N_{\text{hist}}$ 个）；
2. **预测窗耦合事件 $E_{\text{fut}}$**（其扩展影响生命周期 `[start_offset - pre_buffer, end_offset + post_buffer]` 与预测窗口 `[0, H)` 相交的事件，最多取 $N_{\text{fut}}$ 个）。





---

# 3. 核心模型架构实现规范 (`models/`)

## 3.1 浅层时序跳变提取与软词元化 (`time_patch_tokenizer.py`)

为了让 Qwen3.8-27B 在上下文中直接“看懂”当前序列对不同事件的真实物理弹性（而非死记硬背文本向量），需要将历史事件对应的真实时序跳变编码为软词元：

1. **提取未被主干平滑的局部高频残差**：
* 计算历史序列的滑动周期基线 $\text{MA}(X)$ 与稳健标准差 $\sigma_X = \max(\text{Std}(X), 10^{-5})$。
* 构造双通道浅层跳变序列：标准化加性残差 $R(t) = \frac{X_t - \text{MA}(X_t)}{\sigma_X}$ 与局部对数倍率 $U(t) = \ln\left(\frac{\max(X_t, 0) + 0.1\sigma_X}{\max(\text{MA}(X_t), 0) + 0.1\sigma_X}\right)$。


2. **历史事件响应切片与投影**：
* 对每个历史事件 $e_i \in E_{\text{hist}}$，根据其历史区间切片提取 `[事件前蓄水期, 事件进行期, 事件后恢复期]` 的局部跳变轨迹 $[R(t), U(t)]$ 以及 TimesFM 3.0 浅层 Patch 隐状态。
* 通过多尺度 1D-Conv + 自适应时间池化 + 两层 MLP（`TimeToLLMProjector`），将每次历史事件的真实时序响应映射为 $P=4$ 个维度为 $d_{\text{llm}}$ 的连续软词元 `<TS_HIST_PATCH>`。
* 同时将预测起点 $t_0$ 前的近期序列状态映射为 $P=4$ 个 `<TS_RECENT_PATCH>` 软词元。



## 3.2 交错式多模态上下文构造与单次 Prefill (`interleaved_context.py`)

1. **交错式上下文拼装（In-Context Elasticity Deduction）**：
将事件语义词元（由文本 Embedding 或多路向量经 `EventProjector` 映射得到）与时序软词元按严格的时间因果顺序交错排列，构造送入 Qwen3.8-27B 的 `inputs_embeds`：

$$\Big[\underbrace{E_{\text{series\_stat}}}_{\text{序列统计基线}}, \; \underbrace{\big(\text{EventSem}(e_1^{\text{hist}}), \text{TSPatch}(e_1^{\text{hist}})\big), \dots}_{\text{历史“事件语义-实际跳变”交错少样本对}}, \; \underbrace{\text{TSRecent}}_{\text{近期时序状态}}, \; \underbrace{\text{EventSem}(e_1^{\text{fut}}), \dots}_{\text{未来耦合事件语义与相位}}, \; \underbrace{Q_{\text{hor}, 1}, \dots, Q_{\text{hor}, H}}_{\text{H个未来时间步查询词元}}\Big]$$


* 其中 $Q_{\text{hor}} \in \mathbb{R}^{H \times d_{\text{llm}}}$ 是可学习的查询词元基底，叠加了未来时间步 $t \in [0, H)$ 相对于各未来耦合事件的连续生命周期编码（距离事件起止的相对时间差、所处预热/爆发/衰减/透支相位、日内周期正余弦编码）。


2. **单次 Prefill 并行提取控制表征（无自回归生成延迟）**：
* 将交错序列 `inputs_embeds` 输入挂载了 LoRA 的 `QwenBackbone` 执行**单次前向传播**（无需逐词自回归 Decode）。
* 直接提取序列末尾 $H$ 个查询槽位置对应的顶层隐藏状态 $Z_{\text{qwen}} \in \mathbb{R}^{B \times H \times d_{\text{llm}}}$。



## 3.3 跨模态交互与保序分位数 FiLM 调制头 (`distributional_film.py`)

不在 Transformer 内部做会被 `LayerNorm` 抹平的微弱加法，而是在条件输出分布层进行显式仿射变换：

1. **带完整可学习投影的跨模态注意力**：
* 从 `TimesFMBackbone` 获取基线分位数分布 $\hat{Y}_{\text{base}} \in \mathbb{R}^{B \times H \times \vert{}Q\vert{}}$ 与顶层未来隐状态 $H_{\text{fut}} \in \mathbb{R}^{B \times H \times d_{\text{ts}}}$。
* 通过独立的线性投影层 $W_Q, W_K, W_V$，计算 $H_{\text{fut}}$ 对 $Z_{\text{qwen}}$ 的跨模态注意力，输出融合表征 $F_{\text{ctx}}(t) \in \mathbb{R}^{B \times H \times d_{\text{ts}}}$。


2. **连续时间事件激活包络 $m(t) \in [0, 1]$**：
* 对未来每个时间步 $t \in [0, H)$，根据其与 $E_{\text{fut}}$ 的距离计算连续激活权重：事件期内 $m(t)=1.0$，事件前后缓冲区内按指数核 $\exp(-\vert{}\Delta t\vert{}/\tau_{\text{buf}})$ 平滑衰减，无任何耦合事件时 $m(t) \equiv 0.0$。


3. **保序分位数仿射调制算子（Non-Crossing Quantile FiLM）**：
* 由 $F_{\text{ctx}}(t)$ 预测三个分布控制参数（输出层采用 $\mathcal{N}(0, 10^{-3})$ 小权重初始化，避免 `gate=0` 导致的第 1 步梯度死锁）：
* 中位数对数尺度倍率 $u_{0.5}(t) = \alpha_{\max} \cdot \tanh(\text{Linear}_u(F_{\text{ctx}}(t)))$（默认 $\alpha_{\max}=2.5$）；
* 标准化位置平移项 $\beta_{0.5}(t) = \text{Linear}_\beta(F_{\text{ctx}}(t))$；
* 分位数不确定性展宽因子 $s_{\text{spread}}(t) = \text{Softplus}(\text{Linear}_s(F_{\text{ctx}}(t))) + 0.1 > 0$。


* 令 $\Delta_{q, \text{base}}(t) = \hat{Y}_{q, \text{base}}(t) - \hat{Y}_{0.5, \text{base}}(t)$ 为原生分位数相对于中位数的非负间距，合成最终分布：

$$\hat{Y}_{0.5}(t) = \hat{Y}_{0.5, \text{base}}(t) \cdot \exp\big(m(t) \cdot u_{0.5}(t)\big) + m(t) \cdot \beta_{0.5}(t) \cdot \sigma_X$$


$$\hat{Y}_{q}(t) = \hat{Y}_{0.5}(t) + \Delta_{q, \text{base}}(t) \cdot \exp\big(m(t) \cdot u_{0.5}(t)\big) \cdot \Big(1 + m(t)\big(s_{\text{spread}}(t) - 1\big)\Big)$$


* 当 $m(t) = 0$ 时，输出在代数上严格恒等于 $\hat{Y}_{q, \text{base}}(t)$。



---

# 4. 通用物理动力学损失函数 (`losses/dynamics_loss.py`)

彻底摒弃 `arcsinh` 压缩（防止大脉冲梯度衰减）与任何下游特定窗口积分指标，仅由以下通用生成目标驱动：

1. **线性归一化全分位数似然损失 ($\mathcal{L}_{\text{quantile}}$)**：
在线性标准化空间 $\tilde{Y}_t = (Y_t - \mu_X)/\sigma_X$ 上计算全分位数 Pinball Loss（仅在 $\vert{}e\vert{} > 15\sigma$ 处使用 Huber 软过渡防止极端脏数据梯度爆炸，在正常脉冲区间保持 100% 线性梯度）。
2. **一阶速度差分与二阶拐点差分损失 ($\mathcal{L}_{\text{dynamics}}$)**：
* 一阶差分（斜率/阶跃速度）：$\nabla_t \tilde{Y}_t = \tilde{Y}_t - \tilde{Y}_{t-1}$，计算全分位数 Pinball Loss，迫使模型精准捕捉事件起爆时的正向阶跃斜率与结束时的回落斜率。
* 二阶差分（曲率/拐点加速度）：$\nabla^2_t \tilde{Y}_t = \nabla_t \tilde{Y}_t - \nabla_t \tilde{Y}_{t-1}$，使用 Smooth-L1 监督中位数二阶差分，锁定脉冲波峰与波谷的转折点。


3. **反事实对数倍率直通监督 ($\mathcal{L}_{\text{cf}}$)**：
在 $m(t) > 0$ 的时刻，计算真实对数倍率 $u^*(t) = \text{clip}\left(\ln\frac{\max(Y_t, 0) + 0.1\sigma_X}{\max(\hat{Y}_{0.5,\text{base}}(t), 0) + 0.1\sigma_X}, -2.5, 2.5\right)$ 与 $u_{0.5}(t)$ 的 Smooth-L1 Loss。
4. 总损失：$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{quantile}} + \lambda_1 \mathcal{L}_{\text{dynamics}} + \lambda_2 \mathcal{L}_{\text{cf}}$。

---

# 5. 模型机制单元测试与演示 (`tests/test_model_mechanics.py` & `train_demo.py`)

请编写并运行 `pytest`，确保以下 5 项纯模型能力测试全部通过：

1. `test_arbitrary_horizon_and_shapes`：测试不同预测步长（如 $H=12, 24, 72, 96$）下的前向传播，验证输出维度 `[B, H, |Q|]` 完全匹配。
2. `test_zero_event_exact_identity`：当无未来耦合事件（$m(t)=0$）时，断言模型输出与原生 `TimesFMBackbone` 输出的最大绝对误差严格为 `0.0`。
3. `test_causal_time_filtering`：注入 `publish_offset >= 0` 的未发布事件，断言模型将其完全屏蔽，修改其数值不影响输出。
4. `test_end_to_end_gradient_flow`：执行一次反向传播，断言 `QwenBackbone` 的 LoRA 参数、`TimePatchTokenizer`、`Q_hor` 查询词元及 `DistributionalFiLM` 的梯度均非零且无 NaN。
5. `test_in_context_elasticity_and_impulse_fit`：在 `train_demo.py` 中构造两组具有相同未来事件文本、但历史事件跳变幅度截然不同（高弹性 $3.5\times$ vs 低弹性 $1.0\times$，并含事件前蓄水下拉与期内双峰）的合成序列，在本地 Mac 上训练 50 步，验证模型能够通过交错上下文自动区分两种弹性，并精准拟合出预热下拉、起爆阶跃与分位数展宽！

```
# AReaL 架构文档

> A Distributed Reinforcement Learning Training Framework for LLM Alignment

## 目录

1. [项目概述](#1-项目概述)
2. [核心架构](#2-核心架构)
3. [关键技术](#3-关键技术)
4. [核心接口](#4-核心接口)
5. [使用示例](#5-使用示例)
6. [最佳实践](#6-最佳实践)
7. [附录](#7-附录)

---

## 1. 项目概述

### 1.1 项目简介

**AReaL** 是一个用于大语言模型对齐的大规模异步强化学习训练框架。通过解耦的生成和训练过程、智能的陈旧度控制和多种分布式后端支持，实现高效的RL训练。

### 1.2 技术栈

- **语言**: Python 3.12+
- **深度学习框架**: PyTorch
- **分布式训练**: FSDP2, Megatron-LM
- **推理引擎**: SGLang, vLLM
- **集群调度**: Ray, Slurm
- **配置管理**: Hydra

### 1.3 核心特性

- **原生异步RL**: 生成和训练完全解耦，支持持续推理
- **多后端支持**: FSDP2和Megatron-LM用于训练，SGLang/vLLM用于推理
- **算法丰富**: GRPO、PPO、DAPO、LitePPO、Dr.GRPO、RLOO、REINFORCE++、SAPO
- **灵活工作流**: 单轮、多轮、视觉语言和自定义agent工作流
- **高可扩展性**: 从单GPU到1000+ GPU的无缝扩展
- **高性能**: 相比同步系统有2.77×加速比

---

## 2. 核心架构

### 2.1 架构分层

```
┌─────────────────────────────────────────────────────────────┐
│  Entry Point Layer (examples/)                               │
│  - Training scripts                                          │
│  - Configuration management                                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Customization Layer (engine/ppo/, workflow/)               │
│  - Algorithm implementations (PPO, GRPO, etc.)              │
│  - Rollout workflows (single-turn, multi-turn)             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Backend Layer (engine/fsdp_engine.py, engine/megatron/)    │
│  - Training engine adapters (FSDP2, Megatron)               │
│  - Inference engines (SGLang, vLLM)                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  API Layer (areal/api/)                                     │
│  - Abstract interfaces (TrainEngine, InferenceEngine)      │
│  - Configuration dataclasses                                │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 核心目录结构

```
areal/
├── api/                    # 抽象接口和数据类
│   ├── engine_api.py        # TrainEngine, InferenceEngine 抽象基类
│   ├── workflow_api.py      # RolloutWorkflow 抽象基类
│   ├── reward_api.py        # 奖励函数包装器
│   ├── cli_args.py          # 配置数据类（2000+ 行）
│   ├── io_struct.py         # 请求/响应数据结构
│   └── alloc_mode.py        # 并行策略定义
│
├── engine/                 # 训练和推理引擎
│   ├── core/
│   │   └── train_engine.py  # 引擎无状态工具
│   ├── fsdp_engine.py       # 基于FSDP2的训练（1700+ 行）
│   ├── megatron_engine.py   # 基于Megatron-LM的训练（1600+ 行）
│   ├── sglang_remote.py     # SGLang客户端包装器
│   ├── vllm_remote.py       # vLLM客户端包装器
│   ├── ppo/
│   │   ├── actor.py          # PPO/GRPO算法实现（1000+ 行）
│   │   └── critic.py         # PPO评论家
│   ├── sft/
│   │   └── lm_engine.py      # 监督微调
│   └── rw/
│       └── rw_engine.py      # 奖励模型训练
│
├── workflow/               # Rollout工作流实现
│   ├── rlvr.py             # 单轮RLVR工作流
│   ├── multi_turn.py        # 多轮尝试工作流
│   └── vision_rlvr.py       # 视觉语言工作流
│
├── reward/                 # 奖励函数实现
│   ├── gsm8k.py            # 数学验证奖励
│   ├── geometry3k.py        # 几何奖励
│   └── clevr_count_70k.py   # 视觉计数奖励
│
├── dataset/                # 数据集加载器
│   ├── gsm8k.py
│   ├── hhrlhf.py
│   └── geometry3k.py
│
├── infra/                  # 基础设施组件
│   ├── workflow_executor.py # 异步rollout执行（1400+ 行）
│   ├── dist_rollout.py      # 分布式rollout协调
│   ├── staleness_manager.py # 容量/陈旧度控制
│   ├── remote_inf_engine.py # 远程推理后端
│   ├── controller/          # 训练/Rollout控制器
│   └── platforms/          # GPU平台抽象
│
├── launcher/               # 不同后端的启动器
│   ├── local.py            # 本地子进程启动器
│   └── ray.py              # Ray集群启动器
│
├── scheduler/              # 作业调度
│   ├── local.py
│   ├── ray.py
│   └── slurm.py
│
├── utils/                  # 工具（日志、数据、分布式）
└── experimental/            # 实验性功能
    ├── trainer/rl.py       # PPOTrainer（高级API）
    ├── engine/archon_engine.py  # Archon训练后端
    └── openai/             # OpenAI兼容代理
```

### 2.3 设计原则

1. **算法优先、轻量级定制**: 单文件定制RL流水线
2. **系统抽象**: 最小化底层系统概念暴露
3. **PyTorch为中心**: 原生类型，无不必要的抽象
4. **透明编排**: 清晰的操作流程
5. **开发友好**: 易于IDE导航（Ctrl+点击）
6. **生态系统兼容**: 与ML/RL工具集成
7. **组合优于继承**: 通过依赖注入实现模块化设计

---

## 3. 关键技术

### 3.1 分布式训练策略

#### 3.1.1 FSDP2 (PyTorch Fully Sharded Data Parallel)

**位置**: `areal/engine/fsdp_engine.py`

**核心特性**:

- **N维并行**: DP + SP (Ulysses) + TP
- **激活检查点**: 内存效率优化
- **CPU卸载**: 通过 `torch_memory_saver` 支持
- **FSDP2状态字典**: 用于检查点保存
- **梯度裁剪**: FSDP感知的范数计算
- **LoRA支持**: 通过PEFT库
- **树训练**: 高效批处理

**关键实现**:

```python
class FSDPEngine(TrainEngine):
    def __init__(self, config: TrainEngineConfig):
        # 为N维并行创建设备网格
        self.world_mesh = ParallelHelper.from_parallel_strategy(parallel_strategy)
        self.dp_group = self.world_mesh["dp"].get_group()
        self.sp_group = self.world_mesh["sp"].get_group()
        self.mp_group = self.world_mesh["sp_tp"].get_group()

    def forward_backward_batch(self, mb_list, process_output_fn, forward_only=False):
        # 使用FSDP2处理微批次
        for mb_item in mb_list:
            inputs = self._prepare_mb_inputs(mb_list)  # Ulysses SP处理
            outputs = self.model(**inputs)  # FSDP分片前向传播
            loss = process_output_fn(outputs, inputs)
            if not forward_only and loss is not None:
                loss.backward()  # FSDP反向传播
```

**并行策略**:
- `dp_size`: 数据并行大小
- `tp_size`: 张量并行大小
- `sp_size`: 序列并行大小（Ulysses）
- `pp_size`: 流水线并行大小（FSDP2不支持）

#### 3.1.2 Megatron-LM

**位置**: `areal/engine/megatron_engine.py`

**核心特性**:

- **4D并行**: DP + TP + PP + CP + EP
- **流水线并行**: 交错调度
- **专家并行**: 支持MoE模型
- **上下文并行**: 类似Ulysses
- **分布式优化器**: 梯度累积
- **FP8训练**: 低精度训练支持
- **mbridge**: HF↔Megatron权重转换

**关键实现**:

```python
class MegatronEngine(TrainEngine):
    def __init__(self, config: TrainEngineConfig):
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=...,
            pipeline_model_parallel_size=...,
            context_parallel_size=...,
            expert_model_parallel_size=...
        )
        self.model = _MegatronModelList(models)  # 流水线块

    def forward_backward_func(self, forward_step_func, data_iterator, ...):
        # Megatron的流水线并行执行
        forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=data_iterator,
            model=self.model,
            num_microbatches=len(mb_list),
        )
```

#### 3.1.3 并行策略对比

| 特性 | FSDP2 | Megatron |
|------|-------|----------|
| 张量并行 | ✅ | ✅ |
| 序列并行 | ✅ (Ulysses) | ✅ (Ulysses-like) |
| 流水线并行 | ❌ | ✅ |
| 上下文并行 | ✅ | ✅ |
| 专家并行 | ❌ | ✅ |
| FP8训练 | ❌ | ✅ |
| LoRA | ✅ | ❌ |
| 树训练 | ✅ | ✅ |

### 3.2 异步Rollout机制

#### 3.2.1 核心组件

1. **WorkflowExecutor** (`areal/infra/workflow_executor.py`)
   - 使用 **BatchTaskDispatcher** 实现生产者-消费者模式
   - 后台线程用于提交和结果收集
   - 陈旧度控制的异步任务执行

2. **StalenessManager** (`areal/infra/staleness_manager.py`)
   - 根据模型版本控制rollout容量
   - 防止样本过于陈旧（离策略）
   - 公式: `capacity = min(concurrency_limit, (max_staleness + 1) * batch_size - current_samples)`

3. **AsyncTaskRunner** (`areal/infra/async_task_runner.py`)
   - 异步工作流执行的线程池
   - 带容量限制的任务队列
   - 超时处理的优雅关闭

#### 3.2.2 异步Rollout流程

```
1. 提交阶段:
   数据 → WorkflowExecutor.submit()
   → _pending_inputs 队列
   → 生产者线程在容量可用时提交到AsyncTaskRunner
   → StalenessManager跟踪已入队/运行/已接受的计数

2. 执行阶段:
   AsyncTaskRunner执行 workflow.arun_episode()
   → RemoteInferenceEngine.agenerate() 调用 SGLang/vLLM服务器
   → 奖励计算（异步，带超时）
   → 返回轨迹或None（被拒绝）

3. 收集阶段:
   消费者线程收集完成的结果
   根据陈旧度和should_accept_fn过滤
   结果累积在 _pending_results
   → wait() 返回打乱的结果用于训练
```

#### 3.2.3 版本跟踪用于陈旧度

- 每个token有 `version` 字段指示生成时间
- 模型更新递增 `current_version`
- 差异: `staleness = current_version - token_version`
- 根据 `max_staleness` 配置过滤或限制陈旧样本

### 3.3 检查点系统

#### 3.3.1 两种检查点格式

1. **HuggingFace格式** (`weight_format="hf"`)
   - 与HF模型加载兼容
   - 单GPU整合和保存
   - 使用FSDP2的 `get_model_state_dict()`

2. **DCP格式** (`weight_format="dcp"`)
   - PyTorch分布式检查点
   - 分片保存（无需GPU整合）
   - 包含优化器状态（`with_optim=True`）

#### 3.3.2 关键实现（FSDP）

```python
def _save_model_to_hf(self, path: str, tokenizer, processor):
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    state_dict = get_model_state_dict(self.model, options=options)

    if dist.get_rank() == 0:
        self.model.save_pretrained(path, state_dict=state_dict)
        self.model_config.save_pretrained(path)
    dist.barrier(group=self.cpu_group)
```

**Megatron检查点**:
- 使用 `MegatronCheckpointManager`
- 支持分布式优化器状态
- 通过mbridge转换Megatron→HF格式

### 3.4 与SGLang/vLLM集成

#### 3.4.1 远程推理架构

```
训练进程 (GPU 0-N)          推理服务器 (独立GPU)
    │                                         │
    ├─ TrainEngine (FSDP/Megatron)          │
    │   ├─ model.training_step()            │
    │   └─ update_weights() ───┐           │
    │                              │           │
    │                              ↓           │
    │                         权重更新        │
    │                         (NCCL/Disk)    │
    │                              │           │
    └───────────────────────────────┼───────────┘
                                    │
    ┌──────────────────────────────────┴─────────┐
    │                                                │
    ├─ prepare_batch() ──── submit() ──→ ──────────┤
    │   ├─ 持续提交                     │
    │   ├─ 收集异步结果                    │
    │   └─ 维护陈旧度控制                   │
    └────────────────────────────────────────┘
                    │
                    ↓
            HTTP/REST API (异步)
                    │
    ┌────────────────────────────────────────┴─────────┐
    │                                                  │
    ├─ SGLang/vLLM服务器                          │
    │   ├─ 加载模型权重                             │
    │   ├─ 生成响应                                │
    │   ├─ 返回logprobs + token IDs                 │
    │   └─ 支持LoRA切换                       │
    └──────────────────────────────────────────────┘
```

#### 3.4.2 权重更新机制

**1. XCCL (NCCL) 更新**:
- 通过NCCL直接GPU到GPU传输
- 分块参数广播
- 所有推理服务器参与
- 快速但需要内存对齐

**2. 磁盘更新**:
- 训练引擎将检查点保存到磁盘
- 推理服务器从磁盘加载
- 较慢但更灵活
- 支持LoRA路径切换

#### 3.4.3 SGLang集成实现

```python
class SGLangBackend:
    def build_generation_request(self, req: ModelRequest):
        return HttpRequest(
            endpoint="/generate",
            payload={
                "input_ids": req.input_ids,
                "sampling_params": {...},
                "return_logprob": True,
            }
        )

    def build_distributed_weight_update_requests(self, meta, param_specs):
        return HttpRequest(
            endpoint="/update_weights_from_distributed",
            payload={
                "names": [pspec.name for pspec in param_specs],
                "dtypes": [pspec.dtype for pspec in param_specs],
                "shapes": [pspec.shape for pspec in param_specs],
                "group_name": meta.nccl_group_name,
            }
        )
```

---

## 4. 核心接口

### 4.1 Engine契约 (`areal/api/engine_api.py`)

#### 4.1.1 TrainEngine - 分布式训练抽象基类

```python
class TrainEngine(abc.ABC):
    # 进程组设置
    def create_process_group(parallel_strategy: ParallelStrategy | None = None)

    # 模型生命周期
    def initialize(self, *args, **kwargs)
    def destroy(self)

    # 分布式协调
    @property
    def data_parallel_group(self) -> dist.ProcessGroup
    def is_data_parallel_head(self) -> bool

    # 训练操作
    def train_batch(input_, loss_fn, loss_weight_fn) -> dict[str, float]
    def forward_backward_batch(mb_list, process_output_fn, forward_only=False)
    def optimizer_step()

    # Rollout协调
    def prepare_batch(dataloader, workflow, ...) -> dict[str, Any]
    def rollout_batch(data, workflow, ...) -> dict[str, Any]

    # 权重管理
    def update_weights(meta: WeightUpdateMeta)
    def save(meta: SaveLoadMeta)
    def load(meta: SaveLoadMeta)

    # 评估
    @torch.no_grad()
    def eval_batch(input_, loss_fn, loss_weight_fn) -> torch.Tensor | None
```

#### 4.1.2 InferenceEngine - 推理抽象基类

```python
class InferenceEngine(abc.ABC):
    # 生命周期
    def initialize(*args, **kwargs)
    def destroy(self)

    # 异步生成
    async def agenerate(req: ModelRequest) -> ModelResponse

    # 权重更新
    def init_weights_update_group(meta, rank_ids) -> Future[None]
    def update_weights_from_distributed(meta, param_specs) -> Future[None]
    def update_weights_from_disk(meta) -> Future[None]

    # 异步rollout
    def submit(data, workflow, ...) -> int
    def wait(count, timeout, raise_timeout) -> list[dict]
    def wait_for_task(task_id, timeout, raise_timeout) -> dict | None

    # 批操作
    def rollout_batch(data, workflow, ...) -> list[dict]
    def prepare_batch(dataloader, workflow, ...) -> list[dict]

    # 暂停/恢复
    def pause_generation()
    def continue_generation()
    def offload()
    def onload(tags)
```

### 4.2 Workflow契约 (`areal/api/workflow_api.py`)

#### 4.2.1 RolloutWorkflow - 用户定义rollout逻辑

```python
class RolloutWorkflow(ABC):
    @abstractmethod
    async def arun_episode(
        self,
        engine: InferenceEngine,
        data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """运行单个rollout回合。

        返回:
            轨迹字典（包含张量），如果被拒绝则返回None。

        示例:
            req = ModelRequest(input_ids=..., gconfig=...)
            resp = await engine.agenerate(req)
            reward = await compute_reward(resp, data)
            return {
                "input_ids": torch.tensor(...),
                "logprobs": torch.tensor(...),
                "rewards": torch.tensor(...),
                ...
            }
        """
```

#### 4.2.2 AgentWorkflow（已弃用但仍支持）

```python
class AgentWorkflow(ABC):
    @abstractmethod
    async def run(self, data: dict[str, Any], **extra_kwargs) -> dict[str, float] | float:
        """使用OpenAI SDK运行agent。

        extra_kwargs包括:
        - base_url: str
        - http_client: httpx.AsyncClient
        """
```

#### 4.2.3 Workflow解析

- 可以传递类实例、类类型、字符串路径或任何具有 `run()` 方法的对象
- 字符串路径动态导入
- 示例:
  - `RLVRWorkflow(...)`
  - `"areal.workflow.rlvr.RLVRWorkflow"`
  - `MyAgentClass()`

### 4.3 奖励函数API (`areal/api/reward_api.py`)

#### 4.3.1 标准奖励签名

```python
def reward_fn(
    prompt: str,
    completions: str,
    prompt_ids: list[int],
    completion_ids: list[int],
    **kwargs  # 数据集特定字段如"answer"、"images"等
) -> float:
    """计算完成的奖励。

    参数:
        prompt: 原始提示文本
        completions: 模型的完成
        prompt_ids: 标记化的提示
        completion_ids: 标记化的完成
        **kwargs: 任务特定数据（answer、images等）

    返回:
        浮点奖励值（通常为0.0或1.0）
    """
```

#### 4.3.2 AsyncRewardWrapper - 使同步奖励函数异步

```python
async_reward_fn = AsyncRewardWrapper(
    reward_fn,
    timeout_seconds=15,
    max_workers=8,  # 基于CPU/GPU数量自动计算
)
```

**特性**:
- 自动管理ProcessPoolExecutor生命周期
- 超时处理
- 进程池崩溃自动恢复
- 最大重试机制

### 4.4 配置数据类 (`areal/api/cli_args.py`)

#### 4.4.1 GenerationHyperparameters

```python
@dataclass
class GenerationHyperparameters:
    n_samples: int = 1              # 每个提示的样本数
    max_new_tokens: int = 2048      # 最大生成token数
    temperature: float = 1.0        # 采样温度
    top_p: float = 1.0              # nucleus采样
    top_k: int = -1                 # top-k采样
    stop_token_ids: list[int] = None  # 停止token ID
    use_beam_search: bool = False   # 是否使用束搜索
    frequency_penalty: float = 0.0  # 频率惩罚
    lora_name: str | None = None    # LoRA适配器名称
```

#### 4.4.2 TrainEngineConfig

```python
@dataclass
class TrainEngineConfig:
    path: str                       # 模型路径
    dtype: str = "bfloat16"         # float16/bfloat16
    gradient_checkpointing: bool = False  # 激活检查点
    use_lora: bool = False          # 使用LoRA
    target_modules: list[str] = None  # LoRA目标模块
    lora_rank: int = 8              # LoRA秩
    enable_tree_training: bool = False  # 树训练
    mb_spec: MicroBatchSpec = None  # 微批次规格
```

#### 4.4.3 PPOActorConfig

```python
@dataclass
class PPOActorConfig:
    kl_ctl: float = 0.1             # KL散度系数
    discount: float = 1.0           # 折扣因子
    gae_lambda: float = 0.95        # GAE lambda
    eps_clip: float = 0.2           # PPO裁剪参数
    adv_norm: bool = True           # 优势归一化
    reward_norm: bool = True        # 奖励归一化
    use_decoupled_loss: bool = False  # 解耦PPO
    prox_logp_method: str = "old"   # prox策略计算方法
    behav_imp_weight_cap: float | None = None  # 行为重要性权重上限
```

#### 4.4.4 InferenceEngineConfig

```python
@dataclass
class InferenceEngineConfig:
    tokenizer_path: str
    consumer_batch_size: int = 256
    max_concurrent_rollouts: int = 1024
    max_head_offpolicyness: int = 1
    queue_size: int = 100
    # 服务器特定配置（SGLang、vLLM）
```

---

## 5. 使用示例

### 5.1 训练脚本示例

**位置**: `examples/math/gsm8k_rl.py`

**最小GRPO训练**:

```python
def main(args):
    config, _ = load_expr_config(args, GRPOConfig)

    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=load_hf_tokenizer(config.tokenizer_path),
    )

    workflow_kwargs = dict(
        reward_fn="areal.reward.gsm8k.gsm8k_reward_fn",
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
    )

    with PPOTrainer(config, train_dataset=train_dataset) as trainer:
        trainer.train(
            workflow="areal.workflow.rlvr.RLVRWorkflow",
            workflow_kwargs=workflow_kwargs,
        )
```

**启动命令**:

```bash
# 单节点（本地启动器）
python3 examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_grpo.yaml \
    scheduler.type=local

# 多节点（Ray启动器）
python3 -m areal.launcher.ray examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_grpo.yaml \
    cluster.n_nodes=4 cluster.n_gpus_per_node=8 \
    allocation_mode=sglang:d12p1t1+d4p1t1
```

### 5.2 配置模式

**分配模式语法**:

```
allocation_mode=<backend>:<train_dims>+<gen_dims>

示例:
- "sglang:d4p1t1"     # SGLang: 4个GPU用于训练（1D并行），1个GPU用于生成
- "vllm:d8p2t1+d8p2t1" # vLLM: 8个GPU用于训练（2D TP），8个用于生成
- "megatron:d32p4t2+d16" # Megatron: 32个GPU训练（4D TP×PP），16个用于生成
```

**AllocationMode字段**:
- `type_`: TRAINING, LLM_SERVER_ONLY
- `train_backend`: "fsdp"或"megatron"
- `gen_backend`: "sglang"或"vllm"
- `train`: ParallelStrategy（dp_size、tp_size、pp_size等）
- `gen`: 推理服务器的ParallelStrategy

### 5.3 运行不同工作流

#### 5.3.1 单轮RLVR（默认）

```python
workflow="areal.workflow.rlvr.RLVRWorkflow"
workflow_kwargs=dict(
    reward_fn="areal.reward.gsm8k.gsm8k_reward_fn",
    gconfig=config.gconfig,
    tokenizer=config.tokenizer_path,
)
```

#### 5.3.2 多轮带自我纠正

```python
workflow="areal.workflow.multi_turn.MultiTurnWorkflow"
workflow_kwargs=dict(
    reward_fn=reward_fn,
    gconfig=config.gconfig,
    tokenizer=tokenizer,
    max_turns=5,
    turn_discount=0.9,
)
```

#### 5.3.3 视觉语言

```python
workflow="areal.workflow.vision_rlvr.VisionRLVRWorkflow"
workflow_kwargs=dict(
    reward_fn="areal.reward.geometry3k.geometry3k_reward_fn",
    gconfig=config.gconfig,
    tokenizer=tokenizer,
    processor_path=processor_path,  # VLM处理器
)
```

#### 5.3.4 自定义Agent工作流

```python
class MyAgent:
    async def run(self, data, base_url, http_client):
        # 使用任何SDK（OpenAI、Anthropic等）
        client = AsyncOpenAI(base_url=base_url)
        response = await client.chat.completions.create(...)
        return reward

workflow=MyAgent()
```

### 5.4 自定义奖励函数

```python
def my_reward_fn(
    prompt: str,
    completions: str,
    prompt_ids: list[int],
    completion_ids: list[int],
    answer: str,  # 数据集特定字段
    **kwargs
) -> float:
    """自定义奖励函数逻辑

    参数:
        prompt: 输入提示
        completions: 模型生成的文本
        prompt_ids: 提示的token IDs
        completion_ids: 完成的token IDs
        answer: 数据集中的正确答案
        **kwargs: 其他数据集字段

    返回:
        奖励值（通常0.0或1.0）
    """
    try:
        # 实现你的验证逻辑
        is_correct = verify_answer(completions, answer)
        return 1.0 if is_correct else 0.0
    except Exception:
        return 0.0
```

### 5.5 自定义工作流

```python
from areal.api.workflow_api import RolloutWorkflow
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import ModelRequest
import torch

class MyCustomWorkflow(RolloutWorkflow):
    def __init__(self, reward_fn, gconfig, tokenizer):
        self.reward_fn = reward_fn
        self.gconfig = gconfig
        self.tokenizer = tokenizer

    async def arun_episode(
        self,
        engine: InferenceEngine,
        data: dict[str, Any]
    ) -> dict[str, torch.Tensor] | None:
        # 1. 准备输入
        messages = data["messages"]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True
        )

        # 2. 创建请求
        req = ModelRequest(
            rid=str(uuid.uuid4()),
            input_ids=input_ids,
            gconfig=self.gconfig,
            tokenizer=self.tokenizer
        )

        # 3. 生成响应
        resp = await engine.agenerate(req)

        # 4. 计算奖励
        prompt_str = self.tokenizer.decode(input_ids)
        completion_str = self.tokenizer.decode(resp.output_tokens)
        reward = await self.async_reward_fn(
            prompt_str,
            completion_str,
            resp.input_tokens,
            resp.output_tokens,
            **data
        )

        # 5. 构建返回字典
        seq = resp.input_tokens + resp.output_tokens
        logprobs = [0.0] * resp.input_len + resp.output_logprobs
        loss_mask = [0] * resp.input_len + [1] * resp.output_len
        versions = [-1] * resp.input_len + resp.output_versions

        return {
            "input_ids": torch.tensor(seq, dtype=torch.int32).unsqueeze(0),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.int32).unsqueeze(0),
            "logprobs": torch.tensor(logprobs, dtype=torch.float32).unsqueeze(0),
            "versions": torch.tensor(versions, dtype=torch.int32).unsqueeze(0),
            "rewards": torch.tensor(reward, dtype=torch.float32).unsqueeze(0),
        }
```

---

## 6. 最佳实践

### 6.1 代码风格

#### 6.1.1 日志规范

- 使用 `areal.utils.logging.getLogger(name)`，**不是** `print` 或 stdlib `logging`
  - 好的: `getLogger("RLVRWorkflow")`, `getLogger("ArchonEngine")`, `getLogger("GSM8KReward")`
  - 避免: `getLogger(__name__)` 或点分路径如 `getLogger("areal.engine.fsdp")`

#### 6.1.2 命名规范

| 类型 | 模式 | 示例 |
|------|------|------|
| 配置数据类 | `XxxConfig` | `GRPOConfig`, `FSDPConfig` |
| 引擎类 | `XxxEngine` | `FSDPEngine`, `ArchonEngine` |
| 工作流类 | `XxxWorkflow` | `RLVRWorkflow`, `MultiTurnWorkflow` |
| 奖励函数 | `xxx_reward` | `math_reward`, `code_reward` |

#### 6.1.3 设计模式

- **组合优于继承**: 避免深层类层次
  - 好: `Engine` 持有 `Checkpointer` 实例
  - 避免: `CheckpointableEngine(Engine)` → `FSDPCheckpointableEngine(CheckpointableEngine)`
- 保持继承浅层（≤2层）
- 谨慎使用mixin；优先显式委托

#### 6.1.4 性能模式

- **避免GPU-CPU同步**: `.item()`, `.tolist()`, `print(tensor)` 导致同步
- **优先批操作**: 避免Python循环张量元素
- **原地操作**: 安全时使用，但注意autograd（`.add_()` vs `+`）

### 6.2 分布式训练最佳实践

#### 6.2.1 并行策略选择

**FSDP2适用场景**:
- 模型参数量 < 100B
- 需要LoRA支持
- 需要序列并行（Ulysses）
- HuggingFace模型生态

**Megatron适用场景**:
- 超大模型（100B+）
- 需要流水线并行
- 需要专家并行（MoE）
- 需要FP8训练
- NVIDIA Megatron生态

#### 6.2.2 内存优化

```python
# 激活检查点
config.gradient_checkpointing = True

# CPU卸载（FSDP2）
config.use_cpu_offload = True

# 微批次规格
config.mb_spec = MicroBatchSpec(
    micro_batch_size=8,
    gradient_accumulation_steps=4
)
```

#### 6.2.3 陈旧度控制

```python
# 控制rollout容量
config.max_concurrent_rollouts = 1024
config.max_head_offpolicyness = 1  # 最大允许陈旧度

# 动态批大小
config.dynamic_bs = True  # 只接受有效样本
```

### 6.3 工作流设计

#### 6.3.1 返回值规范

```python
# 正确: 返回tensor字典，带batch维度
return {
    "input_ids": torch.tensor([...]).unsqueeze(0),  # [1, seq_len]
    "logprobs": torch.tensor([...]).unsqueeze(0),
    "rewards": torch.tensor(reward).unsqueeze(0),
}

# 错误: 缺少batch维度
return {
    "input_ids": torch.tensor([...]),  # [seq_len]
}
```

#### 6.3.2 错误处理

```python
async def arun_episode(self, engine, data):
    try:
        resp = await engine.agenerate(req)
        reward = await self.compute_reward(resp, data)
    except TimeoutError:
        logger.warning("Generation timeout, rejecting trajectory")
        return None  # 返回None拒绝轨迹
    except Exception as e:
        logger.error(f"Error in episode: {e}")
        return None
```

### 6.4 奖励函数

#### 6.4.1 超时处理

```python
# AsyncRewardWrapper自动处理超时
async_reward_fn = AsyncRewardWrapper(
    reward_fn,
    timeout_seconds=15,  # 15秒超时
    max_retries=3,       # 失败重试
)
```

#### 6.4.2 验证逻辑

```python
def robust_reward_fn(prompt, completions, answer, **kwargs):
    try:
        # 尝试验证
        result = verify(completions, answer)
        return 1.0 if result else 0.0
    except Exception as e:
        logger.warning(f"Verification failed: {e}")
        return 0.0  # 失败时返回0
```

### 6.5 性能调优

#### 6.5.1 批大小配置

```python
# 训练批大小
config.train_batch_size = 512

# 消费者批大小（推理）
config.consumer_batch_size = 256

# 微批次大小
config.mb_spec = MicroBatchSpec(
    micro_batch_size=16,
    gradient_accumulation_steps=32  # 16 * 32 = 512
)
```

#### 6.5.2 并发配置

```python
# 最大并发rollout
config.max_concurrent_rollouts = 1024

# 队列大小
config.queue_size = 100

# 奖励计算worker数
config.reward_workers = 8
```

### 6.6 调试技巧

#### 6.6.1 启用详细日志

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### 6.6.2 性能分析

```python
# 配置性能追踪器
from areal.api.cli_args import PerfTracerConfig

perf_config = PerfTracerConfig(
    enabled=True,
    trace_memory=True,
    trace_time=True,
)
engine.config_perf_tracer(perf_config, rank=0, role="actor")

# 保存追踪数据
engine.save_perf_tracer(step=100)
```

#### 6.6.3 检查点验证

```bash
# 验证安装
python areal/tools/validate_installation.py

# 检查检查点
python -c "
from areal.utils.hf_utils import load_hf_model
model = load_hf_model('./checkpoints/step_1000')
print('Model loaded successfully')
"
```

---

## 7. 附录

### 7.1 常用命令

```bash
# 环境检查
python --version              # 需要3.12+
uv --version

# 同步依赖
uv sync --extra cuda          # 带CUDA支持
uv sync --group dev           # 包含开发/测试包
uv run python3 areal/tools/validate_installation.py

# Pre-commit钩子
pre-commit install            # 设置钩子（运行一次）
pre-commit run --all-files    # 格式化和lint

# 运行测试
uv run pytest areal/tests/test_<topic>.py

# 生成CLI文档
uv run python docs/generate_cli_docs.py
```

### 7.2 项目引用

| 目录 | 用途 |
|------|------|
| `docs/tutorial/quickstart.md` | 快速入门 |
| `docs/tutorial/gsm8k_grpo.md` | GSM8K GRPO深度解析 |
| `docs/customization/agent.md` | Agent工作流 |
| `docs/customization/` | 自定义指南 |
| `docs/algorithms/*.md` | 算法详情 |
| `docs/cli_reference.md` | CLI参考 |

### 7.3 相关技能指南

- `/add-dataset` - 数据集加载器创建指南
- `/add-workflow` - 工作流实现指南
- `/add-reward` - 奖励函数指南
- `/debug-distributed` - 分布式调试指南
- `/add-unit-tests` - 测试开发指南

### 7.4 Git工作流

**提交**: Conventional Commits（`feat:`, `fix:`, `docs:`），~72字符主题，祈使语气

```bash
# 示例提交信息
git commit -m "feat: add support for vision-language workflows

- Implement VisionRLVRWorkflow for VLM training
- Add image processing utilities
- Update documentation with VLM examples

This enables training vision-language models on multimodal datasets.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

**PR要求**:
- 运行pre-commit
- 文档化测试覆盖率
- 注明硬件限制

### 7.5 版本信息

- **当前版本**: v0.3+
- **Python**: 3.12+
- **PyTorch**: 最新稳定版
- **主要特性**: 异步RL、多后端支持、高性能

### 7.6 许可证

请参考项目根目录下的LICENSE文件。

---

## 总结

AReaL提供了一个**全面、生产就绪的框架**，用于异步强化学习训练，具有以下主要亮点：

1. **异步架构**: 解耦的rollout和训练，具备陈旧度感知控制
2. **多后端**: FSDP2和Megatron-LM用于训练，SGLang和vLLM用于推理
3. **算法支持**: GRPO、PPO、DAPO、LitePPO、Dr.GRPO、RLOO、REINFORCE++、SAPO
4. **灵活工作流**: 单轮、多轮、视觉语言和自定义agent工作流
5. **可扩展性**: 从单GPU到1000+ GPU，使用Ray/Slurm启动器
6. **高性能**: 相比同步系统2.77×加速比（v0.3版本）
7. **开发友好**: 单文件定制，清晰、文档完善的API

该系统特别适合**训练推理和agent模型**，其中异步rollout、多轮交互和工具使用是常见需求。

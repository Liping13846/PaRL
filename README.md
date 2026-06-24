# PaRL

**P**aper **A**gent for **R**esearch **L**iterature — 复杂学术查询的智能论文搜索 Agent

> 第八届中国研究生人工智能创新大赛 · 华为赛题三：科研场景下复杂学术查询的智能论文搜索与推荐

基于 [PaSa](https://github.com/bytedance/pasa) 双 Agent 架构改造，复用 PaSa-7B 模型能力，面向 AstaBench 评测与本地可复现部署。

---

## 运行指南

### 1. 安装依赖

```powershell
cd D:\1L\PaRL
pip install -r requirements.txt
copy .env.example .env
```

`.env` 建议填写：

```
OPENALEX_MAILTO=你的邮箱
S2_API_KEY=你的Key          # 可选，用于 CorpusID 对齐与辅助检索
```

### 2. 安装 GPU 版 PyTorch（完整模式需要）

```powershell
# 一般 NVIDIA 显卡
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu124

# RTX 50 系列（如 RTX 5060 Ti）需 CUDA 12.8
pip uninstall torch -y
pip install torch==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

### 3. 下载模型

```powershell
# 轻量开发（约 3GB）
python scripts/download_model.py

# PaSa-7B 双模型，正式评测推荐（约 30GB）
python scripts/download_model.py --pasa
```

### 4. 启动

**Web 界面（推荐）**

```powershell
python run_web.py
```

浏览器打开 http://127.0.0.1:7860 ，可选 lite / dev / pasa 配置。

**命令行**

```powershell
# 快速模式：不加载 LLM，仅检索 + 规则排序（约 10 秒，无需 GPU）
python run_agent.py --mode lite --query "transformer visual question answering"

# 完整模式 — Qwen2.5-1.5B 本地调试
python run_agent.py --config dev --query "你的学术查询"

# 完整模式 — PaSa-7B 双模型（推荐）
python run_agent.py --config pasa --query "你的学术查询"

# 批量 JSONL
python run_agent.py --config pasa --input-file data/test.jsonl
```

结果输出到 `results/{config}/`（如 `results/pasa/0.json`）。

### 5. 配置文件

| 配置 | 模型 | 说明 |
|------|------|------|
| `configs/dev.yaml` | Qwen2.5-1.5B，INT4 | 本地开发、快速验证 |
| `configs/pasa.yaml` | PaSa-7B Crawler + Selector，INT4 | **正式评测推荐** |
| `configs/prod.yaml` | Qwen2.5-7B，INT4 | 云 GPU 备选 |

### 6. 常见问题

| 问题 | 处理 |
|------|------|
| 检索 0 结果 | 换英文关键词；Web 页 Ctrl+F5 刷新 |
| 完整模式 OOM | `pasa.yaml` 使用 `load_mode: sequential`；或改用 `dev` |
| RTX 50 系列报错 | 安装 `torch==2.10.0+cu128` |
| S2 限流拖慢 | `configs/pasa.yaml` 设 `enable_s2: false` |
| 端口占用 | 修改 `run_web.py` 中 `port=7860` |

---

## 相对 PaSa 的创新点

PaSa 原版是 **Google Serper（付费）+ arXiv + ar5iv 全文引文扩展** 的端到端 Agent。PaRL 保留 PaSa-7B 双 Agent 核心（Crawler 生成检索词、Selector 相关性打分），在检索后端、查询理解、扩展策略、排序输出与工程部署上做了系统改造，面向赛题 **F1（70%）+ 效率（20%）+ 结构化输出（10%）** 对齐。

### 1. 免费开源检索栈，替代 Google + arXiv 单一链路

| | PaSa | PaRL |
|---|------|------|
| 主检索 | Google Serper（付费 API） | **OpenAlex**（免费，无需 Key） |
| 覆盖范围 |  mainly arXiv 预印本 | 期刊 + 会议 + 预印本 |
| 辅助源 | 无 | Semantic Scholar（可选，CorpusID 对齐） |
| 参赛成本 | 需 Serper Key + 调用费 | **零 API 费用**可跑通全流程 |

PaRL 不依赖 Google 搜索，可在本地/比赛环境直接复现；OpenAlex 元数据更全，对 ACL/NeurIPS 等正式发表论文的 metadata 类查询更友好。

### 2. Metadata 约束感知检索（硬条件前置过滤）

PaSa 将用户查询整体交给 Crawler 生成 `site:arxiv.org` 检索词，**不显式处理** venue、年份、引用关系等硬约束。

PaRL 新增 **Query Parser**，将查询解析为：

- `intent`：semantic / metadata / navigational
- `constraints`：year、venue、cites、topic、method 等
- `sub_queries`：可独立检索的子查询

检索阶段（不仅是排序阶段）使用 OpenAlex filter：

```
publication_year:2024
primary_location.source.id:S...   （venue → Source ID）
cites:W...                        （「引用某篇论文」类查询）
```

配合 venue 消歧（如 ACL → `computational linguistics`），避免缩写误匹配。对 AstaBench 中 navigational / metadata 类查询（约 27%）针对性更强。

### 3. 智能引文扩展，替代 ar5iv 章节解析

PaSa 通过 **ar5iv HTML 全文** 解析 Related Work 章节里的 `\cite{}`，再按标题搜 arXiv；依赖 HTML 质量，且只覆盖 arXiv 预印本。

PaRL 采用 **引文图 + 相似度预筛**：

```
Top-K 高相关 seed 论文
  → 拉 references（参考文献）+ cited-by（引用 seed 的后续论文）
  → 与 query 做 token 相似度预筛
  → Top 8 送 Selector 打分
```

优势：

- 不依赖 ar5iv 解析，**期刊/会议论文**也可扩展
- 双向引文（backward + forward），补全「经典基础工作」与「后续发展」
- 预筛后再调 Selector，减少 LLM 调用，对齐赛题 **效率 20%** 权重
- **条件触发**：候选数 / 高相关数已够时跳过扩展，避免无效 API 消耗

### 4. F1 导向的多轮搜索与早停（Reflector）

PaSa 固定搜索轮次与扩展层数。PaRL 新增 **Reflector** 模块，每轮检索后评估：

- 当前召回是否覆盖子查询
- 高相关论文数量是否足够
- API 预算是否值得继续

动态决定是否继续搜索、生成新的检索方向，在 F1 与 API/LLM 成本间权衡，直接对齐 **F1 70% + 效率 20%** 的评分结构。

### 5. 多信号融合排序 + 结构化输出

PaSa 输出主要为论文标题列表 + Selector 分数。PaRL 新增完整后处理链路：

| 模块 | 功能 |
|------|------|
| **Ranker** | 语义 + 约束满足 + 权威性 + 时效性 四维融合；Tier 分层；MMR 多样性重排 |
| **Summarizer** | 主题聚类归纳、引文关系图 |
| **输出格式** | 对齐 AstaBench：paper_id、markdown_evidence、分层结果 JSON |

覆盖赛题 **结构化输出 10%** 要求，并提供证据链式呈现。

### 6. 约束感知查询分解（Constraint-Aware Query Decomposition）

PaSa Crawler 擅长生成检索关键词，但对「ACL 2024 + 引用 Transformer + 使用 EMD 评测 VQA」这类**组合约束**不做显式拆分。

PaRL Query Parser 区分：

- **硬约束**（year、venue、cites）→ 走 metadata filter
- **软约束**（topic、method、dataset）→ 走语义 search + Selector

子查询可并行检索、独立反思，提升复杂查询的召回完整性。

### 7. 工程化与本地可部署

| 能力 | 说明 |
|------|------|
| **INT4 量化** | PaSa-7B 双模型 16GB 显存可跑（sequential 加载） |
| **三档配置** | dev / pasa / prod 一键切换 |
| **共享 / 顺序加载** | 避免 Crawler/Selector 重复占显存 |
| **Selector batch 推理** | 多篇论文一次 forward |
| **PaSa 单 token 打分** | Selector 只需生成 True/False 1 token，与 PaSa 训练一致 |
| **Web 界面** | lite 快速验证 + full 模式配置切换 |
| **S2 CorpusID 解析** | 检索结果对齐 AstaBench 评测 ID |

PaSa 原版需 Google Key、主要面向 arXiv；PaRL 可在 **消费级 GPU + 免费 API** 条件下完整运行，降低参赛与复现门槛。

### 8. 创新点总结

```
PaSa-7B 双 Agent（Crawler + Selector，RL 训练）
        +
PaRL 自研增强
  ├── Query Parser      约束感知分解 + metadata filter 检索
  ├── Reflector         F1 导向多轮搜索与早停
  ├── 智能引文扩展       references + cited-by + 预筛
  ├── Ranker            多信号融合 + MMR
  ├── Summarizer        聚类 + 引文图 + evidence
  └── OpenAlex 免费栈    替代 Google Serper，覆盖正式发表文献
```

**一句话**：PaRL = 复用 PaSa-7B 领域模型能力 + 面向复杂学术查询与 AstaBench 评测的开源、可复现、约束感知增强框架。

---

## 参考

- [PaSa 论文与代码](https://github.com/bytedance/pasa)
- [PaSa-7B 模型](https://huggingface.co/bytedance-research/pasa-7b-crawler)（CC-BY-NC-SA-4.0，参赛使用需署名）
- [OpenAlex API](https://docs.openalex.org/)
- [AstaBench / PaperFindingBench](https://github.com/allenai/asta-bench)

# MoHA

这个目录是独立 Git 仓库，只保存一份当前实现。历史修改通过 Git 查看；真实配置、实验输出和冻结运行记录放在仓库外。

```text
moha/
  src/moha/          运行、诊断、选择与校准逻辑
  tests/             单元与集成测试
  calibrated/        已完成校准的模型组合，一组一个当前配置
  scripts/           本地 Whisper 启动入口
  config.example.json
  pyproject.toml
  README.md
  AGENTS.md
```

## 运行

使用已有 Video OS Python 环境。源码模式不需要安装，也不需要修改父目录的 Python 包：

```bash
cd /home/jianghan/video_os/moha
export PYTHONPATH="$PWD/src"
PY=/home/jianghan/miniconda3/envs/llamafactory/bin/python

# 将模板复制到仓库外，填写数据清单、服务地址和凭据引用。
# 示例：/home/jianghan/video_os_runs/moha_current/config.json
$PY -m moha doctor --config /path/outside/repo/config.json
$PY -m moha run --config /path/outside/repo/config.json --output /path/outside/repo/run
$PY -m moha resume --config /path/outside/repo/config.json --output /path/outside/repo/run
```

Video OS 的工具、媒体处理和模型适配器由 `config.runtime.root` 指定；`config.runtime.commit` 必须与该目录的提交完全一致。启动时检查模块实际导入路径，不扫描父目录中的历史版本。模板绑定了已修复视频结尾取整问题的干净运行时。要切换依赖，显式修改配置并创建新实验。

MoHA 与 Video OS 分别记录 Git 提交、源码哈希和干净状态。`doctor` 检查输入、视频哈希与配置，不调用模型；真实推理要求两份代码均已提交。配置只保存凭据文件或环境变量引用，不保存密钥。

无网络演示：

```bash
PYTHONPATH=src python -m moha demo --output /tmp/moha-demo
```

独立安装可使用 `python -m pip install --no-deps -e .`；运行依然使用上述 `python -m moha` 入口。

## 校准流程

所有模型栈从同一 H0 开始：search/observe 两个语义工具、一个 Omni observer。Planner 支持模块与 observer 执行策略由目录中的可执行候选定义。

模板中的 `specialists: ["ocr", "asr"]` 让两种 specialist 成为校准候选，H0 的 `harness.specialists` 仍为空。只有对应能力的 observer 失败经过执行设置 probe 后得到 `no_rescue`，selector 才能提出该 specialist；最终是否保留仍由独立验证决定。只配置模型不代表已经校准或启用。

OCR 通过 QDD 的 `Qwen/Qwen3.5-4B` 执行，`image.key` 引用已有凭据。ASR 使用本地 `whisper-large-v3-turbo` 的转写接口；`asr.language: null` 表示自动识别语言。两者复用固定 VideoOS 的媒体与模型适配器，receipt 分别记录真实 specialist 模型名。缺少后端配置时不能把对应 specialist 加入候选。

检查 GPU 空余显存后，可在仓库根目录启动 Whisper；日志目录必须在仓库外：

```bash
MOHA_WHISPER_GPU=1 bash scripts/serve_whisper.sh > /path/outside/repo/whisper.log 2>&1
```

默认复用已有权重与 vLLM 环境，监听 `127.0.0.1:8093`，最多并发两条请求，显存比例设为 0.08。GPU、端口、vLLM 路径和权重路径可通过脚本中列出的 `MOHA_WHISPER_*` 环境变量指定。

校准 trace 用于诊断和提案。Selector 每次只能选择一个合法候选或放弃；独立的视频级验证集决定是否保留修改。Observer probe 仅在校准期间执行，固定窗口、目标和 observer，测试有界执行设置变化。无可靠证据时保留 `unresolved`；预算耗尽本身不证明某项语义失败。

## Judge 与 selector 的证据

`evidence.py` 是唯一的证据投影入口，不改变 benchmark 时的 planner 行为。

- Judge 收到题目/校准答案、当前 harness、工具 schema、初始消息、每步真实可见消息、planner 已返回的文本、工具结果和用量。缺失的历史记录明确标为缺失。
- 工具与 assistant 的 JSON 正文解析成完整对象，system/user 消息及非 JSON 文本保持原文，保留初始视频时长等元数据；不生成模型未返回的推理。不同时间出现的同一 observation 的不同内容不会按 ID 强行合并。底层文本接口继续过滤原始工具结果中的媒体句柄，不向文本模型传递视频或音频文件。
- 重复 JSON 容器通过 `shared` 表引用。`unpack` 可还原完整语义数据；该结构只是存储去重，没有推断因果边。
- Selector 在全局失败分布、全部诊断、当前 harness、候选与接受历史之外，得到每个失败族/能力组的一条中等长度真实校准 trace。示例选择规则与数量公开，不把示例当作全量统计。
- 示例保留全部工具动作和观察内容，包括后来相反的描述；仅附加的完整上下文摘录限定为最多三步：最早引用步骤、紧随其后的步骤和最后决策上下文。未附完整消息的步骤明确列出，其动作、观察和可见性摘要仍在时间线中。
- Observer probe 的实际结果与判定随代表性 trace 提供。验证/测试样本、标签和轨迹不会传给 selector。
- 较新的观察不自动覆盖旧观察；已纠正的错误仍可能消耗预算。冲突不自动判给 planner，也不自动启用 verification。现有 `verification_basic` 是辅助上下文模块，不是 planner 必须调用的工具。

## 输出与续跑

每个运行目录记录 `manifest.json`、完整 episode、诊断及其证据、选择、验证结果、checkpoint 和最终冻结 harness。单个运行只允许一个写入者；已完成记录不可覆盖。

`resume` 只接受完全一致的源码、依赖、配置和输入身份。修改 judge/selector 后应建立新实验；复用旧 episode 必须单独核验运行语义并记录来源，不能把旧 manifest 改名覆盖。

最终评测：

```bash
PYTHONPATH=src python -m moha evaluate --config /path/config.json \
  --frozen /path/calibration-run/frozen_harness.json \
  --manifest /path/test_manifest.json --output /path/test-run
```

完成校准后，运行 `PYTHONPATH=src python -m moha export --run /path/calibration-run --output calibrated`，将通过身份、完成状态和接受历史核验的模型组合登记到 `calibrated/`。每组保留一个当前 JSON，历史由 Git 管理。配置含冻结 harness、模型规格、预算和来源，不含密钥或逐样本标签。它是校准结果档案；复现实验仍使用来源运行的固定源码和配置。

## 测试

```bash
PYTHONPATH=src:/path/to/pinned/video-os python -m unittest discover -s tests -v
```

测试覆盖证据去重的还原、冲突保留、上下文可见性、selector 示例与校准隔离、原有 schema/预算/缓存/验证门限，以及真实 Video OS registry 的离线集成。

源码最初来自父仓库提交 `bc83ebb` 中的 MoHA 重写。现在的权威源码是这个独立仓库；父目录不再跟踪这里的文件。

# MoHA

这个目录是独立 Git 仓库，只保存一份当前实现。历史修改通过 Git 查看；真实配置、实验输出和冻结运行记录放在仓库外。

```text
moha/
  src/moha/          运行、诊断、计票与校准逻辑
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

Planner 用 `video_player_observe(start_seconds, end_seconds, goal)` 直接选择源视频时间范围。检索候选只提供定位线索：可以沿用其起止时间、扩展前后文，也可以按题目时间直接观察，无需先 search 或提供 `candidate_id`。例如检索命中 107–109 秒后，可以请求：

```json
{"start_seconds": 95, "end_seconds": 120,
 "goal": {"type": "sequence", "target": "Describe the events before and after the object falls."}}
```

接口要求有限数值且 `0 <= start_seconds < end_seconds <= duration_seconds`；非法范围返回工具错误供 planner 修正，不静默移动、扩大、裁剪或取整窗口。帧率、分辨率、采样及 observer/specialist 路由仍由 harness 和固定 Video OS 执行层决定，原有媒体与预算约束继续生效。

`tools.py` 只适配时间选择和 schema，复用固定运行时的观察、specialist、预算与 receipt 路径。直接窗口的 receipt 保留实际时间与目标，`candidate_id` 为 `null`；probe 继续固定该实际窗口。轨迹的 `planner_tool_policy: moha_semantic_windows_v1` 标记此接口。改变时间选择能力需要新实验并重跑 H0，旧的 candidate-only episode 不能作为新接口的基线；既有冻结运行保持原接口。

模板中的 `specialists: ["ocr", "asr"]` 让两种 specialist 成为校准候选，H0 的 `harness.specialists` 仍为空。只有对应能力的 observer 失败经过执行设置 probe 后得到 `no_rescue`，Judge 才能为这条 trace 提出该 specialist；最终是否保留仍由独立验证决定。只配置模型不代表已经校准或启用。

OCR 通过 QDD 的 `Qwen/Qwen3.5-4B` 执行，`image.key` 引用已有凭据。ASR 使用本地 `whisper-large-v3-turbo` 的转写接口；`asr.language: null` 表示自动识别语言。两者复用固定 VideoOS 的媒体与模型适配器，receipt 分别记录真实 specialist 模型名。缺少后端配置时不能把对应 specialist 加入候选。

检查 GPU 空余显存后，可在仓库根目录启动 Whisper；日志目录必须在仓库外：

```bash
MOHA_WHISPER_GPU=1 bash scripts/serve_whisper.sh > /path/outside/repo/whisper.log 2>&1
```

默认复用已有权重与 vLLM 环境，监听 `127.0.0.1:8093`，最多并发两条请求，显存比例设为 0.08。GPU、端口、vLLM 路径和权重路径可通过脚本中列出的 `MOHA_WHISPER_*` 环境变量指定。

校准流程只有一条主线：**Trace → Judge → Aggregate → Validation**。没有全局 LLM selector，也不接受 `models.selector` 配置。

1. Judge 读取每条失败 calibration trace，同时给出归因、证据步骤、`candidate_id` 和 `proposal_reason`。候选只能来自当前可执行 catalog，或为 `null`。`confidence` 仅表示归因置信度，不参与计票；没有失败标签到模块的硬编码映射。
2. 若归因为 observer，首次提案必须为 `null`。先在同一窗口、目标、observer 下做执行设置 probe，再让同一个 Judge 根据该 trace 和实际 probe 结果完成局部提案。只有匹配 text/speech 目标的 `no_rescue` 才开放对应 OCR/ASR specialist。probe 不确定时不能据此宣称能力缺失。
3. 每条有效失败 trace 最多一票。按支持样本数降序排序，同票按完整 candidate ID 的字典序排列。`unresolved`、弃权、错误以及不再合法的提案不投票。
4. 每轮冻结一次提案集合。在预设候选预算内依次验证排名最高的候选；拒绝后移除它，使用原票数的下一名，不要求 Judge 改投。成功后更新 harness，并在下一轮产生新的 trace 和提案。已启用、无实际作用及此前被拒绝的候选不再参与。
5. 验证集只用于原有成对收益门限，只有过门限才 promote。无支持候选时保留原有 patience 与失败分布稳定性停止规则。预算耗尽本身不证明某项语义失败。

这个规则衡量跨 trace 的支持度，不声称求得全局最优或证明局部归因。验证集被多轮使用后仍需独立最终测试集。默认每轮最多验证两个候选；每轮最多接受一个，接受即结束该轮。

## Planner 上下文

`context.py` 在原有历史轮数与 token 上限生效之前投影 planner 输入：完整 observation 正文、事实 ID、否定结果、不确定性、时间范围与候选句柄仍保留；历史工具消息中的重复 player/budget/state 快照只保留最新一份。最新状态中的候选目录、搜索历史、访问窗口和预算继续可见，旧 search 消息单独保留其返回的候选句柄。

没有事实正文的历史 observation ID/fact IDs、观察审计记录和版本标识不再反复进入 planner 上下文。实际观察仍携带其窗口、目标与简要采样信息。完整工具结果、执行 receipt、原始消息都留在轨迹中；每步 context 事件记录真实送给 planner 的消息，`planner_context_policy` 标记投影规则。

这一步只清理已有输入，不总结或找回已被历史截断丢弃的事实，也不隐式开启 memory。事实附带的来源 ID 与空挂的历史 ID 区别处理；相同 observation ID 下出现的矛盾正文仍分别保留。

`memory_basic` 由 `memory.py` 实现，按完整 observation 保留 planner 已收到的事实正文、来源、窗口、目标、采样信息、`missing`、`uncertainties` 和 refinement。没有额外 LLM 总结，也不再按 10 条 fact、每条 240 字符截断。零事实但包含缺失信息的观察仍可保留；相同 ID 下的不同正文或限定条件分别保存，仅完全相同的记录去重。

Memory 最多占 `history_tokens` 的一半，按固定运行时的保守 token 估计计数：先放入能容纳的前两条 observation，再从新到旧填充，展示时恢复发生顺序。超出容量时整条省略，不能只留下事实而删除其限定条件；省略条数明确告知 planner，完整记录仍在审计轨迹中。实际 memory 用量从原历史 token 配额扣除，剩余额度用于原有历史选择，因此不靠另加一份历史预算实现记忆。原历史上限仍是 advisory：保留的锚点和最新完整工具轮可能超额，system/task、工具 schema 及控制反馈也不属于该历史配额；这不是总 API 输入的硬上限。

压缩提示明确区分当前 player/预算状态与观察证据，不再将最新工具结果整体视为权威。否定和缺失信息仅适用于所述窗口与目标，不等于全视频不存在该事件；新观察不能自动覆盖旧的相反证据。context 事件的 `visible_observations` 同时统计保留的工具正文和实际送出的 memory，按内容去重，保留同 ID 的冲突。

所有 harness 都在既定 `max_steps` 内预留最后一次 planner 调用用于回答或明确弃权（默认第 16 次），以免最后一步取得观察后无机会使用。该调用携带 `tool_choice: none` 和明确收尾提示，不增加额外调用。若 provider 仍返回工具调用，记录原始输出和 `tool_calls_on_reserved_final_call`，不执行这些工具，结果保持 `budget_exhausted`。轨迹分别记录 `planner_completion_policy` 和启用时的 `memory_policy`；较早作答不受阻拦。

修改投影会改变实际模型输入与可保留的历史范围，下一次必须从 H0 开始重新进行完整 calibration/validation，不能复用旧投影下的 episode 作为新基线。正在运行的实验继续使用其冻结源码和原有投影。

## Judge 的证据

`evidence.py` 是唯一的完整轨迹投影入口，不改变 benchmark 时的 planner 行为。

- Judge 收到题目/校准答案、当前 harness、工具 schema、初始消息、每步真实可见消息、planner 已返回的文本、工具结果、用量和当前候选。缺失的历史记录明确标为缺失。
- 工具与 assistant 的 JSON 正文解析成完整对象，system/user 消息及非 JSON 文本保持原文，保留初始视频时长等元数据；不生成模型未返回的推理。不同时间出现的同一 observation 的不同内容不会按 ID 强行合并。底层文本接口继续过滤原始工具结果中的媒体句柄，不向文本模型传递视频或音频文件。
- 重复 JSON 容器通过 `shared` 表引用。`unpack` 可还原完整语义数据；该结构只是存储去重，没有推断因果边。
- observer 的最终提案同时看到完整单条 trace、实际 probe 输出及判定理由；没有跨样本代表 trace packet，也不把 validation/test 的样本、标签、轨迹或分数传给 Judge。
- 较新的观察不自动覆盖旧观察；已纠正的错误仍可能消耗预算。冲突不自动判给 planner，也不自动启用 verification。现有 `verification_basic` 是辅助上下文模块，不是 planner 必须调用的工具。

论文方法部分需与此实现一致：将“诊断不提出修复、全局 selector 选择”的描述改为局部候选推荐、等权支持聚合和验证。固定 H0、离散单坐标 catalog、observer probe 与验证门限保持原定义。

## 输出与续跑

每个运行目录记录 `manifest.json`、完整 episode、诊断及提案、每轮冻结提案集合、计票排序及支持样本 ID、验证结果、checkpoint 和最终冻结 harness。单个运行只允许一个写入者；已完成记录不可覆盖。

`resume` 只接受完全一致的源码、依赖、配置和输入身份。修改 Judge、计票规则或运行逻辑后应建立新实验；复用旧 episode 必须单独核验运行语义并记录来源，不能把旧 manifest 改名覆盖。

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

测试覆盖证据去重的还原、冲突保留、上下文可见性、等权计票、同票排序、拒绝后续选、提案缓存与校准隔离、原有 schema/预算/缓存/验证门限，以及真实 Video OS registry 的离线集成。

源码最初来自父仓库提交 `bc83ebb` 中的 MoHA 重写。现在的权威源码是这个独立仓库；父目录不再跟踪这里的文件。

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

后续实验的 Judge 使用 apihy `claude-opus-4-8`（API 模型列表中的准确名称），
地址为 `https://zgc.apihy.com`，响应模式为 `json_text`。`config.example.json`
引用服务器已有的 `apihy_claude_judge_20260902.json` 文件中的 `claude` 凭据；
其他机器可将 `models.judge.key` 替换为自己的文件或环境变量引用。
trace 诊断和 observer probe 判定均使用 `models.judge`。启动时拒绝 `216.*`
地址上的 `gpt-5.5` Judge 配置，避免从旧实验复制配置后误用旧服务。
Planner 的独立模型配置不受此限制。Judge 配置属于运行身份：更换 Judge 时使用新输出目录，
已有实验按原冻结源码和配置续跑，不能将旧 Judge 的诊断缓存标记为 Claude 结果。

新实验模板使用 `f_view=128 / p_view=262144 / p_call=33554432 / b_video=16384`。
这些是 H0 与所有候选共享的上限；单次帧数由窗口、目标帧数、源帧数与预算决定。
固定运行时中的 `f_episode=256` 与 `b_video_episode=65536` 是 episode 累计量的提示目标，不是阻断调用的硬上限。累计消耗继续写入 ledger；单次硬上限与 planner 步数限制保持生效。
已有冻结配置不从模板继承新值，也不应在原实验中途替换服务的媒体读取规则。

Qwen3-Omni 服务还有独立的 vLLM 视频解码上限，默认 32。`mm_processor_kwargs.fps`
不能覆盖它。使用以下入口从同一实验配置生成 `--media-io-kwargs`，其 `video.num_frames`
严格等于 `budget.f_view`；旧 32 帧配置仍生成 32，新 128 帧配置生成 128：

```bash
PYTHONPATH=src python -m moha.serving --config /path/outside/repo/config.json \
  --gpu 0,1 --host 0.0.0.0 \
  --vllm-bin /home/jianghan/miniconda3/envs/video-os-qwen25omni-vllm/bin/vllm-omni \
  --model-path /path/to/Qwen3-Omni-30B-A3B-Instruct --dry-run
```

`--dry-run` 只输出无凭据的启动参数，不访问端点或 GPU。确认 GPU 与端口可用后，
以 `--launch-record /path/outside/repo/omni-launch.json` 替换 `--dry-run` 启动服务。
端口从 `observer.base_url` 读取；多个 observer 端点使用 `--endpoint-index` 选择。
入口保存配置哈希、实际命令、明确的环境变量和 PID；不停止或重启已有服务，不覆盖旧启动记录。
总上下文默认保持 32768，可显式配置 `--max-model-len`，仍需为音频、文本与回答预留空间。
`doctor` 显示所需的 loader 配置，但 `server_verified=false`：离线配置检查和启动记录都不能
证明远端服务已就绪或实际保留了全部帧。上线时仍需核验实际 loader、decoded frames 和 processor grid。

`models.planner.spec.base_url` 和 `observer.base_url` 均支持用逗号列出同一模型的多个端点。两边数量相同时按顺序配对；只有一个端点的一边由所有 lane 共用。例如两个 9B planner 端点和一个 Omni 端点形成两路执行，共用该 Omni。一个 planner 端点和两个 Omni 端点仍形成原有的两路执行；数量既不相等又都大于一时拒绝配置。每路拥有独立的 service、planner client 和 session 目录，即使地址相同也不共享本地可变状态。样本按清单序号轮流固定分配，每路同时最多运行一个 episode；缓存命中、H0 与候选验证均保持相同端点分配，最终结果按原清单顺序对齐。`doctor` 同时显示各路 planner 和 observer 地址。

校准与验证的 episode 批次共用这一条执行路径；单端点就是一路，无另一个串行实现。计票和 promotion 由一个协调器顺序决策；observer probe 使用该 calibration 样本原分配的端点。任一路失败后不再提交新 episode，另一条已在执行的 episode 完成落盘后退出，续跑复用成功缓存。`workers/0.json`、`workers/1.json` 记录每路当前样本与时间，`progress.json` 由协调器更新总进度；完整 attempt 记录端点归属。

外部 LLM 诊断由 `diagnosis_workers` 控制，默认 `8`，设为 `1` 即为串行。它独立于 planner/observer 的执行路数：每个 worker 使用独立 Judge client，同一时刻最多处理一条失败 trace，包含初次诊断、probe 和最终局部提案；默认最多八条在途，无待执行请求积压。probe 与其判定使用样本原执行路的独立 service/client，并按路互斥，因此增加诊断 worker 不会增加同一路 observer 的并发。完成结果按原样本顺序汇总，一样保留每样本一票、健康门限和验证决策；成功诊断立即写入不可变缓存。异常退出时停止新提交，已在途结果仍可落盘，续跑只补缺失或失败的诊断。`diagnosis_workers/<序号>.json` 记录各 worker 状态，`progress.json` 显示诊断进度，`doctor` 显示并发数。并发设置进入运行身份，已有冻结实验继续使用自己的代码与配置。

`observer.py` 在固定感知服务上补充一条输出恢复路径：首次请求保持原样，仅在 observation JSON 无效、`finish_reason=length` 或支持时间越出当前窗口时，使用原媒体、目标、采样和模型重试一次。重试提示按事件/变化报告，合并连续不变的状态，保留真实重复动作与冲突；输出上限为 2048 token（原显式上限更低时沿用更低值）。不对有效 observation 做事后去重、裁剪或改写。原始输出与两次请求哈希均留档，失败输出也计入 provider token 总数，重试另计一次感知与媒体成本。

两次均无效时返回 `ObserverOutputError`，不把残缺文本送入 planner 证据或 memory；planner 可在剩余调用内继续观察或回答。Judge 收到恢复状态与成本，不能仅据输出格式问题归因于 planner 能力；probe 使用同一规则，耗尽重试时记为 inconclusive。网络、鉴权、缺失用量等基础设施错误继续停止批次。恢复记录在 session receipt 的 `observer_output_recoveries`；`observer_calls` 表示逻辑请求，`observer_retry_calls` 单列额外请求，感知 ledger 与 `provider_totals` 已包含重试。此规则对所有 harness 相同，不是 catalog 中的自适应模块。

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

`tools.py` 适配时间选择、schema 与 observer 执行策略，复用固定运行时的观察、specialist、预算与 receipt 路径。直接窗口的 receipt 保留实际时间与目标，`candidate_id` 为 `null`；probe 继续固定该实际窗口。轨迹的 `planner_tool_policy: moha_semantic_windows_v1` 标记此接口。改变时间选择能力需要新实验并重跑 H0，旧的 candidate-only episode 不能作为新接口的基线；既有冻结运行保持原接口。

当前 observer harness 使用 `moha_source_relative_v1`，执行设置按观察目标独立继承默认值：

| 坐标 | 候选值 | 含义 |
| --- | --- | --- |
| `frames` | `auto`、32、64、128 | 目标帧数；`auto` 为窗口秒数向上取整，保持原先 1 fps 的请求密度 |
| `source_scale` | 0.5、0.75、1.0 | 原视频两条边的比例上限，保持宽高比，不上采样 |
| `priority` | `temporal`、`spatial`、`balanced` | 预算受限时保留时间信息、空间信息，或平衡两者 |

H0 为 `auto / 1.0 / balanced`，所有模型从同一 H0 开始。目标级设置只覆盖写出的字段；
例如 `{"execution":{"default":{},"text":{"source_scale":0.75}}}` 只改变 text 的空间目标。
catalog 为八类 typed goal 提供单字段候选，包括 speech；默认值用于继承，不作为全局校准候选。
帧数和比例是目标上限，最终输入还受源帧数、codec 尺寸、`f_view/p_view/p_call/b_video` 约束。

`execution.py` 在同一可行集合中分配预算：枚举不超过目标值的整数帧数，以及从目标边长比例开始每档乘 0.9 的降采样阶梯。
时间优先按帧数、空间比例排序；空间优先反过来；均衡模式最大化两项相对目标完成度的较小值，
同分依次比较完成度总和、空间、时间。三种模式共享最低画质边界：每条边至少保留单帧像素预算内可达到目标尺寸的 25%（按 codec 取偶数），防止时间优先为少量额外帧数把画面压成几十像素。预算充足时三种模式得到相同输入。预算不足时允许比例低于目标档位，但不得低于这条边界。
分配使用固定运行时的渲染尺寸与模型 token 估计，再通过其 `experiment_render` 接口执行；不另建媒体或模型适配器。
Qwen processor 的最小像素要求可能把小画面放大，因此降低源画质不保证降低 token；receipt 分别保存源尺寸、渲染尺寸、帧时间戳和 processor 计量。

固定窗口 probe 先进行不调用模型的分配检查，跳过与基线或已选候选产生相同媒体的配置。
每条 trace 最多检查一个请求、一份新基线和七个不同的单字段候选（3 个帧数、2 个比例、2 个模式），
找到 rescue 即停止。每个逻辑观察仍最多进行一次格式修复；全部实际成本入账。
实际 receipt 再核对源文件、帧时间戳、帧数和完整画面尺寸，不能只按参数名声称发生改变。
只有得到 rescue 的具体执行候选能进入该 trace 的最终提案；匹配类型的 `no_rescue` 才开放已有 OCR/ASR 路由。

本次改变了 H0 与 observer catalog，必须创建新实验并从 H0 完整校准。旧的字符串预设配置显式拒绝加载，
不能把旧 profile 或缓存自动转换为新策略。`calibrated/` 中既有结果继续对应其记录的冻结源码与 catalog；
正在执行的冻结校准及其服务配置不随本次修改改变。

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

`memory_basic` 由 `memory.py` 实现两个追加式账本：`result_memory` 原样保存成功观察的正文、窗口、目标、采样信息与所有限定字段，包括完全重复记录和同ID冲突；`working_memory` 保存 planner 通过 `memory_note` 写下的原文。`memory_read` 按 result/working/both 和可选 source IDs 读取原记录。没有自动摘要、排序、合并、前两条锚点或语义索引。每步只显示账本条数与来源ID，取回的原文与普通工具结果一样经过既有历史投影和预算。

`verification_basic` 开启原生 `verify_fresh(diagnostic_question?, source_ids?)`：同一个配置的文本模型在两个新消息中，读取原问题、待诊断问题、原始观察及其窗口，返回自然语言诊断。输入没有工作笔记、历史planner对话、观察目标、视频截图或选项列表。没有覆盖率规则、判定枚举、JSON诊断输出要求或自动修复重试。结果原样返回，planner自行决定继续观察、修改判断或作答。未开启memory时，诊断只能使用当前历史投影中仍可见的观察，不能借verification找回已被截断的证据。

两个视频工具保留；只在相应模块开启时增加memory读写或verify工具。默认H0不增加任何模块工具。旧的EvidenceLedger覆盖率/complete反馈已退出正式episode循环。

`max_steps` 是共享的planner加diagnosis模型调用额度。诊断花费一次现有调用，必须还留一次planner响应供其使用诊断并作答；预算不足时返回工具错误，不自动调用或扩充额度。所有harness最后一次调用仍禁用工具。轨迹记录`verification_request`（实际两消息输入）、`verification`（原文及provider元数据）、`memory`（最终两个账本）及分别计数的`planner_calls / verification_calls / model_calls`。模型诊断失败不触发额外重试。

修改投影会改变实际模型输入与可保留的历史范围，下一次必须从 H0 开始重新进行完整 calibration/validation，不能复用旧投影下的 episode 作为新基线。正在运行的实验继续使用其冻结源码和原有投影。

## Judge 的证据

`evidence.py` 是唯一的完整轨迹投影入口，不改变 benchmark 时的 planner 行为。

- Judge 收到题目/校准答案、当前 harness、工具 schema、初始消息、每步真实可见消息、planner 已返回的文本、工具结果、用量和当前候选。缺失的历史记录明确标为缺失。
- 工具与 assistant 的 JSON 正文解析成完整对象，system/user 消息及非 JSON 文本保持原文，保留初始视频时长等元数据；不生成模型未返回的推理。不同时间出现的同一 observation 的不同内容不会按 ID 强行合并。底层文本接口继续过滤原始工具结果中的媒体句柄，不向文本模型传递视频或音频文件。
- 重复 JSON 容器通过 `shared` 表引用。`unpack` 可还原完整语义数据；该结构只是存储去重，没有推断因果边。
- observer 的最终提案同时看到完整单条 trace、实际 probe 输出及判定理由；没有跨样本代表 trace packet，也不把 validation/test 的样本、标签、轨迹或分数传给 Judge。
- 较新的观察不自动覆盖旧观察；已纠正的错误仍可能消耗预算。冲突不自动判给 planner，也不自动启用 verification。`verification_basic` 提供可选的独立诊断工具，不要求planner必须调用。

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

### Terminal answer compatibility

All planners retain the same OpenAI-compatible POST contract and final-call
budget. No JSON Schema constraint or extra answer-generation call is added.
A complete answer JSON object at the end of explanatory text or a JSON code
fence can be recovered after the normal structured parse. This shared fallback
validates option labels and explicit abstention, rejects duplicate JSON keys,
and does not read answers from tool-call or reasoning markup. Existing plain
JSON and natural-language answer handling remains unchanged. Trajectories
record answer_parsing_policy, answer_extraction, and terminal_answer_status;
a live process or a completed episode alone does not establish output-format
health.

For Qwen served by vLLM, configure the server to honor the existing
tool_choice=none request with --exclude-tools-when-tool-choice-none.
The configured thinking mode may be set through
--default-chat-template-kwargs '{"enable_thinking":false}' so the host does
not need a Qwen-specific request field. Service changes require a fresh run;
keep old trajectories and validate historical parsing offline.

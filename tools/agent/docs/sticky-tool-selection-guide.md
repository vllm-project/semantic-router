# Trusted-session sticky tool selection 学习与实施指南

这份指南把本次会话中已经完成的仓库理解、需求解释、实施路线和验证规则
集中在一起。它面向需要自己修改代码的贡献者；指南不替代实际 API 设计，
实现前仍要以当前 `main` 和已经合并的 foundation contract 为准。

## 先记住三件事

1. 这是一个 Envoy ExtProc 请求路由器，不是模型服务本身。它把请求解码成
   中立结构，做信号和决策路由，再编码后交给后端。
2. sticky tool selection 只是在可信 session 中保存“工具身份和顺序”的有界
   状态；它不能保存授权结论，也不能替代每次请求的授权、能力和策略检查。
3. 功能默认关闭。没有明确配置、可信身份或可用状态存储时，行为必须与当前
   stateless tool selection 相同，并安全回退。

## 仓库是谁的、谁负责什么

- 当前远端是 `Lcollection/semantic-router`，所以这个本地 fork 的所属账号是
  `Lcollection`。
- 上游项目是 `vllm-project/semantic-router`。
- 根目录 `OWNER` 是仓库级回退维护者：`@rootfs`、`@Xunzhuo`、
  `@WUKUNTAI-0211`、`@AayushSaini101`。
- Router runtime 的 `src/semantic-router/OWNER` 维护者是
  `@FAUST-BENCHOU`、`@shraderdm`、`@drivebyer`、`@ramkrishs`、
  `@WUKUNTAI-0211`、`@AayushSaini101`、`@siloteemu`、`@theohsiung`、
  `@wilsonwu`、`@Peterren`。

最终 reviewer 仍以目标文件最近的 `OWNER` 和 `.github/CODEOWNERS` 为准；
“所属人”不等于某一个功能的唯一代码负责人。

## 项目地图：先从哪儿看

| 层 | 作用 | 首要入口 |
| --- | --- | --- |
| `src/vllm-sr/` | Python CLI、镜像和本地栈编排 | `src/vllm-sr/cli/main.py`、`commands/runtime.py` |
| `src/semantic-router/` | Go Router、Envoy ExtProc、配置和运行时 | `pkg/extproc/`、`pkg/config/` |
| `config/` | canonical 配置、recipe 和运行时示例 | `config/recipes/` |
| `candle-binding/`、`ml-binding/`、`nlp-binding/`、`onnx-binding/` | 原生推理/模型绑定 | 各目录 `Makefile` 和 README |
| `dashboard/` | 管理后台前端和后端 | `dashboard/frontend/`、`dashboard/backend/` |
| `deploy/`、`e2e/` | 部署产物、Kubernetes 和端到端验证 | `deploy/`、`e2e/` |
| `tools/`、`website/` | 构建/测试工具与公开文档 | `tools/make/`、`website/docs/` |

完整的目录边界见 [repo-map.md](repo-map.md)，变更分类见
[change-surfaces.md](change-surfaces.md)。

## 一次请求怎样走到 provider

```text
Envoy headers
  -> captureRequestHeaders
  -> prepareProtocolRequest (wire -> llmprotocol.Request)
  -> signal snapshot / decision / model selection
  -> request plugins and dispatch preparation
  -> tool selection on the neutral request
  -> finalizeProviderDispatchResponse
  -> OpenAI Chat / Responses / Anthropic codec
  -> selected backend
```

与本需求最相关的文件职责如下：

- `pkg/extproc/processor_req_header.go`：捕获 transport headers、request ID 和
  auth 相关信息。
- `pkg/extproc/processor_protocol_contract.go`：协议解码成
  `llmprotocol.Request`，并在末端调用 provider 编码。
- `pkg/extproc/processor_req_body_routing.go`：请求体路由编排。它是热点文件，
  只应接入窄 helper，不应把 state store、fingerprint 和合并规则全塞进来。
- `pkg/extproc/req_filter_tools.go`、`req_filter_tools_history.go`、
  `req_tool_selection_plugin.go`：工具模式、历史处理、语义检索和插件路径。
- `pkg/llmprotocol/types.go`：provider 无关的 request/tool/metadata 类型。
- `pkg/tools/tools.go`、`relevance.go`：工具目录、embedding 检索和排序。
- `pkg/config/tools_plugin.go`、`tool_selection_plugin.go`：全局及 decision 级
  工具策略。
- `pkg/protocolcodec/`：最后把中立 `request.Tools` 按 slice 顺序序列化；这里
  不应重新实现授权或 sticky 合并。

### 当前工具选择链

```text
handleToolSelectionForRequest
  -> handleToolSelection
  -> decision plugin (add/filter) 或 early mode
       none       : 清除工具和可选的 tool history
       filtered   : 按 allow/block 过滤
       passthrough: 保留客户端工具，auto 时继续选择
  -> semantic retrieval / tool_selection plugin
  -> applySelectedTools (更新 request.Tools 和 Generation)
  -> commitToolSelection (写回 ctx.SemanticRequest)
```

sticky 逻辑应围绕这条链的窄 seam 工作：先完成正常 request-time selection，
再对候选集合做“当前请求已授权的历史工具 + 新候选”的有界合并，最后一次性
提交中立 request。不要绕过现有 mode、plugin 或 codec。

## 需求翻译成工程规则

### 必须做到

- **默认关闭**：关闭时不初始化、不读取 state store，保持原 stateless 路径。
- **可信身份**：sticky key 必须由 transport/auth middleware 证明并由服务端派生；
  客户端 `x-session-id`、`RequestContext.SessionID` 的启发式值、消息 hash 和
  客户端 tool history 都不能单独建立信任。
- **每轮重新授权**：每次请求都重新检查 catalog、schema、policy、capability
  和当前身份。历史 state 只提供候选身份和顺序，不能提供“已授权”结论。
- **有界状态**：只存工具 identity/name、definition fingerprint、顺序、called/
  pinned 标记、revision、TTL 和必要的 policy/capability fingerprint；不存 raw
  schema、prompt、参数、结果或完整消息。
- **确定性顺序**：仍然有效的工具按历史顺序保留；新增工具按明确的 append 规则
  加入；超额时按明确、可测试的 replacement 规则淘汰。
- **安全失效和回退**：fingerprint 不匹配、TTL 过期、状态损坏、身份不可信、
  store 不可用或 CAS 冲突时，回到普通选择；不能因 sticky 失败扩大请求失败面。
- **并发有语义**：共享 store 使用 CAS/version 或按 session 串行化；不能让两个
  并发 turn 无界覆盖或把未授权工具带回结果。
- **内容最小化观测**：receipt/metric 只记录 reuse、growth、replacement、
  invalidation、fallback 及原因类别，不记录 prompt、schema、参数或结果。
- **provider 前缀**：保留工具定义的顺序和可复用序列化前缀；工具集合或顺序变更
  时必须正确递增 `request.Generation`，避免 codec replay 旧 body。

### 明确不做

- 不执行工具，不缓存授权决定，不保存无界 catalog。
- 不把 sticky state 当作 capability/execution contract（该边界属于 issue #2361）。
- 不承诺 provider 一定命中 prompt cache；只验证 Router 没有无谓破坏前缀。
- 不在 foundation API 未确认前直接把 `RequestContext.SessionID` 接成 sticky key。

### 依赖状态审计（2026-09-15）

上游 `vllm-project/main` 仍未合并 #3392 的完整 runtime；当前工作分支已经具备
一个等价的窄 foundation seam，并在 runtime 中启用 opt-in sticky selection。当前
分支已有：

- `pkg/sessiontools` 的 identity-only bounded state、`Store`/CAS、TTL、quota 和
  local `MemoryStore`；
- 可选 standalone Redis CAS store、selection manager 和稳定 fingerprint helper；
- trusted identity 判定、每轮重新授权，以及 catalog/schema/policy/capability
  invalidation；
- `tool_selection.sticky` 与 `global.stores.tool_sessions` 的 disabled-by-default
  配置契约。

这些是当前工作分支的实现事实，不是已经合并到上游 `main` 的公共 API。不能把
客户端 `RequestContext.SessionID` 或消息推导值单独当作可信 sticky key。

该 PR 的公开 review 还暴露了实现必须吸收的安全教训：过期 key 的清理需要防止
ABA race；foundation 阶段在 runtime 尚未接入时必须拒绝 `sticky.enabled: true`，
避免配置成功但请求路径静默 no-op；JSON fingerprint 不能把大整数解码成
`float64`，应保留精确数字（例如 `Decoder.UseNumber`）并有回归测试。实现时先
重新确认合并后的 API、测试和 reviewer 决策，不要从候选分支复制未审查代码。

本地 fork 的 `origin/main` 只代表 fork 状态；本次核验时本地基线相对上游
`vllm-project/main` 已落后 90 个提交。开始业务实现前应先按团队约定 fetch、
审查差异并重新记录基线。

## 依赖和实现顺序

父需求是 #2973；#3392 负责 bounded state、storage、fingerprint 和 trusted
identity foundation。上游 `main` 尚未合并完整 runtime，但当前工作分支已经有
可供后续审计使用的窄 foundation 和请求管线接入。按 [PL-0042](plans/pl-0042-sticky-tool-selection.md)
的状态推进：

| 阶段 | 当前状态 | 审计结论 |
| --- | --- | --- |
| STICKY-00 | 已完成 | 已记录基线、依赖、工作树和安全边界 |
| STICKY-01 | 已完成 | foundation、trusted identity、bounded state、store、fingerprint 已具备 |
| STICKY-02 | 已完成 | deterministic reuse、called pin、growth、replacement 有窄测试 |
| STICKY-03 | 已完成 | runtime 接入覆盖主要 mode/plugin 路径，默认关闭 |
| STICKY-04 | 已完成 | 每轮授权、fingerprint invalidation、corrupt/expired/untrusted/store fallback 已覆盖 |
| STICKY-05 | 已完成 | local/shared persistence、CAS、expiry、并发 contract 已覆盖 |
| STICKY-06 | 部分完成 | 三种 provider codec retained-prefix 测试完成；prompt-cache/stateless baseline 待补 |
| STICKY-07 | 部分完成 | E2E 已注册并可编译；live TTL/restart/unavailable-store recovery 待执行 |
| STICKY-08 | 未完成 | repository gates、最终报告和环境限制收敛待完成 |

实现顺序仍遵循“先正常 request-time selection，再做有界历史合并，最后编码”的
边界；剩余工作只补验证证据，不得绕过每轮授权或把 state 扩展成原始 schema、
prompt、参数和结果缓存。

每一步都先做最小可验证改动；不要为了“顺手”重排 import、格式化无关文件或扩大
热点模块职责。

### Runtime 接入边界

运行时合并应放在正常 request-time selection 和 allow/block policy 过滤之后、
`commitToolSelection` 之前。这样历史 state 只能影响最终的有序候选集合，不能
绕过现有授权或 mode 分支：

- `mode=none` 先清空工具，sticky 不得把工具恢复回来；
- `filtered`、`passthrough`、显式 non-auto、外部 gateway 和 looper 路径默认
  保持现有 no-op/短路语义，除非契约明确允许参与；
- semantic retrieval 与 `tool_selection` add/filter 都必须经过同一套重授权和
  bounded merge；没有 decision 时保持原有短路行为；
- store error、身份不可信或当前工具未授权时只回退本轮 stateless 结果，不写入
  中间状态；所有清理和选择结束后才允许提交下一版 state。

纯 reuse 且 retained definitions/order 未变化时不要无条件递增
`request.Generation`，否则 provider codec 可能失去 `Envelope.CanReplay` 的
字节稳定性；只有工具新增、删除、顺序或定义内容真正变化时才按现有 mutation
规则递增。

## 代码风格与最小改动清单

- 先读目标目录最近的 `AGENTS.md` 和相邻实现，再决定新增文件还是扩展 helper。
- 沿用现有 Go 的缩进、空行、注释句式、错误处理和命名；注释解释约束/原因，
  不重复代码表面行为。
- 优先新增单一职责的小文件（state、identity、merge、store 各自窄化），让
  `processor_req_body*.go` 只保留编排。
- 不把 raw request、schema、prompt、参数、结果写入 state 或日志。
- 每次改变 `request.Tools` 或其顺序时检查 `Generation` 与 codec replay 条件。
- 配置默认值必须保持 disabled；新增配置同时更新 canonical/reference config 和
  解析/验证测试。

## 本地启动和验证

仓库规定的 CPU 本地路径是：

```bash
make vllm-sr-dev
vllm-sr serve --image-pull-policy never
```

也可以使用 harness 包装：

```bash
make agent-serve-local ENV=cpu
```

默认拓扑通常包含 Dashboard `8700`、Router 管理 API `8080` 和 routed inference
listener `8899`；`serve` 负责路由栈，不会自动启动物理 vLLM 后端。AMD 使用
`make vllm-sr-dev VLLM_SR_PLATFORM=amd` 后加 `--platform amd`；NVIDIA 流程见
[nvidia-local.md](nvidia-local.md)。

本次勘察环境的限制必须如实记录：Docker/OrbStack socket 不可用，无法声称服务
启动成功；`go` 不在 `PATH`，但本地 toolchain 可用路径为
`/private/tmp/semantic-router-go.3tZHpZ/go/bin/go`，最新 targeted rerun 因缺失
Go modules 和 DNS 无法访问 `proxy.golang.org`；Python `pytest` 未安装；外部
GitHub/raw 查询受 DNS 限制。`pkg/protocolcodec` 的缓存 targeted tests 已通过，
sticky E2E package 已编译，但 live Kubernetes/AI Gateway E2E、Redis recovery 和
provider cache-usage baseline 尚未完成。它们是环境限制，不是功能通过证据。可运行
的 harness 检查仍应优先执行 `make agent-validate`、`git diff --check` 和相应
`make agent-report`。

## 分阶段测试矩阵

| 阶段 | 最小测试 | 需要证明什么 |
| --- | --- | --- |
| Foundation | 新窄 package unit tests | schema 有界、TTL、identity fail-closed、fingerprint 稳定 |
| Merge | merge package `go test` | 顺序、去重、pin/append/replacement、disabled 等价 |
| Runtime | `req_filter_tools` targeted tests | 各 mode/plugin 不越权，授权每轮执行，fallback 安全 |
| Recovery | store/merge tests、可行时 `go test -race` | fingerprint/TTL/corrupt/store unavailable 的失效与恢复 |
| Codec | 三种 provider codec tests | retained prefix 顺序/字节稳定，Generation/replay 正确 |
| E2E | `make build-e2e` 与受影响 profile | 多轮 reuse、growth、并发、重启和回退 |
| Repository | `make agent-lint`、`make agent-ci-gate`、必要时 `make test-semantic-router` | 代码、契约和构建门禁 |

如果新增配置，还要补跑 `src/vllm-sr/tests/test_plugin_tool_selection.py`、
`test_plugin_parsing_advanced_filtering.py` 及相关 config contract tests。

## Git 备份与 push 规则

每个阶段都按下面顺序操作：

```bash
git status --short --branch
# 只修改本阶段的最窄文件
<最小 targeted test>
git diff --check
git diff --stat
git status --short
git add <本阶段文件>
git commit -s -m "<有原因的阶段性提交>"
git log -1 --format=fuller
```

阶段提交是本地可恢复备份，中间版本不 push。只有所有适用 gate/E2E 通过、工作树
干净、远端目标再次确认后，才 push 最新一版。若需要让远端只保留一个最终版本，
先保留本地备份分支，再在得到明确同意后 squash；不要擅自 reset、覆盖或删除用户
已有提交。

## 测试报告模板

每次阶段交付都记录以下字段，最终报告汇总所有阶段：

```text
Stage / commit:
Changed files:
Commands:
Result:
Environment-only limitation (if any):
Known uncovered cases:
Push status:
```

当前审计阶段的证据：`git diff --check` 和对应 `make agent-report ENV=cpu ...`
已通过；`pkg/protocolcodec` targeted Go tests 通过，sticky E2E package 可编译。
Go 其他 targeted rerun、pytest、容器启动、live E2E、provider cache baseline 和
完整 repository gates 分别受上述环境限制或尚未执行。业务 runtime 已实现，但
不能把 STICKY-06～08 或 Sticky Tool Selection 全部宣布为完成。

## 继续阅读

- [PL-0042 执行计划](plans/pl-0042-sticky-tool-selection.md)
- [Repo Map](repo-map.md)
- [Environments](environments.md)
- [Change Surfaces](change-surfaces.md)
- [Testing Strategy](testing-strategy.md)
- [Feature Complete Checklist](feature-complete-checklist.md)
- `src/semantic-router/pkg/extproc/req_filter_tools.go`
- `src/semantic-router/pkg/extproc/req_tool_selection_plugin.go`
- `src/semantic-router/pkg/llmprotocol/types.go`
- `src/semantic-router/pkg/protocolcodec/`

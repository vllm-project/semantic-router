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

## 依赖和实现顺序

父需求是 #2973；#3392 负责 bounded state、storage、fingerprint 和 trusted
identity foundation。当前 `main` 审计显示这些独立 seam 尚未可直接使用，#3392
仍需确认真实 API/合并状态。因此先做设计和窄 contract，不要凭候选分支 API 写运行时代码。

建议按现有执行计划 [PL-0042](plans/pl-0042-sticky-tool-selection.md) 的
`STICKY-00` 到 `STICKY-08` 顺序推进：

1. **Foundation**：冻结 state schema、可信身份解析、TTL/cardinality、local/shared
   store seam 和稳定 fingerprint helper。
2. **纯逻辑 merge**：输入当前 request-time selection 与历史 state，输出有序且有界
   的新集合和 receipt；先写纯函数单测。
3. **runtime 接入**：只在 `req_filter_tools*.go` 邻近 seam 接入，覆盖 `none`、
   `filtered`、`passthrough`、plugin add/filter 和显式 model。
4. **失效/恢复**：每轮重授权；catalog/schema/policy/capability/fingerprint、
   expired/corrupt/untrusted/unavailable 均回退 stateless。
5. **持久化/并发**：加入 revision/CAS 或 session serialization，验证 reload、
   restart、race 和 store 缺失。
6. **provider-prefix**：对 OpenAI Chat、Responses、Anthropic 验证 retained 定义
   的顺序和字节稳定性，并与 stateless baseline 对比。
7. **维护 E2E**：覆盖 reuse、growth、called pin、replacement、expiry、invalidation、
   concurrency、restart 和 unavailable-store fallback。

每一步都先做最小可验证改动；不要为了“顺手”重排 import、格式化无关文件或扩大
热点模块职责。

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
启动成功；`go` 不在 `PATH`，因此 Go targeted test 尚未执行；外部 GitHub/raw
查询受 DNS 限制。它们是环境限制，不是功能通过证据。可运行的 harness 检查仍应
优先执行 `make agent-validate`、`git diff --check` 和相应 `make agent-report`。

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

当前文档阶段的证据：`make agent-validate`、`git diff --check` 和对应
`make agent-report ENV=cpu ...` 已通过；Go、容器启动和外部网络检查分别受上述
环境限制。业务代码尚未修改，故不能把 sticky runtime 功能宣布为完成。

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

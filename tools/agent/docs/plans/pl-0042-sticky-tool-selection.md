# PL-0042: Trusted-session sticky tool selection

## Goal

在可信的 session 内维护一个有界、可失效、确定性排列的 selected-tool
集合。每次请求仍然重新做授权和能力检查；只有当前请求仍允许的工具才可以
从历史集合复用。保留工具的 provider-visible 顺序和序列化前缀，新工具按明确
规则追加或替换。

本文同时是本次工作的学习笔记和可恢复执行计划：它记录仓库目前的请求链路、
需求边界、依赖缺口、分阶段实现方式、测试矩阵，以及每一步的本地 Git 备份规则。

## Current baseline

- 工作仓库：`Lcollection/semantic-router`，远端 `origin` 指向
  `git@github.com:Lcollection/semantic-router.git`。
- 上游项目：`vllm-project/semantic-router`；根目录 `OWNER` 是项目级回退
  maintainer，`src/semantic-router/OWNER` 是 Router runtime owners。
- 基线提交：`a6b932a5`（2026-09-08 克隆时的 `main`，当前与 `origin/main`
  对齐）。执行本计划时先确认 `git status --short --branch` 为 clean。
- Router 是 Envoy ExtProc gRPC 请求路由层：它将公开协议解码为中立请求，
  提取 signals，执行 decision 和 model algorithm，运行 request/response
  plugins，最后由 provider codec 编码并转发。
- 当前工具选择是每请求独立完成的：
  `handleToolSelectionForRequest` → `handleToolSelection` → early mode
  (`none`/`filtered`/`passthrough`) → semantic retrieval 或
  `tool_selection` plugin → `applySelectedTools` → `commitToolSelection`。
  尚不存在 session-scoped selected-tool state。
- 当前本地栈曾验证过 Router `localhost:8080`、Envoy `localhost:8899`、
  Dashboard `localhost:8700` 等入口；`/ready` 可能因 embedding/classifier
  模型下载或 Hugging Face 限流保持 503。真实生成请求还需要配置中的 vLLM
  后端可达。这些是环境事实，不是 sticky-selection 的通过条件。

### Repository ownership

- 当前工作远端是 `Lcollection/semantic-router`，因此本地 fork 的所属账号是
  `Lcollection`。
- 上游项目是 `vllm-project/semantic-router`。仓库级回退 owners（根目录
  `OWNER`）为 `@rootfs`、`@Xunzhuo`、`@WUKUNTAI-0211`、`@AayushSaini101`。
- Router runtime 的目录 owners（`src/semantic-router/OWNER`）为
  `@FAUST-BENCHOU`、`@shraderdm`、`@drivebyer`、`@ramkrishs`、
  `@WUKUNTAI-0211`、`@AayushSaini101`、`@siloteemu`、`@theohsiung`、
  `@wilsonwu`、`@Peterren`。具体变更应以目标目录的 `OWNER` 和 CODEOWNERS
  规则为准。

## Project learning map

### Request lifecycle

```text
Envoy headers
  -> captureRequestHeaders (ctx.Headers, RequestID, auth headers)
  -> prepareProtocolRequest (wire -> llmprotocol.Request)
  -> populateSessionTransitionFields
  -> signal snapshot and decision evaluation
  -> memory/compression and provider dispatch preparation
  -> tool selection on the neutral request
  -> finalizeProviderDispatchResponse
  -> provider codec (OpenAI Chat / Responses / Anthropic)
```

关键文件：

- `src/semantic-router/pkg/extproc/processor_req_header.go`：捕获请求头和
  transport identity。
- `src/semantic-router/pkg/extproc/processor_protocol_contract.go`：协议解码、
  `SemanticRequest` 建立，以及最终 provider 编码入口。
- `src/semantic-router/pkg/extproc/processor_req_body_routing.go`：路由和
  dispatch 编排；此类热点文件不应吸收新的状态逻辑。
- `src/semantic-router/pkg/extproc/req_filter_tools.go`、
  `req_tool_selection_plugin.go`：工具模式、检索和中立请求变更。
- `src/semantic-router/pkg/tools/tools.go`、`relevance.go`：工具目录和检索。
- `src/semantic-router/pkg/config/tools_plugin.go`、
  `tool_selection_plugin.go`：全局/decision 级工具配置。
- `src/semantic-router/pkg/llmprotocol/types.go`：中立 `Tool`、`Request` 和
  `TrustedMetadata` 定义。
- `src/semantic-router/pkg/protocolcodec/`：只在最后一步把中立工具切片编码为
  provider wire JSON；编码器按 slice 顺序输出。

## Scope

本计划覆盖 trusted-session selected-tool state 的 foundation、纯逻辑合并、
请求管线接入、失效/恢复、并发/存储、provider-prefix 验证和维护 E2E。每一阶段
都保持 sticky 功能显式 opt-in；正常 stateless selection、工具执行、授权缓存和
provider prompt-cache 保证均不在范围内。

### Existing session and identity surfaces

`RequestContext.SessionID` 会按不同协议从显式 conversation、`x-session-id`、
Anthropic 信号、消息指纹或 request ID 推导；这些来源中有客户端可控或启发式
值，不能直接作为 sticky state 的可信 key。`llmprotocol.TrustedMetadata` 已
预留 `NamespaceID`、`ActorID`、`SubjectID`、`SessionID` 等字段，但当前
`prepareProtocolRequest` 只填 `SourceFormat` 和 `CorrelationID`。

`sessiontelemetry.RouterSessionStateStore`、Redis store slot、TTL 和 generation
生命周期可作为存储实现参考，但现有 `RouterSessionSnapshot` 是模型/成本遥测
结构，不应把 selected tools 塞进去。工具状态应有独立的窄 seam 和 key prefix。

`cache.FingerprintValue` / `cache.CombineFingerprints` 可参考 SHA-256 实现，
但 response-cache identity 与工具目录/策略 fingerprint 的语义不同，不能直接
复用 response-cache entry。

## Requirement interpretation

这是“有界的 session 状态 + 每轮重新授权”的功能，不是把授权结果缓存起来，也
不是无界地保存完整工具目录。必须满足：

1. 默认关闭；关闭时不初始化 state store，并保持现有 stateless 行为。
2. 只接受经过 transport/authz 证明的 identity；未认证或仅由客户端/消息推导的
   session ID 不得打开跨请求复用。
3. 状态只保存工具 identity、顺序、是否曾调用和有界 metadata/fingerprint；不
   保存 prompt、参数、结果或完整原始 schema。
4. 每个请求重新检查当前 catalog、tool schema、decision policy、capability
   和可信 identity；历史授权结论不能复用。
5. catalog、schema、policy、capability 或 fingerprint 不匹配、TTL 过期、状态
   损坏、identity 不可信、store 不可用时，安全回退到普通 request-time selection。
6. retained tools 保持原顺序；新工具只在明确的 append/growth 规则下加入；总数、
   session 数和状态大小均受配额约束。
7. 状态更新在并发请求下有明确的版本/CAS 或串行化语义；reload、restart 和
   store unavailable 都不能导致越权或请求失败扩大化。
8. receipts/metrics 只记录 content-minimized 的 reuse、growth、replacement、
   invalidation 和 fallback 原因。
9. provider-visible 的 retained 定义应保持 byte-stable；逻辑稳定不等于保证
   上游一定命中 prompt cache。

## Dependency status

当前 `main` 中未找到 #3392 的独立 bounded state、storage、fingerprint 或
trusted-identity seam。可复用的只是上述 session telemetry 参考实现和已有
`TrustedMetadata` 类型。因此正式 runtime work 之前需要先落地一个最小 foundation
contract，或把 #3392 的已审查提交合并到本分支；不能把现有 `RequestContext.SessionID`
或 `sessiontelemetry.RouterSessionSnapshot` 当成替代品。

## Proposed state contract

实现阶段应先冻结下列语义，再写运行时接入。字段名称可以在 #3392 合并后按其
实际 API 调整，但不能扩大保存内容：

```text
ToolSelectionState
  trusted_identity_key   // server-derived, namespace-scoped
  version                 // monotonic revision for concurrency control
  created_at / expires_at / last_seen
  catalog_fingerprint
  policy_fingerprint
  capability_fingerprint
  selected_tools[]
    tool_id               // stable identity/name, never raw schema
    definition_fingerprint
    ordinal               // retained order
    called                // bounded fact used only for explicit pin policy
```

建议的接口边界：

- `Store.Load/Save/Delete`：带超时、TTL 和版本条件；memory store 是测试/单进程
  实现，shared store 是可选实现。
- `AuthorizeAndMerge`（纯逻辑）：输入当前 request-time selection、当前 catalog、
  policy/capability/fingerprints、历史 state 和时间；输出最终有序工具、下一版
  state 与 receipt。
- `TrustedIdentityResolver`：只消费 auth middleware 证明的字段，返回
  `TrustedMetadata` 或明确的“不可信”结果。

## Exit Criteria

- [ ] #3392 foundation 或等价的、经过审查的 trusted identity/state seam 已可用。
- [ ] 重复可信 session turn 产生有界且确定性的工具顺序。
- [ ] 每轮重新授权，catalog/schema/policy/capability 变化会安全失效。
- [ ] expiry、restart、并发和 store 不可用都能安全回退到 stateless selection。
- [ ] provider-visible retained prefix 和维护 E2E 证据已完成。
- [ ] 适用 repository gates 通过，最终测试报告记录提交和已知限制。

## Task List

- [ ] `STICKY-00` 记录基线、确认 #3392 API/依赖和当前工作树；不改业务代码。
- [ ] `STICKY-01` 落地/接入 foundation：trusted identity、独立 state schema、
      TTL/cardinality 限制、local/shared store seam、fingerprint helper。
- [ ] `STICKY-02` 实现纯逻辑 deterministic merge：reuse、去重、called-tool pin、
      append/growth、bounded replacement、disabled 等价路径。
- [ ] `STICKY-03` 在 `req_filter_tools*.go` 邻近 seam 接入；覆盖 `none`、
      `filtered`、`passthrough`、`tool_selection add/filter`、fallback 和显式
      model；每轮仍走授权。
- [ ] `STICKY-04` 增加 catalog/schema/policy/capability/fingerprint invalidation、
      state corrupt/expired/untrusted/store unavailable 的安全恢复与 receipts。
- [ ] `STICKY-05` 增加 local/shared persistence、版本/CAS 或 session serialization、
      reload/restart/expiry/concurrency/race 覆盖。
- [ ] `STICKY-06` 增加三种 provider codec 的 retained-prefix 字节稳定性验证，
      并与 stateless prompt-cache baseline 对比。
- [ ] `STICKY-07` 增加维护 E2E：reuse、growth、called pin、replacement、expiry、
      invalidation、concurrency、restart、unavailable-store fallback。
- [ ] `STICKY-08` 完成适用 gates、更新本文件的测试报告和已知限制；确认最终提交
      是唯一允许 push 的版本。

## Test matrix

| 阶段 | 最小验证 | 重点断言 |
| --- | --- | --- |
| Foundation | foundation package unit tests；`make agent-report ENV=cpu CHANGED_FILES="..."` | schema 有界、TTL、identity fail-closed、fingerprint 稳定 |
| Pure merge | `go test`（新增窄 package） | 顺序稳定、去重、pin/append/replacement 有界、disabled 等价 |
| Runtime seam | `req_filter_tools` targeted tests；`req_filter_tools_retriever_e2e_test.go` | 四种工具路径不越权，授权每轮执行，fallback 安全 |
| Invalidation/recovery | store/merge tests + `go test -race`（可行时） | catalog/schema/policy/capability/TTL/corrupt/store unavailable |
| Provider prefix | `processor_req_body_routing_test.go` 扩展 | OpenAI Chat/Responses/Anthropic retained 定义顺序和字节前缀稳定 |
| Repository gate | `make test-semantic-router`、`make agent-lint CHANGED_FILES="..."`、`make agent-ci-gate CHANGED_FILES="..."` | core build/lint/contract |
| Maintained E2E | `make build-e2e`；`make e2e-test E2E_TESTS=tool-selection` | 多轮 reuse/growth/invalidation/recovery |

新增配置字段时还必须运行 `src/vllm-sr/tests/test_plugin_tool_selection.py`、
`test_plugin_parsing_advanced_filtering.py` 和相关 canonical/reference config
contract tests。若修改 `sessiontelemetry/*memory*`，补跑 `make memory-test-integration`。

## Operating Rules

每一个阶段都按相同顺序执行：

1. 先确认基线 `git status --short --branch`、记录当前 commit，不覆盖用户已有改动。
2. 只在最窄的责任文件中用 `apply_patch` 修改；保持现有缩进、空行、命名和注释
   风格，不做无关格式化或重排。
3. 先跑该阶段最小 targeted test；失败时修复并重复运行，不把首个失败直接交付。
4. 检查 `git diff --check`、`git diff`、`git status`，确认改动范围与任务一致。
5. 阶段完成后执行本地 `git add` + `git commit -s` 作为可恢复备份，记录 hash、
   文件、命令和结果；中间提交不 push。
6. 只有所有适用 gate 和 E2E 完成、工作树干净、远端目标再次确认后，才允许 push
   最后一版提交。若没有明确的最终 push 指令，保留本地提交并报告 hash。

## Test-report template

每个阶段在交付消息和本文件对应勾选项旁记录：

```text
Stage / commit:
Changed files:
Commands:
Result:
Environment-only limitation (if any):
Known uncovered cases:
Push status:
```

## Next Action

先完成 `STICKY-00`：以当前 `main` 为基线，确认 #3392 的真实接口是否可以取得。
在依赖未落地前只提交设计/文档或 foundation contract，不把启发式
`RequestContext.SessionID` 直接接到工具状态存储上。

## Related Docs

- `tools/agent/docs/README.md`
- `tools/agent/docs/repo-map.md`
- `tools/agent/docs/change-surfaces.md`
- `tools/agent/docs/architecture/state-taxonomy-and-inventory.md`
- `tools/agent/docs/testing-strategy.md`
- `src/semantic-router/pkg/extproc/AGENTS.md`
- `src/semantic-router/pkg/extproc/req_filter_tools.go`
- `src/semantic-router/pkg/extproc/processor_req_body_routing.go`
- `src/semantic-router/pkg/extproc/processor_protocol_contract.go`
- `src/semantic-router/pkg/llmprotocol/types.go`
- `src/semantic-router/pkg/sessiontelemetry/router_memory_shared_store.go`
- `src/semantic-router/pkg/cache/request_identity.go`
- [Epic #2973](https://github.com/vllm-project/semantic-router/issues/2973)
- [Foundation #3392](https://github.com/vllm-project/semantic-router/issues/3392)

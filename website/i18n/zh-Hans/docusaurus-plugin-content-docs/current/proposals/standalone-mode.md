---
title: Standalone 模式
description: Router 自己提供 OpenAI 兼容接口并在进程内执行路由流水线，Envoy 变为可选。一个与传输无关的路由核心，配 ext_proc 和原生 HTTP 两个适配器；借鉴 Envoy 的上游层，带 timeout、retry、fallback；进程内请求图；以及支持热更新和回滚的版本化配置快照。
created: 2026-10-06
status: Implemented
translation:
  source_commit: "9156d5bc1ed9edff626b95a2b8260a77cb1712c5"
  source_file: "docs/proposals/standalone-mode.md"
  outdated: false
---

> **状态：** 已在 [#4628](https://github.com/vllm-project/semantic-router/pull/4628) 中实现 - **创建日期：** 2026-10-06 -
> **跟踪 issue：** [#4623](https://github.com/vllm-project/semantic-router/issues/4623)

## 概要

新增 **standalone 模式**：Router 自己作为 OpenAI 兼容的入口和反向代理，路由流水线在进程内执行，不经过 Envoy 和
ext_proc。`vllm-sr serve` 默认使用它。Envoy 仍以显式模式 `--gateway extproc` 提供：`vllm-sr` 在 Router 前面起
Envoy，Router 和今天一样提供 ext_proc。Kubernetes 也默认 standalone 模式；要接入已有的 Envoy 类网关，仍用 extproc。

两种模式共用**同一个路由核心**。核心与传输无关，ext_proc 和 HTTP 是它的两个薄适配器。在核心之外，standalone
gateway 增加：借鉴 Envoy cluster/endpoint 的**上游层**；一等公民的 **timeout、retry、fallback**；在 Router 内部
执行 looper 算法、不再绕回 Envoy 的**请求图执行器**；以及基于不可变、版本化快照的**配置体系**，支持预热、原子
切换、drain、历史和回滚。

本提案取代 [独立 HTTP Gateway](./standalone-http-gateway)。那份设计把网关做成单独的实验性二进制；本设计让 Router
二进制同时承担两种角色，只保留一份配置文档，并把 standalone 设为默认模式。它对共享引擎、统一响应流水线和
looper 执行器的分析继续适用。

## 已确认的决定

1. Router 新增 standalone 模式：接收 OpenAI 兼容流量，在进程内执行路由流水线，并自己代理到模型后端。
2. `--gateway standalone|extproc` 在所有 target 上选择模式，默认 `standalone`。docker 上 standalone 只起 Router 容器，
   模型 runtime 在容器内，由 Router 自己提供 listener；`extproc` 就是今天的栈：`vllm-sr` 在 Router 前面起一个 Envoy
   容器，Router 提供 ext_proc。
3. Kubernetes 也默认 standalone。Helm chart 里 Router pod 提供配置中的 listener，Service 对外暴露它们，readiness 用
   `/ready`。operator 也默认 standalone，它的 `spec.gateway` 集成会选择 extproc。接入 Envoy Gateway、AI Gateway、
   Istio、KServe 时仍可选 `extproc`（50051 端口上的 ext_proc Service 和那条五个 header 的规则）。
4. 一切都跑在容器里：`--target docker`（默认）或 `--target kubernetes`（v0.4.0 已发布的 `k8s` 保留一个版本作为隐藏
   别名，使用时给出警告），没有裸机 target。`--platform cpu|rocm|cuda` 对两个 target 都生效：docker 上选择镜像和
   GPU 透传，kubernetes 上选择同一个镜像，并在生成的 Helm values 里加上 GPU 资源请求（`amd.com/gpu` 或
   `nvidia.com/gpu`）。engine 模式（`vllm-sr serve MODEL --engine`）在 docker target 上用同一个镜像在容器里跑模型 runtime，
   使用发布版本固定的 ROCm 或 CUDA 依赖；数值一致性仍须按硬件与 profile 验证。
5. PyPI 上只发布 `vllm-sr`。Router 二进制和模型 runtime（`vllm-srun`）只随一个镜像家族发布：`vllm-sr`（CPU，
   amd64 和 arm64）、`vllm-sr-rocm` 和 `vllm-sr-cuda`（amd64），覆盖 docker 和 kubernetes、两种模式以及 engine 模式。
   原来的 `extproc` 和 `extproc-rocm` 镜像在一个版本内作为同一 digest 的别名 tag。上游 Envoy 只用于 docker 上的
   `extproc`。
6. Router 与模型 runtime 之间仍是 Unix 域套接字上的 HTTP/JSON；测出开销之后再考虑二进制快路径。在 CPU 上测得
   传输约占一次 runtime 调用的 2%，所以不做快路径（见[结果](#results)）。
7. 首发必须支持 timeout、retry、fallback。
8. looper 请求图完全在 Router 内部闭环，不再绕回 Envoy。
9. 配置体系模块化、版本化，支持热更新和回滚，借鉴 Envoy 配置设计的核心思想。

## 背景

standalone 实现前的请求链路：

```text
client -> Envoy -> ext_proc（gRPC，默认整包缓冲 body）-> Router（Go）-> 通过 Unix 套接字调用模型 runtime
       <- header 和 body 修改 <-
Envoy -> 按 x-selected-model 选中的后端 -> Envoy -> ext_proc（响应阶段）-> Router -> Envoy -> client
```

默认 Envoy 模板会处理请求头、请求体、响应头、响应体，因此每个请求要经历四条 ext_proc 消息，并对请求体和响应体
各做一次整包缓冲。

looper 请求图（fusion、quorum、置信度升级、ReMoM、ratings、grounding、workflows）的每一个中间调用，都带着内部
header `x-vsr-looper-request`、`x-vsr-looper-secret`、`x-vsr-looper-decision`、`x-vsr-looper-iteration` 打回本地
Envoy listener，再进入 ext_proc，被识别为内部请求后才派发到后端：每一跳都是 Router -> Envoy -> Router -> Envoy ->
后端。

这条链路的问题：

- 部署和排障都要同时理解两套配置、两种心智模型：Envoy 的和 Router 的。
- looper 的回环只是因为 ext_proc 无法发起经过 Envoy 路由的上游调用；它拉长了每一跳，让语义变绕，还需要内部密钥
  header 防伪造。
- 每个本地栈都要在 Router 前面起一个由第二套模板渲染的 Envoy 容器，即使只是一台机器、一个用户。

## 模式

网关传输（`--gateway`）与服务能力（`--engine`）是独立选择。默认 Router 模式；传入 `MODEL` 只改变判断模型，不会关闭路由。Engine 模式禁用 recipe 路由，仍保留前端和原生模型 API。

| 命令 | 运行什么 | 客户端流量入口 | 适用 |
| --- | --- | --- | --- |
| `vllm-sr serve`（= `--gateway standalone`，默认） | Router 容器，它托管的 runtime 进程在容器内 | Router 自己的 OpenAI 兼容端口 | 单机、开发、边缘、多数自托管 |
| `vllm-sr serve --gateway extproc` | `vllm-sr` 起的 Envoy 容器，在 Router 容器（ext_proc）前面 | Envoy | 需要限流、mTLS、复杂路由匹配等 Envoy 能力 |
| `vllm-sr serve MODEL --engine` | 持久前端与模型 worker（Engine 模式） | 前端的 System One API | 在自己的代码里调用模型 |

两种 gateway 模式共用一个路由核心：standalone 使用 HTTP 适配器，extproc 模式（下文称 Envoy 模式）使用 ext_proc
适配器。Kubernetes 上 `--target kubernetes` 以任一模式安装 Helm chart，你自己运行的 Envoy 类网关（Envoy
Gateway、AI Gateway、Istio、KServe）以 extproc 模式接入。

正交参数：

- `--target docker|kubernetes`：默认 `docker`。
- `--platform cpu|rocm|cuda`：两个 target 上都选择镜像；docker 上透传 GPU，kubernetes 上加 GPU 资源请求。
- `--container-runtime docker|podman`（原 `--runtime`，保留一个版本作为隐藏别名，使用时给出警告）。
- `serve` 的每个参数属于一个分组，分组写明它适用于哪里：docker、kubernetes、engine 模式或其中几种。
  `--platform`、`--image`、`--log-level` 三者通用；`--minimal`、`--readonly` 两个 target 通用；
  `--image-pull-policy`、`--container-runtime` 用于 docker 和 engine 模式。`--help` 按分组列出，在分组不适用
  的地方使用其参数会报错。
- `--minimal`：不起 dashboard 和观测栈。

## 架构

### 普通请求

```text
Envoy 模式（今天）
client --> Envoy --gRPC ext_proc（缓冲）--> Router --UDS--> runtime
                 <-- header 和 body 修改 --
           Envoy --> 后端 LLM --> Envoy --ext_proc--> Router --> Envoy --> client

Standalone 模式
client --> Router [listener -> 入口安全与鉴权 -> 路由核心 -> 上游层] --> 后端 LLM
                                     |                                  |
                                     +--UDS--> runtime（信号、embedding）  +-- 流式 --> 响应阶段 --> client
```

Standalone 模式少了一跳代理、每个请求至少四条 ext_proc 消息，以及它们所需的整包缓冲和重新序列化；配置从“Envoy 配置
加 Router 配置”变成一份文档。

### 请求图（looper）

```text
今天
client -> Envoy -> ext_proc（Router 开始执行 looper 算法）
                    |- hop 1: Router -HTTP-> Envoy listener -> ext_proc（识别内部请求）-> Envoy -> 后端 A
                    |- hop 2: Router -HTTP-> Envoy listener -> ext_proc -> Envoy -> 后端 B
                    `- 聚合 -> immediate response -> Envoy -> client

Standalone 模式（请求图在 Router 内部闭环）
client -> Router: Plan -> 图执行器
                    |- call A --上游层--> 后端 A   （每跳在进程内跑插件，带 timeout、retry、fallback）
                    |- call B --上游层--> 后端 B   （并行 fan-out）
                    |- aggregate / branch / loop
                    `- respond（最后一跳可流式）--> client
```

envoy 和 extproc 模式复用同一个执行器：中间跳直接调用上游层，不再回环；最后一跳要么作为普通路由请求交给 Envoy
（保留流式），要么作为 immediate response 返回。内部密钥 header 随回环一起退役。

### 配置管理

```text
今天：  config.yaml --fsnotify--> Router 重建 generation（原子切换）
        + vllm-sr CLI 渲染 Envoy 配置（另一套生命周期）

Standalone：配置源（文件 / HTTP API / CRD / 以后的控制面）
          --> compile   （规范文档 -> 类型化资源）
          --> validate  （schema + 交叉引用 + 能力检查）
          --> warm      （上游连接池、runtime 模型就绪、预计算）
          --> activate  （原子切换 snapshot）
          --> drain     （旧 snapshot 上的请求跑完）
          --> history   （保留最近 N 个版本，可回滚）
```

## 详细设计

### 路由核心

今天位于 `pkg/extproc` 的请求与响应处理，成为一个与传输无关的路由核心，放在新包 `pkg/routing` 的一个小契约之后。
ext_proc 服务和原生 HTTP 前端都是这个契约的适配器。

```go
// Processor opens one routing session per request; the routing core implements it.
type Processor interface {
    Open(ctx context.Context) (Session, error)
}

// Session processes one request's phases in order and answers each with an Effect:
// header and body mutations, a route-cache refresh, a streamed response body, or an
// immediate response.
type Session interface {
    RequestHeaders(header Header, endOfStream bool) (*Effect, error)
    RequestBody(body []byte, endOfStream bool) (*Effect, error)
    ResponseHeaders(header Header, endOfStream bool) (*Effect, error)
    ResponseBody(body []byte, endOfStream bool) (*Effect, error)
    Evidence() Evidence
    Close(err error)
}

// Engine is the transport-agnostic routing core shared by every gateway mode.
type Engine interface {
    // Plan resolves the entrypoint and recipe, evaluates signals, the decision and
    // request-side plugins, and returns what to do with the request.
    Plan(ctx context.Context, req *Request) (*Plan, error)
    // Respond runs response-side plugins on a buffered or streamed upstream response.
    Respond(ctx context.Context, plan *Plan, resp *UpstreamResponse) (*Response, error)
}

type Plan struct {
    Immediate *Response // block, cache hit, policy answer, completed Looper result
    Call      *Call     // one upstream call: the request as it leaves the Router, and its route
    Budget    Budget    // deadline, maximum hops, token or cost ceiling
    Evidence  Evidence  // signals, decision, selected candidates (for headers and traces)
}
```

- **一条流水线，一套语义。** 所有模式下请求和响应都经过相同的阶段：请求头、请求体、响应头、响应体。ext_proc 适配器把
  每个效果编码成 ext_proc 消息交给 Envoy 应用。`routing.NewEngine` 按本地 Envoy 模板的处理模式驱动会话（两个方向都
  发送 header；两个方向的 body 都缓冲，除非响应被切换为流式），并按 Envoy 自己的规则应用效果：先删除后设置；路由类
  header（`host`、`:authority`、`:method`、`:scheme`）和 `x-envoy-*` header 不改动；修改后的 body 要与 content-length
  一致；直接响应按 Envoy 生成 local reply 的方式渲染。因此两种模式下的上游请求和客户端响应完全一致。
- **调用与路由。** `Plan.Call` 携带经过 Router 全部修改之后的上游请求（method、path、header、body）和它的路由键：
  Envoy 路由表匹配的 `x-selected-model` 值；与 Envoy 一样，一旦某个效果清除了路由缓存，就取修改后的 header 中的值。
  路由键为空表示默认路由。请求图执行器会为多步请求图增加 `Plan.Program`。
- **每个请求一个生命周期。** 会话在整个请求期间固定一个 router generation（以后是一个配置快照），与 ext_proc 流
  完全相同；`Plan.Finish` 无论结果如何都只结束一次：replay 记录、in-flight 准入、会话遥测和 trace span，在两个适配器下
  走同一条收尾路径。
- **分阶段抽取。** 第一步引入契约，在现有流水线上实现 `routing.Processor`，并让 gRPC 流和会话共用每个阶段的回复逻辑。
  流水线内部暂时仍以 ext_proc 消息作为内部表示，在唯一一个严格的边界上解码，契约无法表达的内容一律失败而不是忽略。
  之后逐个阶段移到边界之后，每一步都由 parity 记录工具把关，保证 ext_proc 行为始终逐字节不变。
- **parity 记录工具。** `pkg/routing/parity` 对一组请求记录：每个阶段的输入和效果、应用修改之后的上游请求、路由、
  客户端响应，以及路由证据（决策、模型、recipe、信号）。归一化只改写易变的值（耗时、trace 上下文、时钟字段）和
  header 删除的顺序（Envoy 总是先执行全部删除再设置）。仓库里的语料以三种方式运行：经过 ext_proc gRPC 适配器并把
  确切消息固定为 golden；经过进程内会话；以及端到端地在线上经过两种 gateway 模式。三者必须一致。

### Standalone 前端

- **listener：** 地址、HTTP/1.1 与 h2c、空闲超时（默认值与本地 Envoy 模板一致）、请求体上限、连接数上限、可选的
  下游 TLS。
- **入口安全，即边缘的信任边界：** 客户端伪造的内部 header（`x-vsr-*` 内部 header、`x-authz-user-*`）既不参与路由
  决策，也不会到达后端。只有受信任代理才能设置的代理控制 header 也一样，例如 `x-envoy-max-retries`、
  `x-envoy-retry-on`、`x-envoy-upstream-rq-timeout-ms`。Router 会把客户端 header 转发给上游，而它后面基于 Envoy
  的组件（sidecar、模型服务前的 AI gateway）会听从受信任调用方给的这些 header，Router 正是这样的调用方。剥掉它们，
  Router 的可靠性策略才是唯一的重试和超时来源。名单沿用 Envoy 对外部请求剥除的那些 header，并保持这么窄；它保护的是
  边缘，不是在模拟 Envoy。
- **API key：** Bearer 或 `api-key`，校验后剥掉，客户端凭据不会到达 provider；失败返回 OpenAI 风格的 401 JSON。
- **接口：** `/v1/chat/completions`、`/v1/completions`、`/v1/responses`、`/v1/models`，以及 Router 今天提供的其他推理
  接口；其余路径按 Envoy 默认路由的语义转发到默认后端。
- **运维接口：** `/health`、`/ready`（路由核心和 runtime 都就绪才算就绪）、`/metrics`；SIGTERM 时优雅 drain。

### 与 Envoy 模式的差异

Standalone 模式复现 Envoy 和 ext_proc 的行为，并由 parity 套件把关。以下差异是有意为之：

- **探针。** standalone listener 自己应答 `/health` 和 `/ready`；Prometheus metrics 仍在 Router 的 metrics 端口。
- **身份 header。** 客户端发来的 `x-authz-*` 身份 header 在路由之前被丢弃，因为 standalone listener 前面没有鉴权组件。
  在 Envoy 后面时，ext_proc 看到的是前方网关转发的内容。
- **代理身份。** Standalone 模式不添加任何 `x-envoy-*` header（例如上游的 `x-envoy-expected-rq-timeout-ms` 和
  `x-envoy-original-host`、下游的 `x-envoy-upstream-service-time`），也不会把后端的 `server` header 替换成
  `server: envoy`。
- **DNS endpoint。** 以 DNS 名称给出的后端是一个 endpoint；Envoy 的 `STRICT_DNS` cluster 会为每个解析出的地址生成
  一个 host。
- **传输分帧。** content length 与 chunked 分帧遵循 Go 的 HTTP 服务器：对于缓冲的 body，Go 可能用长度分帧，而
  Envoy 可能使用 chunked。
- **跨模型 fallback。** 每个候选都由上游层发送，因此候选享有其 provider model 的重试、离群剔除和 TLS。Envoy 模式的
  响应阶段 fallback 则直接用 HTTP 调用候选。
  - 无法连通的候选以 `503` 或 `504` 本地应答交给策略，而不是传输错误。只有策略不含这两个状态码时，结果才会不同。
  - standalone 模式也会在 Envoy 本地应答（如连接被拒、reset、超时）上 fallback。在 Envoy 后面时，ext_proc 遇到 Envoy 自己
    的本地应答就结束处理，因此客户端收到该应答，不会尝试任何候选。两种模式下，送到客户端的本地应答都不经过响应阶段。
  - standalone 候选的请求与主请求的构造方式相同，都从客户端请求出发；Envoy 模式的候选只带 provider 的 header。
  - standalone 模式在候选的成功响应 header 到达时确定使用它；Envoy 模式在翻译完缓冲的 body 之后才确定。
  - `per_attempt_timeout` 和 `total_timeout` 的剩余部分约束 standalone 候选直到其响应开始，因此不会截断已开始的流。

### 上游层

借鉴 Envoy 的 cluster 和 endpoint：

- **cluster** 是一个 provider model，它的 **endpoint** 是该模型的后端。
- **连接池：** HTTP/1.1 keep-alive 和 HTTP/2；上游 TLS 与 SNI、系统 CA。
- **负载均衡：** 带权重的 round robin，以及 least request（P2C）。
- **改写与注入：** host 改写（自动或固定值）、路径前缀改写、provider 需要的 header，以及从 secret 引用注入的凭据。
- **熔断：** 最大并发请求数、最大排队数。
- **离群剔除：** `consecutive_5xx`、`base_ejection_time`、`max_ejection_percent`，规范配置的
  `providers.models[].reliability` 里已有。
- **主动健康检查：** `health_check_path`、`health_check_interval`、`health_check_timeout`，规范配置里同样已有。
- **流式：** SSE 逐块 flush 并有背压；需要完整 body 的响应插件在旁路累积。

默认值复现今天本地 Envoy 模板的配置，因此在模式之间切换不会改变负载均衡、离群剔除或健康检查的行为。

### Timeout、Retry、Fallback

**Timeout** 有 cluster 级默认值，可被 entrypoint、recipe、决策或图节点覆盖：

- `connect`、`first_byte`（首 token）、`per_try`、`total`（整个请求的 deadline）、`idle`（流式空闲）。
- deadline 沿调用链传递：runtime 信号调用和图的每个节点共用同一个预算。

**Retry：**

- 触发条件 `retry_on` 沿用 Envoy 的取值：`connect-failure`、`refused-stream`、`reset`、`gateway-error`（502、503、504）、
  `5xx`，以及配合状态码列表的 `retriable-status-codes`（如 429，在上限内尊重 `Retry-After`）。与 Envoy 一样，单次尝试
  超时在 `5xx`、`gateway-error`、`reset` 条件下重试。
- 重试次数、带抖动的指数退避，以及重试预算（同时处于重试中的请求占比上限）。
- 重试时优先换一个 endpoint。
- **安全边界：** 一旦有任何响应字节写给了客户端，Router 就不再重试，也不再 fallback。

**决策级覆盖**先落地：`routing.decisions[].reliability` 沿用 provider 配置块的 timeout 和 retry 字段。两种模式都按
Envoy 合并逐请求 header 的方式合并覆盖：timeout 和重试次数替换 provider model 的值，重试条件和可重试状态码在其基础上
追加。

- Envoy 模式下，Router 的 gRPC 应答把覆盖写成 `x-envoy-upstream-rq-timeout-ms`、
  `x-envoy-upstream-rq-per-try-timeout-ms`、`x-envoy-max-retries`、`x-envoy-retry-on` 和
  `x-envoy-retriable-status-codes`。standalone 请求从不带这些 header。
- ext_proc 默认忽略对 `x-envoy-*` 的修改，因此仓库提供的每个 ext_proc filter 都用 `mutation_rules.allow_expression`
  只放行这五个 header。这条规则始终存在，因为 Router 热更新配置时并不重新生成 Envoy 配置。`allow_envoy: true` 会让
  Router 能改任何 `x-envoy-*` header，放开的范围远超这些字段所需。
- reliability 配置块是这些 header 的唯一写入者：加载配置时会把它们从决策的 `header_mutation` 插件中去掉，并给出警告。
- Envoy 没有针对单个请求的 idle 或首字节 timeout、退避或 `Retry-After` 上限的 header。设置了这些字段的决策在 standalone
  模式之外会被拒绝：启动时、热更新时，以及 CLI 生成 Envoy 配置时。

**Fallback** 是按序的候选链：

1. 同一 provider model 的其他 endpoint：由 cluster 负责，即优先换未试过 endpoint 的重试、离群剔除和健康检查；
2. 同一模型的其他 provider；
3. 其他模型：决策给出的排序候选；
4. 兜底：缓存响应或静态响应（可选）。

- 跨模型 fallback 可以配置在图节点、决策（`routing.decisions[].fallback`）、recipe（`routing.fallback`）或全局
  （`global.router.fallback`），并按这个顺序逐字段合并。cluster 本身没有跨模型 fallback。

- 触发条件按错误类别配置：超时、5xx、429、连接失败、上游内容拒绝。
- 预算限制 fallback 跳数，并与 total deadline 共享。
- 响应 header 标明实际服务的模型和 fallback 路径；metrics 和 trace span 记录每次尝试。
- 每个请求只有一个 fallback 权威：现有的跨模型策略（`global.router.fallback` 和 recipe 的 `routing.fallback`）仍是
  策略本身，standalone 模式下由上游层执行。

### 请求图执行器（looper v2）

looper 不再是“绕回 Envoy”的特殊流程，而是在 Router 内部执行的通用**请求图**。请求图在引擎的 `Plan` 里、请求体
阶段执行，两种模式都一样。它的结果要么是直接应答，要么是一次普通的路由调用，由前端（或 Envoy）发出，所以两个前端
都不用改。每一跳都是同一请求所固定的 generation 和 snapshot 上的一个路由会话，在进程内标记为 hop，而不是靠 header。

- **节点类型**（通过注册表扩展，插件可以新增节点类型）：
  - `call`：调用一个模型或候选集，自带 timeout、retry、fallback；
  - `parallel`：并行 fan-out，可设并发上限和“先到 k 个即可”；
  - `aggregate`：arbiter、quorum、fusion、ratings 等聚合策略；
  - `branch`：用决策规则引擎的谓词、信号和置信度选择分支；
  - `loop`：直到条件满足或达到最大轮数；
  - `transform`：上下文压缩、prompt 改写、工具结果注入；
  - `respond`：输出结果，最后一跳可流式。
- **执行语义：**
  - 每个请求一个执行上下文，统一 deadline、最大跳数、token 或成本上限；
  - 取消向下传播：客户端断开或预算耗尽时，正在进行的上游调用全部取消；
  - 每一跳在进程内运行该跳的插件链，不需要伪造内部请求；
  - 每个节点一个 trace span，attempt 证据沿用 `looper.AttemptTrace` 的有界结构。
- **声明方式：** 在 recipe 或 algorithm 中用 YAML 声明图，支持子图复用。现有每个 looper 算法都作为内置图模板提供，
  已有配置的行为保持不变。
- **等价验收：** 在确定性 fixture 上，现有每个 looper 算法产生与之前相同的上游调用序列和最终结果。

### 配置体系

借鉴的是 Envoy 配置模型的五个核心思想，而不是照搬 xDS：

1. **类型化资源 + 名字引用：** listener、route、cluster、endpoint、secret 各自独立，按名字互相引用。
2. **类型化扩展：** 过滤器和扩展用 `type` 标识，每个注册的扩展自己校验配置。
3. **原子快照 + 预热：** 新配置先预热再整体生效，失败时旧配置继续服务。
4. **版本 + ACK/NACK：** 每次更新都有版本号，无效更新被明确拒绝。
5. **drain：** 旧配置上的在途请求自然跑完。

落地方式：

- **作者格式不变：** 仍是规范文档 `version/listeners/providers/evaluation/routing/entrypoints/recipes/global`。
- **编译为不可变快照**，由类型化资源组成：

  | 资源 | 来自规范文档 | 类比 Envoy |
  | --- | --- | --- |
  | Listener | `listeners` | Listener + HTTP connection manager |
  | Route | `entrypoints` 的匹配部分 | RouteConfiguration |
  | Program | `recipes`、`routing`、`evaluation` | HTTP filter chain + 扩展配置 |
  | Cluster / Endpoint | `providers` 的模型与后端 | Cluster / ClusterLoadAssignment |
  | Secret | 凭据引用（环境变量、文件、Kubernetes Secret） | Secret（SDS） |
  | RuntimeModel | runtime 部署 | 无（Router 独有） |

- **类型化扩展注册表：** 信号、算法、插件、图节点、过滤器都按 `type` 注册 Go 实现、schema、默认值和校验器；新增
  一种扩展不需要改核心代码。
- **校验：** 由 Go 类型生成的 JSON schema（今天是 `router-config-v0.3.schema.json`）+ 交叉引用检查 + 能力检查。
  当前模式不支持的字段在加载时直接报错，并提示可以支持它的模式（例如 `--gateway extproc`）。
- **生命周期：** compile、validate、warm、activate（原子切换，和今天的 router generation 切换一样）、drain、history。
- **增量重建：** 只重建变化的资源。只改 endpoint 时不重建信号和模型绑定；只改 recipe 时不重建上游连接池。
- **版本与回滚：** 每个快照有单调递增的版本号和内容 hash；保留最近 N 个；`POST /api/v1/config/rollback` 回到指定
  版本；当前版本出现在 `/api/v1/config`、metrics 和响应 header 中。
- **配置源可插拔：** 文件监听（已有）、HTTP API（已有的 `/api/v1/config`，补上 ACK/NACK 语义和审计记录）、通过
  operator 的 Kubernetes CRD，以后还可以加按资源下发的类 xDS 控制面。
- **扩容：** Router 实例无状态，从同一个配置源取配置；runtime 可以接入共享的 GPU runtime 池。
- **升级与迁移：** `version` 字段管理 schema 版本；布局变化走 `vllm-sr config migrate`（运行时解析器只接受规范
  布局）。dry-run 校验（`/api/v1/config/recipes/validate`）和路由预览（`/api/v1/routing/preview`）已经存在。

### Router 与 runtime

- 不变：runtime 是 Router 托管的子进程，通过 Unix 域套接字上的 HTTP/JSON 访问，每个请求阶段每个 runtime 进程调用
  一次 `/v1/bundle`。
- 这和 vLLM 自己的前端与 EngineCore 是同一种关系：vLLM 的 Rust 前端是独立进程，通过 ZMQ 和 MessagePack 与
  EngineCore 通信；只有离线的 `LLMEngine` 在进程内运行。
- runtime 不放进 Router 进程，是因为在 Go 里嵌入 PyTorch 需要 cgo（Router 刚刚去掉它），Python GIL 和 Go 调度器
  互相干扰，GPU 故障还会带走 Router。
- 先测量 Router 到 runtime 的开销；只有数据需要时才加 MessagePack 或共享内存。

### 分发与 CLI

- PyPI 上只有 `vllm-sr`：在装了 Docker 或 Podman 的主机上 `pip install vllm-sr` 就是全部安装步骤。
- Router 二进制和模型 runtime（`vllm-srun`）都在同一个镜像家族里（`vllm-sr`、`vllm-sr-rocm`、`vllm-sr-cuda`），
  CLI 栈、Helm 和 operator 共用一个 entrypoint，二者都不单独发布。Router 二进制自身的默认值仍是 extproc，所以不传
  `-gateway` 参数的清单行为不变；每个启动方都显式传入模式。
- Kubernetes 上 Helm chart 接受 `gateway.mode: standalone|extproc`（默认 standalone），CLI 的 kubernetes target
  会连同镜像以及 GPU 平台的 GPU 资源请求一起写入。
- standalone 模式下 `vllm-sr serve` 只起 Router 容器，由它对外提供 listener，并在容器内托管 runtime 进程。dashboard
  直接连 Router，Router 的 `/ready` 回答 dashboard 的探活。
- engine 模式 `vllm-sr serve MODEL` 用同一个镜像在容器里起 `vllm-srun serve`，映射端口并挂载模型缓存。没有宿主机
  上的 engine 路径，所以 macOS 上 engine 模式只能用 CPU。
- release notes 写明默认模式的变化；`--gateway extproc`（Helm 里是 `gateway.mode: extproc`）恢复原来的行为。

### 可观测性

- **访问日志：** 保留今天 Envoy 访问日志的字段。
- **metrics：** 请求、延迟分位、上游错误、重试、fallback、离群剔除、图节点耗时，与现有 Router metrics 合并。
- **tracing：** OpenTelemetry，每个请求一个根 span，路由核心、每次 runtime 调用、每个图节点、每次上游尝试各一个子
  span；向后端透传 `traceparent`。

## 包结构与依赖规则

| 包 | 职责 |
| --- | --- |
| `pkg/routing` | 与传输无关的契约：`Engine`、中立的请求与响应、阶段效果及其应用规则、`Plan`、`Budget`、`Evidence` |
| `pkg/routing/parity` | parity 记录、记录器、归一化、比较器、golden 文件 |
| `pkg/routing/graph` | 请求图执行器、节点注册表、内置 looper 模板 |
| `pkg/upstream` | cluster、endpoint、连接池、负载均衡、健康检查、离群剔除，以及 timeout、retry、fallback 的执行 |
| `pkg/gateway` | 原生 HTTP 前端 |
| `pkg/extproc` | ext_proc 适配器；在抽取完成之前，也承载实现 `routing.Engine` 的流水线 |

- `pkg/routing` 及其子包、`pkg/upstream`、`pkg/gateway` 既不引用 Envoy 类型，也不引用 `pkg/extproc`。
- `pkg/gateway` 依赖 `pkg/routing` 和 `pkg/upstream`；没有包依赖 `pkg/gateway`。
- 只有组合根（Router 的 `cmd` 和服务装配代码）把 ext_proc 引擎绑定到原生前端。
- 由依赖测试强制执行这些规则。

## 首发能力与 follow-up

| 能力 | 首发 | 说明 |
| --- | --- | --- |
| 监听、HTTP/1.1、h2c、超时、body 与连接上限 | 是 | 对齐本地 Envoy 模板 |
| 下游 TLS（单向） | 是 | Go 标准库 |
| 剥离内部 header、API key 校验 | 是 | 对齐 Envoy Lua 过滤器与 header 规则 |
| 进程内路由流水线、直接响应、流式请求体 | 是 | 核心价值 |
| 上游：负载均衡、连接池、host 与 path 改写、header 与凭据注入、TLS/SNI | 是 | 对齐 Envoy cluster |
| 熔断、离群剔除、主动健康检查 | 是 | 规范配置已有这些字段 |
| timeout、retry、fallback | 是 | |
| SSE 流式与背压 | 是 | |
| 请求图执行器（looper v2），与现有 looper 算法等价 | 是 | |
| 配置快照、预热、原子切换、drain、版本、回滚、增量重建、ACK/NACK、审计 | 是 | |
| 访问日志、metrics、tracing、健康与就绪、优雅退出 | 是 | |
| 两个 target 上的 `--gateway standalone` 和 `--gateway extproc`、容器里的 engine 模式、`serve` 参数分组 | 是 | |
| Helm 和 operator 默认 standalone；两个 target 共用一个镜像家族 | 是 | |
| 从代码、CI、镜像、Helm 和当前文档中移除 OpenClaw 和 fleet simulator（`vllm-sr-sim`） | 是 | 历史博客和已发布版本的文档保持不变 |
| 限流（按 key 或路由的令牌桶） | follow-up | 期间用 `--gateway extproc` |
| mTLS、JWT/OIDC、外部鉴权 | follow-up | 同上 |
| 正则或多 header 路由匹配、按权重分流、流量镜像、请求 hedging | follow-up | 同上 |
| WebSocket（realtime）、HTTP/3 | follow-up | |
| 类 xDS 控制面 | follow-up | 首发只提供文件、API、CRD 三种配置源 |
| macOS 上通过宿主机桥接做 GPU 加速（`--platform apple`） | follow-up（[#4636](https://github.com/vllm-project/semantic-router/issues/4636)） | macOS 上内置模型在 arm64 镜像里用 CPU 运行 |

需要 follow-up 能力的配置，在 standalone 模式下启动即报错，并提示使用 `--gateway extproc`。

## 测试与验收

- **parity 套件：** 同一组请求（覆盖每类信号、决策、插件和 looper 算法）分别在 standalone 和 Envoy 模式下运行，路由
  决策、上游请求和响应变换必须一致。
- **端到端：** CLI 集成测试覆盖 docker 上的 standalone 模式和 engine 模式；新增一个用 Helm 默认值（standalone）的
  Kind profile；现有接入网关的 profile 固定为 extproc，结果不变。
- **故障注入：** 假后端模拟超时、5xx、429、连接重置、半途断流，验证 retry、fallback、熔断、离群剔除，以及“字节
  写给客户端之后不再重试”的边界。
- **looper 等价：** 现有每个 looper 算法在确定性 fixture 上的调用序列和结果与之前一致。
- **配置：** 热更新、被拒绝的更新（NACK，旧版本继续服务）、回滚，以及 drain 期间在途请求不受影响。
- **性能记录：** 两种模式的端到端延迟和吞吐（至少 5 轮交错，带 95% 区间），以及 Router 到 runtime 的耗时占比，
  用来决定是否值得做 runtime 快路径。
- **打包：** 在只装了 Docker 的干净主机上，`pip install vllm-sr` 之后 `vllm-sr serve`（standalone 模式）和
  `vllm-sr serve MODEL`（engine 模式）在 CPU 上直接可用。

### 结果 {#results}

以下测量记录当时的实现和硬件，不代表当前默认模型、副本池或所有 profile 的性能。

合入前在最终代码上记录，单节点（AMD EPYC 9575F、Docker 29.8.1、Envoy 1.35.3），后端为 fake backend：

- **一致性：** 18 个用例在 Envoy + ext_proc 与 standalone 模式下的状态码、响应体、路由和 Router 头都相同，
  只出现上文列出的差异：`server`、`x-envoy-*` 头和响应体分帧。
- **故障注入：** 一次 503、429、reset 或 per-try timeout，400，流式 503，流中途断开，以及每次尝试都 503：
  两种模式的后端调用和客户端结果相同。每次尝试都 reset 或超时的情况按上文所述不同：standalone 模式会对
  local reply 做 fallback。
- **Looper：** Confidence（普通、流式、小模型失败）、Ratings、ReMoM 和 Fusion（普通、流式）发出相同的模型调用，
  返回相同的响应。
- **配置：** 负载下约 209,000 个请求，经历 7 次激活、一个被拒绝的文件和两次回滚：没有失败，没有 worker 看到
  版本倒退，每个版本只服务一份文档，30 个流全部完成，其中 14 个跨越了变更。文件里是被拒绝的文档时，回滚
  依然成功。
- **打包：** 在只有 Docker 的主机上，用 CLI 的 wheel 装进干净的 venv，`vllm-sr serve` 两种网关模式都可用，
  `vllm-sr serve MODEL` 在 CPU 上可用。ROCm 镜像里的 Router 按 chart 的方式运行（uid 65532、只读根文件系统、
  无 capability），只通过 render 组访问 GPU。
- **性能：** 每种模式、每个并发度 10 轮交错，请求不命中任何 decision；取各轮均值，带 95% 区间。Envoy 用 8 个核，
  ext_proc 模式的 Router 用 16 个。
  - **合入时的代码**（每轮 2,000 个请求），standalone 的 Router 在 1 和 8 个客户端时用 16 个核，在 32 和 64 个时
    用 24 个：

    | 客户端数 | p50，ext_proc → standalone | p99，ext_proc → standalone | 每秒请求数，ext_proc → standalone |
    | --- | --- | --- | --- |
    | 1 | 0.74 → 0.57 ms | 2.02 → 1.82 ms | 1,280 → 1,644 |
    | 8 | 0.87 → 0.64 ms | 1.77 → 1.56 ms | 7,771 → 9,737 |
    | 32 | 2.17 → 2.20 ms | 4.55 → 6.28 ms | 14,067 → 13,435 |
    | 64 | 4.20 → 5.06 ms | 9.36 → 10.92 ms | 14,561 → 13,080 |

    从 32 个客户端起，standalone 模式在比 Envoy 模式低 5–10% 的吞吐处饱和，加核也几乎不提高上限。
  - **[#4666](https://github.com/vllm-project/semantic-router/issues/4666) 之后**，同一节点（每轮 2,000 个请求），
    standalone 的 Router 在各并发度都用 24 个核，与 Envoy 模式的总核数相同：

    | 客户端数 | p50，ext_proc → standalone | p99，ext_proc → standalone | 每秒请求数，ext_proc → standalone |
    | --- | --- | --- | --- |
    | 1 | 0.57 → 0.41 ms | 1.12 → 0.95 ms | 1,692 → 2,318 |
    | 8 | 0.66 → 0.43 ms | 1.40 → 1.34 ms | 10,831 → 14,523 |
    | 32 | 1.03 → 0.93 ms | 2.54 → 3.46 ms | 26,744 → 27,935 |
    | 64 | 1.97 → 1.70 ms | 4.52 → 8.43 ms | 30,152 → 29,473 |

    上限来自两种模式共用的路由核心。每个请求都在一把全局锁下更新 TTFT 历史，随后又在这把锁下把历史复制出来，
    于是一次被垃圾回收拖住的复制会卡住排在它后面的所有请求。每个请求还分配约 300 KB，大部分是 provider 目录的
    防御性副本，以及随后被采样器丢弃的日志字段。去掉这些之后，两种模式都快了约一倍。standalone 模式在各负载下
    都响应更快，到 32 个客户端为止每秒处理的请求也更多。64 个客户端时，在每轮 2,000 个请求（按这个速率约 70 ms）
    的测量里两者持平（−680 ± 721 请求/秒）；每轮 10,000 个请求时，standalone 模式在 64 个客户端下每秒处理
    30,319 个请求，Envoy 模式 28,837 个，32 个客户端下为 28,830 对 26,370。从 32 个客户端起它的 p99 更高：两个
    Router 此时都受 CPU 限制，垃圾回收仍占它们三分之一以上的 CPU。`internal/gatewayparity` 里的
    `BenchmarkNativeGatewayClients` 在进程内从 1、8、32 和 64 个客户端经 standalone 路径发送同样的请求，不需要
    Envoy 或 Docker 就能看出请求路径上的退化。
- **Router 到 runtime 的占比：** 一个 CPU 上的 jailbreak 信号（307M 的 Vela Guard）下，standalone 请求耗时
  13.9 ms，其中 12.4 ms（89%）在 runtime 调用里。runtime 现在为每个请求报告自己的耗时，Router 记录每次调用的
  传输时间（[#4667](https://github.com/vllm-project/semantic-router/issues/4667)）。在 CPU 上，推理占一次调用的
  95–99%；传输在 Vela 2.0 0.3B 默认信号上占 0.6%，在 Vela 1.0 信号上占 1.9–2.2%，单独的 Guard 上占 2.1%，runtime
  自身的 HTTP 和 JSON 处理占 0.4–1.6%。快路径最多只能让一个请求快约 0.8 ms，所以 Router 继续使用 HTTP/JSON
  （[记录](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/runtime-transport-cpu.md)）。

## 风险与对策

| 风险 | 对策 |
| --- | --- |
| 默认模式切换影响现有用户（包括 Helm 和 operator 部署） | release notes、启动日志提示、`--gateway extproc` 或 `gateway.mode: extproc` 恢复原行为；不支持的配置快速失败 |
| 自研代理不如 Envoy 久经考验 | 基于成熟的 Go 标准库组件；故障注入测试；生产网关场景仍推荐 Envoy 模式 |
| 重写 looper 引入行为差异 | 先做等价测试再切换；旧路径在 Envoy 模式保留，直到等价得到证明 |
| 配置 schema 变化 | `vllm-sr config migrate` 负责迁移；运行时解析器只接受规范布局 |

## 实施阶段

1. **P0 设计：** 跟踪 issue 与本文档。
2. **P1 路由核心：** `pkg/routing` 与 ext_proc 适配器，纯重构；现有端到端 profile 全部不变；parity 记录工具。
3. **P2 上游层：** cluster 与 endpoint、连接池、负载均衡、host 与 path 改写、header 与凭据注入、TLS/SNI、熔断、
   离群剔除、主动健康检查、流式代理。
4. **P3 timeout、retry、fallback：** 语义、预算、安全边界、可观测性、故障注入测试。
5. **P4 Standalone 前端：** listener、TLS、API key、header 剥离、OpenAI 接口、health、ready 与 metrics、访问日志、
   tracing、优雅退出。
6. **P5 请求图执行器：** 节点注册表、执行器、预算与取消、每跳插件、trace；looper 算法成为内置模板并通过等价测试；
   ext_proc 模式改用执行器，回环和内部密钥 header 随之移除。
7. **P6 配置体系：** 类型化资源快照、扩展注册表、校验、预热、原子切换、drain、版本历史与回滚、增量重建、API 的
   ACK/NACK 与审计。
8. **P7 CLI 与部署：** 两个 target 上的 `--gateway standalone|extproc`（默认 standalone）、`--target kubernetes`、
   容器里的 engine 模式、`serve` 参数分组和 `--container-runtime`；Helm 和 operator 的默认值；一个镜像家族；在只有
   Docker 的主机上跑 CLI 集成测试，以及一个 standalone 的 Kind profile。
9. **P8 验证与文档：** parity 套件、端到端 profile、性能记录、用户文档（模式选择、可靠性、请求图、配置管理）、
   release notes。

## 与其他提案的关系

- [独立 HTTP Gateway](./standalone-http-gateway)：被本提案取代。它对共享引擎、单一中立响应流水线和进程内 looper
  执行器的分析继续适用；单独的二进制、单独的 bootstrap 配置和“仅限实验”的范围不再沿用。
- [模型执行回退](./model-execution-fallback)：这里的 fallback 链是那条边界的具体落地，每个请求只有一个
  fallback 权威。
- [多协议适配器架构](./multi-protocol-adaptor)：两个适配器使用同一套中立编解码。
- [Router Flow 工作流](./router-flow-workflows)：workflows 成为行为不变的请求图。
- [统一配置契约 v0.3](./unified-config-contract-v0-3)：规范文档仍是唯一的作者格式，快照由它编译而来。
- [vLLM Production Stack 集成](./production-stack-integration)：分层关系不变；standalone 模式负责某个逻辑模型的传输
  和后端选择，服务平台继续负责副本的生命周期。

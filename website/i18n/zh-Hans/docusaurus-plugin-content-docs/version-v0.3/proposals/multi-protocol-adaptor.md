---
translation:
  source_commit: "15da5536"
  source_file: "docs/proposals/multi-protocol-adaptor.md"
  outdated: false
---

# 设计文档：多协议适配器架构

**作者：** vLLM Semantic Router 团队  
**状态：** 待实现  
**创建：** 2026 年 2 月  
**最后更新：** 2026 年 2 月

## 概述

本文描述 vLLM Semantic Router 的多协议适配器架构设计与实现思路，在 Envoy ExtProc 之外抽象 API 层，以支持多种前端协议。

## 背景

Semantic Router 曾通过 gRPC 与 Envoy External Processor（ExtProc）紧耦合。这虽能与 Envoy 深度集成，但对以下用户形成门槛：

- 希望在不部署 Envoy 的情况下使用路由器
- 偏好直接 HTTP/REST 集成
- 使用 Nginx 或其他反向代理
- 在开发或测试时需要更简化的部署拓扑

### 动机

- **灵活性**：无需 Envoy 基础设施即可获得直连 HTTP API
- **测试**：无需完整 Envoy 部署即可轻量测试
- **可扩展性**：支持 nginx、原生 gRPC 与自定义协议
- **可复用性**：所有协议共享同一路由引擎
- **部署形态**：支持 serverless、边缘与简化部署

## 目标

### 主要目标

1. **协议抽象**：将路由逻辑与协议相关代码分离
2. **多协议支持**：允许多种协议同时工作
3. **向后兼容**：保留现有 ExtProc 能力
4. **共享状态**：缓存、重放与路由决策的单一事实来源
5. **易于扩展**：新增协议适配器的固定模式

### 非目标

1. 替换或弃用 Envoy ExtProc
2. 改变路由决策算法或分类逻辑
3. 除适配器相关节外修改配置格式
4. 支持破坏抽象层的协议专属特性

## 设计原则

### 1. 单一路由流水线

**关键约束：** 所有路由逻辑**必须**流经 `RouterEngine.Route()`，无一例外。

- 适配器将协议翻译为 `RouteRequest` → 调用 `RouterEngine.Route()`
- `RouterEngine.Route()` 返回 `RouteResponse` → 适配器再翻译回协议
- 适配器**不得**重复实现分类、安全、缓存、重放逻辑
- 适配器**不得**直接调用分类器、缓存或重放记录器

### 2. 薄适配器层

适配器**仅做协议翻译**：

- 解析协议专属请求格式
- 转换为 `RouteRequest`
- 调用 `RouterEngine.Route()`
- 将 `RouteResponse` 转换为协议格式
- 返回给客户端

### 3. RouterEngine 拥有全部路由

`RouterEngine.Route()` 是**唯一**发生以下行为之处：

- 分类
- PII/越狱检测
- 缓存读/写
- 工具选择
- 重放记录
- 后端选择
- 代理（或返回代理信息）

## 设计

### 架构概览

```
┌────────────────────────────────────────────────────────────┐
│                    Application Layer                       │
│                                                            │
│  ┌───────────────────────────────────────────────────┐     │
│  │                Adapter Manager                    │     │
│  │  - Reads adapter config                           │     │
│  │  - Creates protocol adapters                      │     │
│  │  - Manages lifecycle                              │     │
│  └──────┬────────┬────────┬───────────┬──────────────┘     │
│         │        │        │           │                    │
│  ┌──────▼──┐ ┌───▼─── ┐ ┌─▼──────┐ ┌──▼─────┐              │
│  │ ExtProc │ │ HTTP   │ │ gRPC   │ │ Nginx  │              │
│  │ Adapter │ │Adapter │ │Adapter │ │Adapter │              │
│  │ ┌─────┐ │ │ ┌─────┐│ │ ┌─────┐│ │ ┌─────┐│              │
│  │ │Parse│ │ │ │Parse││ │ │Parse││ │ │Parse││              │
│  │ │ExtP │ │ │ │HTTP ││ │ │gRPC ││ │ │NJS  ││              │
│  │ └──┬──┘ │ │ └─┬───┘│ │ └─┬───┘│ │ └──┬──┘│              │
│  │    │Conv│ │   │Con │ │   │Con │ │    │Con│              │
│  │    ▼    │ │   ▼    │ │   ▼    │ │    ▼   │              │
│  │ ┌─────┐ │ │ ┌────┐ │ │ ┌────┐ │ │ ┌─────┐│              │
│  │ │Req  │ │ │ │Req │ │ │ │Req │ │ │ │Req  ││              │
│  │ └──┬──┘ │ │ └─┬──┘ │ │ └─┬──┘ │ │ └──┬──┘│              │
│  └────┼────┘ └───┼────┘ └───┼────┘ └────┼───┘              │
│       │          │          │          │                   │
│       └──────────┴──────────┴──────────┘                   │
│                    Single Entry Point                      │
│                             │                              │
│                             ▼                              │
│        ┌──────────────────────────────────────────┐        │
│        │           RouterEngine.Route()           │        │
│        │  1. Classify request                     │        │
│        │  2. Check PII / jailbreak                │        │
│        │  3. Check cache                          │        │
│        │  4. Select tools                         │        │
│        │  5. Select model/backend                 │        │
│        │  6. Record replay                        │        │
│        │  7. Proxy to backend (via Backend Layer) │        │
│        │  8. Update cache                         │        │
│        └──────────────┬───────────────────────────┘        │
│                       │                                    │
│                       ▼                                    │
│                  RouteResponse                             │
│                       │                                    │
│        ┌──────────────┼──────────────┬───────────┐         │
│        │              │              │           │         │
│  ┌─────▼─────┐ ┌──────▼────┐ ┌───────▼───┐ ┌─────▼─────┐   │
│  │ ExtProc   │ │ HTTP      │ │ gRPC      │ │ Nginx     │   │
│  │ Adapter   │ │ Adapter   │ │ Adapter   │ │ Adapter   │   │
│  │ ┌───────┐ │ │ ┌───────┐ │ │ ┌───────┐ │ │ ┌───────┐ │   │
│  │ │Convert│ │ │ │Convert│ │ │ │Convert│ │ │ │Convert│ │   │
│  │ │to gRPC│ │ │ │to HTTP│ │ │ │gRPC   │ │ │ │to NJS │ │   │
│  │ └───────┘ │ │ └───────┘ │ │ └───────┘ │ │ └───────┘ │   │
│  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘   │
│        │             │             │             │         │
└────────┼─────────────┼─────────────┼─────────────┼─────────┘
         │             │             │             │
         └─────────────┴─────────────┴─────────────┘
                            │
                            ▼
         ┌─────────────────────────────────────────┐
         │      Backend Abstraction Layer          │
         └──────┬──────────────────┬───────────────┘
                │                  │
       ┌────────▼────────┐  ┌──────▼──────────┐
       │ Envoy Proxy     │  │ Direct Proxy    │
       │ (ExtProc mode)  │  │ (HTTP/gRPC)     │
       │ - Dynamic fwd   │  │ - HTTP client   │
       │ - Headers only  │  │ - Full response │
       └────────┬────────┘  └──────┬──────────┘
                │                  │
                └──────────┬───────┘
                           ▼
             ┌────────────────────────────┐
             │      Inference Backends    │
             │  ┌────────┐  ┌────────┐    │
             │  │ vLLM   │  │Ollama  │    │
             │  │Server  │  │Server  │    │
             │  └────────┘  └────────┘    │
             └────────────────────────────┘
```

**要点：** 适配器是**薄翻译层**，智能全部在 RouterEngine。

### 组件设计

#### 1. RouterEngine（核心）

**位置：** `pkg/router/engine/`

**职责：**

- 与协议无关的路由逻辑
- 请求分类与决策求值
- 语义缓存操作
- 工具选择与嵌入
- 路由器重放记录
- PII 与越狱检测
- 模型选择

**主要方法：**（与英文原文一致，见代码块）

```go
type RouterEngine struct {
    Config               *config.RouterConfig
    Classifier           *classification.Classifier
    PIIChecker           *pii.PolicyChecker
    Cache                cache.CacheBackend
    ToolsDatabase        *tools.ToolsDatabase
    ModelSelector        *selection.Registry
    ReplayRecorders      map[string]*routerreplay.Recorder
}

func (e *RouterEngine) Route(ctx context.Context, req *RouteRequest) (*RouteResponse, error)
func (e *RouterEngine) ClassifyRequest(ctx context.Context, messages []Message) (*ClassificationResult, error)
func (e *RouterEngine) CheckCache(ctx context.Context, model, query, decisionName string) (string, bool, error)
func (e *RouterEngine) UpdateCache(ctx context.Context, model, query, response, decisionName string) error
func (e *RouterEngine) SelectTools(ctx context.Context, query string, topK int) ([]openai.ChatCompletionToolParam, error)
func (e *RouterEngine) RecordReplay(ctx context.Context, decisionName string, record *routerreplay.RoutingRecord) error
```

**设计决策：**

- 所有适配器共享单实例
- 有状态（维护缓存、重放记录器）
- 无协议专属逻辑
- 返回与协议无关的数据结构

#### 2. 适配器接口

**位置：** `pkg/adapter/manager.go`

```go
type Adapter interface {
    Start() error                      // Start the adapter (blocks)
    Stop() error                       // Graceful shutdown
    GetEngine() *engine.RouterEngine  // Access to shared engine
}
```

**设计决策：** 接口最小化以保留灵活性；各适配器自管生命周期；接口中无协议专属方法；适配器在独立 goroutine 中运行。

#### 3. Adapter Manager

**位置：** `pkg/adapter/manager.go`

**职责：** 解析适配器配置、按配置实例化、在独立 goroutine 中启动、协调优雅关闭。

**主要方法：**

```go
func (m *Manager) CreateAdapters(cfg *config.RouterConfig, eng *engine.RouterEngine, configPath string) error
func (m *Manager) StartAll() error
func (m *Manager) StopAll() error
func (m *Manager) Wait()
```

#### 4. ExtProc 适配器

**位置：** `pkg/adapter/extproc/`

包装现有 Envoy ExtProc 实现，保持向后兼容，处理 gRPC/Envoy 细节，支持 TLS。

#### 5. HTTP 适配器

**位置：** `pkg/adapter/http/`

提供 OpenAI 兼容 REST API，无需 Envoy，处理 CORS、请求头等 HTTP 关注点。

**端点：** `POST /v1/chat/completions`、`POST /v1/completions`（未来）、`GET /v1/models`、`POST /v1/classify`、`POST /v1/route`、`GET /v1/router_replay`、`GET /v1/router_replay/{id}`、`GET /health`、`GET /ready`。

#### 6. gRPC 适配器

**位置：** `pkg/adapter/grpc/`

提供原生 gRPC API，较 ExtProc 直连 gRPC 客户端更高效，支持流式与一元 RPC。

**服务定义示例：**

```protobuf
service SemanticRouter {
  rpc Route(RouteRequest) returns (RouteResponse);
  rpc Classify(ClassifyRequest) returns (ClassifyResponse);
  rpc StreamRoute(stream RouteRequest) returns (stream RouteResponse);
}
```

#### 7. Nginx 适配器

**位置：** `pkg/adapter/nginx/`

通过 NJS（JavaScript）模块与 Nginx 集成，支持 OpenResty Lua，基于头的路由类似 ExtProc。

**集成方式：** NJS 模块、Lua/OpenResty、HTTP 子请求（内部调用 HTTP 适配器）、共享内存 IPC。

### 配置设计

```yaml
adapters:
  - type: "envoy" # ExtProc adapter
    enabled: true
    port: 50051
    tls:
      enabled: true
      cert_file: "/path/to/cert.pem"
      key_file: "/path/to/key.pem"

  - type: "http" # HTTP REST API
    enabled: true
    port: 9000

  - type: "grpc" # Native gRPC API
    enabled: true
    port: 50052
    tls:
      enabled: true
      cert_file: "/path/to/cert.pem"
      key_file: "/path/to/key.pem"

  - type: "nginx" # Nginx integration
    enabled: true
    port: 9001
    mode: "njs" # Options: njs, lua, http
    config:
      upstream_variable: "backend_upstream"
      header_prefix: "x-vsr-"
```

### 数据流（HTTP 适配器示例）

1. 客户端请求 → 2. HTTP 适配器接收 `POST /v1/chat/completions` → 3. 解析 OpenAI 请求 → 4. `RouterEngine.Route` → 5. 分类、缓存、工具、重放等 → 6. 返回 `RouteResponse` → 7. 代理到选定后端 → 8. 返回客户端。

**共享状态：** 多适配器共享同一缓存条目、重放记录器、分类决策与模型选择状态。

## 实现细节

### 初始化顺序

1. 加载配置 → 2. 初始化嵌入模型 → 3. 创建 RouterEngine → 4. 创建 Adapter Manager → 5. `CreateAdapters` → 6. `StartAll` → 7. `Wait()` 阻塞。

### 错误处理

- 适配器创建失败：致命，进程退出
- 适配器启动失败：致命，进程退出
- 运行时错误：记录日志，适配器在可能情况下继续
- RouterEngine 错误：返回适配器做协议相关处理

### 并发模型

- **RouterEngine：** 线程安全，多适配器可并发调用
- **Cache：** 由后端处理并发
- **Replay Recorders：** 线程安全 map + 按决策加锁
- **Adapters：** 独立 goroutine，无共享适配器状态

## 权衡与替代方案

### 选用共享 RouterEngine 而非每适配器一实例

**理由：** 跨协议决策一致、共享缓存提高命中率、重放记录单一来源、内存更小。  
**代价：** 潜在热点（通过线程安全设计缓解）。

### 适配器接口：选用最小接口（见上文），而非富接口（含 HandleRequest、GetMetrics 等）。

**理由：** 不同协议的请求/响应模型差异极大，最小接口最灵活。

### 配置：选用单一配置文件中的 `adapters` 节，而非每适配器独立文件或纯环境变量。

## 已知局限

1. 抽象层难以做协议专属优化  
2. 适配器之间不能直接通信（有意设计）  
3. 若 RouterEngine 非线程安全则共享状态存在竞态风险  
4. 用户可配置项增多  

## 测试策略

- **单元测试：** RouterEngine 方法、各适配器逻辑、配置解析  
- **集成测试：** 多适配器同时运行、跨适配器缓存一致性、双协议重放  
- **E2E：** Envoy ExtProc 8801、直连 HTTP 9000，验证相同路由决策与重放可见性  

## 后续工作

### 短期

优雅关闭、适配器指标（QPS、延迟直方图、错误率）、增强 Nginx 集成（OpenResty、Nginx Plus、共享内存 IPC）。

### 长期

gRPC 双向流、WebSocket 适配器、插件化动态加载适配器、每适配器限流/认证/中间件。

## 迁移指南

**以前：** `extproc.NewServer(...); server.Start()`

**以后：** `NewRouterEngine` → `adapter.NewManager` → `CreateAdapters` → `StartAll` → `Wait()`，并在 `config.yaml` 中增加 `adapters` 节（见英文原文示例）。

### 新增适配器步骤

1. 新建 `pkg/adapter/myprotocol/`  
2. 实现 `Adapter` 接口（见英文原文 `MyAdapter` 示例）  
3. 在 `manager.go` 中 `case "myprotocol":` 注册  
4. 增加 YAML 配置  

## 参考

- [OpenAI API Specification](https://platform.openai.com/docs/api-reference)
- [Envoy ExtProc Documentation](https://www.envoyproxy.io/docs/envoy/latest/configuration/http/http_filters/ext_proc_filter)
- [vLLM Semantic Router Documentation](https://vllm-semantic-router.com)

## 附录

### 性能

RouterEngine 单实例省内存但可能成为瓶颈；缓存后端选型关键；重放建议异步写；适配器开销主要来自序列化。

### 安全

每适配器 TLS；认证在适配器层（未来可抽象外部鉴权）；PII/越狱检测跨适配器共享。

### 监控

每适配器与 RouterEngine 指标、分布式追踪跨适配器 span、结构化日志、每适配器健康检查端点。

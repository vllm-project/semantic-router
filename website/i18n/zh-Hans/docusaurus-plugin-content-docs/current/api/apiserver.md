# Router Apiserver API 参考

Router Apiserver 是运行在 `:8080` 上的 HTTP 控制面。

## 基础 URL

`http://localhost:8080`

## 实时 Schema

- `GET /api/v1`：发现索引
- `GET /openapi.json`：实时 OpenAPI schema
- `GET /docs`：Swagger UI

精确的请求和响应结构以 `/openapi.json` 或 `/docs` 为准。本页只保留当前已记录接口的简明参考。

## 发现与健康检查

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/health` | 健康检查 |
| `GET` | `/ready` | 就绪检查，只有启动完成后才返回 ready |
| `GET` | `/api/v1` | API 发现与文档索引 |
| `GET` | `/openapi.json` | OpenAPI 3.0 schema |
| `GET` | `/docs` | 交互式 Swagger UI |

## 分类接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/api/v1/classify/intent` | 对请求做路由类别分类 |
| `POST` | `/api/v1/classify/pii` | 检测文本中的 PII |
| `POST` | `/api/v1/classify/security` | 检测 jailbreak 与安全威胁 |
| `POST` | `/api/v1/classify/fact-check` | 判断文本是否需要事实核查 |
| `POST` | `/api/v1/classify/user-feedback` | 识别用户反馈类型 |
| `POST` | `/api/v1/classify/combined` | 组合分类接口 |
| `POST` | `/api/v1/classify/batch` | 带 `task_type` 的批量分类 |

## 模型与信息接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/info/models` | 返回已加载模型信息 |
| `GET` | `/info/classifier` | 返回 classifier 状态与信息 |
| `GET` | `/v1/models` | OpenAI 兼容模型列表 |
| `GET` | `/metrics/classification` | 返回分类指标 |

## Router Config 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/config/router` | 以 JSON 返回当前 router config |
| `PATCH` | `/config/router` | 以 merge 语义更新 router config |
| `PUT` | `/config/router` | 以 replace 语义替换 router config |
| `GET` | `/config/router/versions` | 列出可回滚的备份版本 |
| `POST` | `/config/router/rollback` | 回滚到指定历史版本 |

## Config 语义

- `GET /config/router` 返回当前 router config 文档。
- `PATCH /config/router` 使用 merge 语义。
- `PUT /config/router` 使用 replace 语义。
- `PATCH` 和 `PUT` 都会在返回前完成校验、备份、写入和 hot-reload。

## 说明

- 上面的接口列表直接对应 `GET /api/v1` 和 `GET /openapi.json` 暴露出来的 router apiserver surface。
- 当前运行时里，部分已记录接口可能仍是占位实现，例如 `POST /api/v1/classify/combined` 和 `GET /metrics/classification`。

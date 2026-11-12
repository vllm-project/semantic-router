---
title: Router 配置参考
sidebar_label: Router 配置
sidebar_position: 3
description: 配置 model_runtime 部署的 Router 字段，以及 Router 托管运行时的环境变量。
is_mtpe: true
translation:
  source_commit: "bc5c6bfb747ffe9c5f35d98b203427dada8c4c56"
  source_file: "docs/model-runtime/reference/configuration.md"
  outdated: true
---

# Router 配置参考 {#router-configuration-reference}

这些是描述 `model_runtime` 部署及其功能如何读取它的 Router 字段，以及改变 Router 如何运行其托管 worker 的环境变量。

## 部署与绑定字段 {#router-configuration}

| 字段 | 位置 | 含义 |
| --- | --- | --- |
| `provider: model_runtime` | 部署 | 模型跑在模型 runtime 里。 |
| `artifact`、`revision` | 部署 | Hub 仓库和 commit，或本地绝对路径。 |
| `device` | 部署 | 见[让 router 来跑模型](../deploy/connect-the-router.md#describe-a-deployment)。 |
| `max_num_seqs`、`max_num_batched_tokens`、`batch_window_ms`、`dtype`、`enable_shared_context` | 部署 | 部署的[引擎参数](../deploy/engine-args.md)；没设的保持 runtime 默认值。 |
| `input.max_tokens`、`input.overflow` | 部署 | 任务模型的输入上限。答问题的部署只收 `overflow: window` 加 `max_tokens`：它的那些问题把长的部分整读到的扫描预算（见[长输入](./http-api.md#long-inputs)）。这也包括默认决策模型、直接发布给 System One 的部署，以及本地 System One 别名的部署，不论监听器是否发布该别名。路由的 System One 入口不是一个部署。不属于这些情况的绑定部署仍保留任务模型的输入上限。 |
| `signal_timeout_ms` | `global.model_catalog` | 请求的模型 runtime 信号的截止时间。默认取请求的截止时间扣掉剩余时间的十分之一（至少一秒）；router 看不到截止时间的受服务请求取 45 秒。还在跑的信号按它的政策收尾：越狱或 PII 规则按没有答案处理，除非规则设置了 `on_unknown: no_match`，否则匹配；其他信号都按失败的信号处理，走 `on_error` 或 `rules.on_unknown`。 |
| `on_unscanned` | `recipes[].signals.systemone[]`、`jailbreak[]`、`pii[]` | 在 `coverage: full` 下，一个没能读完整个输入的答案意味着什么：`match`、`no_match`、`unknown`（默认）或 `best_effort`。`jailbreak` 和 `pii` 规则分窗口读取长输入，默认为 `best_effort`，即根据已读窗口作出判断。 |
| `routing_view`、`scan_window` | 部署 | `routing_view: {max_tokens, head_share}` 把路由问题读取的长输入限定为开头和结尾（默认 8,192 个 token，开头占四分之一）；`scan_window: {tokens, overlap, max_parallel}` 设定防护问题分窗口读取时的窗口大小（8,192，重叠 512），每个窗口单独一次调用，同时最多 `max_parallel`（4）个。两者都按字节计算，每个 token 四个字节，并且不超过模型的验证长度减去 1,024（按每个 token 一个字节计算）；参见[长输入](../../system-one/deploy/performance.md#long-inputs)。 |
| `process` | 部署 | 同名的托管部署共用一个进程。 |
| `endpoint`、`model` | 部署 | 挂到你自己跑的 runtime；`model` 是所服务模型的 ID。[连接块](../../installation/model-catalog.md#connection-block)另有 `auth`、`headers`、`timeout_ms` 和 `tls`。 |
| `bindings.<type>`、`recipes[].model_bindings.<type>` | `global.model_catalog`、配方 | 一个信号类型使用的部署：写部署名，或写 `{deployment, head}` 以选择头。 |
| `deployment` | 规则 | `decision`、`classifiers`、`safety`（及 `hazard`）或 `complexity` 规则询问的部署。 |

绑定键是信号类型 `domain`、`jailbreak`、`pii`、`fact_check`、`user_feedback`、`modality`、`hallucination`、`preference` 和 `reask`。Embedding 在 `global.model_catalog.embeddings.default` 中指定部署。各信号如何解析见[为信号选择模型](../../installation/choose-a-model-for-a-signal.md)；各信号读取的答案见各任务指南。

## router 托管的 runtime {#router-managed-runtimes}

| 环境变量 | 默认 | 含义 |
| --- | --- | --- |
| `VLLM_SRUN_COMMAND` | `vllm-srun` | router 跑托管进程用的命令。 |
| `VLLM_SRUN_DIR` | 私有临时目录 | router 放各进程 Unix socket 和 models 文件的地方。 |
| `VLLM_SRUN_CACHE_DIR` | router 镜像里的 `/app/models/model-runtime` | 托管 runtime 的 Hugging Face 缓存。 |
| `VLLM_SRUN_CPU_PROCESSES` | 每个 CPU 模型一个，至多每两核一个 | 没写 `process` 的 CPU 模型最多摊到多少个进程；无 GPU 的宿主机上，`device: auto` 的也算在内。 |
| `VLLM_SRUN_READY_TIMEOUT` | `10m` | router 启动或重载时等一个部署就绪多久，形如 `30m` 的时长。首次启动可能要下载校验大模型。 |
| `VLLM_SRUN_RESULT_CACHE` | `4096` | router 给每个模型留多少条近期分类和决策结果，重复请求就不下 runtime 了；`0` 关掉。 |
| `VLLM_SRUN_AUTOTUNE_CACHE` | | runtime 的 `--autotune-cache` 目录。 |
| `HF_TOKEN` | | 访问 gated 或私有仓库的令牌。 |

router 和托管 runtime 之间只走 Unix socket，目录只有它自己的用户能开（0700）。进程退出它就重启，头一回等 1 秒，反复失败最多等到 60 秒；它自己退出时把每个进程都停掉（先 SIGTERM 再 SIGKILL）。一个进程里所有模型都加载失败的，它也照样重启——因为偶发原因失败的模型，不用重载配置就能回来。还在服务任何模型的进程，不停。

---
title: 档位
description: 在发布的答案和更快、近似的设置之间做选择。
translation:
  source_commit: "abae8ff99df2fdab372f0fb6d032b305907b9f44"
  source_file: "docs/model-runtime/profiles.md"
  outdated: false
is_mtpe: true
---

# 档位 {#profiles}

档位说的是模型怎么跑：和发布时一模一样，还是快一点、数字略有出入。按部署逐个选。

| 档位 | 大白话 | 什么时候用 |
| --- | --- | --- |
| `exact`（默认） | 给出的答案和模型发布方测出来的一致，Decision 2.0 是逐比特一致。任务模型在任何设备上都跑满 32 位精度。 | 永远用它，除非你测出自己确实需要更快。 |
| `shared_context` | 就一个请求问决策模型多个问题时，它把请求读一遍，所有问题都从这次读取里答。 | GPU 上每个请求有很多决策问题。 |
| `batching` | 几毫秒内先后到达的请求，凑到一起跑。 | GPU 上流量大，合并请求能抬吞吐。 |
| `max_speed` | 在前面这些基础上，硬件支持就用更快的 kernel 和 16 位数值。 | 延迟比最后一位小数更重要的 GPU 部署。 |

除 `exact` 之外都是**近似**：答案可能和发布的略有出入，边界答案可能翻面。每个近似档位的实测精度记录，见 runtime 的[记录](https://github.com/vllm-project/semantic-router/tree/main/src/model-runtime/docs/records)。

## 设档位 {#set-a-profile}

router 托管的模型，在部署上设 `profile`：

```yaml
global:
  model_catalog:
    deployments:
      decision-lux:
        provider: model_runtime
        artifact: vllm-sr/Decision-2.0-Lux-9B
        device: rocm:0
        profile: shared_context
```

runtime 自己起的，传 `--runtime-profile`（给 `vllm-srun serve` 时才用 `--profile`）：

```bash
vllm-sr serve vllm-sr/Decision-2.0-Lux-9B --platform amd --device rocm:0 --runtime-profile shared_context
```

用近似档位起的 runtime，照样能答要求 `exact` 的请求（`"options": {"profile": "exact"}`），切换之前可以先在自己的流量上对一对两边。

## 不改答案也能跑得快 {#performance-without-changing-answers}

默认档位下 runtime 也很快。下面这些它从来不拿正确性换：

- 一个请求的各路信号，一次打包调用送到每个 runtime 进程；
- 多个功能就同一段文本问同一个模型时，共享一次模型前向；
- 不同模型的工作并行跑，在不同进程和设备上；共用一块 GPU 的模型排队轮着用；
- 重复的输入不重算：每个模型留一份近期结果缓存，按精确输入和档位做键（`--result-cache-entries`）；
- 长输入只在你要求开窗时才切窗。

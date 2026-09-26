---
title: 使用同一个服务
translation:
  source_commit: "31fa0fdab6787b3b149894db259af13c8ce42f5a"
  source_file: "docs/benchmarking/sr-bench/shared-worker.md"
  outdated: true
---

# 使用同一个服务

`vllm-sr serve` 会独立启动核心评测 worker。存储位置为 `<state-root>/.sr-bench/<stack>/store`，主机 API 仅监听回环地址的 `8090 + 端口偏移`。同一工作目录中的 CLI 自动发现该存储和私有服务 token。Dashboard 或配置重载不会中断 worker；`vllm-sr stop` 停止 worker，但保留结果。

核心容器不挂载 Docker socket 或 GPU，也不包含全部上游运行环境。代码和 Agent 任务应使用准备好的独立 worker 主机：

```bash
vllm-sr benchmark setup --benchmark all
vllm-sr benchmark setup --benchmark all --install
```

默认只检查；`--install` 显式安装固定版本的可选解释器和源码，并获取经过 SHA256 校验的 SciCode 测试数据。`--build-sandbox` 构建离线代码评分镜像，回执保存镜像和基础镜像 digest 及固定依赖。这些操作不调用模型。缓存默认为 `~/.cache/vllm-sr/sr-bench-1.0`，可通过 `SR_BENCH_HOME` 覆盖；SciCode 数据默认位于缓存内的 `assets/scicode/test_data.h5`，也可通过 `SR_BENCH_SCICODE_TEST_DATA` 指定已准备文件。终端任务镜像、数据源权限以及裁判/模拟器仍需满足前置条件。

通过 `SR_BENCH_URL` 选择外部 worker 后，不会再创建本地 worker 容器。地址需要从 Dashboard 容器可达；CLI 的主机地址与 Dashboard 的容器地址可以不同，但必须指向同一服务。

```bash
# 在服务和客户端环境中私下配置相同的 SR_BENCH_TOKEN。
vllm-sr benchmark --store ./data/sr-bench serve --host 127.0.0.1 --port 8090
```

非回环监听必须设置服务 token。浏览器通过已认证的 Dashboard 访问代理，不直接访问 worker。`SR_BENCH_TOKEN_ENV` 可以指定自定义服务凭据变量名；模型使用独立的 `api_key_env`。不要把密钥值写入清单或命令参数。

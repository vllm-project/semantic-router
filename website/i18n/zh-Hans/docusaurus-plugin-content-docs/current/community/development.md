# 开发指南

本指南介绍 vLLM Semantic Router 的环境要求、初始化设置和测试流程。

## 环境要求

请确保已安装以下工具：

- **Docker**（或 Podman）
- **Make**（构建自动化）
- **Python** 3.10+（推荐，用于训练和测试）

## 快速开始

1. **克隆仓库：**

   ```bash
   git clone https://github.com/vllm-project/semantic-router.git
   cd semantic-router
   ```

2. **启动开发环境：**

   ```bash
   make vllm-sr-start
   ```

   这一个命令会完成所有操作：
   - 构建包含所有依赖的 Docker 镜像
   - 从 Hugging Face 下载所需模型
   - 安装 `vllm-sr` CLI 工具
   - 启动所有服务（semantic router、envoy、dashboard）

3. **安装 Python 依赖（可选）：**

   ```bash
   # 用于训练和开发
   pip install -r requirements.txt
   
   # 用于端到端测试
   pip install -r e2e/testing/requirements.txt
   ```

## 调试技巧

- **Rust：** 设置 `RUST_LOG=debug`
- **Go：** 设置 `SR_LOG_LEVEL=debug`

## 运行测试

### 单元测试

- **Rust bindings：**

  ```bash
  make test-binding
  ```

- **Go Router：**

  ```bash
  make test-semantic-router
  ```

- **分类器：**

  ```bash
  make test-category-classifier
  make test-pii-classifier
  make test-jailbreak-classifier
  ```

### 手动测试

使用以下命令测试特定场景：

```bash
# 模型自动选择
make test-auto-prompt-no-reasoning
make test-auto-prompt-reasoning
make test-pii          # PII 检测
make test-prompt-guard # jailbreak 检测
make test-tools        # Tools 自动选择
```

### 端到端测试

确保服务已启动，然后：

```bash
# 运行所有端到端测试
python e2e/testing/run_all_tests.py

# 运行特定测试
python e2e/testing/00-client-request-test.py
```

# 代码规范

为保持高质量代码库，我们遵循特定的代码风格指南，并使用自动化工具强制执行。

## 强制质量检查

我们使用 `pre-commit` hooks 确保一致性。**提交 PR 前必须通过这些检查。**

### 配置 Pre-commit

**安装 pre-commit：**

```bash
pip install pre-commit
# 或
brew install pre-commit
```

**安装 hooks：**

```bash
pre-commit install
```

**手动运行检查：**

```bash
pre-commit run --all-files
# 或
make precommit-local
```

## 语言规范

### Go 代码

- 使用 `gofmt` 格式化
- **命名：** 使用有意义的变量和函数名
- **注释：** 为导出的函数和类型添加文档
- **模块：** 运行 `make check-go-mod-tidy` 检查所有模块是否整洁
- **Lint：** 运行 `make go-lint` 检查问题，或 `make go-lint-fix` 自动修复

### Rust 代码

- 使用 `cargo fmt` 格式化
- 使用 `cargo clippy` 进行 lint
- 使用 `Result` 类型处理错误
- 为公共 API 编写文档

### Python 代码

- 遵循 **PEP 8** 风格指南
- 使用类型注解
- 为类和函数编写 docstrings

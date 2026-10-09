---
title: 接入自己的模型家族
sidebar_label: 接入自己的模型家族
description: 装一个小 Python 插件就能服务一种新模型，附完整可跑的示例。
translation:
  source_commit: "abae8ff99df2fdab372f0fb6d032b305907b9f44"
  source_file: "docs/model-runtime/plugins.md"
  outdated: false
is_mtpe: true
---

# 接入自己的模型家族 {#add-your-own-model-family}

runtime 是插件搭起来的。**模型家族**懂一种模型格式：怎么读模型包、怎么把请求变成模型输入、怎么把模型输出变成答案。**引擎**真正跑网络，**加速器**驱动一类硬件，**档位**定数值和批处理。内置的家族、引擎、加速器和档位，注册方式和你的一模一样，所以插件是 runtime 的一等公民。

想服务一个内置家族不认识的新模型，才需要写插件。Hugging Face ModernBERT 和 mmBERT 的分类器、嵌入模型不用——它们直接就能加载。

## 示例插件 {#the-example-plugin}

仓库里有个完整插件，小到坐下来一会儿就能读完：[`src/model-runtime/examples/third_party_plugin`](https://github.com/vllm-project/semantic-router/tree/main/src/model-runtime/examples/third_party_plugin)。它加了一个关键词模型家族和跑它的引擎，外加一个加速器和一个档位，四种插件都齐了。它的模型包是一个 JSON 文件，把标签映到关键词；家族就靠词频答 `/v1/classify`、`/v1/embeddings` 和 `/v1/rerank`。runtime 的测试套件会构建安装它的 wheel、通过 entry points 发现它，并让它在同一个进程里和一个决策模型并肩服务，每个端点上都是，用的还是示例自带的加速器和档位。

试一把：

```bash
pip install ./src/model-runtime ./src/model-runtime/examples/third_party_plugin
mkdir -p /tmp/keywords && cat > /tmp/keywords/example_model.json <<'JSON'
{"format": "vllm-sr-example/1", "labels": ["billing", "shipping", "other"],
 "keywords": {"billing": ["refund", "invoice", "charge"], "shipping": ["parcel", "delivery"]}}
JSON
vllm-srun serve /tmp/keywords --engine example_counts --device cpu --port 8100
```

```bash
curl -s localhost:8100/v1/classify -H 'content-type: application/json' \
  -d '{"input": ["Please refund the invoice", "Where is my parcel?"]}'
vllm-srun plugins
```

第一个请求对第一段文本返回 `billing`，对第二段返回 `shipping`。`vllm-srun plugins` 会在家族里列出 `example_keywords`、引擎里列出 `example_counts`、加速器里列出 `example_host`、档位里列出 `example_one_by_one`；`GET /v1/models` 还会显示它们各自的来源发行版和版本。

示例的加速器把宿主机 CPU 当成一种设备提供，它的档位让每个请求按到达顺序单独跑。名字跟内置的一样起：

```bash
vllm-srun serve /tmp/keywords --engine example_counts --device example_host --profile example_one_by_one --port 8100
```

`vllm-sr serve` 跑的是 router 镜像里的 runtime，所以它服务的插件来自一个装了这个插件的镜像。示例带一个 Dockerfile，负责把它加进 router 镜像。从仓库根目录：

```bash
docker build -t vllm-sr-example src/model-runtime/examples/third_party_plugin
vllm-sr serve /tmp/keywords --image vllm-sr-example --image-pull-policy ifnotpresent --device example_host --runtime-profile example_one_by_one --port 8100
```

容器通过只读挂载读 `/tmp/keywords`，CLI 把同样的名字传给 runtime，runtime 自己就挑中了示例的引擎。CLI 只查 `--runtime-profile` 是不是个档位名、内置加速器是不是这个镜像跑得动的。runtime 对没有插件支持的设备或档位会直接拒绝，并列出它有的那些名字。

## 写自己的插件 {#write-your-own}

插件就是一个普通的 Python 发行包。

**1. 家族。** 从 `vllm_srun.plugins.base` 继承 `ModelFamily` 和 `LoadedModel`：

| 方法 | 干什么 |
| --- | --- |
| `detect(package)` | 便宜地判断：这个目录或仓库是你的吗？读一个小文件就够，别碰权重。 |
| `verify(package)` | 读包里的文件、做检查，返回它们的身份（`model_sha256`）、上限和许可证。 |
| `describe(package)` | 告诉引擎该跑什么：骨干网络、权重文件和数值类型。 |
| `load(package, spec, engine_model)` | 返回加载好的模型及其 `ModelInfo`：它服务哪些界面、有哪些头和标签、嵌入和重排视图长什么样。 |
| `plan_surface(surface, request)` | 校验一个请求，把它变成工作项。 |
| `run(items)` | 引擎对一批工作项跑一次前向。 |
| `finish_surface(plan, results)` | 把结果变成该端点的响应体。 |

在 `surfaces` 里声明你服务哪些端点，在 `descriptor()` 里描述插件；`/v1/models` 会把它展示给客户端。

在 `/v1/decisions` 上答题的家族，改继承 `vllm_srun.plugins.decisions` 里的 `DecisionModel`（不是 `LoadedModel`）。它写 `plan`（把一个请求的若干问题变成工作项）和 `answer`（从结果里取一个问题的答案）；端点、启动自检和按问题的指标都由 `DecisionModel` 负责。

家族也可以自带模型，钉在某个 revision 上：在 `builtin_table` 里点名一个模块，用 `MODELS` 列出它们（`BuiltinModel` 条目带仓库、revision、身份和记录在案的基准答案）。runtime 之后就能按仓库 ID 服务它们；在没有别的组织的内置模型占这个名字时，也可以直接用裸模型名，并在启动时核对它们的答案。表导入失败等于什么都没钉，runtime 会记下原因。一张表钉了别的家族的表也钉了的仓库、或列了别的家族的模型，会让这个进程里的所有模型都加载不了——内置的也跑不掉——直到你消掉冲突。生成微型测试包的模块写在 `fixture_writer` 里，`vllm-srun fixture --family <name>` 就能写出一个。内置家族也是同样两套声明。

**2. 引擎，需要的话。** 多数家族在 `ModelSpec` 里描述好自己的网络，复用内置的 `native`（PyTorch）或 `onnxruntime` 引擎就够了。只有新种类的网络或新的执行库才值得写引擎（`Engine` 和 `EngineModel`：`supports`、`load`、`forward` 或 `encode`）。想让 `engine: auto` 优先试你的引擎，就设 `auto_priority`（值小的先试；内置 `native` 引擎是 0）；不设的话，`auto` 会在设有优先级的引擎都试过之后，按名字试到它。你的 `descriptor()` 要建在 `super().descriptor()` 之上——它会在模型卡上列出 `auto_priority`。如果你的 `load` 要从磁盘读权重，那就连 `read` 一起覆写：它抢在 runtime 拿下设备之前做这些宿主侧的活儿，把收尾的设备侧活儿交回去，这样设备上的其他模型在读的时候照常答。不覆写的话，整个 `load` 都是设备侧活儿。

**3. 加速器或档位，需要的话。** 新硬件就继承 `Accelerator`（`available`、`devices`、`torch_device`、`kernels`），想让 `device: auto` 可能挑中它就设 `auto_priority`；不设，只有点名这个设备的请求会用到它。新的请求合并方式，就继承 `Profile`（`plan`，以及它从模型读什么的 `bind`）。

**4. 在 `pyproject.toml` 里注册：**

```toml
[project.entry-points."vllm_srun.families"]
example_keywords = "vllm_sr_example.family:KeywordFamily"

[project.entry-points."vllm_srun.engines"]
example_counts = "vllm_sr_example.engine:CountsEngine"

[project.entry-points."vllm_srun.accelerators"]
example_host = "vllm_sr_example.accelerator:HostAccelerator"

[project.entry-points."vllm_srun.profiles"]
example_one_by_one = "vllm_sr_example.profile:OneByOneProfile"
```

组名是 `vllm_srun.families`、`vllm_srun.engines`、`vllm_srun.accelerators` 和 `vllm_srun.profiles`。名字撞了，启动时直接被拒。

**5. 让它快。** 两个开关打开 runtime 的共享优化：

- 在结果只取决于内容的工作项上设 `cache_key`，重复输入就走结果缓存；
- 一次前向能顺带服务一批打包到达的请求时，给加载好的模型设 `fuse_bundled_jobs = True`。

**6. 测它**，照示例的样子：装上发行包、在一个小模型包上起 runtime，逐个端点和 OpenAPI 契约会对（[`tests/test_third_party_plugin.py`](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/tests/test_third_party_plugin.py)）。

## 从 router 用它 {#use-it-from-the-router}

把插件装在 router 所用的 runtime 旁边，然后把一个功能绑到你模型的部署上。runtime 你自己跑的话，挂上去：

```yaml
global:
  model_catalog:
    deployments:
      ticket-topics:
        provider: model_runtime
        endpoint: http://runtime.internal:8100
        served_name: keywords
```

自定义分类器信号随后就能读它的标签；见[分类请求](./guides/classify.md#use-your-own-classifier)。router 会把绑定和模型在 `/v1/models` 里报告的标签对一遍。部署用 `device` 点名你的加速器、用 `profile` 点名你的档位，方式一样；router 只查格式，名字没有对应插件，runtime 拒绝并列出它有的一些。

## 插件的规矩 {#rules-for-plugins}

- 插件在 runtime 进程里跑。只装你信得过的插件。
- runtime 绝不执行模型包里夹带的代码。格式真需要代码，那段代码就该放在你的插件里。
- 层级要分清：家族永不导入引擎，引擎永不读模型包。正因如此，你的家族才能跑在内置引擎上，你的引擎才能服务内置家族。

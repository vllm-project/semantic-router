# APUS OpenJev 4B / 9B 原生基线：统一开发评测

本记录只使用开发集与 CSS pilot。最终保留集的 gold 未参与。执行代码为签名提交 `4513ca8` 的 `inference/apus.py`，模型固定为 [APUS 4B revision `422b3741`](https://huggingface.co/apus-ailab/APUS-OpenJev-v1-4B/tree/422b3741f8b5c092eeefef847c1ca89d78337d45) 与 [APUS 9B revision `82c9c56c`](https://huggingface.co/apus-ailab/APUS-OpenJev-v1-9B/tree/82c9c56cfa9de8d36704ed91948d4726ef111635)。两者均使用发布权重中的 `OpenJet.from_pretrained(...).decide(..., effort="high")`，即完整 32 层。HF 本地下载版本元数据、发布 manifest 中除后续更新的模型卡外的所有文件及原生代码均通过 SHA-256 验证。4B 对应训练 step 5949，9B 对应 step 3000。

## 接口覆盖与公平性

原生 [APUS runtime](https://huggingface.co/apus-ailab/APUS-OpenJev-v1-4B/blob/422b3741f8b5c092eeefef847c1ca89d78337d45/RUNTIME.md) 接收 2–16 项 Choice 与二元 Noul；`score_level` 只是单个命题的判断，无法表达完整的序数 Score。统一开发集中的 400 道 Score 因此均为 `unsupported_native_ordinal_score`，保留在 1600 题完整分母中。**APUS 不应列入全题型准确率的同能力排名**；Choice+Noul 的 1200 题可回答分母单独比较。CSS pilot 的 1430 题全是原生 Choice，覆盖率 100%。

为了传递统一题面的完整信息，JSON state 以不转义 Unicode 的紧凑 JSON 文本输入；Choice 原标签写入候选描述，因为 APUS 的原生渲染器只显示候选描述；Noul 的 true/false 描述写入指令，再使用其固定 yes/no 候选。候选顺序保留。未传入 gold 或 split 元数据。概率直接使用原生输出，未重新校准。

| DEV1600 指标 | APUS 4B | APUS 9B |
|---|---:|---:|
| 完整分母准确率 | 920/1600 = 57.50% | 877/1600 = 54.81% |
| 可回答 Choice+Noul | 920/1200 = 76.67% | 877/1200 = 73.08% |
| Choice | 683/800 = 85.38% | 696/800 = 87.00% |
| Noul | 237/400 = 59.25% | 181/400 = 45.25% |
| Score | 0/400 有效 | 0/400 有效 |
| Choice Brier | 0.1314 | 0.1067 |
| Noul Brier | 0.3213 | 0.3621 |

两模型在不同任务上交叉：4B 的 `attribute_gate` 是 99.25%，`transition_table` 是 71.5%；9B 分别是 93.0% 和 81.0%。更大的 9B Choice 总体更好，却在 Noul 规则优先级题上明显变差。Noul gold 中 true 208、false 192；下表说明两者有强烈 No 偏向，9B 不是 4B 的全面升级。

| Noul 混淆计数 | TP | FN | TN | FP | 预测 No / 400 |
|---|---:|---:|---:|---:|---:|
| APUS 4B | 64 | 144 | 174 | 18 | 318 |
| APUS 9B | 37 | 171 | 146 | 46 | 317 |

| CSS pilot1430 指标 | APUS 4B | APUS 9B |
|---|---:|---:|
| 有效题 | 1430/1430 | 1430/1430 |
| 微平均准确率 | 49.09% | 53.36% |
| 任务宏 F1 中位数 | 0.4366 | 0.5218 |
| 任务 Brier-sum 中位数 | 0.8523 | 0.6544 |
| `semeval_stance` 准确率 | 68.51% | 69.20% |
| `discourse` 准确率 | 43.86% | 54.33% |
| `implicit_hate` 准确率 | 37.35% | 38.55% |

9B 在 CSS pilot 比 4B 高 4.27 个百分点，同时 DEV Noul 低 14 个百分点。单一开发面板或某一题型不能替代这两个维度。CSS pilot 是已使用的开发材料，不是盲测结论。

## 可复现身份

- `inference/apus.py` SHA-256 `69da66f2f8a4d29c7eede6494ffb6fafe508004b47765333ecca2dbbade06974`，`inference/Dockerfile.apus-rocm` SHA-256 `93aad88c71cecc0592edc37c5cd8a505a50a802c4ff8d49fcbde4715b0b0ef8a`；collector `apus-native-openjet-v1`。
- 4B / 9B `release-manifest.json` SHA-256 分别为 `9e8263363fe13fa815c0acf0bdd7d5dd5d7da4dcf430dba8b96c7ae1d57c5f6f` / `45806c9cc88a4782e0fcb0b493b94bbdac184038ecc6ba5400688f7c8f7df3a2`。共同原生 `runtime.py` SHA-256 `6e7b0b131cb14ab0d25cc8fd6c7fc41738b799cfe6de1ccdbae2d09d7c64313c`。
- DEV prompts/gold SHA-256：`a17ec4b675bbc3da96dba8f31af8f25c9b02cc96ff048fb7de899bdd8b6cf79a` / `c7a8b86bda0d0d6120e572b94dfc756bf10264108554af76307141ae02fbf5dc`；CSS pilot prompts/gold：`598319a429de16c659b59ede0eac3c269939356b0599f4e08d1983e44def3dda` / `9a7274760dc4ced5ce5219b300974a1cf54c7d5e7e0c7de05d78bb33f2959391`。
- 4B / 9B DEV prediction SHA-256：`b2ffa5e884d17794a3d56b7564b449838be769c4ec5371051b1c52118e0ae529` / `08b78cebf4a35f6e59cc36c5fb1be7c664fb4a93d188711278fb97d6bfc82de6`；CSS pilot：`91688f2ecaa53455b1635ab41e2d1d0634a9c230358971967000913d929eaafb` / `ccce03dd51d9a4822b6a2d2fbe7b15c7b429d48502a198e61cc556467e7f6d0a`。
- DEV scorer `typed-decision-report/2`，源文件 SHA-256 `d3fe4719ee78aa004136be23799e421d6f7cf87e8a786e6131665e0e514b8af9`；CSS scorer `css-transfer-score/2`，源文件 SHA-256 `e87dc4feadd00eb651065f9e09a04f29250b9c6232b55826e844d974f6813dcd`。四份机器可读报告在任务运行目录 `runs/apus{4b,9b}-{dev,css-pilot}.report.json`；对应 SHA-256 依次为 `30fa38e5ac8103c8357272e471905288b755456ce11887aab2ca701cd66c357f`、`53681e672994cc8e8d243a0d5f69aba97b449885d3ec3686d34cbd2cf5ed7809`、`5842f7beccb7ad068827748c9f89aa2feadf9eaea85f93b05b1d12e4d25e592d`、`4846975cb8b65de40139fba25ce027e65f78ca37b5e3409f2494507d5eb5f74b`。
- 运行时：ROCm PyTorch `2.12.0+git6bbd260`、HIP `7.2.53211`、Transformers `5.16.1`、BF16。发布方验证的是 CUDA PyTorch 2.8；这里的 ROCm 数值一致性未验证，延迟不作为跨模型官方速度结论。

## 解释边界

[4B](https://huggingface.co/apus-ailab/APUS-OpenJev-v1-4B/blob/422b3741f8b5c092eeefef847c1ca89d78337d45/README.md) 与 [9B](https://huggingface.co/apus-ailab/APUS-OpenJev-v1-9B/blob/82c9c56cfa9de8d36704ed91948d4726ef111635/README.md) 卡均标注 Apache-2.0 权重；[训练说明](https://huggingface.co/apus-ailab/APUS-OpenJev-v1-9B/blob/82c9c56cfa9de8d36704ed91948d4726ef111635/training.md)列出多来源数据及不同授权制度，原始训练数据没有随权重仓库提供，因此无法证明逐样本无重叠。作者的 Frozen80 是复用开发面板，其 4B 66/80、9B 68/80 不应和本次 DEV1600 / CSS pilot 混为同一评测。发布方指出 BF16 merge 后决策对齐但概率数值不等价；上述 Brier 是直接重评当前 merged 版本，未经额外校准。

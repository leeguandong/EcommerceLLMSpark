# Spark-X2.5-4B LoRA 微调实验

基于 Spark-X2.5-4B 的文本指令微调实验记录，提供公开的训练、推理和评测入口，并同步脱敏后的结果摘要。

<p align="center">
  <a href="https://github.com/leeguandong/EcommerceLLMSpark"><img src="https://img.shields.io/badge/Project-Spark--X2.5--4B-green"></a>
  <a href="https://github.com/leeguandong/EcommerceLLMSpark/issues"><img src="https://img.shields.io/github/issues/leeguandong/EcommerceLLMSpark?color=0088ff"></a>
  <a href="https://github.com/leeguandong/EcommerceLLMSpark/pulls"><img src="https://img.shields.io/github/issues-pr/leeguandong/EcommerceLLMSpark?color=0088ff"></a>
  <a href="https://github.com/leeguandong/EcommerceLLMSpark/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-lightgrey.svg"></a>
</p>

## 本文贡献

- 使用 Spark-X2.5-4B 完成一次 SFT + LoRA 微调，并在固定留出集上比较微调前后的行为。
- 覆盖意图识别、商品信息抽取、标题生成、场景应答、搜索摘要和短视频文案等文本任务。
- 提供路径参数化的模型下载、数据转换、训练、推理、评测和预检脚本。
- 发布按任务指标、代表性安全示例和训练曲线；不发布原始数据、完整预测或模型权重。

## 模型与训练方法

| 模型权重 | 下载链接 | 微调方法 |
| :- | :- | :- |
| Spark-X2.5-4B | [ModelScope](https://www.modelscope.cn/models/XHToken/Spark-X2.5-4B) | LoRA |

- LoRA 配置：`rank=16`、`alpha=32`、`dropout=0.05`
- 对话模板：`spark`；训练和评测使用 `enable_thinking=false`
- 训练设置：1 epoch、19,555 条样本、4 × A800 80GB，训练主体约 27.24 分钟
- 训练框架：[LlamaFactory](https://github.com/hiyouga/LLaMA-Factory)

历史结果中的 `qwen1.5-1.8b`、`qwen1.5-7b`、`qwen2.5-7b`、`qwen3-8b` 和 `llama3-chinese-sft` 均表示对应模型的**微调版本**，仅作为对比背景，不在本仓库重新发布权重。

## 数据集

训练数据不随仓库发布。数据处理脚本接收用户自行准备的 JSON 或 JSONL 文件，每条记录包含：

```json
{"instruction": "任务说明", "input": "用户输入", "output": "目标回答"}
```

转换为 LlamaFactory 使用的 ShareGPT JSONL：

```bash
python scripts/prepare_data.py ./private/train.jsonl ./data/train.jsonl
python scripts/prepare_data.py ./private/validation.jsonl ./data/validation.jsonl
```

使用前请完成数据授权、脱敏、重复样本检查和质量审核。

## 快速上手

### 1. 安装环境

```bash
pip install -r requirements.txt
```

### 2. 下载模型

```bash
python scripts/download_model.py --output-dir ./models/Spark-X2.5-4B
```

### 3. 模型推理

使用 Spark-X2.5-4B：

```bash
python scripts/chat.py \
  --model-path ./models/Spark-X2.5-4B \
  --prompt "请用一句话介绍这项服务。"
```

使用 LoRA 适配器：

```bash
python scripts/chat.py \
  --model-path ./models/Spark-X2.5-4B \
  --adapter-path ./outputs/spark4b_lora_v1 \
  --prompt "请用一句话介绍这项服务。"
```

## 模型训练

训练配置见 [`configs/spark4b_lora.yaml`](configs/spark4b_lora.yaml)。路径通过命令行覆盖：

```bash
python scripts/train.py \
  --config configs/spark4b_lora.yaml \
  --model-path ./models/Spark-X2.5-4B \
  --dataset-dir ./data \
  --output-dir ./outputs/spark4b_lora_v1
```

多卡运行请根据硬件使用 LlamaFactory 或 Accelerate 的标准启动方式。仓库脚本不包含机器名、容器名、GPU 编号或内部缓存路径。

## 效果展示

固定留出集共 1,014 条，两个版本采用相同 prompt、模板、greedy 解码和任务长度预算。

| 指标 | Spark-X2.5-4B | Spark-X2.5-4B + LoRA v1 | 变化 |
| :- | -: | -: | -: |
| 意图识别集合完全匹配准确率 | 70.93% | 90.70% | +19.77 个百分点 |
| 意图识别 Micro-F1 | 82.93% | 92.51% | +9.58 个百分点 |
| 商品抽取 Micro-F1 | 75.79% | 82.62% | +6.83 个百分点 |
| 商品抽取 Micro-Precision | 66.47% | 78.82% | +12.36 个百分点 |
| 商品抽取 Micro-Recall | 88.17% | 86.80% | -1.37 个百分点 |
| 平均生成长度 | 430.0 token | 177.8 token | -252.2 token |
| 生成上限命中 | 271/1,014 | 11/1,014 | 明显减少 |

基模在商品召回、解释展开和部分文案多样性方面保留优势；LoRA 版本在意图识别、商品抽取精度、回答收敛长度以及部分场景应答的结构化程度上更好。两者没有全面胜负：长标题 6 条中，长度上限命中由 0 增至 2，重复 4-gram 均值由 30.42% 升至 55.49%；SEO 重复率由 10.56% 升至 35.68%，商品文案、直播、短视频等任务也出现不同程度的重复上升。

### 案例对比

以下案例来自固定测试集中的安全样例，仅用于观察输出行为，不代表整体质量胜率。

| 问题 | Spark-X2.5-4B | Spark-X2.5-4B + LoRA v1 | 观察 |
| :- | :- | :- | :- |
| “早上好”在西班牙语中怎么说？ | 给出 `Buenos días`，随后扩展内容较长，并出现问候语混用。 | 直接回答“可以说 `Buenos días`”。 | LoRA 版本保留核心答案，减少无关扩展。 |
| 从钢琴选购说明中识别产品词，以 JSON 列表返回。 | 将品牌、质量、音色、价格等属性或选购维度误当作商品。 | 返回钢琴、立式钢琴、三角钢琴，仍漏掉部分词项。 | LoRA 版本误抽减少，但召回仍需改进。 |
| 打网球需要哪些防晒用品？ | 覆盖防晒霜、遮阳帽、太阳镜和防晒衣，回答较长。 | 回答更短，但一项防晒用品说明出现重复。 | LoRA 版本更易收敛，同时暴露重复问题。 |
| 解释“波澜不惊的流量曲线”。 | 从多个应用场景展开，约 808 token。 | 用约 45 token 说明流量平稳、没有明显波动。 | LoRA 版本更直接，但省略部分背景。 |

逐例内容和任务标签见 [`evaluation/spark_x2_5_4b/examples.md`](evaluation/spark_x2_5_4b/examples.md)；示例不含内部标识、个人信息或未公开材料。

更多按任务指标见 [`evaluation/spark_x2_5_4b/metrics.json`](evaluation/spark_x2_5_4b/metrics.json)，安全示例见 [`examples.md`](evaluation/spark_x2_5_4b/examples.md)，训练曲线见 [`training_curves.png`](evaluation/spark_x2_5_4b/training_curves.png)。

## 评测命令

评测输入为 JSONL，每行至少包含 `id` 和 `prompt`；输出支持中断后继续：

```bash
python scripts/evaluate.py \
  --model-path ./models/Spark-X2.5-4B \
  --adapter-path ./outputs/spark4b_lora_v1 \
  --input ./data/test_prompts.jsonl \
  --output ./reports/predictions.jsonl
```

长任务运行前可先检查模型和对话模板：

```bash
python scripts/preflight.py --model-path ./models/Spark-X2.5-4B
```

## 项目致谢

1. [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory) 提供训练框架。
2. [Spark-X2.5-4B](https://www.modelscope.cn/models/XHToken/Spark-X2.5-4B) 提供基础模型。

## 免责声明

本项目仅用于研究和工程验证。参考文本重合度不等于事实正确性或文案质量；模型输出可能存在遗漏、重复或事实错误，使用者应自行审核。长标题仅 6 条，短标题和小红书文案各 5 条，样本量较小；尚未完成独立人工盲评、系统性事实准确率评估、外部通用基准或多随机种子复验，逐例核查不能替代这些评估。

## 发布边界

仓库不包含训练、验证或测试原始数据，完整预测、运行日志、checkpoint、LoRA 权重、历史实验压缩包、服务器路径、个人信息或访问凭据。公开脚本只接受用户自行提供的路径和数据。

## 使用许可

代码遵循仓库根目录 [MIT License](LICENSE)。模型权重及其许可证请以 ModelScope 原页面为准。

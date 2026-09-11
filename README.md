# Spark-X2.5-4B LoRA 微调实验

本仓库记录一次基于 **Spark-X2.5-4B** 的 LoRA 指令微调实验，并提供与参考仓库相同的公开入口：模型下载、数据格式转换、训练、推理和评测脚本。仓库中的结果是脱敏后的实验摘要，便于复核训练方法和指标变化。

## 模型与方法

- 模型：[XHToken/Spark-X2.5-4B](https://www.modelscope.cn/models/XHToken/Spark-X2.5-4B)
- 训练框架：[LlamaFactory](https://github.com/hiyouga/LLaMA-Factory)
- 方法：监督微调（SFT）+ LoRA，rank=16、alpha=32、dropout=0.05
- 对话模板：`spark`；训练和评测均使用 `enable_thinking=false`
- 本次运行：1 epoch，19,555 条训练样本，4 × A800 80GB，训练主体约 27.24 分钟

历史结果中出现的 `qwen1.5-1.8b`、`qwen1.5-7b`、`qwen2.5-7b`、`qwen3-8b` 和 `llama3-chinese-sft` 均表示对应模型的既有微调版本；本仓库不重新发布它们的权重。

## 安装

建议使用 Python 3.10+、PyTorch、Transformers、PEFT、Datasets、PyYAML 和 LlamaFactory。依赖安装方式以目标硬件和 CUDA 版本为准；示例：

```bash
pip install torch transformers peft datasets pyyaml modelscope
pip install llamafactory
```

下载模型到自定义目录：

```bash
python scripts/download_model.py --output-dir ./models/Spark-X2.5-4B
```

## 数据格式

训练数据不随仓库发布。准备一个 JSON 或 JSONL 文件，每条记录包含 `instruction`、`input`、`output` 字段，然后转换为 ShareGPT JSONL：

```bash
python scripts/prepare_data.py ./private/train.jsonl ./data/train.jsonl
```

将验证集写入 `./data/validation.jsonl`，并按 LlamaFactory 的数据集注册方式配置 `dataset` 与 `eval_dataset`。请在使用前完成脱敏、授权和质量检查。

## 训练

`configs/spark4b_lora.yaml` 保存了本次实验的关键超参数。路径通过命令行覆盖，不依赖任何特定服务器目录：

```bash
python scripts/train.py \
  --config configs/spark4b_lora.yaml \
  --model-path ./models/Spark-X2.5-4B \
  --dataset-dir ./data \
  --output-dir ./outputs/spark4b_lora_v1
```

多卡训练请根据硬件使用 LlamaFactory 或 Accelerate 的标准启动方式；训练入口不包含机器名、容器名或 GPU 编号。

## 推理与评测

单条推理：

```bash
python scripts/chat.py --model-path ./models/Spark-X2.5-4B \
  --adapter-path ./outputs/spark4b_lora_v1 \
  --prompt "请用一句话介绍这项服务。"
```

评测输入为 JSONL，每行至少包含 `id` 和 `prompt`。脚本支持中断后继续写入：

```bash
python scripts/evaluate.py --model-path ./models/Spark-X2.5-4B \
  --adapter-path ./outputs/spark4b_lora_v1 \
  --input ./data/test_prompts.jsonl \
  --output ./reports/predictions.jsonl
```

长任务运行前可先做模板和模型加载检查：

```bash
python scripts/preflight.py --model-path ./models/Spark-X2.5-4B
```

## 本次实验结果

固定留出集共 1,014 条，两个版本使用相同 prompt、模板、greedy 解码和任务长度预算。

| 指标 | Spark-X2.5-4B | Spark-X2.5-4B + LoRA v1 | 变化 |
|---|---:|---:|---:|
| 意图识别集合完全匹配准确率 | 70.93% | 90.70% | +19.77 个百分点 |
| 意图识别 Micro-F1 | 82.93% | 92.51% | +9.58 个百分点 |
| 商品抽取 Micro-F1 | 75.79% | 82.62% | +6.83 个百分点 |
| 商品抽取 Micro-Precision | 66.47% | 78.82% | +12.36 个百分点 |
| 商品抽取 Micro-Recall | 88.17% | 86.80% | -1.37 个百分点 |
| 平均生成长度 | 430.0 token | 177.8 token | -252.2 token |
| 生成上限命中 | 271/1,014 | 11/1,014 | 明显减少 |

基模在商品召回、解释展开和部分文案多样性方面保留优势；LoRA 版本在意图识别、商品抽取精度、回答收敛长度以及部分场景应答的结构化程度上更好。两者没有全面胜负：长标题 6 条中，长度上限命中由 0 增至 2，重复 4-gram 均值由 30.42% 升至 55.49%；SEO 重复率由 10.56% 升至 35.68%，商品文案、直播、短视频等任务也出现不同程度的重复上升。

完整的按任务指标、示例和训练曲线见 [`evaluation/spark_x2_5_4b/`](evaluation/spark_x2_5_4b/)。示例只展示不含内部标识或个人信息的内容。

## 发布边界与局限

本仓库不包含训练、验证或测试原始数据，完整预测、运行日志、checkpoint、LoRA 权重和历史实验压缩包也未上传。公开脚本只接受用户自行提供的路径和数据。

长标题仅 6 条，短标题和小红书文案各 5 条，样本量较小。尚未完成独立人工盲评、系统性事实准确率评估、外部通用基准或多随机种子复验；逐例核查仅用于说明行为差异，不能替代这些评估。因此，指标变化应理解为本次固定测试集上的实验观察，不应直接外推为通用能力或生产质量结论。

## 许可证

代码按仓库根目录 [LICENSE](LICENSE) 发布。模型权重及其许可证请以 ModelScope 原页面为准。

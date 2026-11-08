# 新数据管线

本项目在 `rwkv-pi` 仓库的 **`data` 分支**开发，独立工作目录为
`/home/chase/GitHub/rwkv-pi-data`。`main` 保留不含私有题目的通用代码基线；原目录的未提交工作仍在
`/home/chase/GitHub/rwkv-pi`，没有自动复制进新分支。

目标是降低每道合格新题的人工修复成本，复用已有题库和成熟的 DeepSeek 作答链。
第一版实现题库继承、版本与重复检查、审核记录、冻结题目发布包。它没有付费调用、
远程服务控制、训练入口，也不把题目计作训练轨迹。

## 已实现的流程

```text
原题库（只读） → 最终题目快照 → 独立语义审核 + 真实沙盒校准
                                      ↓ 版本哈希一致
                                冻结题目发布包
                                      ↓ 后续显式接入
                         现有 DeepSeek / Pi 作答与训练导出链
```

- 按原始字节 SHA-256 保存完整题目 bundle；保留作者、来源、ID。同 ID 修订不增加题目数。
- 多个修订共存时必须明确选择版本，不按修改时间自动覆盖。语义审核与校准都绑定同一哈希。
- 检查机制、场景、完整题面与初始文件的精确重复。大小写、全半角和空白归一化。
  这些检查不能发现全部语义近似题；仍需独立判断任务机制是否真的不同。
- 只读取评测保留集的 family 清单进行排除，不导入评测题、答案或验收内容。
- 新清单只表示“已收录”。不会把旧 bundle 的 `review` 或作者自己的通过声明当成独立审批。
  新项目的状态也不会修改原项目已经完成的审核、轨迹与训练数据状态。
- 旧题用“在 repo 目录创建 ops.sh”表达 `repo/ops.sh`，或把标准输出列作交付物时，保留原文并标出
  `deliverablesNotLiteralInPrompt` 供语义审核，不用字面路径匹配代替判断。
- 发布检查完整的 baseline/reference/每个 wrong control 执行结果。超时/信号退出不能充当错误解被拒证据。
- 发布包兼容现有 `newBundles` / `allowedAuthors` / `bundles/*.json` 布局。
  `authorizedToStart=false`，需要后续明确接入既有 admission；不同作者可能还需既有 admission 的来源适配。
  本模块不导入原工作区未提交的运行时代码，也不重写 teacher 或 XML 渲染器。
- 发布包报告数据管理＋Agent 调度与其余八领域的实际比例，目标为 **1:2**；不自动补齐、不伪造达标。

## 使用

仅使用 Python 标准库及仓库现有 `pipeline/datakit.py`。从此工作目录执行：

```bash
python3 -B -m data_pipeline inventory \
  --source claude=/mnt/f/rwkv-pi-data/datasets/private-claude/inbox \
  --source deepseek=/mnt/f/rwkv-pi-data/datasets/private-deepseek/bundles \
  --heldout-manifest /mnt/f/rwkv-pi-data/datasets/eval-holdout-v1/manifest.json \
  --out /mnt/f/rwkv-pi-data/datasets/data-pipeline-v1/inventory-NEW

python3 -B -m data_pipeline status \
  /mnt/f/rwkv-pi-data/datasets/data-pipeline-v1/inventory-NEW/catalog.json

python3 -B -m data_pipeline review-template \
  --catalog /mnt/f/rwkv-pi-data/datasets/data-pipeline-v1/inventory-NEW/catalog.json \
  --id QUESTION_ID \
  --out /mnt/f/rwkv-pi-data/datasets/data-pipeline-v1/reviews/batch-NEW.json
```

审核者逐题阅读题面、输入文件、参考控制和检查源码，填写 `reviewer`、具体 `notes`、
六项审核结果及 `verdict`。`calibrationReceipt` 使用既有 82 校准器产生的单题 JSON
绝对路径，`calibrationSha256` 为该文件实际 SHA-256；它必须包含相同题目 ID、bundle 哈希、
执行主机、baseline/reference/all wrong controls 的结果。不能用人工编写的模拟 receipt
冒充真实执行。这里验证证据一致性，不是对恶意伪造证据的鉴定系统。

```bash
python3 -B -m data_pipeline prepare-release \
  --catalog /mnt/f/rwkv-pi-data/datasets/data-pipeline-v1/inventory-NEW/catalog.json \
  --reviews /mnt/f/rwkv-pi-data/datasets/data-pipeline-v1/reviews/batch-NEW.json \
  --out /mnt/f/rwkv-pi-data/datasets/data-pipeline-v1/releases/batch-NEW

python3 -B -m unittest data_pipeline.test_pipeline -v
```

源目录只读取直接子级 `*.json` 最终 bundle，不递归扫描作者请求目录。
所有输出必须在已挂载的 F: 上。快照、审核文件和发布目录拒绝隐式覆盖；测试使用 F: 临时目录，
测试 fixture 不进入真实清单或训练数据。任何不完整清单都会记录错误，并阻止发布。

## 接下来的出题改造

1. **先定义任务机制与验收契约。** 继承已审查的环境工具，按 L1/L2 为主设计不同的实际任务。
   多步是一次中文请求后的真实工具观察循环，不编造用户轮次，也不凑工具次数。
2. **缩小 API 作者职责。** 模型输出结构化任务设计和必要素材；受审查的本地构建器负责
   环境初始化与固定业务规则。独立的边界/错误控制用于发现检查器漏洞。
   第一版还没有实现该构建器，也没有证明这种方法能稳定生成五千道不同题目。
3. **接入现有校准和作答链。** 82 执行环境，DeepSeek Flash 单一作答教师，完整 Pi system prompt、
   中文问题与约束、高思考、`zh-reasoning-v2`、XML+JSON 连续轨迹及独立验收保持固定。
   待显式接入后才运行模型，最后继续使用既有原生 tokenizer/renderer 打包。
4. **衡量合格产出。** 记录首次语义通过率、人工修复分钟数、每道合格题的费用与每小时
   通过轨迹数。API 能调用、生成 JSON、控制校准通过，都不能替代这些指标。

既有私有预算账本仍有效，包含既往消耗和未知请求保留额。本模块不重新授权支出。

## 数据保密

GitHub 只保存通用管线代码和不来自题库的工程测试。题目、生成器、参考答案、验收内容、
轨迹、清单、报告与历史备份都放在私有 F: 存储，不能提交到 GitHub，包括私有仓库。
确认训练有效后，由用户决定向 Hugging Face 发布哪些数据；没有自动上传入口。

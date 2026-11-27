# RWKV-Pi 本地部署指南

在一张 NVIDIA 显卡上运行 RWKV-7 7.2B 编程 Agent，用 `pi` 直接调用。Linux 和 Windows 都只需一条命令。

**准备**：NVIDIA 显卡（显存 ≥ 12 GB，驱动支持 CUDA 12.9，无需 CUDA Toolkit），磁盘 ≥ 20 GB。

脚本会自动补齐 Node.js / Git 等依赖，按 GPU 0 的**空闲显存**选择权重，下载引擎、模型与 Demo，并注册 `pi`、`rwkv-engine`、`rwkv-demos` 三个命令。

| 空闲显存 | 权重 | 下载 | 显存占用 |
| :--- | :--- | :---: | :---: |
| < 16 GB | `RWKV7-Pi-7.2B-v0.2.w8a16.rwkvq` | 7.2 GB | ≈ 8.5 GB |
| ≥ 16 GB | `RWKV7-Pi-7.2B-v0.2.pth`（BF16） | 13.4 GB | ≈ 14.2 GB |

> [!TIP]
> 安装前先关闭游戏、浏览器等占用显存的程序，避免误判。

> [!NOTE]
> W8A16 量化版相比 BF16 原版存在一定能力损失，复杂任务的成功率会略低。显存允许时建议使用 BF16（`--precision bf16` / `RWKV_PRECISION=bf16`）。

---

## 一、安装

### Linux / WSL2

```bash
curl -fsSL https://hf-mirror.com/RWKV-Agent-Lab/RWKV7-Agent-7.2B-OpenSource/resolve/main/quickstart-linux.sh | bash
```

装完后新开一个终端（或 `source ~/.bashrc`）。

<details>
<summary>可选参数</summary>

参数写在 `bash -s --` 之后，例如 `... | bash -s -- --dir /data/rwkv-agent --precision bf16`。

| 参数 | 作用 |
| :--- | :--- |
| `--dir <路径>` | 安装目录，默认 `~/rwkv-agent` |
| `--precision w8a16\|bf16` | 手动指定权重 |
| `--deepseek-key <KEY>` | 安装时写入 DeepSeek Key |
| `--cuda 13.2` | 使用 CUDA 13.2 版引擎 |
| `--official-hf` | 海外网络，改用官方源 |

</details>

### Windows 10 / 11（PowerShell）

```powershell
curl.exe -fsSL https://hf-mirror.com/RWKV-Agent-Lab/RWKV7-Agent-7.2B-OpenSource/resolve/main/quickstart-windows.ps1 | Out-String | iex
```

缺少的依赖会通过 `winget` 自动安装（可能弹出一次授权），装完在当前窗口即可使用。

<details>
<summary>可选参数</summary>

在命令前设置环境变量，例如 `$env:RWKV_AGENT_DIR="D:\rwkv-agent"; curl.exe ...`。

| 环境变量 | 作用 |
| :--- | :--- |
| `RWKV_AGENT_DIR` | 安装目录，默认 `$HOME\rwkv-agent` |
| `RWKV_PRECISION` | `w8a16` 或 `bf16` |
| `DEEPSEEK_API_KEY` | 安装时写入 DeepSeek Key |
| `RWKV_CUDA_VER` | 设为 `13.2` 使用 CUDA 13.2 版引擎 |
| `HF_ENDPOINT` | 海外网络设为 `https://huggingface.co` |

</details>

---

## 二、使用

在任意项目目录运行 `pi` 即可，首次运行会在后台自动启动推理引擎。先跑一个内置 Demo 试试：

```bash
cd ~/rwkv-agent/demos/06-textindex        # Windows：cd $HOME\rwkv-agent\demos\06-textindex
pi
```

输入：

```text
QA 报了一个回归问题：运行 python3 -m unittest discover -s tests 时有 5 个搜索测试失败——任何非空词查询实际都返回了空列表 []。请读取 textindex/index.py 后半段的 search() 函数（第 195–230 行）定位根本原因，用最小改动修复源代码（不要修改测试），运行 python3 -m unittest discover -s tests 确认全部 30 个测试通过并简要总结。
```

**常用命令**（Linux / Windows 相同）：

| 命令 | 作用 |
| :--- | :--- |
| `pi` | 在当前目录启动 Agent |
| `rwkv-engine status` / `logs` | 查看引擎状态 / 日志 |
| `rwkv-engine stop` / `restart` | 停止（释放显存）/ 重启引擎 |
| `rwkv-demos list` / `reset` | 列出 / 还原内置 Demo |

---

## 三、内置 Demo

位于安装目录的 `demos/`，全部题目见 `demos/PROMPTS.zh-CN.md`，测完执行 `rwkv-demos reset` 还原。

| 目录 | 场景 | 难度 |
| :--- | :--- | :---: |
| `06-textindex` | 倒排索引回归修复 | 推荐入门 |
| `01-portal-security` | 两轮安全审计与加固 | 中等 |
| `02-nginx-gateway` | Nginx 配置漂移排障 | 中等 |
| `04-python-web-research` | 联网核验官方文档 | 需 DeepSeek Key |
| `05-routemap` | 13 文件跨模块排障 | 建议配合 `/rwkv-subagent` |
| `03-warehouse-reconcile` | CSV + SQLite 对账 | 挑战 |
| `07-cachebox-git` | Git 历史定位回归 | 挑战 |

> [!NOTE]
> 7B 本地模型不保证每次一次成功，没过时可补充提示，或 `rwkv-demos reset` 后重试。

---

## 四、DeepSeek API Key（可选）

纯本地写代码、修 Bug **无需配置**。以下两项功能需要 [DeepSeek 官方 API Key](https://platform.deepseek.com/api_keys)：

- **联网检索**：`web_search` / `web_fetch`
- **强弱协同**：在 `pi` 中输入 `/rwkv-subagent`，由 DeepSeek 规划审查、多个本地子 Agent 并发写代码；`/rwkv-subagent cost` 查看省钱账单

一条命令同时开启两项功能：

```text
rwkv-engine set-deepseek <你的 DeepSeek API Key>
```

---

## 五、常见问题

- **开着代理连不上本地引擎**：`pi` 已自动直连 `127.0.0.1`；若开启了 TUN 全局模式，请在规则中放行 `127.0.0.1`。
- **Windows 首次加载模型偏慢**：在 Windows 安全中心的排除项中加入安装目录。
- **Linux 未安装 `bubblewrap`**：不影响使用，子 Agent 会改用工作区副本隔离。
- **提高子 Agent 并发**：把安装目录下 `engine.conf` 的 `DECODE_BATCH`（Windows 为 `engine.json` 的 `decodeBatch`）改为 `8` 或 `16`，再 `rwkv-engine restart`。

---

## 六、进阶：训练与量化

本文只覆盖部署。训练需使用 [Hugging Face 仓库](https://huggingface.co/RWKV-Agent-Lab/RWKV7-Agent-7.2B-OpenSource) 中的 BF16 权重：在编程 Agent 上继续训练选 `RWKV7-Pi-7.2B-v0.2.pth`，从通用底座出发选 `RWKV7-Agent-7.2B-v0.1.pth`。训练、合并、量化工具已随引擎装在 `bundle/` 下，用法参见：

- [rwkv_lightning_cuda](https://github.com/Alic-Li/rwkv_lightning_cuda)：引擎与工具总览（[NoRA](https://github.com/Alic-Li/rwkv_lightning_cuda/blob/main/src/nora/README.md) / [MiSS](https://github.com/Alic-Li/rwkv_lightning_cuda/blob/main/src/miss/README.zh-CN.md) 训练、[适配器合并](https://github.com/Alic-Li/rwkv_lightning_cuda/blob/main/docs/peft-weight-merge.zh-CN.md)）
- [rwkv-pi](https://github.com/w1c2j3/rwkv-pi)：提示模板与数据管线（训练数据需与 `pi` 的提示模板一致）

量化：`bundle/rwkv_quantize --format w8a16 <输入.pth> <输出.rwkvq>`（量化后会有一定能力损失）。换模型后修改 `engine.conf` / `engine.json` 中的模型路径，再 `rwkv-engine restart`。

> [!NOTE]
> 仓库中保留了旧版 `RWKV7-Pi-7.2B-v0.1-final`，它使用旧编辑协议，需在 `.env` 中设置 `RWKV_EDIT_MODE=text`。

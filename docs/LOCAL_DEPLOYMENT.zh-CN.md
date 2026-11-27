# RWKV-Pi 本地部署指南

在一张 NVIDIA 显卡上本地运行 RWKV-7 7.2B 编程 Agent，并在 `pi` 中直接调用。**Linux 和 Windows 都只需要一条命令。**

| 你的系统 | 跳转 |
| :--- | :--- |
| Ubuntu / Debian / WSL2 | [一、Linux / WSL2](#一linux--wsl2) |
| Windows 10 / 11 | [二、Windows](#二windows) |

## 准备工作

- **显卡**：NVIDIA 显卡，显存 ≥ 12 GB；驱动支持 CUDA 12.9 即可，**不需要安装 CUDA Toolkit**。
- **磁盘**：预留 20 GB 以上。

安装脚本会自动完成以下工作：

1. 读取 GPU 0 的空闲显存，自动选择权重；
2. 补齐 Node.js、Git 等依赖；
3. 下载推理引擎、模型权重和 7 个 Demo 项目；
4. 注册 `pi`、`rwkv-engine`、`rwkv-demos` 三个命令。

| 空闲显存 | 自动选择的权重 | 下载体积 | 运行占用 |
| :--- | :--- | :---: | :---: |
| 低于 16 GB | `RWKV7-Pi-7.2B-v0.2.w8a16.rwkvq`（W8A16 量化） | 7.2 GB | 约 8.5 GB |
| 16 GB 及以上 | `RWKV7-Pi-7.2B-v0.2.pth`（BF16 原始权重） | 13.4 GB | 约 14.2 GB |

> [!TIP]
> 安装前请先关闭游戏、浏览器等占用显存的程序，否则可能被误判为显存不足。空闲显存不够时脚本会给出提醒；也可以用参数手动指定权重。

---

## 一、Linux / WSL2

适用于 Ubuntu 22.04+、Debian 12+，以及 Windows 下的 WSL2（Ubuntu）。

### 1. 安装

```bash
curl -fsSL https://hf-mirror.com/RWKV-Agent-Lab/RWKV7-Agent-7.2B-OpenSource/resolve/main/quickstart-linux.sh | bash
```

安装完成后，**新开一个终端**（或执行 `source ~/.bashrc`）。

### 2. 开始使用

```bash
cd ~/rwkv-agent/demos/06-textindex
pi
```

在任意项目目录输入 `pi` 都可以使用。第一次运行时会在后台自动启动推理引擎。

### 3. 可选参数

参数写在 `bash -s --` 后面，例如：

```bash
curl -fsSL https://hf-mirror.com/RWKV-Agent-Lab/RWKV7-Agent-7.2B-OpenSource/resolve/main/quickstart-linux.sh | bash -s -- --dir /data/rwkv-agent --precision bf16
```

| 参数 | 作用 |
| :--- | :--- |
| `--dir <路径>` | 安装目录，默认 `~/rwkv-agent` |
| `--precision w8a16\|bf16` | 手动指定权重，默认按显存自动选择 |
| `--deepseek-key <KEY>` | 安装时一并写入 DeepSeek Key |
| `--cuda 13.2` | 改用 CUDA 13.2 版引擎 |
| `--official-hf` | 海外网络：改用 huggingface.co 和 nodejs.org |
| `--install-only` | 只安装，暂不启动引擎 |

### 4. 常见问题

- **请求本地引擎返回 `502`**：终端设置了 `http_proxy`。`pi` 已自动直连 `127.0.0.1`；如果你自己用 `curl` 测试，请加上 `--noproxy "*"`。
- **没有安装 `bubblewrap`**：不影响使用，`/rwkv-subagent` 会自动改用工作区副本隔离。想用沙盒隔离可以执行 `sudo apt install bubblewrap`。

---

## 二、Windows

适用于 Windows 10（1809+）/ Windows 11 64 位，不需要 WSL，也不需要 Visual Studio。

### 1. 安装

按 `Win + X` 打开「终端」或「PowerShell」，执行：

```powershell
curl.exe -fsSL https://hf-mirror.com/RWKV-Agent-Lab/RWKV7-Agent-7.2B-OpenSource/resolve/main/quickstart-windows.ps1 | Out-String | iex
```

缺少 Node.js、Git 或 Python 时，脚本会通过 `winget` 自动安装（可能弹出一次管理员授权）。**装完后直接在当前窗口使用即可，不用重开。**

### 2. 开始使用

```powershell
cd $HOME\rwkv-agent\demos\06-textindex
pi
```

在任意项目目录输入 `pi` 都可以使用，PowerShell 和 CMD 均可。第一次运行时会在后台自动启动推理引擎，不弹窗口。

### 3. 可选参数

想指定安装盘符或模型精度时，在命令前加上环境变量即可：

```powershell
$env:RWKV_AGENT_DIR="D:\rwkv-agent"; $env:RWKV_PRECISION="bf16"; curl.exe -fsSL https://hf-mirror.com/RWKV-Agent-Lab/RWKV7-Agent-7.2B-OpenSource/resolve/main/quickstart-windows.ps1 | Out-String | iex
```

| 环境变量（或脚本参数） | 作用 |
| :--- | :--- |
| `RWKV_AGENT_DIR`（`-InstallDir`） | 安装目录，默认 `$HOME\rwkv-agent` |
| `RWKV_PRECISION`（`-Precision`） | `w8a16` 或 `bf16`，默认按空闲显存自动选择 |
| `DEEPSEEK_API_KEY`（`-DeepSeekKey`） | 安装时一并写入 DeepSeek Key |
| `RWKV_CUDA_VER`（`-Cuda`） | 设为 `13.2` 可改用 CUDA 13.2 版引擎 |
| `HF_ENDPOINT`（`-HfEndpoint`） | 海外网络可设为 `https://huggingface.co` |

### 4. 常见问题

- **开着 Clash / v2rayN 时连不上本地引擎**：`pi` 已自动直连 `127.0.0.1`。如果开启了 TUN 全局模式，请在代理规则中放行 `127.0.0.1`。
- **首次加载模型偏慢**：在「Windows 安全中心 → 病毒和威胁防护 → 排除项」中加入安装目录。

---

## 三、配置 DeepSeek API Key（可选）

只在本地读写代码、修 Bug、跑测试时**不需要配置**，完全零费用。以下两项功能需要 [DeepSeek 开放平台](https://platform.deepseek.com/api_keys) 的官方 API Key：

| 功能 | 用途 |
| :--- | :--- |
| 联网检索 `web_search` / `web_fetch` | 通过 DeepSeek 官方检索接口查资料、读网页 |
| `/rwkv-subagent` 强弱协同 | DeepSeek 负责规划和审查，多个本地 RWKV 子 Agent 并发写代码 |

在终端执行下面这一条命令即可，Linux 和 Windows 写法相同，两项功能会同时生效：

```text
rwkv-engine set-deepseek <你的 DeepSeek API Key>
```

也可以直接编辑配置文件，填写 `DEEPSEEK_API_KEY=` 一行：

- Linux：`~/rwkv-agent/rwkv-pi/.env`
- Windows：`$HOME\rwkv-agent\rwkv-pi\.env`

---

## 四、常用命令

以下命令在 Linux 和 Windows 上完全一样：

| 命令 | 作用 |
| :--- | :--- |
| `pi` | 在当前目录启动编程 Agent |
| `rwkv-engine status` | 查看推理引擎状态 |
| `rwkv-engine logs` | 查看引擎日志 |
| `rwkv-engine stop` | 停止引擎，释放显存 |
| `rwkv-engine restart` | 重启引擎 |
| `rwkv-demos list` | 列出内置 Demo |
| `rwkv-demos reset` | 把所有 Demo 恢复到初始状态 |

---

## 五、内置 Demo 与实测

Demo 位于安装目录下的 `demos/`，完整题目见 `demos/PROMPTS.zh-CN.md`。测完后执行 `rwkv-demos reset` 即可恢复原样，可以反复体验。

下表中的「本地实测」是 `v0.2` W8A16 只用本地模型、不开子 Agent 时的通过次数（通过次数 / 运行次数）：

| 目录 | 场景 | 本地实测 | 是否需要 DeepSeek Key |
| :--- | :--- | :---: | :---: |
| `06-textindex` | 倒排索引 `search()` 回归修复（30 项测试） | **4/5**，成功时约 20 秒 | 否 |
| `01-portal-security` | 两轮安全审计：先只读定位漏洞，再修复并跑通测试 | 2/3（第 2 轮修复） | 否 |
| `02-nginx-gateway` | 排查 Nginx 生产配置漂移 | 2/4 | 否 |
| `04-python-web-research` | 联网核验 Python 官方文档 | — | **是** |
| `05-routemap` | 13 个文件的跨模块排障（33 项测试） | 0/3，建议配合 `/rwkv-subagent` | 推荐配置 |
| `03-warehouse-reconcile` | CSV 数据清洗 + SQLite 对账报表 | 0/3，挑战题 | 否 |
| `07-cachebox-git` | 翻 Git 历史定位引入回归的提交 | 0/3，挑战题 | 否 |

> [!NOTE]
> 7B 本地模型不保证每次都一次成功。没过时可以在对话里补充提示，或者执行 `rwkv-demos reset` 后重试。

### 实测 1：回归定位修复（纯本地，推荐先跑）

进入 `demos/06-textindex` 目录运行 `pi`，输入：

```text
QA 报了一个回归问题：运行 python3 -m unittest discover -s tests 时有 5 个搜索测试失败——任何非空词查询实际都返回了空列表 []。请读取 textindex/index.py 后半段的 search() 函数（第 195–230 行）定位根本原因，用最小改动修复源代码（不要修改测试），运行 python3 -m unittest discover -s tests 确认全部 30 个测试通过并简要总结。
```

### 实测 2：两轮安全审计（纯本地）

进入 `demos/01-portal-security` 目录运行 `pi`，依次输入下面两轮指令：

```text
第 1 轮（只读安全与架构审计，请勿修改任何文件）：请扫描 app/ 目录，梳理 PortalService 的模块调用关系，并精确定位其中存在的 2 处高危安全漏洞（指出具体文件路径、函数名、行号与原理）。
```

```text
第 2 轮（安全加固修复）：现在请修复 app/db.py 的 SQL 注入漏洞（改为参数化占位符查询）与 app/storage.py 的目录穿越漏洞（校验解析后的路径严格位于 base_dir.resolve() 内部，越界抛 ValueError），并运行 python3 -m unittest discover -s tests 确保测试全部通过。
```

### 实测 3：联网检索（需要 DeepSeek Key）

进入 `demos/04-python-web-research` 目录运行 `pi`，输入：

```text
请使用 web_search 与 web_fetch 查阅 Python 3 官方标准库文档（docs.python.org），准确对比 json 与 tomllib 两个模块：1) 哪个模块支持序列化写入（dump/dumps），哪个是纯只读解析？2) 各自的 load(fp) 函数对文件对象的打开模式（文本模式 'r' 还是二进制模式 'rb'）分别有什么强制要求？每项结论请附上核验过的官方文档 URL。
```

### 实测 4：`/rwkv-subagent` 强弱协同（需要 DeepSeek Key）

进入 `demos/05-routemap` 目录运行 `pi`：

1. 输入 `/rwkv-subagent` 开启协同模式；
2. 粘贴下面的任务：

   ```text
   这个仓库的夜间构建失败了。python3 -m unittest discover -s tests 报告有 6 个测试失败，例如 test_avoid_accepts_repeated_flags_and_multiple_values, test_end_to_end_lifecycle_exit_codes, test_rejects_non_positive_and_bool_costs, test_route_prints_only_path_and_cost。根本原因可能不止一个。请排查并修复源代码（不要修改或删除测试），并确认整个测试套件全部通过，完成后简要总结。
   ```

3. 测试全部通过后，输入 `/rwkv-subagent cost` 查看省钱账单；输入 `/rwkv-subagent off` 可切回纯本地模式。

> [!TIP]
> 想提高子 Agent 的并发数，可以修改安装目录下的引擎配置：Linux 改 `engine.conf` 里的 `DECODE_BATCH`，Windows 改 `engine.json` 里的 `decodeBatch`，设为 `8` 或 `16`，然后执行 `rwkv-engine restart`。

---

## 六、进阶：训练与量化（参考链接）

本文只介绍部署。如果你想在这些权重上继续退火训练或自己量化，请按下面的入口查阅官方文档。

**基模权重**：训练必须使用 BF16 `.pth`，在 [Hugging Face 仓库](https://huggingface.co/RWKV-Agent-Lab/RWKV7-Agent-7.2B-OpenSource) 下载：

| 你的目标 | 下载哪个文件 |
| :--- | :--- |
| 在 Pi 编程 Agent 上继续训练（推荐） | `RWKV7-Pi-7.2B-v0.2.pth` |
| 从通用 Agent 底座训练自己的方向 | `RWKV7-Agent-7.2B-v0.1.pth` |

**参考文档**：训练、合并和量化工具都已包含在一键安装的引擎包里（安装目录下的 `bundle/`），用法见：

| 文档 | 内容 |
| :--- | :--- |
| [rwkv_lightning_cuda 仓库](https://github.com/Alic-Li/rwkv_lightning_cuda) | 推理引擎总览与各工具说明 |
| [NoRA 训练](https://github.com/Alic-Li/rwkv_lightning_cuda/blob/main/src/nora/README.md) / [MiSS 训练](https://github.com/Alic-Li/rwkv_lightning_cuda/blob/main/src/miss/README.zh-CN.md) | 低秩适配器训练、数据格式与参数 |
| [适配器合并](https://github.com/Alic-Li/rwkv_lightning_cuda/blob/main/docs/peft-weight-merge.zh-CN.md) | 把训练结果合并回 `.pth` |
| [rwkv-pi 仓库](https://github.com/w1c2j3/rwkv-pi) | 提示模板与数据管线（训练数据需与 `pi` 的提示模板一致） |

量化可以直接运行 `bundle/rwkv_quantize --format w8a16 <输入.pth> <输出.rwkvq>`（Windows 上是 `rwkv_quantize.exe`）。换上新模型后，把 `engine.conf`（Linux）或 `engine.json`（Windows）里的模型路径改为新文件，再执行 `rwkv-engine restart`。

> [!NOTE]
> HF 仓库里仍保留旧版 `RWKV7-Pi-7.2B-v0.1-final`。它使用旧的编辑协议，如果要用它，需要在 `.env` 中改为 `RWKV_EDIT_MODE=text`。新安装默认使用 `v0.2`，不需要改动。

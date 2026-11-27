# RWKV-Pi 本地极速部署与调用测试指南（Linux 篇 / Windows 篇独立分章）

本文档面向希望在本地单张消费级显卡（如 **16GB RTX 4060 Ti / 4070 Ti Super / 5070 Ti**）或工作站显卡（**24GB+**）上部署 **RWKV-7 7.2B Coding Agent** 并在 **`pi`** 中直接调用的开发者。

为避免不同操作系统的命令混淆，本文按操作系统**上下彻底分章**，请直接点击你的系统进入对应完整流程：

- 👉 **[第一篇：Linux / WSL2 端完整部署与调用流程](#第一篇linux--wsl2-端完整部署与调用流程)**（纯 Bash 命令，含 `.sh` 一键脚本 + DeepSeek API 配置 + 内置真实靶场实测）
- 👉 **[第二篇：原生 Windows 端完整部署与调用流程](#第二篇原生-windows-端完整部署与调用流程)**（纯 PowerShell / CMD 命令，含 `.ps1` 一键脚本 + DeepSeek API 配置 + 内置真实靶场实测）

---

## 零、通用显存选型与内置真实 Demo 靶场概览

### 1. 显卡显存与模型权重对照

| 显卡显存规格 | 典型显卡型号 | 推荐权重文件 | 磁盘体积 | 推理常驻显存 | 一键脚本参数 |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **12 GB – 16 GB（主力甜点）** | RTX 4060 Ti 16G / 4070 TiS / 5070 Ti 16G | **`RWKV7-Pi-7.2B-v0.1-final.w8a16.rwkvq`** | `~7.2 GB` | `~7.8 – 8.5 GB` | 默认值（无需传参） |
| **24 GB 及以上** | RTX 3090 / 4090 / 5090 / A100 | **`RWKV7-Pi-7.2B-v0.1-final.pth`** | `13.4 GiB` | `~14.2 GB` | Linux: `--precision bf16`<br>Win: `-Precision bf16` |
| **8 GB – 10 GB** | RTX 3070 / 4060 8G | 自行导出 `SB6 W6 / SB6 W4 (.rwkvq)` | `~4.3 – 6.0 GB` | `~5.2 – 6.8 GB` | 见各篇「手动分步与自定义量化」 |

- **Hugging Face 官方开源仓**：[`RWKV-Agent-Lab/RWKV7-Agent-7.2B-OpenSource`](https://huggingface.co/RWKV-Agent-Lab/RWKV7-Agent-7.2B-OpenSource)（国内镜像：[`hf-mirror.com/RWKV-Agent-Lab/RWKV7-Agent-7.2B-OpenSource`](https://hf-mirror.com/RWKV-Agent-Lab/RWKV7-Agent-7.2B-OpenSource)）
- **显卡驱动要求**：支持 **CUDA 12.9**（默认）或 **CUDA 13.2** 的 NVIDIA 官方显卡驱动（使用官方预编译引擎包，**无需安装 CUDA Toolkit**）。

### 2. 随脚本自动内置的 7 组真实评测靶场（`demos/`）

一键安装脚本会自动将我们实测与录制演示视频使用的 **7 个真实多文件项目靶场**解压到安装目录下的 **`demos/`**，并生成 **`rwkv-demos reset`** 一键还原命令，方便随时反复实测：

| 靶场子目录 | 对应能力与演示场景 | 是否需要外部 API |
| :--- | :--- | :---: |
| `demos/01-portal-security` | **两轮安全审计与加固**（第 1 轮只读查出 SQL 注入与目录穿越，第 2 轮修复跑通单测） | 否（100% 纯本地） |
| `demos/02-nginx-gateway` | **Nginx 反代与生产配置漂移排障**（跨 4 个配置文件对齐端口/健康检查/超时并跑通巡检） | 否（100% 纯本地） |
| `demos/03-warehouse-reconcile` | **CSV 脏数据清洗 + SQLite 多表关联区域对账报表** | 否（100% 纯本地） |
| `demos/04-python-web-research` | **原生联网双文档对比**（查阅 Python 官方文档核验 `json` vs `tomllib` 读写与打开模式） | 需配 `DEEPSEEK_API_KEY` |
| `demos/05-routemap` | **13 文件路由规划库跨模块排障（33 测全绿）** · 推荐体验 **`/rwkv-subagent` 并发省钱账单** | 纯本地或配 `DEEPSEEK_API_KEY` 开子 Agent |
| `demos/06-textindex` | **12 文件 Python 倒排索引库 `search()` 回归修复（30 测全绿）** | 否（100% 纯本地） |
| `demos/07-cachebox-git` | **Git 历史考古定位回归 Commit 并提交无损修复** | 否（100% 纯本地） |

---
---

# 第一篇：Linux / WSL2 端完整部署与调用流程

> **适用环境**：Ubuntu 22.04 / 24.04（glibc `>= 2.35`）、Debian 12+，或 Windows 10/11 下的 **WSL2 (Ubuntu)**。本篇所有命令均为纯 Linux Bash 命令。

## 1.1 前置环境准备

```bash
# 1) 安装基础工具与 bubblewrap（bubblewrap 用于 /rwkv-subagent 并发子 Agent 沙盒隔离；aria2 用于 16 线程加速下载）
sudo apt update && sudo apt install -y curl git tar aria2 bubblewrap python3

# 2) 确认 Node.js 版本 >= 22.19.0
node -v
# 若未安装或版本低于 22.19，可用 nvm 一键安装 Node 22：
# curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash && source ~/.bashrc && nvm install 22
```

---

## 1.2 一键极速部署（自动下引擎 + 下权重 + 装 `demos/` 靶场 + 注册全局 `pi`）

### 1. 下载并运行 Linux 一键脚本

**12GB–16GB 显卡（默认安装到 `~/rwkv-agent`，下载 `7.2 GB` W8A16 预量化包）：**
```bash
curl -fsSL https://hf-mirror.com/RWKV-Agent-Lab/RWKV7-Agent-7.2B-OpenSource/resolve/main/quickstart-linux.sh -o quickstart-linux.sh
bash quickstart-linux.sh
```

**24GB+ 显卡（下载 `13.4 GiB` 原生 BF16 `.pth` 权重）：**
```bash
curl -fsSL https://hf-mirror.com/RWKV-Agent-Lab/RWKV7-Agent-7.2B-OpenSource/resolve/main/quickstart-linux.sh -o quickstart-linux.sh
bash quickstart-linux.sh --precision bf16
```

> **可选参数**：`--dir /data/rwkv-agent`（自定义安装路径）、`--deepseek-key sk-xxxx`（安装时顺手写入 DeepSeek Key）、`--cuda 13.2`、`--official-hf`（海外源）、`--port 8000`。

### 2. 刷新当前终端环境变量（新开终端自动生效）

```bash
export PATH="$HOME/rwkv-agent/bin:$PATH"
```

安装完成后获得 3 个全局命令：
- **`pi`**：在**任意目录**直接输入 `pi` 即可编程（若后台推理引擎未启动，会自动后台拉起引擎、自动对齐模型 ID、自动绕过本地代理并进入交互界面）。
- **`rwkv-engine {status|logs|stop|restart|set-deepseek}`**：随时查看推理状态或执行 `rwkv-engine stop` 一键释放全部 GPU 显存。
- **`rwkv-demos {list|reset}`**：查看或一键重置 `~/rwkv-agent/demos/` 下的 7 组真实测试靶场。

---

## 1.3 配置 DeepSeek 官方 API Key（解锁「联网检索」与「`/rwkv-subagent` 子 Agent 协同」）

- **是否必填？** 只跑本地代码读写、修 Bug、跑单测时 **无需填写**（100% 本地零费用）。当你需要使用以下两项能力时，需配置从 [DeepSeek 开放平台 (`platform.deepseek.com`)](https://platform.deepseek.com/api_keys) 申请的官方 API Key：
  1. **原生联网检索（`web_search` / `web_fetch`）**：底层调用 DeepSeek 官方检索工具接口 `https://api.deepseek.com/v1/tools` 获取权威网页正文；
  2. **`/rwkv-subagent` 强弱协同子 Agent 模式**：由 DeepSeek 官方模型负责只读架构规划与代码审查，并发调度多路本地 RWKV-7 7.2B 子 Agent 零成本写代码与跑单测。

- **写在哪里？（一处写入，两项功能同时永久生效，无需重启 GPU 引擎）**
  - **方法一（最推荐 · 一行命令自动写入）**：
    ```bash
    rwkv-engine set-deepseek sk-你的DeepSeek官方密钥
    ```
  - **方法二（手动编辑配置文件）**：
    打开 **`~/rwkv-agent/rwkv-pi/.env`**，填入文件底部的两行（填同一个 `sk-...` 密钥即可）：
    ```dotenv
    DEEPSEEK_API_KEY=sk-你的DeepSeek官方密钥
    RWKV_WEB_API_KEY=sk-你的DeepSeek官方密钥
    ```

> 配置保存后，`pi` 启动时会自动检测并为你开启 `web_search`、`web_fetch` 工具及会话内 `/rwkv-subagent` 命令！

---

## 1.4 Linux 端三组真实 Demo 靶场端到端实测

全部实测均直接运行在已为你内置好的 **`~/rwkv-agent/demos/`** 真实靶场中（所有完整提示词也可随时查看 `~/rwkv-agent/demos/PROMPTS.zh-CN.md`；测完随时敲 **`rwkv-demos reset`** 秒级还原初始代码）：

### 实测 1（纯本地 RWKV 独跑）：Nginx 生产排障 & 两轮安全审计加固

**A. 一行命令非交互自动排障验收（`02-nginx-gateway`，约 20–35s）：**
```bash
cd ~/rwkv-agent/demos/02-nginx-gateway

pi --thinking high --print "线上发布巡检挂了。请结合 logs/incident.log、config/、deploy/nginx.conf 与 scripts/deploy_check.py 排查所有配置漂移（端口、健康检查路径、请求体上限与生产超时阈值），最小化修改配置文件使 python3 scripts/deploy_check.py 输出 DEPLOY_CHECK_OK（不要修改 scripts/deploy_check.py），并简要说明排障结论。"
```

**B. 交互式 TUI 两轮安全审计与加固（`01-portal-security`，同演示视频 1）：**
```bash
cd ~/rwkv-agent/demos/01-portal-security
pi
```
进入 `pi` 界面后依次输入两轮指令：
1. 第 1 轮：*`第 1 轮（只读安全与架构审计，请勿修改任何文件）：请扫描 app/ 目录，梳理 PortalService 的模块调用关系，并精确定位其中存在的 2 处高危安全漏洞（指出具体文件路径、函数名、行号与原理）。`*
2. 第 2 轮：*`第 2 轮（安全加固修复）：现在请修复 app/db.py 的 SQL 注入漏洞（改为参数化占位符查询）与 app/storage.py 的目录穿越漏洞（校验解析后的路径严格位于 base_dir.resolve() 内部，越界抛 ValueError），并运行 python3 -m unittest discover -s tests 确保测试全部通过。`*

### 实测 2（纯本地 RWKV + DeepSeek 联网检索）：Python 官方双文档核验对比

> 请先按 `1.3` 执行 `rwkv-engine set-deepseek sk-...` 配好密钥。

```bash
cd ~/rwkv-agent/demos/04-python-web-research
pi
```
在 `pi` 输入框粘贴提问（同演示视频 3）：
- *`请使用 web_search 与 web_fetch 查阅 Python 3 官方标准库文档（docs.python.org），准确对比 json 与 tomllib 两个模块：1) 哪个模块支持序列化写入（dump/dumps），哪个是纯只读解析？2) 各自的 load(fp) 函数对文件对象的打开模式（文本模式 'r' 还是二进制模式 'rb'）分别有什么强制要求？每项结论请附上核验过的官方文档 URL。`*

### 实测 3（杀手锏 `/rwkv-subagent` 强弱协同）：13 文件跨模块并发修复 + 实时省钱账单

> 请先按 `1.3` 配好 `DEEPSEEK_API_KEY`。若希望体验多路子 Agent 更高并发吞吐，可在 `~/rwkv-agent/engine.conf` 中将 `DECODE_BATCH` 调至 `8` 或 `16` 后执行 `rwkv-engine restart`。

```bash
cd ~/rwkv-agent/demos/05-routemap
pi
```
进入 `pi` 会话后按三步体验（同演示视频 4）：
1. 先输入 **`/rwkv-subagent`** 回车开启强弱协同模式（输入框上方会出现实时工作量与省钱看板）；
2. 粘贴任务指令：
   *`这个仓库的夜间构建失败了。python3 -m unittest discover -s tests 报告有 6 个测试失败，例如 test_avoid_accepts_repeated_flags_and_multiple_values, test_end_to_end_lifecycle_exit_codes, test_rejects_non_positive_and_bool_costs, test_route_prints_only_path_and_cost。根本原因可能不止一个。请排查并修复源代码（不要修改或删除测试），并确认整个测试套件全部通过，完成后简要总结。`*
3. 33 项单元测试全部修通变绿后，输入 **`/rwkv-subagent cost`** 输出完整省钱结算表；输入 `/rwkv-subagent off` 可随时切回纯本地零成本模式；想重玩一遍只需退回终端执行 **`rwkv-demos reset`**！

---

## 1.5 Linux 端手动分步部署与自定义量化（备选进阶）

<details>
<summary>点击展开：不使用一键脚本的纯手动分步命令 & 自定义 .pth 量化导出</summary>

```bash
# 1) 下载并解压 Linux 预编译引擎（注意二进制名为 rwkv_lighting_cuda）
mkdir -p ~/rwkv-agent/models ~/rwkv-agent/data && cd ~/rwkv-agent
curl -fLO https://github.com/Alic-Li/rwkv_lightning_cuda/releases/download/v1.8.15/rwkv-lightning-v1.8.15-linux-x64-cuda12.9.tar.gz
curl -fLO https://github.com/Alic-Li/rwkv_lightning_cuda/releases/download/v1.8.15/rwkv-lightning-v1.8.15-linux-x64-cuda12.9.tar.gz.sha256
sha256sum --check rwkv-lightning-v1.8.15-linux-x64-cuda12.9.tar.gz.sha256
tar -xzf rwkv-lightning-v1.8.15-linux-x64-cuda12.9.tar.gz

# 2) 下载官方 W8A16 预量化包并校验 SHA-256
cd ~/rwkv-agent/models
aria2c -x 16 -s 16 -c \
  "https://hf-mirror.com/RWKV-Agent-Lab/RWKV7-Agent-7.2B-OpenSource/resolve/main/RWKV7-Pi-7.2B-v0.1-final.w8a16.rwkvq" \
  -o RWKV7-Pi-7.2B-v0.1-final.w8a16.rwkvq
curl -fLO "https://hf-mirror.com/RWKV-Agent-Lab/RWKV7-Agent-7.2B-OpenSource/resolve/main/RWKV7-Pi-7.2B-v0.1-final.w8a16.rwkvq.source.json"
curl -fLO "https://hf-mirror.com/RWKV-Agent-Lab/RWKV7-Agent-7.2B-OpenSource/resolve/main/SHA256SUMS.txt"
grep "RWKV7-Pi-7.2B-v0.1-final.w8a16.rwkvq" SHA256SUMS.txt | sha256sum --check

# [可选] 若你微调了自己的 .pth 权重，15 秒零校准一键导出 W8A16：
# ~/rwkv-agent/bundle/rwkv_quantize ./my-model.pth ./my-model.w8a16.rwkvq

# 3) 下载解压内置真实评测靶场 demos/
cd ~/rwkv-agent
curl -fsSL "https://hf-mirror.com/RWKV-Agent-Lab/RWKV7-Agent-7.2B-OpenSource/resolve/main/rwkv-pi-demos.tar.gz" -o rwkv-pi-demos.tar.gz
tar -xzf rwkv-pi-demos.tar.gz

# 4) 启动推理服务端（终端 1）
LD_LIBRARY_PATH="$PWD/bundle/lib:${LD_LIBRARY_PATH:-}" ./bundle/rwkv_lighting_cuda \
  --model-path ./models/RWKV7-Pi-7.2B-v0.1-final.w8a16.rwkvq \
  --vocab-path ./bundle/rwkv_vocab_v20230424.txt \
  --host 127.0.0.1 --port 8000 \
  --decode-batch-size 4 --chunk-size 512 \
  --wkv32 --chunk-load --cmix-sparse off \
  --state-db-path ./data/rwkv_sessions.db \
  --password rwkv-local

# 5) 安装配置 rwkv-pi（终端 2）
cd ~/rwkv-agent && git clone https://github.com/w1c2j3/rwkv-pi.git && cd rwkv-pi
npm ci
# 在 ~/rwkv-agent/rwkv-pi/.env 中填写 RWKV_MODEL=RWKV7-Pi-7.2B-v0.1-final.w8a16、RWKV_EDIT_MODE=text，以及可选的 DEEPSEEK_API_KEY
```
</details>

## 1.6 Linux 端常见问题排查

1. **手动请求 `127.0.0.1:8000` 报错 `HTTP 502 Bad Gateway`？**
   - 终端开了 `http_proxy` 代理。一键脚本生成的 `pi` 与 `rwkv-engine` 已自动注入 `NO_PROXY="127.0.0.1,localhost"`；自己手敲 `curl` 时加上 `--noproxy "*"` 即可。
2. **手动运行 `./bundle/rwkv_lighting_cuda` 提示缺少 `.so` 动态库？**
   - 保持 `bundle/` 与 `bundle/lib/` 同级目录完整，或执行 `export LD_LIBRARY_PATH="$HOME/rwkv-agent/bundle/lib:${LD_LIBRARY_PATH:-}"`。
3. **联网检索提示 `Web search service HTTP 401`？**
   - 检查 `DEEPSEEK_API_KEY` 是否填写正确且账户余额充足，可通过 `rwkv-engine set-deepseek sk-你的新密钥` 重新覆盖更新。

---
---

# 第二篇：原生 Windows 端完整部署与调用流程

> **适用环境**：Windows 10（1809+）/ Windows 11 64 位原生环境（无需安装 WSL2、无需安装 Visual Studio / C++ 编译器）。本篇所有命令均在 **Windows PowerShell** 中直接复制执行。

## 2.1 前置环境准备（Node.js 22 LTS、Git 与 Python 3）

按 `Win + X` 打开 **终端 (PowerShell)**，若尚未安装 `node`（`>= 22.19`）、`git`（自带 `pi` 所需的 Git Bash）或 `python`（跑 Python 测试靶场需要），用 Windows 自带的 `winget` 一键装齐：

```powershell
winget install --id OpenJS.NodeJS.LTS -e
winget install --id Git.Git -e
winget install --id Python.Python.3.12 -e

# 安装完成后请【关闭并重新打开一个新的 PowerShell 窗口】，验证版本：
node -v
git --version
python --version
```

---

## 2.2 一键极速部署（自动下引擎 + 下权重 + 装 `demos\` 靶场 + 注册全局 `pi`）

### 1. 在 PowerShell 中一键下载并运行安装脚本

**12GB–16GB 显卡（默认安装到 `$HOME\rwkv-agent`，下载 `7.2 GB` W8A16 预量化包）：**
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
curl.exe -fsSL "https://hf-mirror.com/RWKV-Agent-Lab/RWKV7-Agent-7.2B-OpenSource/resolve/main/quickstart-windows.ps1" -o "$HOME\quickstart-windows.ps1"
& "$HOME\quickstart-windows.ps1"
```

**若希望安装到其他盘符（如 `D:\rwkv-agent`）或 24GB+ 显卡使用原生 BF16 `.pth` 权重：**
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
curl.exe -fsSL "https://hf-mirror.com/RWKV-Agent-Lab/RWKV7-Agent-7.2B-OpenSource/resolve/main/quickstart-windows.ps1" -o "$HOME\quickstart-windows.ps1"

# 示例 A：16G 显卡指定安装到 D 盘
& "$HOME\quickstart-windows.ps1" -InstallDir "D:\rwkv-agent"

# 示例 B：24G+ 显卡下载满血 BF16 权重到 D 盘
& "$HOME\quickstart-windows.ps1" -InstallDir "D:\rwkv-agent" -Precision bf16
```

> **可选参数**：`-InstallDir "D:\rwkv-agent"`、`-DeepSeekKey "sk-xxxx"`（安装时顺手写入 DeepSeek Key）、`-Cuda 13.2`、`-HfEndpoint "https://huggingface.co"`、`-Port 8000`。

### 2. 日常单窗口调用与后台显存管理

安装脚本已自动将 `pi`、`rwkv-engine`、`rwkv-demos` 注册进当前 Windows 用户的 `PATH`（在 **PowerShell** 或 **CMD** 任意目录均可直接敲）：
- **`pi`**：进入任意代码目录直接敲 `pi`（若后台 `rwkv_lighting_cuda.exe` 尚未启动，会自动无黑框静默拉起、自动对齐模型 ID、自动绕过系统代理并进入交互终端）。
- **`rwkv-engine {status|logs|stop|restart|set-deepseek}`**：查看推理日志，或执行 `rwkv-engine stop` 一键释放全部显卡显存。
- **`rwkv-demos {list|reset}`**：查看或一键重置 `demos\` 下的 7 组真实评测靶场。

---

## 2.3 配置 DeepSeek 官方 API Key（解锁「联网检索」与「`/rwkv-subagent` 子 Agent 协同」）

- **是否必填？** 只跑本地代码读写与修 Bug 时 **无需填写**（100% 本地零费用）。当你需要开启 **原生联网检索（`web_search` / `web_fetch`，调用 DeepSeek 官方检索接口）** 或 **`/rwkv-subagent` 强弱协同子 Agent 模式** 时，请前往 [DeepSeek 开放平台 (`platform.deepseek.com`)](https://platform.deepseek.com/api_keys) 获取官方 `sk-...` 密钥。

- **写在哪里？（一处写入，两项功能同时永久生效，无需重启 GPU 引擎）**
  - **方法一（最推荐 · 一行命令自动写入）**：在 PowerShell 或 CMD 中直接执行：
    ```powershell
    rwkv-engine set-deepseek sk-你的DeepSeek官方密钥
    ```
  - **方法二（用记事本打开 `.env` 配置文件填入）**：
    ```powershell
    # 若自定义了 -InstallDir "D:\rwkv-agent"，请将下方路径对应改为 D:\rwkv-agent\rwkv-pi\.env
    notepad "$HOME\rwkv-agent\rwkv-pi\.env"
    ```
    在文件底部填入同一个密钥并保存：
    ```dotenv
    DEEPSEEK_API_KEY=sk-你的DeepSeek官方密钥
    RWKV_WEB_API_KEY=sk-你的DeepSeek官方密钥
    ```

> 保存后下次执行 `pi` 会自动识别该密钥，同时启用 `web_search`、`web_fetch` 与 `/rwkv-subagent` 协同命令！

---

## 2.4 Windows 端三组真实 Demo 靶场端到端实测（纯 PowerShell 命令）

> 注：若安装时使用了 `-InstallDir "D:\rwkv-agent"`，请将下方 `$HOME\rwkv-agent` 替换为 `D:\rwkv-agent`。完整题目列表位于 `demos\PROMPTS.zh-CN.md`，测完随时敲 **`rwkv-demos reset`** 一键恢复干净初始代码！

### 实测 1（纯本地 RWKV 独跑）：Nginx 生产排障 & 两轮安全审计加固

**A. 一行命令非交互自动排障验收（`02-nginx-gateway`，约 20–35s）：**
```powershell
Set-Location "$HOME\rwkv-agent\demos\02-nginx-gateway"

pi --thinking high --print "线上发布巡检挂了。请结合 logs/incident.log、config/、deploy/nginx.conf 与 scripts/deploy_check.py 排查所有配置漂移（端口、健康检查路径、请求体上限与生产超时阈值），最小化修改配置文件使 python scripts/deploy_check.py 输出 DEPLOY_CHECK_OK（不要修改 scripts/deploy_check.py），并简要说明排障结论。"
```

**B. 交互式 TUI 两轮安全审计与加固（`01-portal-security`，同演示视频 1）：**
```powershell
Set-Location "$HOME\rwkv-agent\demos\01-portal-security"
pi
```
进入 `pi` 界面后依次输入两轮指令：
1. 第 1 轮：*`第 1 轮（只读安全与架构审计，请勿修改任何文件）：请扫描 app/ 目录，梳理 PortalService 的模块调用关系，并精确定位其中存在的 2 处高危安全漏洞（指出具体文件路径、函数名、行号与原理）。`*
2. 第 2 轮：*`第 2 轮（安全加固修复）：现在请修复 app/db.py 的 SQL 注入漏洞（改为参数化占位符查询）与 app/storage.py 的目录穿越漏洞（校验解析后的路径严格位于 base_dir.resolve() 内部，越界抛 ValueError），并运行 python -m unittest discover -s tests 确保测试全部通过。`*

### 实测 2（纯本地 RWKV + DeepSeek 联网检索）：Python 官方双文档核验对比

> 请先按 `2.3` 执行 `rwkv-engine set-deepseek sk-...` 配好密钥。

```powershell
Set-Location "$HOME\rwkv-agent\demos\04-python-web-research"
pi
```
在 `pi` 输入框粘贴提问（同演示视频 3）：
- *`请使用 web_search 与 web_fetch 查阅 Python 3 官方标准库文档（docs.python.org），准确对比 json 与 tomllib 两个模块：1) 哪个模块支持序列化写入（dump/dumps），哪个是纯只读解析？2) 各自的 load(fp) 函数对文件对象的打开模式（文本模式 'r' 还是二进制模式 'rb'）分别有什么强制要求？每项结论请附上核验过的官方文档 URL。`*

### 实测 3（杀手锏 `/rwkv-subagent` 强弱协同）：13 文件跨模块并发修复 + 实时省钱账单

> 请先按 `2.3` 配好 `DEEPSEEK_API_KEY`。若希望调高并发子 Agent 吞吐，可将 `$HOME\rwkv-agent\engine.json` 中的 `"decodeBatch"` 改为 `8` 或 `16` 后执行 `rwkv-engine restart`。

```powershell
Set-Location "$HOME\rwkv-agent\demos\05-routemap"
pi
```
进入 `pi` 会话后按三步体验（同演示视频 4）：
1. 先输入 **`/rwkv-subagent`** 回车开启强弱协同模式；
2. 粘贴任务指令：
   *`这个仓库的夜间构建失败了。python -m unittest discover -s tests 报告有 6 个测试失败，例如 test_avoid_accepts_repeated_flags_and_multiple_values, test_end_to_end_lifecycle_exit_codes, test_rejects_non_positive_and_bool_costs, test_route_prints_only_path_and_cost。根本原因可能不止一个。请排查并修复源代码（不要修改或删除测试），并确认整个测试套件全部通过，完成后简要总结。`*
3. 33 项单元测试全部通过后，输入 **`/rwkv-subagent cost`** 查看完整省钱结算表；输入 `/rwkv-subagent off` 切回纯本地模式；退出后执行 **`rwkv-demos reset`** 即可将靶场一键恢复原状！

---

## 2.5 Windows 端手动分步部署与自定义量化（备选进阶）

<details>
<summary>点击展开：不使用一键脚本的纯 PowerShell 手动分步命令 & 自定义 .pth 量化导出</summary>

```powershell
# 1) 创建目录并下载解压 Windows 预编译引擎（DLL 已随包内置于 bundle\ 同级目录）
New-Item -ItemType Directory -Force -Path "C:\rwkv-agent\models", "C:\rwkv-agent\data" | Out-Null
Set-Location "C:\rwkv-agent"
curl.exe -fLO "https://github.com/Alic-Li/rwkv_lightning_cuda/releases/download/v1.8.15/rwkv-lightning-v1.8.15-windows-x64-cuda12.9.zip"
Expand-Archive -Path ".\rwkv-lightning-v1.8.15-windows-x64-cuda12.9.zip" -DestinationPath "C:\rwkv-agent" -Force
.\bundle\rwkv_lighting_cuda.exe --version

# 2) 下载官方 W8A16 预量化权重并校验 SHA-256
curl.exe -fL -C - "https://hf-mirror.com/RWKV-Agent-Lab/RWKV7-Agent-7.2B-OpenSource/resolve/main/RWKV7-Pi-7.2B-v0.1-final.w8a16.rwkvq" -o "C:\rwkv-agent\models\RWKV7-Pi-7.2B-v0.1-final.w8a16.rwkvq"
curl.exe -fLO --output-dir "C:\rwkv-agent\models" "https://hf-mirror.com/RWKV-Agent-Lab/RWKV7-Agent-7.2B-OpenSource/resolve/main/RWKV7-Pi-7.2B-v0.1-final.w8a16.rwkvq.source.json"
Get-FileHash -Algorithm SHA256 "C:\rwkv-agent\models\RWKV7-Pi-7.2B-v0.1-final.w8a16.rwkvq"

# [可选] 若你微调了自己的 .pth 权重，15 秒零校准一键导出 W8A16：
# C:\rwkv-agent\bundle\rwkv_quantize.exe C:\path\to\my-model.pth C:\rwkv-agent\models\my-model.w8a16.rwkvq

# 3) 下载解压内置真实评测靶场 demos\
curl.exe -fsSL "https://hf-mirror.com/RWKV-Agent-Lab/RWKV7-Agent-7.2B-OpenSource/resolve/main/rwkv-pi-demos.zip" -o "C:\rwkv-agent\rwkv-pi-demos.zip"
Expand-Archive -Path "C:\rwkv-agent\rwkv-pi-demos.zip" -DestinationPath "C:\rwkv-agent" -Force

# 4) 启动推理服务端（PowerShell 窗口 1）
C:\rwkv-agent\bundle\rwkv_lighting_cuda.exe `
  --model-path C:\rwkv-agent\models\RWKV7-Pi-7.2B-v0.1-final.w8a16.rwkvq `
  --vocab-path C:\rwkv-agent\bundle\rwkv_vocab_v20230424.txt `
  --host 127.0.0.1 --port 8000 `
  --decode-batch-size 4 --chunk-size 512 `
  --wkv32 --chunk-load --cmix-sparse off `
  --state-db-path C:\rwkv-agent\data\rwkv_sessions.db `
  --password rwkv-local

# 5) 安装配置 rwkv-pi（PowerShell 窗口 2）
Set-Location "C:\rwkv-agent"
git clone https://github.com/w1c2j3/rwkv-pi.git
Set-Location "C:\rwkv-agent\rwkv-pi"
npm.cmd ci
Copy-Item .env.example .env
# 在 .env 中设置 RWKV_BASE_URL=http://127.0.0.1:8000/v1、RWKV_MODEL=RWKV7-Pi-7.2B-v0.1-final.w8a16、RWKV_EDIT_MODE=text、DEEPSEEK_API_KEY=sk-...
notepad .env
```
</details>

## 2.6 Windows 端常见问题排查

1. **运行 `.ps1` 提示“因为在此系统上禁止运行脚本”？**
   - 先在当前 PowerShell 窗口执行 `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force`（仅当前窗口生效），再运行安装脚本；安装完成后日常运行 `pi` 走的是 `.cmd` 包装器，不再受执行策略限制。
2. **开启 Clash / v2rayN 系统代理后本地请求报错？**
   - 一键脚本生成的 `pi.cmd` 与 `rwkv-engine.cmd` 已内置 `--noproxy "*"` 与 `$env:NO_PROXY = "127.0.0.1,localhost"` 自动直连本地回环；若开启了虚拟网卡 TUN 全局劫持，请在代理规则中放行 `127.0.0.1`。
3. **运行测试提示找不到 `python` 或弹出微软应用商店？**
   - 进入 Windows「设置 → 应用 → 高级应用设置 → 应用执行别名」，关闭 `python.exe` / `python3.exe` 的应用商店别名，确保使用的是 `winget install Python.Python.3.12` 安装的真实 Python。
4. **Windows Defender 实时扫描导致首次加载 `7.2 GB` 模型偏慢？**
   - 可在「Windows 安全中心 → 病毒和威胁防护 → 管理设置 → 排除项」中将安装目录（如 `D:\rwkv-agent`）加入排除列表，冷启动加载速度可提升一倍以上。

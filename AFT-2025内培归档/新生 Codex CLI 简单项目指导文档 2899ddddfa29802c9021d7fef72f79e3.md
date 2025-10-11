# 新生 Codex CLI 简单项目指导文档

> 这份材料带你从 0 到 1：完成 Codex CLI 的安装配置、熟悉核心命令，并在提供好的脚手架上动手实现一个“未来信息泄露”检测工具。请循序渐进，边操作边记录问题。
如果对量化兴趣不大，主要参考Codex的使用即可，安装好codex之后可以查看下如何构建一个简单的产品文档，参考如下链接，思考如何使用codex完成一个类似的网站并部署到github pages上：
[https://chenrunnan-cooper.github.io/Ancient-Style-Young-Man/](https://chenrunnan-cooper.github.io/Ancient-Style-Young-Man/)
> 

---

## **1. 本次练习目的（无需交付）**

- 一套可运行的本地环境：虚拟环境、依赖、Codex CLI 均能正常工作。
- 理解项目结构，熟悉我们提供的模拟数据与辅助代码。
    - 在 `leak_detection/detector.py` 中补全 `detect_leaks` 函数，识别 `factors/factor_future_return` 等故意埋入的泄露型因子代码。
- 使用 Codex CLI 完成需求分解、代码生成或代码评审，并记录自己的操作习惯。
- 最终产物：
    - 本地运行 `python -m pytest` 通过。
    - `scripts/run_pipeline.py` 能生成 `outputs/manual/summary.csv`、`outputs/manual/violations.json`、`outputs/manual/report.md`。
    - Markdown 小结（可直接在 Codex CLI 中 `/save` 或手动整理）。

---

## **2. 环境前置条件（建议必做的）**

| **项** | **Windows 建议** | **macOS 建议** |
| --- | --- | --- |
| 终端 | [Git Bash 安装指南](https://zhuanlan.zhihu.com/p/599044770)（含下载步骤） | 系统自带 zsh，可参考 [Oh My Zsh 配置图文](https://a1049145827.github.io/2019/05/15/Mac-%E7%8E%AF%E5%A2%83%E5%AE%89%E8%A3%85%E5%B9%B6%E9%85%8D%E7%BD%AE%E7%BB%88%E7%AB%AF%E7%A5%9E%E5%99%A8-oh-my-zsh/) |
| Python | 3.10–3.12，安装时勾选 “Add to PATH” | 使用 [python.org](https://www.python.org/downloads/) 安装或 `brew install python` |
| Node.js | 建议使用 18 LTS，可从 [nodejs.org](https://nodejs.org/) 下载 .msi 安装包；或用 [nvm-windows](https://github.com/coreybutler/nvm-windows) 管理多版本 | `brew install node` 或使用 [nvm](https://github.com/nvm-sh/nvm) 安装 18.x LTS |
| Git | 安装 Git for Windows 后自带 | `xcode-select --install` 后自带 |

> 提示：若需要代理，请在继续前完成第 3 节的网络配置。
> 

---

## **3. 网络与代理准备（可选但推荐先完成）**

1. **确认代理端口**
    - **Windows PowerShell**
        
        ```
        Get-NetTCPConnection -State Listen |
          Where-Object { $_.LocalAddress -in @("127.0.0.1","::1") } |
          Sort-Object LocalPort |
          Select-Object LocalAddress,LocalPort,OwningProcess
        ```
        
    - **macOS / Linux**
        
        ```
        scutil --proxy            # 查看系统代理
        lsof -i -P | grep LISTEN  # 查看本地监听端口
        ```
        
    - 补充：`echo $http_proxy`、`git config --global --get http.proxy` 等命令可检查已有设置。
2. **在当前终端临时走代理**（安装/测试前优先设置）
    
    ```
    export http_proxy=http://127.0.0.1:7890
    export https_proxy=http://127.0.0.1:7890
    export all_proxy=socks5://127.0.0.1:7890
    export no_proxy=localhost,127.0.0.1,*.local
    ```
    
    将上述语句写入 `~/.bashrc` 或 `~/.zshrc` 可持久化。
    
3. **让 Git 单独走代理**（在下载仓库前配置）
    
    ```
    git config --global http.proxy  http://127.0.0.1:7890
    git config --global https.proxy http://127.0.0.1:7890
    # 取消代理
    git config --global --unset http.proxy
    git config --global --unset https.proxy
    ```
    
4. **pip / conda 代理示例**
    - pip：`pip install --proxy=http://127.0.0.1:7890 <package>`
    - conda：在 `~/.condarc` 中设置 `proxy_servers` 节点。

---

## **4. Codex CLI：从安装到首个指令（如果选择插件可以不用配置）**

![image.png](%E6%96%B0%E7%94%9F%20Codex%20CLI%20%E7%AE%80%E5%8D%95%E9%A1%B9%E7%9B%AE%E6%8C%87%E5%AF%BC%E6%96%87%E6%A1%A3%202899ddddfa29802c9021d7fef72f79e3/image.png)

![image.png](%E6%96%B0%E7%94%9F%20Codex%20CLI%20%E7%AE%80%E5%8D%95%E9%A1%B9%E7%9B%AE%E6%8C%87%E5%AF%BC%E6%96%87%E6%A1%A3%202899ddddfa29802c9021d7fef72f79e3/image%201.png)

> 如果选择“在IDE中尝试”可以跳过以下cli的配置过程，但是cli一般都很帅，以下是gemini和claude code 的cli
> 

![image.png](%E6%96%B0%E7%94%9F%20Codex%20CLI%20%E7%AE%80%E5%8D%95%E9%A1%B9%E7%9B%AE%E6%8C%87%E5%AF%BC%E6%96%87%E6%A1%A3%202899ddddfa29802c9021d7fef72f79e3/image%202.png)

1. **安装**（需 Node.js ≥ 18）
    
    在运行下面命令之前，请先确认 `node -v` 与 `npm -v` 输出已存在且版本满足要求(不满足可以随便问问大模型看看怎么安装哦)。
    
    ```
    npm install -g @openai/codex-cli
    ```
    
    如果之前用 `pip install codex-cli`，请先 `pip uninstall codex-cli` 避免命令冲突。
    
2. **登录授权（第一次会默认打开浏览器要求登录认证，需要账号是plus，如果没有的话参考补充说明）**
    
    ```
    codex auth login
    ```
    
    - 默认会打开浏览器完成认证；若无法自动打开，请复制提示链接到浏览器。
    - `codex auth status` 查看当前账号与授权模式。
    
    > 没有账号的话，千万不要去闲鱼搜索“开发成长之路”然后点击codex团队套餐，然后付款哈（相对稳定的低成本使用方式）
    如果比较喜欢白嫖可以使用gemini cli和claude code router，但免费的性能会差很多，需要教程可以私戳会长
    > 
3. **首检**
    
    ```
    codex --version
    codex /status
    ```
    
4. **常用快捷命令（可以自行探索下）**
    
    
    | **场景** | **命令** | **说明** |
    | --- | --- | --- |
    | 查看/切换模型 | `/model gpt-5-codex` | 会话内随时切换模型与推理等级 |
    | 管理授权策略 | `/approvals` | 切换 `on-request`、`never` 等模式 |
    | 新建会话 | `/new` | 清理上下文重新开始 |
    | 初始化说明文件 | `/init` | 自动生成 `AGENTS.md`（可选） |
    | 一键执行命令 | `codex exec "ls"` | 非交互运行 shell 命令 |
    | 联网检索 | `codex --search` | 先检索再回答，携带引用 |
5. **遇到网络问题？** 重新检查第 3 节的代理设置，并确认 `npm config get proxy` 与终端环境一致。

---

## **5. 下载/更新练习项目**

> 课程仓库中已经放入脚手架。如果你是直接解压压缩包，请跳过 git clone。
> 

```
git clone git@github.com:PKU-AFT/leak-detector-starter.git
cd leak-detector-starter
# 如未配置 SSH，可改用：
# git clone https://github.com/PKU-AFT/leak-detector-starter.git
```

项目目录结构与关键文件：

```
.
├── data/
│   └── sample_factors.csv        # 仅含 OHLCV + ret_1d 的模拟数据
├── factors/
│   ├── __init__.py               # 因子注册表（含函数元数据）
│   └── sample_factors.py         # 示例因子函数，含一个故意泄露的实现
├── leak_detection/
│   ├── __init__.py
│   ├── data.py                   # 数据加载 + 描述
│   ├── detector.py               # 需要你补全的“因子代码”检测逻辑
│   ├── pipeline.py               # 运行检测并输出报告
│   ├── reporting.py              # CSV / JSON / Markdown 导出
│   └── simulator.py              # 重新生成模拟数据（仅行情）
├── scripts/
│   ├── regenerate_data.py        # 重制数据集
│   ├── preview_data.py           # 打印样本
│   └── run_pipeline.py           # 命令行入口
├── tests/
│   └── test_dataset.py           # 数据与因子注册表的基本校验
├── requirements.txt
└── 新生codex cli简单项目指导文档.md
```

---

## **6. 创建虚拟环境并安装依赖**

1. 创建虚拟环境
    
    ```
    # macOS / Linux / Git Bash
    python -m venv .venv
    source .venv/bin/activate
    
    # Windows PowerShell
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    ```
    
2. 安装依赖
    
    ```
    pip install -r requirements.txt
    ```
    
3. 验证基础脚本
    
    ```
    python scripts/preview_data.py -n 3
    python -m pytest
    ```
    
    现在 `tests/test_dataset.py` 应全部通过，表示环境无误。
    

---

## **7. 理解我们已经替你实现的部分**

| **组件** | **功能** | **如何尝试** |
| --- | --- | --- |
| `data/sample_factors.csv` | 3 只股票、60 个交易日的 OHLCV + `ret_1d` 数据（不含任何因子值） | `python scripts/preview_data.py` |
| `factors/sample_factors.py` | 提供 3 个因子函数：两个合规、一个故意使用未来数据 | 阅读源码或 `from factors import get_registered_factors` |
| `leak_detection/detector.py` | 你要补全的检测逻辑，输入行情 + 因子函数，输出疑似泄露清单 | 在编辑器中打开并实现 |
| `leak_detection/pipeline.py` | 调用数据加载、因子注册表和检测器，生成报告文件 | `python scripts/run_pipeline.py`（需先实现检测逻辑） |
| `leak_detection/reporting.py` | 辅助把检测结果写成 CSV / JSON / Markdown | 检查 `export_all` 输出 |
| `leak_detection/simulator.py` | 使用随机种子生成新的行情数据 | `python scripts/regenerate_data.py` |

这些代码均添加了简洁注释，便于阅读。

---

## **8. 你需要完成的核心任务：实现 `detect_leaks`**

`leak_detection/detector.py` 提供了需要补全的函数签名：

```
def detect_leaks(
    df: pd.DataFrame,
    *,
    factor_definitions: Mapping[str, FactorDefinition],
    entity_col: str = "symbol",
    time_col: str = "date",
    tolerance: float = 1e-9,
) -> LeakDetectionResult:
    raise NotImplementedError
```

你要做的是**检测因子“代码”是否泄露未来信息**，而不是直接对因子数值做静态分析。建议至少实现以下两类检验：

1. **前推重算（forward-only replay）**
    - 对每个因子函数，以 `symbol` 为分组，逐日增加可见数据窗口。
    - 在只包含过去数据的子集上运行因子函数，取末行结果，与完整数据运行得到的同一行值比较。
    - 若差异超过 `tolerance`，说明函数引用了未来信息（示例：`factor_future_return`）。
2. **稳健性/一致性校验**
    - 可把数据分成训练 / 验证时间段，对两个窗口分别进行前推重算或统计对比。
    - 也可以补充静态规则（例如解析源码是否包含 `shift(-1)`、`lead` 等可疑操作）。

将发现的问题以 `LeakViolation` 形式记录，并返回 `LeakDetectionResult`。`result.inspected_columns` 建议填入已检查的因子函数名，报告中会用到。

> 最低目标：让检测器准确标记出 factor_future_return 为泄露因子，同时不过度误报其它两个因子。
> 

---

## **9. 如何借助 Codex CLI 高效完成开发**

1. **需求分解**
    
    ```
    codex exec "rg 'NotImplementedError'"
    ```
    
    确认只剩 `detect_leaks` 未完成，必要时 `/plan` 生成实现步骤。
    
2. **和 Codex 讨论方案**
    - `/new` 开新会话，粘贴 `factor_future_return` 与其它因子函数代码，请它解释差异。
    - 示例提问：“怎么用滚动回放的方式检测一个因子函数是否用到了未来数据？”
3. **生成实验脚本**
    
    ```
    codex exec "python scripts/preview_data.py -n 5"
    ```
    
    把数据样本和因子函数贴给 Codex，让它协助写出前推重算的雏形；再结合自己的理解完善。
    
4. **实现并自测**
    - 在本地实现第一版 `detect_leaks`。
    - 使用 `codex exec "python -m pytest"`、`python scripts/run_pipeline.py --output outputs/manual`，确认 `outputs/manual/summary.csv` 能标记出 `factor_future_return` 等泄露因子。
5. **让 Codex 做代码评审**
    - 将核心逻辑粘贴回 Codex，请它扮演 reviewer 提出潜在漏洞或边界情况。

> 建议使用 /save MyNotes.md 或手写笔记记录学习过程。
> 

---

## **10. 最终自检清单**

- [ ]  `python -m pytest` 全部通过。
- [ ]  `python scripts/run_pipeline.py` 生成的 `summary.csv` 指向至少一个泄露因子。
- [ ]  自己能解释检测逻辑、阈值含义、为何该因子算泄露。
- [ ]  熟悉 Codex CLI 的常用命令：`/model`、`/approvals`、`codex exec`、`codex --search`。
- [ ]  已整理学习笔记或提交报告。

---

## **11. FAQ**

**Q1. 数据集太小会不会影响实验？** 这是训练用的入门集，逻辑跑通后即可迁移到更大数据。

**Q2. 如果 `codex exec` 返回权限错误？** 检查当前路径是否在项目根目录，以及 `/approvals` 是否被设成 `never`。

**Q3. 能否直接让 Codex 写完整代码？** 鼓励把 Codex 当作助手：请它提供思路、排查缺陷，而不是完全代写，写的代码一定要自己看一遍，每次提示词感觉很笼统的话记得先让他拆分成小提示词，这样来得及看。

**Q4. 想重置数据？** 运行 `python scripts/regenerate_data.py` 会覆盖旧的 CSV。

**Q5. 还不会写测试？** 可以请 Codex 帮忙补充针对 `detect_leaks` 的测试用例，再自行跑通。

---
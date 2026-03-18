# 项目规则 — Cape Town Solar Panel Detection

## Git 大文件保护

**严禁将以下类型的文件提交到 git 仓库：**

1. **影像瓦片** — `tiles/` 下的所有 `.tif`, `.vrt`, `.jp2` 文件（每个 ~12MB，全部约 3GB）
2. **模型权重** — `checkpoints/` 下的 `.pth` 文件（每个 ~170MB）
3. **训练数据** — `data/coco*/` 目录（COCO 数据集，可重新生成）
4. **检测结果** — `results/` 目录（包括 masks, vectors, predictions，可重新生成）
5. **缓存文件** — `cache/` 目录

这些文件已在 `.gitignore` 中配置排除。**修改 `.gitignore` 时必须确保上述规则不被删除或放宽。**

### 为什么这条规则重要

2026-03-18 事件：`tiles/` 和 `cache/` 目录曾被误提交到 git 历史，导致 `.git/` 膨胀至 3GB+，`git push` 持续超时。后通过 `git-filter-repo` 清理历史才恢复正常，但过程中 tiles 工作目录文件也被删除，需要重新下载。

### 如何检查

提交前运行：
```bash
git diff --cached --stat | grep -E '\.(tif|vrt|pth|jp2)$'
```
如有匹配，**不要提交**，先检查 `.gitignore` 是否正确。

## 文件移动与目录治理

### 数据目录禁止放源码

以下目录仅存放运行时数据/产物，**严禁放置 Python 脚本或其他源码**：
- `tiles/` — 影像瓦片
- `results/` — 检测结果
- `checkpoints/` — 模型权重
- `data/coco*/` — 训练数据

源码应放在 `core/`、`scripts/<domain>/`、或项目根目录。

### 文件移动必须同步更新文档

移动文件或新建目录时，**必须在同一个 commit 中**同步更新：
1. `docs/architecture.md` 的目录结构和路径映射
2. `CLAUDE.md` / `AGENTS.md` 的 Key References（如指针变化）
3. 所有引用该文件的 import 语句和 shell 脚本

不允许出现"代码已移动，但文档仍指向旧路径"的中间状态。

### Git 历史清理安全规则

对含工作数据的目录（如 `tiles/`、`results/`）执行 `git-filter-repo` 或类似历史清理操作前，
**必须先备份该目录的工作文件**。`git-filter-repo` 会同时删除工作目录中被跟踪的文件。

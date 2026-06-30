# GitHub Actions CI

本目录与 [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml) 共用 `Makefile` 的 `make ci` 门禁逻辑。

## 快速开启

1. 在 [GitHub](https://github.com) 创建仓库（或把本地仓库 `git remote add origin` 指向 GitHub）
2. 推送包含 `.github/workflows/ci.yml` 的代码到 `develop` / `main` / `master`
3. 打开仓库 **Actions** 页，确认 **CI** workflow 已启用（公开仓库默认开启）
4. （推荐）在 **Settings → Branches → Branch protection rules** 中勾选 **Require status checks to pass before merging**，并选择 **Quality gate**

## 触发时机

| 事件 | 分支 | 说明 |
|------|------|------|
| `pull_request` | `develop` / `main` / `master` | feature → develop 或 develop → main 等 PR 时跑门禁 |
| `push` | `develop` / `main` / `master` | 合并到上述分支后跑门禁 |

若希望**任意 feature 分支 push** 也触发 CI，可在 `ci.yml` 的 `on.push` 中去掉 `branches` 限制，或增加 `branches: ['**']`。

## 质量门禁内容（`make ci`）

- `uv sync --frozen --all-groups --extra ui`（锁定 `uv.lock`）
- Python 3.12 版本校验
- `ruff check` 静态检查
- `pytest -m "not integration"` 单元测试
- `uv build` 打 wheel 制品

GitHub Actions runner 使用 `astral-sh/setup-uv` 缓存依赖，二次构建会更快。

## 本地复现

```bash
make ci
# 或
./deploy/ci/run-quality-gate.sh
```

## 可选：主分支 Docker 镜像推送（CD）

需要容器部署时，可新建 `.github/workflows/docker.yml`，在 push 到 `main` / `master` 时构建并推送到 GitHub Container Registry（GHCR）或自有 Registry。在仓库 **Settings → Secrets and variables → Actions** 中配置：

| Secret | 说明 |
|--------|------|
| `REGISTRY_USER` | 镜像仓库用户名（GHCR 可用 `${{ github.actor }}`） |
| `REGISTRY_PASSWORD` | 仓库 token 或密码 |

推送 tag 建议使用 `${{ github.sha }}`。服务器部署：

```bash
export IMAGE=ghcr.io/<owner>/langgraph-skill-agent
export TAG=<commit-sha>
make docker-prod-up
```

## 分支保护建议

- `develop`：feature PR 合并前必须通过 **Quality gate**
- `main` / `master`：develop → main 等 PR 合并前必须通过 **Quality gate**
- 禁止绕过 required checks 直接合并

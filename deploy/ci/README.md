# CI / CD 流水线

本目录说明 GitHub Actions 与本地 Makefile 的对齐关系。

## 流水线总览

```text
PR / push ──► CI (Quality gate + Security scan)
                    │
merge main ─────────┘
                    ▼
              Build and push image (GHCR + Trivy)
                    │
                    ├─► [可选] Deploy → staging (AUTO_DEPLOY_STAGING=true)
                    └─► Deploy → staging / production (手动 workflow_dispatch)
```

| Workflow | 文件 | 触发 | 职责 |
|----------|------|------|------|
| **CI** | `.github/workflows/ci.yml` | PR + push `develop`/`main`/`master` | 质量门禁 + 安全扫描 |
| **Build and push image** | `.github/workflows/build-image.yml` | `main`/`master` 上 CI 成功后；可手动 | 构建 Docker 镜像 → Trivy → 推 GHCR |
| **Deploy** | `.github/workflows/deploy.yml` | 手动；或 Build 成功后自动 staging | SSH 部署不可变镜像 |

## CI 质量门禁（`make ci`）

- `uv sync --frozen --all-groups --extra ui`
- Python 3.12 版本校验
- `ruff check` 静态检查
- `pytest -m "not integration"` 单元测试（CI 产出 JUnit 报告）
- `uv build` 验证 wheel 可构建

## CI 安全扫描（`make security`）

- `pip-audit`：依赖已知漏洞
- `bandit`：Python 源码安全规则（`-ll`）

分支保护建议同时勾选 **Quality gate** 与 **Security scan**。

## Build 制品

- **Registry**：`ghcr.io/<owner>/langgraph-skill-agent`（owner/repo 自动小写）
- **Tag（不可变）**：Git commit 完整 SHA，例如 `abc1234...`
- **Tag（浮动）**：`latest`（仅方便调试，生产部署请用 SHA）

首次 push 后，在 GitHub **Packages** 中将 package 可见性设为组织策略所需（private 仓库默认 private package）。

服务器拉 private 镜像：

```bash
echo <PAT_with_read:packages> | docker login ghcr.io -u <github-user> --password-stdin
```

## Deploy 配置

### 1. GitHub Environments

在 **Settings → Environments** 创建：

- `staging`
- `production`（建议开启 **Required reviewers**）

### 2. Secrets（每个 Environment 或 Repository 级）

| Secret | 说明 |
|--------|------|
| `DEPLOY_HOST` | 目标服务器 IP / 域名 |
| `DEPLOY_USER` | SSH 用户名 |
| `DEPLOY_SSH_KEY` | SSH 私钥（对应服务器 `authorized_keys`） |

### 3. Variables（Repository → Actions → Variables）

| Variable | 默认值 | 说明 |
|----------|--------|------|
| `DEPLOY_PATH` | `/opt/langgraph-skill-agent` | 服务器部署目录 |
| `DEPLOY_PORT` | `22` | SSH 端口 |
| `AUTO_DEPLOY_STAGING` | （不设） | 设为 `true` 时，Build 成功后自动部署 staging |

### 4. 手动部署

1. Actions → **Deploy** → Run workflow
2. **tag** 填 Build job summary 中的 commit SHA
3. **environment** 选 `staging` 或 `production`

### 5. 服务器准备

见 [deploy/cd/README.md](../cd/README.md)。

## 本地复现

```bash
make ci
make security
./deploy/ci/run-quality-gate.sh
```

## 回滚

在 Deploy workflow 中选择**上一个已知 good 的 SHA**，或于服务器：

```bash
export IMAGE=ghcr.io/<owner>/langgraph-skill-agent
export TAG=<previous-sha>
make deploy-remote
```

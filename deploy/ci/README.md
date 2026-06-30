# Gitee Go CI 说明

本目录为 **Gitee Go** 流水线入口，与 `Makefile` 的 `make ci` 共用同一套门禁逻辑。

## 快速开启

1. 登录 Gitee 仓库 → **流水线** → **开通 Gitee Go**（需绑定手机号）
2. 将 `.workflow/` 下三个 YAML 提交到仓库
3. 在 **设置 → 保护分支** 中勾选「合并前流水线必须通过」

## 流水线文件

| 文件 | 触发时机 | 阶段 |
|------|----------|------|
| `PRPipeline.yml` | 发起 Pull Request | 质量门禁 |
| `BranchPipeline.yml` | push 到非 master/main 分支 | 质量门禁 |
| `MasterPipeline.yml` | push 到 master/main | 质量门禁 |

## 质量门禁内容（`make ci`）

- `uv sync --frozen --all-groups`（锁定 `uv.lock`；不含 `ui` extra，单测无需 Streamlit）
- Python 3.12 版本校验
- `ruff check` 静态检查
- `pytest -m "not integration"` 单元测试
- `uv build` 打 wheel 制品

依赖下载走 **清华 PyPI 镜像**（`pyproject.toml` 的 `[[tool.uv.index]]` + CI 脚本 `UV_DEFAULT_INDEX`），`uv.lock` 内 wheel URL 亦指向镜像，避免 Gitee runner 直连 `files.pythonhosted.org` 超时。

三条流水线均缓存 `~/.cache/uv` 与 `.venv`，二次构建显著加速。

## 本地复现

```bash
make ci
# 或
./deploy/ci/run-quality-gate.sh
```

## 可选：Master 镜像推送（CD 阶段）

配置镜像仓库后，可在 `MasterPipeline.yml` 追加 `build@docker` 阶段。在 Gitee Go **流水线变量/密钥** 中配置：

| 变量 | 说明 |
|------|------|
| `CI_IMAGE_REGISTRY` | 镜像仓库前缀，如 `registry.cn-hangzhou.aliyuncs.com/your-ns` |
| `CI_REGISTRY_USER` | 仓库用户名 |
| `CI_REGISTRY_PASSWORD` | 仓库密码或 token |

推送 tag 为 `${GITEE_COMMIT}`（完整 commit SHA）。服务器部署时使用：

```bash
export IMAGE=${CI_IMAGE_REGISTRY}/langgraph-skill-agent
export TAG=<commit-sha>
make docker-prod-up
```

示例片段（追加到 `MasterPipeline.yml` 的 `stages` 末尾）：

```yaml
  - name: docker
    displayName: 镜像构建
    strategy: fast
    trigger: auto
    steps:
      - step: build@docker
        name: docker_build
        displayName: Docker 构建并推送
        dockerfile: ./Dockerfile
        repository: ${CI_IMAGE_REGISTRY}/langgraph-skill-agent
        username: ${CI_REGISTRY_USER}
        password: ${CI_REGISTRY_PASSWORD}
        tag: ${GITEE_COMMIT}
```

## 分支保护建议

- PR 必须关联 `PRPipeline.yml` 成功
- master/main 必须关联 `MasterPipeline.yml` 成功
- 禁止跳过流水线合并

# 服务器 CD 部署

部署目录只保留 **Compose 配置 + 脚本 + `.env`**，应用在容器内由 Registry 不可变镜像提供。

## 目录结构（推荐）

```text
/opt/langgraph-skill-agent/
├── docker-compose.yml
├── docker-compose.prod.yml
├── Makefile
├── deploy/
│   ├── cd/deploy-remote.sh
│   └── postgres/
├── .env                 # 仅存在于服务器，不进 git
└── （无需 src/ 源码）
```

可从仓库 sparse checkout 或 rsync 同步上述文件：

```bash
rsync -av --exclude='.git' \
  docker-compose.yml docker-compose.prod.yml Makefile deploy/ \
  user@server:/opt/langgraph-skill-agent/
```

## 首次初始化

```bash
# 1. 安装 Docker Engine + Compose plugin

# 2. 登录 GHCR（private package 时）
echo "$GITHUB_PAT" | docker login ghcr.io -u YOUR_USER --password-stdin

# 3. 创建 .env（参考仓库 .env.example）
cp .env.example .env
# 必填: POSTGRES_PASSWORD, DEEPSEEK_API_KEY, Milvus/Embedding 等

# 4. 首次部署（TAG 来自 CI Build summary）
export IMAGE=ghcr.io/unlimitwave/langgraph-skill-agent
export TAG=<git-commit-sha>
make deploy-remote
```

## 日常运维

```bash
# 查看状态
docker compose -f docker-compose.yml -f docker-compose.prod.yml ps

# 日志
docker compose -f docker-compose.yml -f docker-compose.prod.yml logs -f app

# 回滚
export TAG=<previous-good-sha>
make deploy-remote
```

## 健康检查

Streamlit 内置端点：`http://127.0.0.1:8501/_stcore/health`

`deploy-remote.sh` 部署后会自动 curl 该地址，最多等待 60 秒。

## 安全约定

- API Key、`POSTGRES_PASSWORD` 等只写在服务器 `.env`，**不**写入镜像、**不**提交 git
- 生产部署使用 **commit SHA** 作为 TAG，不用 `latest`
- Postgres 端口默认仅绑定 `127.0.0.1`（见 `docker-compose.prod.yml`）

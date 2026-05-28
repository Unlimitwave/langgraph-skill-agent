"""应用级 logging：诊断走 stderr，避免与 stdin/stdout 上的对话流混在一起。"""

from __future__ import annotations

import logging
import os
import sys


def configure_logging() -> None:
    """从环境变量 LOG_LEVEL 读取级别（默认 INFO），格式含时间与 logger 名。"""
    raw = (os.environ.get("LOG_LEVEL") or "INFO").strip().upper()
    level = getattr(logging, raw, None)
    if not isinstance(level, int):
        level = logging.INFO
    fmt = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    datefmt = "%H:%M:%S"
    kwargs: dict = {
        "level": level,
        "format": fmt,
        "datefmt": datefmt,
        "stream": sys.stderr,
    }
    # Python 3.8+
    try:
        logging.basicConfig(**kwargs, force=True)
    except TypeError:
        logging.root.handlers.clear()
        logging.basicConfig(**kwargs)

    # 第三方与工具预览：默认不刷屏（仍可通过子 logger 单独调低级别）
    for name in ("httpx", "httpcore", "mcp"):
        logging.getLogger(name).setLevel(logging.WARNING)

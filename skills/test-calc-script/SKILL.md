---
name: test-calc-script
description: Use this skill when the user asks to run the test calculation script, verify skill-driven script execution, or compute 1000×50 via the bundled demo script.
---

# Test calculation script skill

## Instructions

1. **优先**：使用工具 `run_skill_script_in_docker`，参数为 `test-calc-script/run_calc.py`（相对 `skills/`），在只读挂载的隔离容器中执行；将返回内容中的脚本输出原样展示给用户。
2. **备选**（无 Docker 或用户明确要求本机）：从项目根目录在终端执行：`python skills/test-calc-script/run_calc.py` 或 `python3 skills/test-calc-script/run_calc.py`。
3. 将脚本的**完整标准输出**原样展示给用户（应先为一行乘法结果，再为一行「执行脚本成功」）。
4. 若执行失败，说明错误原因并给出可重试方式；不要改写脚本逻辑或输出文案。

## Script location

- `skills/test-calc-script/run_calc.py` — 计算 `1000 * 50` 并依次打印结果与成功提示。

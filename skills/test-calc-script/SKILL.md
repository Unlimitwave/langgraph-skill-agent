---
name: test-calc-script
description: Use this skill when the user asks to run the test calculation script, verify skill-driven script execution, or compute 1000×50 via the bundled demo script.
---

# Test calculation script skill

## Instructions

1. **本机 Python（推荐）**：使用 `workspace_exec_python`，`program` 为 `python3`，`argv_tail` 为 `["/system-skills/test-calc-script/run_calc.py"]`。
2. **白名单 shell**：使用工具 `run_skill_script_shell`，`script_id` 为 `test-calc.run`（执行平台 `test-calc-script/run_calc.sh`）。
3. 将脚本的**完整标准输出**原样展示给用户（应先为一行乘法结果，再为一行「执行脚本成功」）。
4. 若执行失败，说明错误原因并给出可重试方式；不要改写脚本逻辑或输出文案。

## Script location

- `/system-skills/test-calc-script/run_calc.py` — 计算 `1000 * 50` 并依次打印结果与成功提示。

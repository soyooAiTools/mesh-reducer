# CLAUDE.md

开始工作前先读 `HANDOFF.md`，里面有项目背景、当前指标、已知弱点和下一步计划。

## 要点

- 这是一个"像美术一样删环线"的四边面减面工具。核心操作是环带塌陷（`core/character.py`）。**不要改成 QEM 三角塌陷或重新布线**，之前的失败就是这么来的。唯一的附加操作是 `heal_quads`（塌陷后把"被对角线劈开的四边面"的两个三角面拼回四边面，不移动顶点），用户已于 2026-11-14 确认保留。
- 全三角面的源模型（如范海辛）不要用 Decimate 交付（美术已否决：破面、无四边面）。改用 `core/retopo.py` 的"拓扑烘焙"原型（拼回四边面 → 删环线 → 重新烘焙 BC/N/AO → 传权重），见 HANDOFF 2c。头发这类分散发丝自动处理效果差，交给美术。
- 用户需要独立的 Windows exe（`app/`）。它在后台调用 Blender 5.2（`runner.py`），**不用 Blender MCP**。
- 用户沟通用中文。

## 开发

```bash
pip install bpy==5.2.4 numpy PySide6
python3 dev/bench.py 1500            # 回归基线（Mac arm64）：faces 1499 / quads 1309 / Hausdorff≈0.81% / 翻转 0 / UV 岛 99/99 / 闭合四边环 148
python3 runner.py -- --in testdata/jier_high.fbx --out /tmp/o.fbx --faces 1500 --mode character --renders /tmp/r
python3 evaluate.py testdata/jier_high.fbx /tmp/o.fbx --texture testdata/jier_diffuse.png --regions testdata/jier_regions.json
QT_QPA_PLATFORM=offscreen python3 app/main.py --selftest --blender python3 --in testdata/jier_high.fbx --outdir /tmp/st
```

- 脚本不要用标准库模块的名字（如 `inspect.py`），否则 `import bpy` 会崩溃。
- 每个算法改动都要给出改动前后的指标，并亲自看渲染图（左边是结果，右边是原模型）。
- 推送到 `main` 会触发 Windows CI。结果在 `ci-results` 分支，看 `RUN_INFO.txt` 里的 `job_status`。

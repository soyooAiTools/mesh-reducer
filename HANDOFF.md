# MeshReducer 交接文档

> 更新：2026-11-14。仓库：`soyooAiTools/mesh-reducer`（私有）。分支 `main` 是代码；分支 `ci-results` 是 Windows CI 自动推送的产物（exe 分卷 + 测试结果），不要手动改。

## 1. 项目目标

做一个能达到"人工减面"水平的工具，处理游戏角色和带 UV 的游戏资产。

- **形态**：独立的 Windows 桌面软件（exe）。它在后台调用用户本机安装的 **Blender 5.2**（`blender -b`，不弹窗口）完成几何运算和 FBX 读写。另外附带一个可选的 Blender 插件，和 exe 共用同一套核心代码。
- **明确不用 Blender MCP**：MCP 需要 Blender 开着并装好监听插件，在 exe 里多一层只会多一个故障点。
- **当前验收样例**：吉尔（`testdata/jier_high.fbx`），从 2639 面减到 **1500 面**。

### 背景：之前为什么失败

用户先用 Codex、Kimi 跑了一个多月都没成功。失败的原因是它们用的是通用的 QEM 三角塌陷，结果全是三角面，环线全部被毁掉。美术的 9 条反馈见 `docs/feedback/`（docx 原件和截图，Unity 截图，左边是旧 AI 结果，右边是原模型）：

1. 手的形状跑偏
2. 手臂没有环线，左臂有严重黑块（法线坏了）
3. 手指缺失或变形
4. 背部扭曲拉伸，布线不支持动画
5. 膝盖没有环线，关节处三角面太多
6. 裙子变形严重
7. 肩部布线不是圈状环线
8. 靴子贴图扭曲严重，布线不对，形状和原模型差很多
9. 腰部布线不对，缺少四边形环线

**关键事实**：所谓"高模"其实只有 2639 面，是美术手做的、以四边面为主的干净模型（2113 个四边面 + 526 个三角面）。所以正确的做法是**像美术一样整条删除环线**，而不是重新布线或打碎成三角面。

## 2. 当前状态

| 项目 | 状态 |
|---|---|
| 角色模式算法（1500 面） | ✅ 可用，指标见下 |
| 道具模式（全三角面的石头） | ✅ 可用 |
| PySide6 中文 GUI，以及 `--cli`、`--selftest` 模式 | ✅ 在 Windows CI 上通过 |
| Windows exe（PyInstaller） | ✅ CI 自动打包，在真实 Blender 5.2.4 上跑通，中文和空格路径 OK |
| Blender 插件 | ✅ 只在 bpy 5.2.4 模块里装过、跑过，**没有在真实 Blender 界面里测过** |
| 美术在 Unity 里复核 1500 面结果 | ⏳ **还没做**，这是下一步最重要的输入 |
| 带骨骼和蒙皮的模型、形态键、顶点色 | ❌ 没有测试样例；权重转移写了，但没测 |

### 吉尔 1500 面的指标

距离单位是包围盒对角线的百分比。

| 面数（四边/三角） | Unity 三角面 | Hausdorff | 平均距离 | 法线偏差 | UV 岛（通道 1） | 贴图误差 均值 / p95 | 闭合四边环 | 用时 |
|---|---|---|---|---|---|---|---|---|
| 1498（1219/279） | 2717（原 4752） | 0.86 | 0.073 | 5.8° | 99/99 | 2.1 / 4.9 | 146/197 | 约 10 s（Linux），28 s（Windows CI，含预览） |

- 0 个多边面、0 个非流形、0 个退化面、0 个翻转面、0 个翻转 UV 面。
- 关节环线（结果/原始，括号内是结果/原始的三角面数）：膝 2/4（0/2），腕 3/3 和 2/3，肘 2/1 和 2/4（三角面 6→0、4→1），肩 1/2 和 2/3，腰 4/7（三角面 24→8），颈 1/2。

### 9 条反馈对照（1500 面）

- 第 1、2、4、5、6 条：已解决。
- 第 3 条：基本解决。手指保住了，但指尖的倒角环线没了。
- 第 7、9 条：部分解决。有圈状环线，但数量比原模型少。
- 第 8 条：部分解决。UV 和整体形状正确，但**鞋跟被压平成斜坡**，绑带倒角也没了。这是目前最明显的问题。

### 已知弱点（按优先级）

1. **靴子侧面轮廓（鞋跟）被压平。** 建议加一个"轮廓保护"代价：对侧视/正视剪影边上的顶点提高代价，或者检测高曲率凸起的环。
2. **面数压到 40% 以下时质量明显下降。** 40% 时一侧手臂从肩到腕没有环线，25% 时形状和贴图都明显损坏。现在的定位是"适度减面"。
3. **关节检测是几何启发式。** 它只能识别比相邻区域更密的闭合环带。吉尔有一侧手肘的原始环线是以三角面结束的，所以检测不到。原模型的左右手臂拓扑也不对称，导致结果也不对称。
4. **裙摆到膝盖之间只剩一长段。** `tube_aspect=1.5` 只部分起效。
5. **UVChannel_2 被改动了。** 原数据像垃圾数据（430 个零面积面，坐标范围 −32 到 60），所以被 `seam_layers='AUTO'` 跳过约束，只做插值。如果用户实际用这个通道，需要改成约束。
6. **贴图感知代价**（`core/texture.py`、`Params.tex_weight`）试过两次都没有降低贴图误差，默认关闭，属于实验代码。
7. **速度。** 纯 Python 实现，吉尔约 10 s，大模型会慢。
8. **evaluator 的翻转计数用最近面启发式**，在 40% 和 25% 时报了 1–4 个翻转。reducer 自己的精确检查是 0，推测是匹配误差，但没有证明。

## 3. 算法概要（`core/character.py`）

- **唯一的操作是环带塌陷（ring / poly-chord collapse）**：删掉一整条四边形带，相邻两条环线合并。环带在三角面处终止（三角面随带一起删）、在边界处终止，或者自己闭合。不会产生多边面，四边面不会变成三角面。
- **代价** = (相对原始环线间距的拉伸)² × (1 + 相对误差²) × 保护系数。
  - 误差是到原模型的实测双向距离，除以局部特征尺寸（管径或射线测厚，上限是对角线的 3.5%）。所以手指比躯干受到的保护强得多。
  - 开放边界（裙摆、开衩、发梢）的判定严格 2.5 倍（`border_weight`）。
  - 误差和四边形扭曲都有硬上限，只有在没有其他选择时才放宽。
- **无骨骼的关节检测**（`core/joints.py`）：把闭合四边环带串成"管"，明显比同管典型环带更窄的带（膝、腕、肘）代价 ×`joint_strength`。另外支持用户手刷的 protect 顶点组。
- **约束**：UV 接缝、边界和硬边上的顶点只能沿着它们滑动，角点固定，两条接缝不能合并。所有 UV 层按 wedge 插值。任何操作如果导致面或 UV 翻转、非流形、重复或 doublet，都会被撤销（`undo`）。
- **部分环带**：整条带受阻时可以只删一段，两端各留一个三角面。闭合环、关节带和保护区内禁止这样做，三角面比例不超过原模型。
- **对称**：自动检测镜像轴（吉尔是 X 轴），镜像环带成对塌陷。
- **拟合**（`core/fit.py`）：最后做最小二乘拟合，顶点只沿法线移动，用来抵消收缩。
- **法线**：吉尔是全平直着色，输出保持平直；平滑模型从原模型转移法线。
- **道具模式**（`core/prop.py`）：默认 strict 模式是自写的单边塌陷，三个 UV 层都带接缝约束和翻转约束。`--prop-method fast` 改用 Blender Decimate，几何偏差约小一半，但接缝附近会有 14–17 个翻转或拉伸的 UV 三角面。

参数都在 `core/character.py` 的 `Params` 类里，每个参数都有注释。

## 4. 代码结构

```
core/          算法（只依赖 bpy + bmesh + mathutils + numpy），不依赖 UI
  api.py         对外入口 reduce_character / reduce_prop，被插件、runner、CLI 共用
  rmesh.py       轻量可编辑多边形网格（纯 Python + numpy）
  character.py   角色模式：Params、Reducer（环带塌陷主循环 run()）
  joints.py      闭合环带、管、关节分析
  fit.py         最小二乘回贴
  prop.py        道具模式
  metrics.py     评估指标（evaluate.py 和 runner 共用）
  preview.py     Workbench 无头渲染对比图；缺贴图时用灰色，无 GPU 时回退到 Cycles
  texture.py     贴图采样（实验）
  io_util.py     FBX / OBJ / GLB 导入导出
runner.py      在 Blender 里运行的无头入口；输出 PROGRESS / LOG / RESULT {json} / ERROR 行
reduce_cli.py  开发用命令行；evaluate.py 用于独立评估
app/           PySide6 GUI（中文）；jobs.py 用 QProcess 调 Blender；blender_finder.py 自动查找 Blender
addon/         Blender 插件；python addon/build_addon.py 生成扩展 zip
dev/           开发脚本：bench.py（减面→评估→关节区统计），closeups.py（局部特写渲染）
testdata/      jier_high.fbx、jier_diffuse.png、rock.fbx、jier_regions.json（evaluator 用的关节区域）
docs/feedback/ 美术对旧 AI 结果的反馈（docx + 截图）
mesh_reducer.spec / build_windows.bat / requirements.txt   打包
.github/workflows/windows.yml   Windows CI
```

## 5. 本地开发环境

需要 Python 3.13 和 bpy 模块，版本和用户的 Blender 5.2 一致：

```bash
pip install bpy==5.2.4 numpy PySide6        # bpy 包很大，安装要几分钟
```

> ⚠️ **坑**：脚本不要和标准库模块同名（比如 `inspect.py`、`types.py`），否则 `import bpy` 会直接崩溃，报 glog / Bus error。

常用命令：

```bash
# 减面（不需要安装 Blender，直接用 bpy 模块）
python3 runner.py -- --in testdata/jier_high.fbx --out /tmp/out.fbx --faces 1500 --mode character --report /tmp/r.json --renders /tmp/r

# 独立评估（带关节区统计）
python3 evaluate.py testdata/jier_high.fbx /tmp/out.fbx --texture testdata/jier_diffuse.png --regions testdata/jier_regions.json

# 开发基准：一次性输出 info、指标行、几何误差、关节区（环数/原环数/三角/原三角）
python3 dev/bench.py 1500

# 用真实 Blender 跑
blender -b --factory-startup --python runner.py -- --in X.fbx --out Y.fbx --faces 1500 --mode character

# GUI（Linux 无头自测）
QT_QPA_PLATFORM=offscreen python3 app/main.py --selftest --blender python3 --in testdata/jier_high.fbx --outdir /tmp/st
```

`--blender` 指向的路径如果文件名以 `python` 开头，app 会直接用 `python runner.py`，不走 `blender -b`，方便在没有 Blender 的环境里测试。

**回归基线**：在 1500 面上，`dev/bench.py 1500` 应该得到 faces 1498、quads 1219、Hausdorff 约 0.86%、翻转面 0、UV 岛 99/99。改算法后先跑这一项对比。

## 6. Windows 构建与发布（CI）

- 每次推送到 `main`，都会触发 `.github/workflows/windows.yml`（windows-latest）：
  1. 用 PyInstaller 打包 `dist/MeshReducer/MeshReducer.exe`；
  2. 自动下载最新的 Blender 5.2.x；
  3. 在带中文和空格的路径下跑三项测试：吉尔 1500 面（`--cli`）、石头 50%（`--cli --mode prop`）、GUI 自测（`--selftest`，会截图 `gui.png`）；
  4. 上传 artifacts；
  5. 把结果和 exe 分卷（每卷 45MB 以内）强制推送到 `ci-results` 分支。
- 取结果：
  ```bash
  git clone --depth 1 -b ci-results https://github.com/soyooAiTools/mesh-reducer ci
  cat ci/RUN_INFO.txt   # job_status=success 表示通过
  cat ci/MeshReducer-win64.zip.part* > MeshReducer-win64.zip
  ```
  也可以在 GitHub Actions 页面直接下载 artifact。
- 最近一次成功运行：commit `a082109`，三项测试全部 exit=0，GUI 字体为 Microsoft YaHei UI，Blender 5.2.4。
- 窗口版 exe 没有 stdout，所以 CI 只能通过 `--report` / `--log` 文件和退出码判断结果。

## 7. 建议的下一步

1. **先拿到美术对 1500 面结果的 Unity 复核**（`ci-results` 分支里的 `ci_results/jier_1500.fbx`）。按反馈驱动修改，不要盲调参数。
2. 修鞋跟和轮廓保护（弱点 1），用 `dev/closeups.py` 的 `bootside` 视角验证。
3. 向用户要一个**带骨骼的角色**，测试权重转移，并考虑用骨骼位置代替几何启发式来检测关节（弱点 3）。
4. 确认 UVChannel_2 是否在用（弱点 5）。
5. 性能：需要时把 `Reducer.run` 的热点（`_eval_edge`、`true_error`）向量化，或改写成 C 扩展。

## 8. 约定

- 每个改动都要用 `evaluate.py` 或 `dev/bench.py` 量化对比，并用 `preview` / `closeups` 看图。**不要只看代码判断效果。**
- 渲染图里**左边是结果，右边是原模型**。
- 用户的 Blender 版本是 5.2，用户的电脑是 Mac mini，Windows 只在 CI 上跑。

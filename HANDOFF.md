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
| 1499（1204/295） | 2703（原 4752） | 0.99 | 0.072 | 5.7° | 99/99 | 2.0 / 5.5 | 142/197 | 约 3 s（Mac arm64），28 s（Windows CI，含预览） |

> 以上是 **Mac arm64**（bpy 5.2.4）上加了轮廓保护（`profile_protect`）之后的数字。同一台机器上改动前（`profile_protect=False`）是：1498（1216/282）、Hausdorff 1.07、平均 0.070、法线 5.7°、贴图误差 2.0、闭合四边环 154/197。旧文档里的 1219 / 0.86 是 Linux 上的数字，和 Mac 有浮点差异，不要混着比。

- 0 个多边面、0 个非流形、0 个退化面、0 个翻转面、0 个翻转 UV 面。
- 关节环线（结果/原始，括号内是结果/原始的三角面数）：膝 2/4（0/2），腕 3/3 和 3/3，肘 2/1 和 2/4（三角面 6→0、4→1），肩 1/2 和 2/3，腰 4/7（三角面 24→8），颈 1/2（三角面 6→0）。和加轮廓保护之前完全一样（只有颈部三角面 3→0）。
- 40%（1056 面）对照，改动前 → 改动后：Hausdorff 2.85 → 1.24，平均 0.155 → 0.142，法线 10.1° → 9.7°，四边面 858 → 860，UV 岛 99/99，翻转 UV 0 → 0，evaluator 翻转 2 → 2，闭合四边环 119 → 112。

### 9 条反馈对照（1500 面）

- 第 1、2、4、5、6 条：已解决。
- 第 3 条：基本解决。手指保住了，但指尖的倒角环线没了。
- 第 7、9 条：部分解决。有圈状环线，但数量比原模型少。
- 第 8 条：大部分解决。UV 和整体形状正确，**鞋跟现在保住了台阶形状**（前壁是竖直的，见 `dev/tmp/new_bootside.png`，改动前是 `dev/tmp/base_bootside.png`）。绑带的凸起倒角还是没了（贴图还在），鞋底前缘的包边也变薄了。

### 已知弱点（按优先级）

1. **靴子绑带的凸起倒角没了（鞋跟已修）。**
   - 鞋跟被压平的原因：误差是除以"局部特征尺寸"来判断的，而靴子的特征尺寸是靴筒直径，被封顶在对角线的 3.5%（6.4 cm）。鞋跟台阶只有 2.9 cm 高，把它抹成斜坡的实测误差约 1.1 cm，相对误差只有 0.17，低于硬上限 0.3，所以很早就被删了。
   - 修法（`Reducer._init_relief`，参数 `profile_protect` / `crease_angle` / `profile_min`）：原模型里夹在一条凹折痕和一条对面的凸折痕之间的四边面是"台阶立面"。它的顶点上的边（沿这两条折痕走的边除外）改用台阶高度作特征尺寸（存在 `Reducer.eprofile` 里，随塌陷重新映射，和 `etube` 一样）。不是靠坐标写死的。
   - 绑带没保住是面数预算问题，不是检测问题：绑带立面不到 1 cm 高，`profile_min=0.008`（对角线的 0.8%，约 1.5 cm）把它当倒角忽略了。设 `profile_min=0.003` 能保住绑带（见 `dev/tmp/strap_bootside.png`），但两只靴子要多占约 60 面，这些面只能从头发等大面积部件上扣：1500 面时平均距离 0.072 → 0.083、Hausdorff 0.99 → 1.87；1056 面时更差（平均 0.142 → 0.204，翻转 UV 3 个，一侧膝盖环线丢失）。所以默认不开。
   - 代价：1500 面时平均距离比改动前高 3%（0.070 → 0.072），四边面少 12 个，闭合四边环 154 → 142，拇指少了一条纵向环线。Hausdorff 和 40% 档的指标变好。30%（792 面）档好坏参半，本来就不在定位范围内。
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
  - 轮廓保护（`profile_protect`）：跨过台阶立面（凹折痕和凸折痕之间的四边面）的边，特征尺寸改用台阶高度，鞋跟这类台阶不会被抹成斜坡。
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

**回归基线**（Mac arm64，bpy 5.2.4）：在 1500 面上，`dev/bench.py 1500` 应该得到 faces 1499、quads 1204、Hausdorff 约 0.99%、平均 0.072%、法线 5.7°、翻转面 0、UV 岛 99/99、翻转 UV 0。Linux / Windows 上会有小的浮点差异（四边面数差几个、Hausdorff 差 0.1 左右）。改算法后先跑这一项对比。

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
2. 鞋跟已修（弱点 1）。剩下的绑带倒角要等面数预算有余量再说，用 `dev/closeups.py` 的 `bootside` 视角验证。
3. 向用户要一个**带骨骼的角色**，测试权重转移，并考虑用骨骼位置代替几何启发式来检测关节（弱点 3）。
4. 确认 UVChannel_2 是否在用（弱点 5）。
5. 性能：需要时把 `Reducer.run` 的热点（`_eval_edge`、`true_error`）向量化，或改写成 C 扩展。

## 8. 约定

- 每个改动都要用 `evaluate.py` 或 `dev/bench.py` 量化对比，并用 `preview` / `closeups` 看图。**不要只看代码判断效果。**
- 渲染图里**左边是结果，右边是原模型**。
- 用户的 Blender 版本是 5.2，用户的电脑是 Mac mini，Windows 只在 CI 上跑。

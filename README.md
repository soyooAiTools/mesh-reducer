# MeshReducer 模型减面工具

面向游戏美术的减面工具：**角色模型按“美术手工减面”的方式删除整条环线**（保持四边形布线、关节环线、UV 接缝和轮廓），静态道具走带 UV 约束的三角面减面。
它是一个独立桌面程序（Windows 下为 `MeshReducer.exe`），在后台调用本机安装的 **Blender 5.2**（无窗口）完成计算，不需要打开 Blender。

```
core/            减面算法（bmesh / mathutils / numpy，纯 Python，与界面无关）
runner.py        无界面入口，由 Blender 执行：blender -b --factory-startup --python runner.py -- ...
reduce_cli.py    简化命令行：python3 reduce_cli.py in.fbx out.fbx --ratio 0.4 --mode character
evaluate.py      结果评测（偏差、UV、环线等指标 + 对比渲染）
app/             独立桌面程序（PySide6，中文界面）
addon/           可选的 Blender 插件（复用 core/，N 面板“减面”）
mesh_reducer.spec / build_windows.bat   PyInstaller 打包
testdata/        测试模型（吉尔、石头）
```

## 1. 安装与运行

### Windows 独立程序
1. 安装 [Blender 5.2](https://www.blender.org/download/)（默认路径即可）。
2. 解压 `MeshReducer` 文件夹，双击 `MeshReducer.exe`。
3. 程序会自动查找 `blender.exe`（`C:\Program Files\Blender Foundation\Blender 5.x`、Steam 库、注册表、PATH）。找不到时点“浏览…”手动选择，路径会被记住。
4. 把 FBX / OBJ / GLB 拖进窗口（可多个、可拖文件夹）→ 设定目标面数或比例 → “开始减面”。
5. 完成后右侧显示 **减面结果（左）/ 原始模型（右）** 的线框对比图和指标表；输出文件在输出文件夹（默认与输入同目录，文件名加 `_reduced`），旁边有 `*_report.json` 和 `previews/`。

自己打包 exe：在 Windows 上运行 `build_windows.bat`（需要 Python 3.10+），得到 `dist\MeshReducer\MeshReducer.exe`（one-folder，`runner.py` 与 `core/` 作为数据文件一起打包，由 exe 交给 blender.exe 执行）。

### 命令行（无窗口，适合批处理 / CI）
```
MeshReducer.exe --cli --blender "C:\...\blender.exe" --in 模型.fbx --out 结果.fbx --faces 1500 --mode character --report 报告.json --log 日志.txt
MeshReducer.exe --selftest --blender "C:\...\blender.exe" --in 模型.fbx --outdir 目录     （真实主窗口跑一遍并截图 gui.png / selftest.json）
blender -b --factory-startup --python runner.py -- --in X.fbx --out Y.fbx --faces 1500 --mode character --report r.json --renders 预览目录
python3 reduce_cli.py in.fbx out.fbx --ratio 0.4 --mode character        （需要 pip 的 bpy 模块）
python3 evaluate.py 原始.fbx 结果.fbx --texture 贴图.png --out 指标.json --renders 渲染目录
```
`--cli` 成功返回 0。路径可以包含中文和空格。`runner.py` 的标准输出协议：`PROGRESS 0.42 文本`、`LOG ...`、最后一行 `RESULT {json}`，失败时 `ERROR ...`。

### Blender 插件（可选）
`python addon/build_addon.py` 生成 `dist/mesh_reducer_addon.zip`；Blender 5.2 → 偏好设置 → 获取扩展 → 右上角菜单“从磁盘安装”（或把 zip 拖进 Blender）。3D 视图按 N → “减面”页签。操作结果是一个新物体，原模型只被隐藏、不会被修改。

### 为什么用“后台 Blender 子进程”而不是 Blender MCP
MCP 需要一个已经打开并装好插件的 Blender 和一条始终在线的连接，任何一环断开任务就失败，也无法批处理。子进程方式每次任务启动一个全新的无界面 Blender（`--factory-startup`，不受用户插件和配置影响），用完即退，可以取消、排队、在没有显示器的机器上运行，结果完全可复现。

## 2. 参数

| 界面 | 命令行 | 说明 |
|---|---|---|
| 目标面数 / 目标比例 | `--faces N` / `--ratio 0.4` | 面数指导出的多边形数（四边形算 1 个面）。报告里同时给出三角化后的三角面数（Unity 里看到的数）。文件里有多个网格时按比例分配。 |
| 模式 | `--mode character\|prop\|auto` | 角色：环线减面；静态道具：三角面减面；自动：四边形超过 30% 当作角色。 |
| 对称 | `--symmetry auto\|x\|y\|z\|none` | 镜像的两条环线同时删除，结果保持对称（只对原本严格镜像的部分生效）。 |
| 保护顶点组 | `--protect-group 名称` | 顶点组权重越大，穿过这些顶点的环线越晚被删（可用来手工标记关节、脸等）。 |
| 保持 UV 接缝 | `--no-seams` 关闭 | 接缝上的点只能沿接缝滑动，接缝不会被跨越或合并，所有 UV 层按“楔”插值。 |
| 保持硬边 | `--no-sharp` 关闭 | 拆分法线形成的硬边同样作为约束。 |
| 自动保护关节环线 | `--no-joints` 关闭，`--joint-strength 5` | 没有骨架时，自动找四肢上“明显比相邻环更密”的闭合环（膝、肘、腕）并尽量保留。 |
| 生成预览图 | `--renders 目录` | Workbench 渲染；没有可用显卡时自动改用 Cycles(CPU)，仍失败只影响预览，不影响结果。 |
| — | `--prop-method strict\|fast` | 道具模式：strict（默认，自带的带约束边折叠，UV 不翻转）；fast（Blender 自带 Decimate，更快、几何略好，但接缝旁会有个别 UV 三角形翻转）。 |

## 3. 算法（角色模式）

美术给低模减面时不会重新拓扑，而是**删环线**。本工具唯一的操作就是“边环塌陷”：从一条边出发，沿四边形的对边一路走出一圈边环（poly-chord），把这一圈边全部塌陷 = 删掉一整条四边形带、相邻两条环线合成一条。边环遇到三角形（三角形随之消失）或开放边界结束，或者自己闭合。因此：

* 四边形永远不会被拆成三角形，不产生 N 边形，剩下的环线仍然是完整环线；
* 代价 = （相对原始布线间距的拉伸）² ×（1 + 相对误差²）× 保护系数。误差用**实测**的双向距离（合并进来的原始顶点到新面、新面到原始表面），并除以“局部特征尺寸”（四肢/手指的粗细），所以手指、小配件比躯干受保护得多；另含贴图滑动量、开放边界（裙摆、袖口、发梢）偏移的加权；
* 布线间距按美术原始间距成比例变稀，而不是先把密的地方删光；自动识别的关节环（膝、肘、腕）和“保护顶点组”进一步推迟删除；四肢上一段四边形不允许比肢体直径长太多（保证有环线可弯）；
* UV 接缝 / 开放边界 / 硬边：其上的点只沿该特征线滑动，角点固定，两条不同的接缝永不合并；每个 UV 层的每个角按楔插值；会翻转面法线或 UV 三角形、产生非流形/重复面/退化面的操作直接拒绝；
* 某条环线只在局部被挡住或局部太贵时，允许只删其中一段，在两端各收成一个三角形（美术的“收线”）；这种收尾不允许落在闭合的四边形环内、关节环和保护区里，三角形比例不超过原模型；
* 对称模型成对执行；
* 最后做一次“贴回原始表面”的最小二乘拟合（只沿法线移动，UV 不变），补偿删线造成的体积收缩；
* 原模型是硬表面平直着色（如吉尔）时保持平直着色；带自定义法线的平滑模型从原模型按最近面插值传递法线；有蒙皮权重时从原模型传递顶点组并保留骨架修改器。

静态道具模式（strict）用同一套约束做逐边折叠（二次误差 + 贴图滑动），三套 UV 的接缝都作为约束。

## 4. 评测指标（evaluate.py / 报告 JSON）
面数、四边形/三角形/N 边形、三角化后三角面数、极点数；双向 Hausdorff 与平均偏差（占包围盒对角线的百分比，另按部件给出）；法线偏差、翻转面、缠绕方向不一致的边、非流形边、退化面；每个 UV 层：UV 岛数量（结果/原始）、相对原模型翻转的 UV 面、拉伸、贴图颜色误差（在结果表面采样，对比该点贴图颜色与原模型最近点的贴图颜色）；闭合四边形环数量，以及指定关节区域（`--regions`，见 `testdata/jier_regions.json`）内的环数和三角形数。

## 5. 已知限制（请务必阅读）
* **减面幅度有下限。** 吉尔 2639 面 → 1500 面（57%）效果接近手工；40% 时手臂/腿的环线明显变少、靴子和背部贴图开始走样；25% 时形状和贴图都有明显损坏，只能当远景 LOD。原模型本身已经是很省的手工低模，再往下需要美术重新设计布线，本工具做不到。
* 关节识别是几何启发式（只认“闭合且比相邻环更密”的环）。原模型肘部如果是三角形收线而不是闭合环（吉尔的一侧手臂就是），就识别不到；要保证效果请绑定后用权重、或刷一个“保护顶点组”。
* 不会新建环线、不会改善原布线，只会删；原模型布线不对称/不规整的地方，结果也不对称。
* 贴图内容不参与计算：一条环线如果正好压在贴图的颜色分界上（鞋底边、带子边），删掉后分界会略微变形。
* 删“收线”段会留下三角形（比例不超过原模型）。
* 第二套 UV 如果是无效数据（大量零面积面，如吉尔的 UVChannel_2）只做插值、不作为接缝约束。
* 速度：纯 Python。吉尔约 10–20 秒，7 千三角面的石头约 10 秒；几万面以上的角色会比较慢。
* 形态键（Shape Keys）不支持；多材质可以；顶点色不保留。
* 预览渲染需要 Blender 能创建 GPU 上下文，否则退回 Cycles CPU；都失败时只是没有预览图。
* Windows exe 无法在 Linux 开发机上生成，只验证了：PyInstaller 规格在 Linux 上能打出同结构的程序并跑通 `--cli` / `--selftest`；Windows 构建由 `.github/workflows/windows.yml` 完成。

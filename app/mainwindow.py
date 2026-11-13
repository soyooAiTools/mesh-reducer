# -*- coding: utf-8 -*-
"""Main window of MeshReducer (Chinese UI)."""
import os

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QFont
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QLabel, QLineEdit, QPushButton,
    QFileDialog, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QRadioButton, QProgressBar, QPlainTextEdit,
    QTableWidget, QTableWidgetItem, QTabWidget, QSplitter, QHeaderView, QAbstractItemView, QMessageBox,
    QScrollArea, QSizePolicy)

from . import paths, blender_finder
from .jobs import Job, JobController, MODEL_EXT, default_output

STATUS_CN = {'queued': '排队中', 'running': '处理中', 'done': '完成', 'failed': '失败', 'cancelled': '已取消'}
MODES = [('角色（保持四边形与环线）', 'character'), ('静态道具（三角面）', 'prop'), ('自动判断', 'auto')]
SYMS = [('自动检测', 'auto'), ('X 轴', 'x'), ('Y 轴', 'y'), ('Z 轴', 'z'), ('关闭', 'none')]
FORMATS = [('与输入相同', ''), ('FBX', '.fbx'), ('OBJ', '.obj'), ('GLB', '.glb')]

METRIC_ROWS = [
    ('faces', '面数（多边形）'), ('quads', '四边形'), ('tris', '三角形'), ('ngons', 'N 边形'),
    ('unity_tris', '三角化后三角面数（Unity）'), ('quad_pct', '四边形比例 %'),
    ('hausdorff_pct', '最大偏差 Hausdorff（% 包围盒对角线）'), ('mean_pct', '平均偏差（% 包围盒对角线）'),
    ('normal_dev_deg', '法线平均偏差（度）'), ('flipped', '翻转面'), ('nonmanifold', '非流形边'),
    ('degenerate', '退化面'), ('uv_islands', 'UV 岛 结果/原始'), ('uv_flipped', 'UV 翻转面'),
    ('tex_err', '贴图误差（0-255）'), ('quad_rings', '闭合四边形环 结果/原始'),
]


class PreviewLabel(QLabel):
    def __init__(self):
        super().__init__('完成后在此显示 减面结果（左） / 原始模型（右） 对比图')
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self._pix = None

    def set_image(self, path):
        pm = QPixmap(path) if path and os.path.isfile(path) else QPixmap()
        self._pix = pm if not pm.isNull() else None
        self._update()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._update()

    def _update(self):
        if self._pix is None:
            return
        self.setPixmap(self._pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


class MainWindow(QMainWindow):
    def __init__(self, blender_path=''):
        super().__init__()
        self.setWindowTitle('MeshReducer 模型减面工具')
        self.resize(1240, 800)
        self.setAcceptDrops(True)
        self.settings = paths.load_settings()
        self.controller = JobController(self)
        self.controller.jobChanged.connect(self.on_job_changed)
        self.controller.logLine.connect(self.on_log)
        self.controller.queueFinished.connect(self.on_queue_finished)
        self.jobs = []
        self.inputs = []
        self.blender_version = None
        self._build()
        self._load_settings()
        if blender_path:
            self.ed_blender.setText(blender_path)
            self.check_blender()
        elif self.ed_blender.text():
            QTimer.singleShot(50, self.check_blender)
        else:
            QTimer.singleShot(50, self.detect_blender)

    # ------------------------------------------------------------------ ui
    def _build(self):
        cw = QWidget()
        self.setCentralWidget(cw)
        root = QVBoxLayout(cw)

        # Blender
        gb = QGroupBox('Blender（后台运行，不会弹出窗口）')
        h = QHBoxLayout(gb)
        self.ed_blender = QLineEdit()
        self.ed_blender.setPlaceholderText('blender.exe 的路径（需要 Blender 5.2）')
        self.ed_blender.editingFinished.connect(self.check_blender)
        b1 = QPushButton('浏览…')
        b1.clicked.connect(self.browse_blender)
        b2 = QPushButton('自动检测')
        b2.clicked.connect(self.detect_blender)
        self.lb_blender = QLabel('未检测')
        h.addWidget(self.ed_blender, 1)
        h.addWidget(b1)
        h.addWidget(b2)
        h.addWidget(self.lb_blender)
        root.addWidget(gb)

        split = QSplitter(Qt.Horizontal)
        root.addWidget(split, 1)

        # ---------------- left
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        gin = QGroupBox('输入模型（FBX / OBJ / GLB，可拖入文件，支持批量）')
        v = QVBoxLayout(gin)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(['文件', '状态', '面数', '用时'])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for c in (1, 2, 3):
            self.table.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.itemSelectionChanged.connect(self.on_select)
        v.addWidget(self.table)
        hb = QHBoxLayout()
        for text, fn in (('添加文件…', self.add_files), ('添加文件夹…', self.add_folder),
                         ('移除所选', self.remove_selected), ('清空', self.clear_inputs)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            hb.addWidget(b)
        v.addLayout(hb)
        lv.addWidget(gin, 1)

        gout = QGroupBox('输出')
        g = QGridLayout(gout)
        self.ed_out = QLineEdit()
        self.ed_out.setPlaceholderText('留空 = 与输入文件相同的文件夹')
        bo = QPushButton('浏览…')
        bo.clicked.connect(self.browse_out)
        self.cb_format = QComboBox()
        for t, d in FORMATS:
            self.cb_format.addItem(t, d)
        self.ed_suffix = QLineEdit('_reduced')
        self.ck_preview = QCheckBox('生成前后对比预览图')
        self.ck_preview.setChecked(True)
        g.addWidget(QLabel('输出文件夹'), 0, 0)
        g.addWidget(self.ed_out, 0, 1, 1, 2)
        g.addWidget(bo, 0, 3)
        g.addWidget(QLabel('格式'), 1, 0)
        g.addWidget(self.cb_format, 1, 1)
        g.addWidget(QLabel('文件名后缀'), 1, 2)
        g.addWidget(self.ed_suffix, 1, 3)
        g.addWidget(self.ck_preview, 2, 0, 1, 4)
        lv.addWidget(gout)

        gp = QGroupBox('减面参数')
        g = QGridLayout(gp)
        self.rb_faces = QRadioButton('目标面数')
        self.rb_ratio = QRadioButton('目标比例 %')
        self.rb_faces.setChecked(True)
        self.sp_faces = QSpinBox()
        self.sp_faces.setRange(4, 10000000)
        self.sp_faces.setValue(1500)
        self.sp_faces.setToolTip('Target polygon count (quads count as one face)')
        self.sp_ratio = QDoubleSpinBox()
        self.sp_ratio.setRange(1.0, 100.0)
        self.sp_ratio.setValue(50.0)
        self.sp_ratio.setSuffix(' %')
        self.rb_faces.toggled.connect(lambda on: (self.sp_faces.setEnabled(on), self.sp_ratio.setEnabled(not on)))
        self.sp_ratio.setEnabled(False)
        self.cb_mode = QComboBox()
        for t, d in MODES:
            self.cb_mode.addItem(t, d)
        self.cb_mode.setToolTip('角色：只删除整条环线，保持四边形布线、关节环线、UV 接缝；静态道具：三角面减面')
        self.cb_sym = QComboBox()
        for t, d in SYMS:
            self.cb_sym.addItem(t, d)
        self.ed_group = QLineEdit()
        self.ed_group.setPlaceholderText('可选：顶点组名称，权重越大越不删')
        self.ck_seams = QCheckBox('保持 UV 接缝')
        self.ck_seams.setChecked(True)
        self.ck_sharp = QCheckBox('保持硬边')
        self.ck_sharp.setChecked(True)
        self.ck_joints = QCheckBox('自动保护关节环线')
        self.ck_joints.setChecked(True)
        g.addWidget(self.rb_faces, 0, 0)
        g.addWidget(self.sp_faces, 0, 1)
        g.addWidget(self.rb_ratio, 0, 2)
        g.addWidget(self.sp_ratio, 0, 3)
        g.addWidget(QLabel('模式'), 1, 0)
        g.addWidget(self.cb_mode, 1, 1)
        g.addWidget(QLabel('对称'), 1, 2)
        g.addWidget(self.cb_sym, 1, 3)
        g.addWidget(QLabel('保护顶点组'), 2, 0)
        g.addWidget(self.ed_group, 2, 1, 1, 3)
        hb = QHBoxLayout()
        hb.addWidget(self.ck_seams)
        hb.addWidget(self.ck_sharp)
        hb.addWidget(self.ck_joints)
        hb.addStretch(1)
        g.addLayout(hb, 3, 0, 1, 4)
        lv.addWidget(gp)

        hb = QHBoxLayout()
        self.bt_start = QPushButton('开始减面')
        self.bt_start.setMinimumHeight(38)
        f = QFont()
        f.setBold(True)
        self.bt_start.setFont(f)
        self.bt_start.clicked.connect(self.start)
        self.bt_cancel = QPushButton('取消')
        self.bt_cancel.setMinimumHeight(38)
        self.bt_cancel.setEnabled(False)
        self.bt_cancel.clicked.connect(self.cancel)
        hb.addWidget(self.bt_start, 3)
        hb.addWidget(self.bt_cancel, 1)
        lv.addLayout(hb)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        lv.addWidget(self.bar)
        self.lb_status = QLabel('就绪')
        lv.addWidget(self.lb_status)
        split.addWidget(left)

        # ---------------- right
        self.tabs = QTabWidget()
        pv = QWidget()
        pl = QVBoxLayout(pv)
        hb = QHBoxLayout()
        hb.addWidget(QLabel('视图'))
        self.cb_view = QComboBox()
        self.cb_view.currentIndexChanged.connect(self.show_preview)
        hb.addWidget(self.cb_view, 1)
        self.lb_legend = QLabel('左：减面结果　右：原始模型')
        hb.addWidget(self.lb_legend)
        pl.addLayout(hb)
        self.preview = PreviewLabel()
        pl.addWidget(self.preview, 1)
        self.tabs.addTab(pv, '预览')
        self.metrics = QTableWidget(len(METRIC_ROWS), 2)
        self.metrics.setHorizontalHeaderLabels(['指标', '数值'])
        self.metrics.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.metrics.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.metrics.verticalHeader().setVisible(False)
        self.metrics.setEditTriggers(QAbstractItemView.NoEditTriggers)
        for i, (k, name) in enumerate(METRIC_ROWS):
            self.metrics.setItem(i, 0, QTableWidgetItem(name))
            self.metrics.setItem(i, 1, QTableWidgetItem('—'))
        self.tabs.addTab(self.metrics, '指标')
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        self.tabs.addTab(self.log, '日志')
        split.addWidget(self.tabs)
        split.setStretchFactor(0, 4)
        split.setStretchFactor(1, 5)
        split.setSizes([540, 700])

    # ------------------------------------------------------------ settings
    def _load_settings(self):
        s = self.settings
        self.ed_blender.setText(s.get('blender', ''))
        self.ed_out.setText(s.get('outdir', ''))
        self.sp_faces.setValue(int(s.get('faces', 1500)))
        self.sp_ratio.setValue(float(s.get('ratio', 50.0)))
        (self.rb_faces if s.get('by_faces', True) else self.rb_ratio).setChecked(True)
        for cb, key in ((self.cb_mode, 'mode'), (self.cb_sym, 'symmetry'), (self.cb_format, 'format')):
            i = cb.findData(s.get(key, cb.itemData(0)))
            cb.setCurrentIndex(max(0, i))
        self.ed_suffix.setText(s.get('suffix', '_reduced'))

    def _save_settings(self):
        s = self.settings
        s.update({'blender': self.ed_blender.text().strip(), 'outdir': self.ed_out.text().strip(),
                  'faces': self.sp_faces.value(), 'ratio': self.sp_ratio.value(),
                  'by_faces': self.rb_faces.isChecked(), 'mode': self.cb_mode.currentData(),
                  'symmetry': self.cb_sym.currentData(), 'format': self.cb_format.currentData(),
                  'suffix': self.ed_suffix.text()})
        paths.save_settings(s)

    def closeEvent(self, ev):
        self._save_settings()
        if self.controller.is_running():
            self.controller.cancel()
        super().closeEvent(ev)

    # ------------------------------------------------------------- blender
    def browse_blender(self):
        flt = 'Blender (blender.exe blender Blender);;所有文件 (*)'
        p, _ = QFileDialog.getOpenFileName(self, '选择 blender 可执行文件', self.ed_blender.text(), flt)
        if p:
            self.ed_blender.setText(p)
            self.check_blender()

    def detect_blender(self):
        self.lb_blender.setText('检测中…')
        p, v = blender_finder.find_blender(self.ed_blender.text().strip())
        if p:
            self.ed_blender.setText(p)
        self._set_version(v)

    def check_blender(self):
        self._set_version(blender_finder.query_version(self.ed_blender.text().strip()))

    def _set_version(self, v):
        self.blender_version = v
        if not v:
            self.lb_blender.setText('未找到 Blender，请手动选择 blender.exe')
            self.lb_blender.setStyleSheet('color:#c0392b')
        elif not v.startswith('5.'):
            self.lb_blender.setText('Blender %s（建议使用 5.2）' % v)
            self.lb_blender.setStyleSheet('color:#b9770e')
        else:
            self.lb_blender.setText('Blender %s ✓' % v)
            self.lb_blender.setStyleSheet('color:#1e8449')

    # -------------------------------------------------------------- inputs
    def add_paths(self, ps):
        for p in ps:
            if os.path.isdir(p):
                for root, _d, files in os.walk(p):
                    self.add_paths([os.path.join(root, f) for f in sorted(files)
                                    if f.lower().endswith(MODEL_EXT)])
                continue
            if not p.lower().endswith(MODEL_EXT) or p in self.inputs:
                continue
            self.inputs.append(p)
            r = self.table.rowCount()
            self.table.insertRow(r)
            it = QTableWidgetItem(os.path.basename(p))
            it.setToolTip(p)
            self.table.setItem(r, 0, it)
            self.table.setItem(r, 1, QTableWidgetItem('待处理'))
            self.table.setItem(r, 2, QTableWidgetItem(''))
            self.table.setItem(r, 3, QTableWidgetItem(''))

    def add_files(self):
        ps, _ = QFileDialog.getOpenFileNames(self, '选择模型文件', self.settings.get('last_dir', ''),
                                             '模型文件 (*.fbx *.obj *.glb *.gltf);;所有文件 (*)')
        if ps:
            self.settings['last_dir'] = os.path.dirname(ps[0])
            self.add_paths(ps)

    def add_folder(self):
        d = QFileDialog.getExistingDirectory(self, '选择文件夹', self.settings.get('last_dir', ''))
        if d:
            self.settings['last_dir'] = d
            self.add_paths([d])

    def remove_selected(self):
        if self.controller.is_running():
            return
        rows = sorted({i.row() for i in self.table.selectedItems()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)
            del self.inputs[r]
        self.jobs = []

    def clear_inputs(self):
        if self.controller.is_running():
            return
        self.table.setRowCount(0)
        self.inputs = []
        self.jobs = []

    def browse_out(self):
        d = QFileDialog.getExistingDirectory(self, '选择输出文件夹', self.ed_out.text())
        if d:
            self.ed_out.setText(d)

    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        self.add_paths([u.toLocalFile() for u in ev.mimeData().urls()])

    # ----------------------------------------------------------------- run
    def make_job(self, inp):
        outdir = self.ed_out.text().strip()
        out = default_output(inp, outdir, self.ed_suffix.text() or '_reduced', self.cb_format.currentData())
        renders = os.path.join(os.path.dirname(out), 'previews') if self.ck_preview.isChecked() else ''
        by_faces = self.rb_faces.isChecked()
        return Job(inp, out, faces=self.sp_faces.value() if by_faces else 0,
                   ratio=0.0 if by_faces else self.sp_ratio.value() / 100.0,
                   mode=self.cb_mode.currentData(), symmetry=self.cb_sym.currentData(),
                   protect_group=self.ed_group.text().strip(), seams=self.ck_seams.isChecked(),
                   sharp=self.ck_sharp.isChecked(), joints=self.ck_joints.isChecked(), renders_dir=renders)

    def start(self):
        if self.controller.is_running():
            return False
        if not self.blender_version:
            self.check_blender()
        if not self.blender_version:
            QMessageBox.warning(self, '缺少 Blender', '没有找到可用的 Blender。请安装 Blender 5.2，或点击“浏览…”手动选择 blender.exe。')
            return False
        if not self.inputs:
            QMessageBox.information(self, '没有输入', '请先添加要减面的模型文件。')
            return False
        self._save_settings()
        self.controller.blender = self.ed_blender.text().strip()
        self.controller.queue = []
        self.jobs = []
        for r, inp in enumerate(self.inputs):
            j = self.make_job(inp)
            j.row = r
            self.jobs.append(j)
            self.controller.add(j)
        self.bt_start.setEnabled(False)
        self.bt_cancel.setEnabled(True)
        self.log.appendPlainText('==== 开始：%d 个文件 ====' % len(self.jobs))
        self.controller.start()
        return True

    def cancel(self):
        self.controller.cancel()

    def on_log(self, line):
        self.log.appendPlainText(line)

    def on_job_changed(self, j):
        r = getattr(j, 'row', None)
        if r is None or r >= self.table.rowCount():
            return
        st = STATUS_CN.get(j.status, j.status)
        if j.status == 'running':
            st += ' %d%%' % int(j.progress * 100)
        self.table.item(r, 1).setText(st)
        if j.status == 'failed':
            self.table.item(r, 1).setToolTip(j.error)
        if j.result:
            self.table.item(r, 2).setText('%s → %s' % (j.result.get('faces_before'), j.result.get('faces_after')))
        if j.status in ('done', 'failed', 'cancelled'):
            self.table.item(r, 3).setText('%.1f s' % j.seconds)
        n = max(1, len(self.jobs))
        done = sum(1.0 if x.status in ('done', 'failed', 'cancelled') else x.progress for x in self.jobs)
        self.bar.setValue(int(1000 * done / n))
        if j.status == 'running':
            self.lb_status.setText('%s：%s' % (os.path.basename(j.inp), j.message))
        elif j.status == 'failed':
            self.lb_status.setText('失败：%s' % j.error)
            self.log.appendPlainText('!! %s 失败：%s' % (os.path.basename(j.inp), j.error))
        elif j.status == 'done':
            self.show_job(j)

    def on_queue_finished(self):
        self.bt_start.setEnabled(True)
        self.bt_cancel.setEnabled(False)
        ok = sum(1 for j in self.jobs if j.status == 'done')
        self.lb_status.setText('全部结束：成功 %d / %d' % (ok, len(self.jobs)))
        self.log.appendPlainText('==== 结束：成功 %d / %d ====' % (ok, len(self.jobs)))

    def on_select(self):
        rows = {i.row() for i in self.table.selectedItems()}
        for j in self.jobs:
            if getattr(j, 'row', -1) in rows and j.status == 'done':
                self.show_job(j)
                break

    def show_job(self, j):
        res = j.result or {}
        self.cb_view.blockSignals(True)
        self.cb_view.clear()
        for p in res.get('renders', []):
            self.cb_view.addItem(os.path.basename(p), p)
        self.cb_view.blockSignals(False)
        self.show_preview()
        s = res.get('summary') or {}
        for i, (k, name) in enumerate(METRIC_ROWS):
            v = s.get(k)
            self.metrics.item(i, 1).setText('—' if v is None else str(v))
        if not res.get('renders') and res.get('render_error'):
            self.preview.setText(res['render_error'])

    def show_preview(self):
        p = self.cb_view.currentData()
        if p:
            self.preview.set_image(p)

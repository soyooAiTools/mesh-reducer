# -*- coding: utf-8 -*-
"""Job model + controller: runs runner.py inside a headless Blender process.

The same JobController drives the GUI, --cli and --selftest."""
import os
import sys
import json
import time

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal, QTimer

from . import paths
from .blender_finder import is_python_fallback

MODEL_EXT = ('.fbx', '.obj', '.glb', '.gltf')


class Job:
    def __init__(self, inp, out, faces=0, ratio=0.0, mode='character', symmetry='auto', protect_group='',
                 seams=True, sharp=True, joints=True, renders_dir='', report='', texture='', prop_method='strict'):
        self.inp = inp
        self.out = out
        self.faces = int(faces)
        self.ratio = float(ratio)
        self.mode = mode
        self.symmetry = symmetry
        self.protect_group = protect_group
        self.seams = seams
        self.sharp = sharp
        self.joints = joints
        self.renders_dir = renders_dir
        self.report = report or (os.path.splitext(out)[0] + '_report.json')
        self.texture = texture
        self.prop_method = prop_method
        # state
        self.status = 'queued'       # queued | running | done | failed | cancelled
        self.progress = 0.0
        self.message = ''
        self.result = None
        self.error = ''
        self.log = []
        self.seconds = 0.0

    def runner_args(self):
        a = ['--in', self.inp, '--out', self.out, '--mode', self.mode, '--symmetry', self.symmetry,
             '--report', self.report, '--prop-method', self.prop_method]
        if self.faces > 0:
            a += ['--faces', str(self.faces)]
        else:
            a += ['--ratio', '%.4f' % (self.ratio if self.ratio > 0 else 0.5)]
        if self.protect_group:
            a += ['--protect-group', self.protect_group]
        if not self.seams:
            a.append('--no-seams')
        if not self.sharp:
            a.append('--no-sharp')
        if not self.joints:
            a.append('--no-joints')
        if self.texture:
            a += ['--texture', self.texture]
        return a

    def render_args(self, engine='auto'):
        a = ['--render-only', '--orig', self.inp, '--result', self.out, '--renders', self.renders_dir,
             '--engine', engine, '--report', os.path.splitext(self.report)[0] + '_renders.json']
        if self.texture:
            a += ['--texture', self.texture]
        return a


def build_command(blender, runner_args):
    """argument list (never a shell string: paths may contain spaces / CJK)"""
    runner = paths.runner_path()
    if is_python_fallback(blender):
        return [blender, runner] + list(runner_args)
    return [blender, '-b', '--factory-startup', '--python-exit-code', '1', '--python', runner, '--'] + list(runner_args)


class JobController(QObject):
    """Runs queued jobs one after another.  Stage 1 = reduce + measure (must
    succeed), stage 2 = preview renders in a second process (optional: a GPU
    less machine may fail there, the job still counts as done)."""
    jobChanged = Signal(object)          # job
    logLine = Signal(str)
    queueFinished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.blender = ''
        self.queue = []
        self.current = None
        self.proc = None
        self.stage = 0
        self._buf = b''
        self._t0 = 0.0
        self._render_try = 0
        self._cancel = False

    # ---------------------------------------------------------------- api
    def add(self, job):
        self.queue.append(job)
        self.jobChanged.emit(job)

    def is_running(self):
        return self.current is not None

    def start(self):
        if self.current is None:
            self._cancel = False
            self._next()

    def cancel(self):
        self._cancel = True
        for j in self.queue:
            if j.status == 'queued':
                j.status = 'cancelled'
                self.jobChanged.emit(j)
        if self.proc is not None and self.proc.state() != QProcess.NotRunning:
            self.proc.kill()

    # ----------------------------------------------------------- internals
    def _next(self):
        self.current = next((j for j in self.queue if j.status == 'queued'), None)
        if self.current is None or self._cancel:
            self.current = None
            self.queueFinished.emit()
            return
        j = self.current
        j.status = 'running'
        j.progress = 0.0
        self._t0 = time.time()
        try:
            os.makedirs(os.path.dirname(os.path.abspath(j.out)), exist_ok=True)
            if os.path.isfile(j.report):
                os.remove(j.report)
        except OSError:
            pass
        self.stage = 1
        self._launch(build_command(self.blender, j.runner_args()))
        self.jobChanged.emit(j)

    def _launch(self, cmd):
        self._buf = b''
        self.logLine.emit('> ' + ' '.join('"%s"' % c if ' ' in c else c for c in cmd))
        p = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert('PYTHONIOENCODING', 'utf-8')
        env.insert('PYTHONUTF8', '1')
        env.insert('PYTHONUNBUFFERED', '1')
        p.setProcessEnvironment(env)
        p.setProcessChannelMode(QProcess.MergedChannels)
        p.readyReadStandardOutput.connect(self._on_output)
        p.finished.connect(self._on_finished)
        p.errorOccurred.connect(self._on_error)
        self.proc = p
        p.start(cmd[0], cmd[1:])

    def _on_error(self, err):
        if err == QProcess.FailedToStart and self.current is not None:
            j = self.current
            j.error = '无法启动 Blender：%s' % self.blender
            self._finish_job('failed')

    def _on_output(self):
        if self.proc is None:
            return
        self._buf += bytes(self.proc.readAllStandardOutput())
        while b'\n' in self._buf:
            line, self._buf = self._buf.split(b'\n', 1)
            self._handle_line(line.decode('utf-8', errors='replace').rstrip('\r'))

    def _handle_line(self, line):
        j = self.current
        if j is None or not line:
            return
        if line.startswith('PROGRESS '):
            parts = line.split(' ', 2)
            try:
                f = float(parts[1])
            except (ValueError, IndexError):
                return
            if self.stage == 1:
                j.progress = 0.9 * f
            else:
                j.progress = 0.9 + 0.1 * f
            j.message = parts[2] if len(parts) > 2 else ''
            self.jobChanged.emit(j)
            return
        if line.startswith('RESULT '):
            try:
                data = json.loads(line[7:])
            except ValueError:
                data = None
            if self.stage == 1:
                j.result = data
            elif data and j.result is not None:
                j.result['renders'] = data.get('renders', [])
                j.result['render_engine'] = data.get('engine')
            self.logLine.emit('RESULT (%d bytes)' % len(line))
            return
        if line.startswith('ERROR '):
            if self.stage == 1:
                j.error = line[6:]
        j.log.append(line)
        if not line.startswith(('FBX version', 'EGL Error')):
            self.logLine.emit(line)

    def _on_finished(self, code, status):
        j = self.current
        if j is None:
            return
        self._on_output()
        if self._buf:
            self._handle_line(self._buf.decode('utf-8', errors='replace'))
            self._buf = b''
        if self._cancel:
            self._finish_job('cancelled')
            return
        if self.stage == 1:
            if j.result is None and os.path.isfile(j.report):
                # Blender printed nothing we could parse but wrote the report
                try:
                    with open(j.report, 'r', encoding='utf-8') as f:
                        j.result = json.load(f)
                except Exception:
                    pass
            ok = bool(j.result and j.result.get('ok') and os.path.isfile(j.out))
            if not ok:
                if not j.error:
                    j.error = 'Blender 进程异常退出 (exit code %s)' % code
                self._finish_job('failed')
                return
            if j.renders_dir:
                self.stage = 2
                self._render_try = 0
                j.message = '渲染预览图'
                self.jobChanged.emit(j)
                self._launch(build_command(self.blender, j.render_args('auto')))
                return
            self._finish_job('done')
            return
        # stage 2: previews are optional
        have = bool(j.result.get('renders'))
        if not have and self._render_try == 0:
            self._render_try = 1
            self.logLine.emit('预览渲染失败，改用 Cycles(CPU) 重试…')
            self._launch(build_command(self.blender, j.render_args('cycles')))
            return
        if not have:
            j.result['render_error'] = '预览图渲染失败（不影响减面结果）'
        try:
            with open(j.report, 'w', encoding='utf-8') as f:
                json.dump(j.result, f, ensure_ascii=False, indent=1)
        except OSError:
            pass
        self._finish_job('done')

    def _finish_job(self, status):
        j = self.current
        j.status = status
        j.seconds = time.time() - self._t0
        if status == 'done':
            j.progress = 1.0
        self.proc = None
        self.current = None
        self.jobChanged.emit(j)
        QTimer.singleShot(0, self._next)


def default_output(inp, outdir, suffix, ext=''):
    stem, e = os.path.splitext(os.path.basename(inp))
    ext = ext or e
    return os.path.join(outdir or os.path.dirname(inp), stem + suffix + ext)

# -*- coding: utf-8 -*-
"""MeshReducer desktop app.

    MeshReducer                      -> GUI
    MeshReducer --cli --blender B --in X --out Y --faces 1500 --mode character --report R --log L
    MeshReducer --selftest --blender B --in X --outdir DIR
"""
import os
import sys
import json
import time
import argparse
import traceback

if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = 'app'

from PySide6.QtCore import QCoreApplication, QTimer, QEventLoop      # noqa: E402

from app import paths, blender_finder                                # noqa: E402
from app.jobs import Job, JobController                              # noqa: E402


def parse(argv):
    ap = argparse.ArgumentParser(prog='MeshReducer')
    ap.add_argument('--cli', action='store_true')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--blender', default='')
    ap.add_argument('--in', dest='inp', default='')
    ap.add_argument('--out', default='')
    ap.add_argument('--outdir', default='')
    ap.add_argument('--faces', type=int, default=0)
    ap.add_argument('--ratio', type=float, default=0.0)
    ap.add_argument('--mode', default='character')
    ap.add_argument('--symmetry', default='auto')
    ap.add_argument('--protect-group', default='')
    ap.add_argument('--no-seams', action='store_true')
    ap.add_argument('--no-sharp', action='store_true')
    ap.add_argument('--no-joints', action='store_true')
    ap.add_argument('--report', default='')
    ap.add_argument('--renders', default='')
    ap.add_argument('--texture', default='')
    ap.add_argument('--log', default='')
    ap.add_argument('--timeout', type=int, default=1800)
    args, _unknown = ap.parse_known_args(argv)
    return args


class FileLog:
    def __init__(self, path):
        self.f = None
        if path:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
                self.f = open(path, 'w', encoding='utf-8')
            except OSError:
                self.f = None

    def __call__(self, line):
        if self.f:
            self.f.write(line + '\n')
            self.f.flush()
        try:
            if sys.stdout:
                sys.stdout.write(line + '\n')
                sys.stdout.flush()
        except Exception:
            pass


def resolve_blender(arg, log):
    saved = arg or paths.load_settings().get('blender', '')
    if arg:
        v = blender_finder.query_version(arg)
        if not v:
            log('ERROR cannot run Blender at: %s' % arg)
            return '', None
        return arg, v
    return blender_finder.find_blender(saved)


def wait_for(controller, app, timeout):
    loop = QEventLoop()
    controller.queueFinished.connect(loop.quit)
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    timer.start(timeout * 1000)
    loop.exec()
    timed_out = not timer.isActive()
    timer.stop()
    if timed_out and controller.is_running():
        controller.cancel()
        t0 = time.time()
        while controller.is_running() and time.time() - t0 < 10:
            app.processEvents()
    return not timed_out


def run_cli(args):
    log = FileLog(args.log)
    try:
        app = QCoreApplication.instance() or QCoreApplication(sys.argv[:1])
        blender, ver = resolve_blender(args.blender, log)
        if not blender:
            log('ERROR Blender not found. Use --blender <path to blender.exe>')
            return 3
        log('Blender %s at %s' % (ver, blender))
        if not args.inp or not args.out:
            log('ERROR --in and --out are required')
            return 2
        job = Job(os.path.abspath(args.inp), os.path.abspath(args.out), faces=args.faces, ratio=args.ratio,
                  mode=args.mode, symmetry=args.symmetry, protect_group=args.protect_group,
                  seams=not args.no_seams, sharp=not args.no_sharp, joints=not args.no_joints,
                  renders_dir=os.path.abspath(args.renders) if args.renders else '',
                  report=os.path.abspath(args.report) if args.report else '', texture=args.texture)
        ctl = JobController()
        ctl.blender = blender
        ctl.logLine.connect(log)
        last = [-1]

        def changed(j):
            pc = int(j.progress * 20)
            if pc != last[0]:
                last[0] = pc
                log('[%3d%%] %s %s' % (int(j.progress * 100), j.status, j.message))
        ctl.jobChanged.connect(changed)
        ctl.add(job)
        QTimer.singleShot(0, ctl.start)
        finished = wait_for(ctl, app, args.timeout)
        if not finished:
            log('ERROR timeout after %d s' % args.timeout)
            return 4
        if job.status != 'done':
            log('ERROR job %s: %s' % (job.status, job.error))
            return 1
        log('OK %s -> %s faces, %.1f s' % (job.result.get('faces_before'), job.result.get('faces_after'), job.seconds))
        log('SUMMARY ' + json.dumps(job.result.get('summary'), ensure_ascii=False))
        return 0
    except Exception:
        log('ERROR ' + traceback.format_exc())
        return 5


def run_selftest(args):
    """Open the real main window, push one job through the GUI's own
    controller, screenshot the window."""
    outdir = os.path.abspath(args.outdir or '.')
    os.makedirs(outdir, exist_ok=True)
    log = FileLog(args.log or os.path.join(outdir, 'selftest_log.txt'))
    info = {'ok': False}
    code = 1
    try:
        from PySide6.QtWidgets import QApplication
        from app.mainwindow import MainWindow
        app = QApplication.instance() or QApplication(sys.argv[:1])
        blender, ver = resolve_blender(args.blender, log)
        info['blender'] = blender
        info['blender_version'] = ver
        win = MainWindow(blender_path=blender)
        win.show()
        app.processEvents()
        win.ed_out.setText(outdir)
        win.ck_preview.setChecked(True)
        win.rb_faces.setChecked(True)
        win.sp_faces.setValue(args.faces or 1500)
        i = win.cb_mode.findData(args.mode)
        win.cb_mode.setCurrentIndex(max(0, i))
        win.add_paths([os.path.abspath(args.inp)])
        info['inputs'] = list(win.inputs)
        win.controller.logLine.connect(log)
        started = win.start()
        info['started'] = bool(started)
        finished = wait_for(win.controller, app, args.timeout) if started else False
        app.processEvents()
        job = win.jobs[0] if win.jobs else None
        if job is not None:
            info.update({'status': job.status, 'error': job.error, 'seconds': round(job.seconds, 1),
                         'output': job.out, 'output_exists': os.path.isfile(job.out),
                         'summary': (job.result or {}).get('summary'),
                         'renders': (job.result or {}).get('renders', []),
                         'render_error': (job.result or {}).get('render_error')})
        info['finished'] = bool(finished)
        info['table_status'] = win.table.item(0, 1).text() if win.table.rowCount() else ''
        info['metric_faces_cell'] = win.metrics.item(0, 1).text()
        info['preview_shown'] = win.preview._pix is not None
        win.tabs.setCurrentIndex(0)
        app.processEvents()
        shot = os.path.join(outdir, 'gui.png')
        info['screenshot'] = shot
        info['screenshot_saved'] = bool(win.grab().save(shot))
        win.tabs.setCurrentIndex(1)
        app.processEvents()
        win.grab().save(os.path.join(outdir, 'gui_metrics.png'))
        info['ok'] = bool(job is not None and job.status == 'done' and info['output_exists'] and info['screenshot_saved'])
        code = 0 if info['ok'] else 1
        win.close()
    except Exception:
        info['exception'] = traceback.format_exc()
        log('ERROR ' + info['exception'])
        code = 5
    try:
        with open(os.path.join(outdir, 'selftest.json'), 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=1)
    except OSError:
        pass
    log('SELFTEST ' + json.dumps(info, ensure_ascii=False))
    return code


def run_gui():
    from PySide6.QtWidgets import QApplication
    from app.mainwindow import MainWindow
    app = QApplication(sys.argv)
    app.setApplicationName(paths.APP_NAME)
    win = MainWindow()
    win.show()
    return app.exec()


def main(argv=None):
    args = parse(sys.argv[1:] if argv is None else argv)
    if args.cli:
        return run_cli(args)
    if args.selftest:
        return run_selftest(args)
    return run_gui()


if __name__ == '__main__':
    sys.exit(main())

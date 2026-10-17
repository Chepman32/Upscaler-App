#!/usr/bin/env python3
"""Upscaler 16K — десктопное приложение (PySide6). Всё считается локально."""
import os
import subprocess
import sys
import traceback
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# torch и spandrel импортируем ОДИН раз, в главном потоке и до Qt: ленивый импорт torch из рабочего потока
# даёт «partially initialized module 'torch' has no attribute '_logging'» (гонка импорта / конфликт с Qt).
try:
    import torch  # noqa: F401
    import spandrel  # noqa: F401
    _TORCH_ERR = None
except Exception:  # noqa: BLE001
    _TORCH_ERR = traceback.format_exc()

from PySide6.QtCore import QObject, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QFrame, QGroupBox,
                               QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
                               QSlider, QSpinBox, QVBoxLayout, QWidget)

from upscaler.engine import (Cancelled, PRESETS, Settings, human_size, load_image, plan_passes, plan_target,
                             run_upscale)
from upscaler.models import MODELS

IMG_FILTER = ("Изображения (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff *.gif *.heic *.heif *.avif *.jp2 "
              "*.ppm *.tga);;Все файлы (*)")


class Worker(QObject):
    progress = Signal(float, str)
    log = Signal(str)
    done = Signal(dict)
    failed = Signal(str)

    def __init__(self, path, st):
        super().__init__()
        self.path, self.st = path, st
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            info = run_upscale(self.path, self.st, progress=self.progress.emit, log=self.log.emit,
                               cancel=lambda: self._cancel)
            self.done.emit(info)
        except Cancelled:
            self.failed.emit("__cancelled__")
        except Exception:  # noqa: BLE001
            self.failed.emit(traceback.format_exc())


class DropLabel(QLabel):
    dropped = Signal(str)

    def __init__(self):
        super().__init__("Перетащите сюда изображение\nили нажмите «Выбрать изображение…»")
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(300, 220)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("QLabel{border:2px dashed #888; border-radius:10px; color:#888;}")

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for u in e.mimeData().urls():
            if u.isLocalFile():
                self.dropped.emit(u.toLocalFile())
                break


class Main(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Upscaler 16K — локальный нейросетевой апскейл")
        self.resize(980, 720)
        self.path = None
        self.thread = None
        self.worker = None
        self.last_info = None

        # --- выбор файла
        self.btn_pick = QPushButton("Выбрать изображение…")
        self.btn_pick.clicked.connect(self.pick)
        self.preview = DropLabel()
        self.preview.dropped.connect(self.set_image)
        self.lbl_info = QLabel("Файл не выбран")
        self.lbl_info.setWordWrap(True)

        left = QVBoxLayout()
        left.addWidget(self.btn_pick)
        left.addWidget(self.preview, 1)
        left.addWidget(self.lbl_info)

        # --- настройки
        self.cb_target = QComboBox()
        self.cb_target.addItems(["16K — 15360 px (по длинной стороне)", "8K — 7680 px", "4K — 3840 px", "Другое…"])
        self.sp_custom = QSpinBox()
        self.sp_custom.setRange(256, 32768)
        self.sp_custom.setValue(15360)
        self.sp_custom.setSuffix(" px")
        self.sp_custom.setEnabled(False)
        self.cb_target.currentIndexChanged.connect(self.on_target)
        self.sp_custom.valueChanged.connect(lambda *_: self.refresh_info())

        self.cb_model = QComboBox()
        for k, v in MODELS.items():
            self.cb_model.addItem(v["title"], k)
        self.cb_model.currentIndexChanged.connect(self.on_model)

        self.sl_fid = QSlider(Qt.Horizontal)
        self.sl_fid.setRange(0, 100)
        self.sl_fid.setValue(80)
        self.lbl_fid = QLabel("80%")
        self.sl_fid.valueChanged.connect(lambda v: self.lbl_fid.setText(f"{v}%"))
        fid_row = QHBoxLayout()
        fid_row.addWidget(QLabel("верность"))
        fid_row.addWidget(self.sl_fid, 1)
        fid_row.addWidget(QLabel("детали"))
        fid_row.addWidget(self.lbl_fid)
        self.fid_w = QWidget()
        self.fid_w.setLayout(fid_row)
        fid_row.setContentsMargins(0, 0, 0, 0)

        self.cb_tile = QComboBox()
        for t in (256, 384, 512, 768):
            self.cb_tile.addItem(f"{t} px", t)
        self.cb_tile.setCurrentIndex(1)
        self.cb_dev = QComboBox()
        for lbl, v in (("Авто (GPU, если есть)", "auto"), ("Apple GPU (MPS)", "mps"), ("NVIDIA CUDA", "cuda"),
                       ("Процессор (CPU)", "cpu")):
            self.cb_dev.addItem(lbl, v)
        self.cb_fmt = QComboBox()
        self.cb_fmt.addItem("PNG (без потерь)", "png")
        self.cb_fmt.addItem("TIFF (без потерь, deflate)", "tiff")
        self.chk_cmp = QCheckBox("Сделать сравнение на 100%-кропе")
        self.chk_cmp.setChecked(True)
        self.chk_color = QCheckBox("Сохранять цвет и свет оригинала (рекомендуется)")
        self.chk_color.setChecked(True)
        self.chk_color.setToolTip("Нейросети слегка смещают цвет, а при нескольких проходах сдвиг копится.\n"
                                  "Коррекция берёт крупный цвет/свет из оригинала, а детали — от нейросети.")

        self.ed_out = QLineEdit()
        self.ed_out.setPlaceholderText("По умолчанию — рядом с исходным файлом")
        btn_out = QPushButton("Папка…")
        btn_out.clicked.connect(self.pick_out)
        out_row = QHBoxLayout()
        out_row.addWidget(self.ed_out, 1)
        out_row.addWidget(btn_out)
        out_w = QWidget()
        out_w.setLayout(out_row)
        out_row.setContentsMargins(0, 0, 0, 0)

        tgt_row = QHBoxLayout()
        tgt_row.addWidget(self.cb_target, 1)
        tgt_row.addWidget(self.sp_custom)
        tgt_row.setContentsMargins(0, 0, 0, 0)
        tgt_w = QWidget()
        tgt_w.setLayout(tgt_row)

        form = QFormLayout()
        form.addRow("Разрешение:", tgt_w)
        form.addRow("Модель:", self.cb_model)
        form.addRow("Баланс:", self.fid_w)
        form.addRow("Тайл:", self.cb_tile)
        form.addRow("Устройство:", self.cb_dev)
        form.addRow("Формат:", self.cb_fmt)
        form.addRow("Сохранить в:", out_w)
        form.addRow("", self.chk_color)
        form.addRow("", self.chk_cmp)
        box = QGroupBox("Настройки")
        box.setLayout(form)

        self.btn_go = QPushButton("Запустить апскейл")
        self.btn_go.setMinimumHeight(38)
        self.btn_go.setEnabled(False)
        self.btn_go.clicked.connect(self.start)
        self.btn_cancel = QPushButton("Отмена")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel)
        btns = QHBoxLayout()
        btns.addWidget(self.btn_go, 2)
        btns.addWidget(self.btn_cancel, 1)

        right = QVBoxLayout()
        right.addWidget(box)
        right.addLayout(btns)

        top = QHBoxLayout()
        top.addLayout(left, 1)
        top.addLayout(right, 1)

        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.lbl_status = QLabel("Готов")
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.btn_show = QPushButton("Показать результат в папке")
        self.btn_show.clicked.connect(self.show_result)
        self.btn_cmp = QPushButton("Открыть сравнение 100%")
        self.btn_cmp.clicked.connect(self.open_cmp)
        self.btn_show.setEnabled(False)
        self.btn_cmp.setEnabled(False)
        bot_btns = QHBoxLayout()
        bot_btns.addWidget(self.btn_show)
        bot_btns.addWidget(self.btn_cmp)

        lay = QVBoxLayout(self)
        lay.addLayout(top, 3)
        lay.addWidget(self.bar)
        lay.addWidget(self.lbl_status)
        lay.addWidget(self.log_view, 2)
        lay.addLayout(bot_btns)

    # ---- события
    def on_target(self, i):
        self.sp_custom.setEnabled(i == 3)
        self.refresh_info()

    def on_model(self, *_):
        self.sl_fid.setEnabled(MODELS[self.cb_model.currentData()]["has_fidelity"])

    def target_long(self):
        i = self.cb_target.currentIndex()
        return [PRESETS["16K"], PRESETS["8K"], PRESETS["4K"], self.sp_custom.value()][i]

    def pick(self):
        start = str(Path(self.path).parent) if self.path else str(Path.home() / "Pictures")
        f, _ = QFileDialog.getOpenFileName(self, "Выберите изображение", start, IMG_FILTER)
        if f:
            self.set_image(f)

    def pick_out(self):
        d = QFileDialog.getExistingDirectory(self, "Папка для результата")
        if d:
            self.ed_out.setText(d)

    def set_image(self, path):
        try:
            rgb, _, _, _ = load_image(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Не удалось открыть", f"{path}\n\n{e}")
            return
        self.path = path
        self.src_size = rgb.size
        th = rgb.copy()
        th.thumbnail((640, 480))
        from PIL.ImageQt import ImageQt
        pm = QPixmap.fromImage(ImageQt(th))
        self.preview.setStyleSheet("")
        self.preview.setPixmap(pm.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.btn_go.setEnabled(self.thread is None)
        self.refresh_info()

    def refresh_info(self):
        if not self.path:
            return
        w, h = self.src_size
        tw, th = plan_target(w, h, self.target_long())
        n = max(plan_passes(w, h, self.target_long()), 1)
        mb = tw * th * 3 / 1e6
        self.lbl_info.setText(
            f"{Path(self.path).name}\n{w}×{h} px → {tw}×{th} px  ({tw * th / 1e6:.0f} Мпикс, ≈{mb:.0f} МБ без сжатия)\n"
            f"Проходов нейросети: {n}")

    def settings(self):
        out = self.ed_out.text().strip() or None
        return Settings(target_long=self.target_long(), tile=self.cb_tile.currentData(),
                        device=self.cb_dev.currentData(), model=self.cb_model.currentData(),
                        fidelity=self.sl_fid.value() / 100.0, fmt=self.cb_fmt.currentData(),
                        make_compare=self.chk_cmp.isChecked(), out_dir=out,
                        color_fix=self.chk_color.isChecked())

    def start(self):
        if not self.path:
            return
        self.log_view.clear()
        self.bar.setValue(0)
        self.btn_go.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.btn_show.setEnabled(False)
        self.btn_cmp.setEnabled(False)
        self.thread = QThread()
        self.worker = Worker(self.path, self.settings())
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_progress)
        self.worker.log.connect(self.log_view.appendPlainText)
        self.worker.done.connect(self.on_done)
        self.worker.failed.connect(self.on_failed)
        self.thread.start()

    def cancel(self):
        if self.worker:
            self.worker.cancel()
            self.lbl_status.setText("Отменяю…")

    def on_progress(self, f, s):
        self.bar.setValue(int(f * 1000))
        self.lbl_status.setText(s)

    def finish_thread(self):
        if self.thread:
            self.thread.quit()
            self.thread.wait()
        self.thread = None
        self.worker = None
        self.btn_cancel.setEnabled(False)
        self.btn_go.setEnabled(bool(self.path))

    def on_done(self, info):
        self.finish_thread()
        self.last_info = info
        w, h = info["size_px"]
        msg = (f"Готово: {info['name']}\nРазрешение: {w}×{h} px\nРазмер файла: {human_size(info['size_bytes'])}\n"
               f"Время: {info['seconds']:.0f} с")
        self.log_view.appendPlainText("\n" + msg)
        if info.get("compare"):
            self.log_view.appendPlainText("Сравнение 100%: " + info["compare"])
            self.btn_cmp.setEnabled(True)
        self.btn_show.setEnabled(True)
        self.bar.setValue(1000)
        self.lbl_status.setText("Готово")
        QMessageBox.information(self, "Готово", msg)

    def on_failed(self, tb):
        self.finish_thread()
        if tb == "__cancelled__":
            self.lbl_status.setText("Отменено")
            self.log_view.appendPlainText("Отменено пользователем.")
            return
        self.lbl_status.setText("Ошибка")
        self.log_view.appendPlainText(tb)
        QMessageBox.critical(self, "Ошибка", tb.strip().splitlines()[-1])

    def show_result(self):
        if not self.last_info:
            return
        p = self.last_info["path"]
        if sys.platform == "darwin":
            subprocess.Popen(["open", "-R", p])
        elif sys.platform.startswith("win"):
            subprocess.Popen(["explorer", "/select,", p])
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(p).parent)))

    def open_cmp(self):
        if self.last_info and self.last_info.get("compare"):
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.last_info["compare"]))


def main():
    app = QApplication(sys.argv)
    if _TORCH_ERR:
        QMessageBox.critical(None, "Upscaler 16K",
                             "Не удалось загрузить PyTorch. Удалите папку\n"
                             "~/Library/Application Support/Upscaler16K и запустите приложение снова "
                             "(зависимости поставятся заново).\n\n" + _TORCH_ERR[-1500:])
        sys.exit(1)
    w = Main()
    w.show()
    if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
        w.set_image(sys.argv[1])
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

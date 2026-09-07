import sys
import math
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np

MIN_SCALE = 0.2
MAX_SCALE = 4.0

# ------------------------ Матрицы преобразований ----------------------------
def translation_matrix(tx: float, ty: float, tz: float) -> np.ndarray:
    M = np.eye(4, dtype=float)
    M[0, 3] = tx
    M[1, 3] = ty
    M[2, 3] = tz
    return M

def scale_matrix(sx: float, sy: float, sz: float) -> np.ndarray:
    M = np.eye(4, dtype=float)
    M[0, 0] = sx
    M[1, 1] = sy
    M[2, 2] = sz
    return M

def rotation_x(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    M = np.eye(4, dtype=float)
    M[1, 1], M[1, 2] = c, -s
    M[2, 1], M[2, 2] = s, c
    return M

def rotation_y(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    M = np.eye(4, dtype=float)
    M[0, 0], M[0, 2] = c, s
    M[2, 0], M[2, 2] = -s, c
    return M

def rotation_z(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    M = np.eye(4, dtype=float)
    M[0, 0], M[0, 1] = c, -s
    M[1, 0], M[1, 1] = s, c
    return M

def look_at_matrix(camera_pos: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    C = np.array(camera_pos, dtype=float)
    T = np.array(target, dtype=float)
    UP = np.array(up, dtype=float)

    zc = T - C
    zn = np.linalg.norm(zc)
    if zn < 1e-9:
        zc = np.array([0.0, 0.0, 1.0])
    else:
        zc = zc / zn

    xc = np.cross(zc, UP)
    xn = np.linalg.norm(xc)
    if xn < 1e-9:
        xc = np.array([1.0, 0.0, 0.0])
    else:
        xc = xc / xn

    yc = np.cross(xc, zc)

    V = np.eye(4, dtype=float)
    V[0, 0:3] = xc
    V[1, 0:3] = yc
    V[2, 0:3] = zc
    V[0:3, 3] = -V[0:3, 0:3] @ C
    return V

def project_with_camera(points4, camera_pos, target, up, focal, mode='perspective', ortho_dist=1.0):
    pts = np.array(points4, dtype=float)
    V = look_at_matrix(camera_pos, target, up)
    pts_cam = (V @ pts.T).T
    res = []
    eps = 1e-9
    for p in pts_cam:
        x, y, z, w = p
        if z <= eps:
            res.append(None)
            continue
        if mode == 'perspective':
            t = focal / z
        else:  # orthographic
            t = focal / ortho_dist   # масштабируем так, чтобы размер совпадал с перспективой на расстоянии ortho_dist
        res.append((x * t, y * t))
    return np.array(res, dtype=object)

# ------------------------ Модель буквы M ----------------------------
def make_letter_M(size=1.0, depth=0.2):
    h = size
    w = size * 0.5
    z0, z1 = depth / 2, -depth / 2

    verts = [
        (-w, -h/2, z0, 1.0),   # 0
        (-w,  h/2, z0, 1.0),   # 1
        (0,  0, z0, 1.0),      # 2  средняя нижняя
        ( w,  h/2, z0, 1.0),   # 3
        ( w, -h/2, z0, 1.0),   # 4
    ]
    verts += [
        (-w, -h/2, z1, 1.0),   # 5
        (-w,  h/2, z1, 1.0),   # 6
        (0,  0, z1, 1.0),      # 7
        ( w,  h/2, z1, 1.0),   # 8
        ( w, -h/2, z1, 1.0),   # 9
    ]

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # перед
        (5, 6), (6, 7), (7, 8), (8, 9),  # зад
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
    ]
    return np.array(verts, dtype=float), edges

# ------------------------ Виджет с рисованием ----------------------------
class GLWidget(QtWidgets.QWidget):
    cameraChanged = QtCore.pyqtSignal(float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(800, 600)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

        self.model_vertices, self.model_edges = make_letter_M(size=1.0, depth=0.25)

        self.tx, self.ty, self.tz = 0.0, 0.0, 0.0
        self.rx, self.ry, self.rz = 0.0, 0.0, 0.0
        self.s = 1.0

        self.last_mouse = None
        self.mouse_mode = None

        self.camera_pos = np.array([4.0, 4.0, 4.0], dtype=float)
        self.camera_target = np.array([0.0, 0.0, 0.0], dtype=float)
        self.camera_up = np.array([0.0, 1.0, 0.0], dtype=float)
        self.focal = 4.0

        self.projection_mode = 'perspective'

        self._recalc_camera_spherical()

        self.line_pen = QtGui.QPen(QtGui.QColor(40, 40, 180), 2)
        self.axis_pen_x = QtGui.QPen(QtGui.QColor(200, 30, 30), 2)
        self.axis_pen_y = QtGui.QPen(QtGui.QColor(30, 160, 30), 2)
        self.axis_pen_z = QtGui.QPen(QtGui.QColor(30, 120, 200), 2)

    def _recalc_camera_spherical(self):
        v = self.camera_pos - self.camera_target
        r = np.linalg.norm(v)
        if r < 1e-6:
            r = 1e-6
            v = np.array([r, 0.0, 0.0])
        el = math.asin(v[1] / r)
        az = math.atan2(v[2], v[0])
        self.cam_radius = r
        self.cam_azimuth = az
        self.cam_elevation = el

    def _update_camera_from_spherical(self):
        r = max(1e-6, self.cam_radius)
        el = self.cam_elevation
        az = self.cam_azimuth
        x = r * math.cos(el) * math.cos(az)
        y = r * math.sin(el)
        z = r * math.cos(el) * math.sin(az)
        self.camera_pos = np.array([x, y, z], dtype=float) + self.camera_target
        self.cameraChanged.emit(float(self.camera_pos[0]), float(self.camera_pos[1]), float(self.camera_pos[2]))

    def sizeHint(self):
        return QtCore.QSize(900, 700)

    def model_matrix(self):
        T = translation_matrix(self.tx, self.ty, self.tz)
        Rx, Ry, Rz = rotation_x(self.rx), rotation_y(self.ry), rotation_z(self.rz)
        S = scale_matrix(self.s, self.s, self.s)
        return T @ (Rz @ (Ry @ (Rx @ S)))

    def paintEvent(self, event):
        qp = QtGui.QPainter(self)
        qp.setRenderHint(QtGui.QPainter.Antialiasing)
        qp.fillRect(self.rect(), QtGui.QColor(245, 245, 245))

        w, h = self.width(), self.height()
        cx, cy = w / 2.0, h / 2.0

        M = self.model_matrix()
        verts4_world = (M @ self.model_vertices.T).T

        # Расстояние от камеры до цели (для масштабирования в ортографии)
        ortho_dist = self.cam_radius

        pts2d_model = project_with_camera(
            verts4_world,
            self.camera_pos,
            self.camera_target,
            self.camera_up,
            self.focal,
            mode=self.projection_mode,
            ortho_dist=ortho_dist
        )

        scale_screen = min(w, h) * 0.5

        # Рёбра модели
        qp.setPen(self.line_pen)
        for (i, j) in self.model_edges:
            p1 = pts2d_model[i]
            p2 = pts2d_model[j]
            if p1 is None or p2 is None:
                continue
            sx1, sy1 = cx + p1[0] * scale_screen, cy - p1[1] * scale_screen
            sx2, sy2 = cx + p2[0] * scale_screen, cy - p2[1] * scale_screen
            qp.drawLine(int(sx1), int(sy1), int(sx2), int(sy2))

        # Мировые оси
        axis_len = 1.2
        origin = np.array([0.0, 0.0, 0.0, 1.0])
        axis_points = [
            (origin, np.array([axis_len, 0.0, 0.0, 1.0])),
            (origin, np.array([0.0, axis_len, 0.0, 1.0])),
            (origin, np.array([0.0, 0.0, axis_len, 1.0])),
        ]
        pts_for_axes = np.array([p for pair in axis_points for p in pair], dtype=float)
        proj_axes = project_with_camera(
            pts_for_axes,
            self.camera_pos,
            self.camera_target,
            self.camera_up,
            self.focal,
            mode=self.projection_mode,
            ortho_dist=ortho_dist
        )
        for idx, pen in enumerate((self.axis_pen_x, self.axis_pen_y, self.axis_pen_z)):
            p_origin = proj_axes[2 * idx]
            p_end = proj_axes[2 * idx + 1]
            if p_origin is None or p_end is None:
                continue
            sx1, sy1 = cx + p_origin[0] * scale_screen, cy - p_origin[1] * scale_screen
            sx2, sy2 = cx + p_end[0] * scale_screen, cy - p_end[1] * scale_screen
            qp.setPen(pen)
            qp.drawLine(int(sx1), int(sy1), int(sx2), int(sy2))
            dx = sx2 - sx1
            dy = sy2 - sy1
            L = math.hypot(dx, dy)
            if L > 1.0:
                ux, uy = dx / L, dy / L
                vx, vy = -uy, ux
                arrow_len = min(12, L * 0.15)
                ax1 = sx2 - ux * arrow_len + vx * arrow_len * 0.4
                ay1 = sy2 - uy * arrow_len + vy * arrow_len * 0.4
                ax2 = sx2 - ux * arrow_len - vx * arrow_len * 0.4
                ay2 = sy2 - uy * arrow_len - vy * arrow_len * 0.4
                qp.drawLine(int(sx2), int(sy2), int(ax1), int(ay1))
                qp.drawLine(int(sx2), int(sy2), int(ax2), int(ay2))

        # ----- ИНФОРМАЦИЯ В ЛЕВОМ ВЕРХНЕМ УГЛУ (без фона, без рамок) -----
        qp.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0)))
        qp.setFont(QtGui.QFont("Consolas", 10))
        x0, y0 = 10, 20
        line_height = 18

        qp.drawText(x0, y0, f"Позиция: ({self.tx:.2f}, {self.ty:.2f}, {self.tz:.2f})")
        y0 += line_height
        qp.drawText(x0, y0, f"Поворот (град): ({math.degrees(self.rx):.1f}, {math.degrees(self.ry):.1f}, {math.degrees(self.rz):.1f})")
        y0 += line_height
        qp.drawText(x0, y0, f"Масштаб: {self.s:.2f}")
        y0 += line_height
        proj_text = "Перспективная" if self.projection_mode == 'perspective' else "Ортографическая"
        qp.drawText(x0, y0, f"Проекция: {proj_text}")
        y0 += line_height
        qp.drawText(x0, y0, f"Камера: ({self.camera_pos[0]:.2f}, {self.camera_pos[1]:.2f}, {self.camera_pos[2]:.2f})")

    # ------------------ Обработка мыши и клавиатуры ------------------
    def mousePressEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            mods = QtWidgets.QApplication.keyboardModifiers()
            if mods & QtCore.Qt.ControlModifier:
                self.mouse_mode = "orbit_cam"
                self._orbit_start_az = self.cam_azimuth
                self._orbit_start_el = self.cam_elevation
                self._orbit_start_mouse = ev.pos()
            else:
                self.mouse_mode = "rotate"
            self.last_mouse = ev.pos()
        elif ev.button() == QtCore.Qt.RightButton:
            self.mouse_mode = "pan"
            self.last_mouse = ev.pos()

    def mouseMoveEvent(self, ev):
        if self.last_mouse is None:
            return
        pos = ev.pos()
        dx = pos.x() - self.last_mouse.x()
        dy = pos.y() - self.last_mouse.y()
        mods = QtWidgets.QApplication.keyboardModifiers()

        if self.mouse_mode == "rotate":
            if mods & QtCore.Qt.ShiftModifier:
                self.rz += dx * 0.01
            else:
                self.ry += dx * 0.01
                self.rx += dy * 0.01

        elif self.mouse_mode == "pan":
            if mods & QtCore.Qt.ShiftModifier:
                self.tz += dy * 0.005
            else:
                self.tx += dx * 0.005
                self.ty -= dy * 0.005

        elif self.mouse_mode == "orbit_cam":
            start = self._orbit_start_mouse
            ddx = pos.x() - start.x()
            ddy = pos.y() - start.y()
            az = self._orbit_start_az + ddx * 0.01
            el = self._orbit_start_el + ddy * 0.01
            max_el = math.radians(89.0)
            el = max(-max_el, min(max_el, el))
            self.cam_azimuth = az
            self.cam_elevation = el
            self._update_camera_from_spherical()

        self.last_mouse = pos
        self.update()

    def mouseReleaseEvent(self, ev):
        self.mouse_mode = None
        self.last_mouse = None

    def wheelEvent(self, ev):
        delta = ev.angleDelta().y() / 120.0
        factor = 1.1 ** delta
        self.s *= factor
        self.s = max(MIN_SCALE, min(MAX_SCALE, self.s))
        self.update()

    def keyPressEvent(self, ev):
        k = ev.key()
        if k == QtCore.Qt.Key_R:
            self.reset_transform()
        elif k == QtCore.Qt.Key_P:
            self.toggle_projection()
        self.update()

    def toggle_projection(self):
        if self.projection_mode == 'perspective':
            self.projection_mode = 'orthographic'
        else:
            self.projection_mode = 'perspective'
        self.update()

    def reset_transform(self):
        self.tx, self.ty, self.tz = 0.0, 0.0, 0.0
        self.rx, self.ry, self.rz = 0.0, 0.0, 0.0
        self.s = 1.0
        self.camera_pos = np.array([4.0, 4.0, 4.0], dtype=float)
        self.camera_target = np.array([0.0, 0.0, 0.0], dtype=float)
        self._recalc_camera_spherical()
        self.cameraChanged.emit(float(self.camera_pos[0]), float(self.camera_pos[1]), float(self.camera_pos[2]))
        self.update()

# ------------------------ Главное окно ----------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Лабораторная работа 1 – 3D проволочная буква M")
        self.gl = GLWidget(self)
        self.setCentralWidget(self.gl)
        self.create_controls()

    def create_controls(self):
        dock = QtWidgets.QDockWidget("Управление", self)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        # ---- Радиокнопки для выбора проекции ----
        layout.addWidget(QtWidgets.QLabel("<b>Проекция</b>"))
        self.proj_group = QtWidgets.QButtonGroup(self)
        self.radio_persp = QtWidgets.QRadioButton("Перспективная")
        self.radio_ortho = QtWidgets.QRadioButton("Ортографическая")
        self.radio_persp.setChecked(True)
        self.proj_group.addButton(self.radio_persp)
        self.proj_group.addButton(self.radio_ortho)
        self.radio_persp.toggled.connect(self.on_proj_changed)
        self.radio_ortho.toggled.connect(self.on_proj_changed)
        layout.addWidget(self.radio_persp)
        layout.addWidget(self.radio_ortho)

        layout.addWidget(QtWidgets.QLabel("<b>Параметры камеры</b>"))
        cam_layout = QtWidgets.QGridLayout()
        cam_layout.addWidget(QtWidgets.QLabel("Cam X:"), 0, 0)
        self.cam_x = QtWidgets.QDoubleSpinBox()
        self.cam_x.setRange(-100.0, 100.0)
        self.cam_x.setValue(self.gl.camera_pos[0])
        cam_layout.addWidget(self.cam_x, 0, 1)
        cam_layout.addWidget(QtWidgets.QLabel("Cam Y:"), 1, 0)
        self.cam_y = QtWidgets.QDoubleSpinBox()
        self.cam_y.setRange(-100.0, 100.0)
        self.cam_y.setValue(self.gl.camera_pos[1])
        cam_layout.addWidget(self.cam_y, 1, 1)
        cam_layout.addWidget(QtWidgets.QLabel("Cam Z:"), 2, 0)
        self.cam_z = QtWidgets.QDoubleSpinBox()
        self.cam_z.setRange(-100.0, 100.0)
        self.cam_z.setValue(self.gl.camera_pos[2])
        cam_layout.addWidget(self.cam_z, 2, 1)

        def cam_changed(_):
            self.gl.camera_pos = np.array([self.cam_x.value(), self.cam_y.value(), self.cam_z.value()], dtype=float)
            self.gl._recalc_camera_spherical()
            self.gl.cameraChanged.emit(float(self.gl.camera_pos[0]), float(self.gl.camera_pos[1]), float(self.gl.camera_pos[2]))
            self.gl.update()

        self.cam_x.valueChanged.connect(cam_changed)
        self.cam_y.valueChanged.connect(cam_changed)
        self.cam_z.valueChanged.connect(cam_changed)

        def on_camera_changed(x, y, z):
            self.cam_x.blockSignals(True)
            self.cam_y.blockSignals(True)
            self.cam_z.blockSignals(True)
            self.cam_x.setValue(x)
            self.cam_y.setValue(y)
            self.cam_z.setValue(z)
            self.cam_x.blockSignals(False)
            self.cam_y.blockSignals(False)
            self.cam_z.blockSignals(False)

        self.gl.cameraChanged.connect(on_camera_changed)
        layout.addLayout(cam_layout)

        btn_row = QtWidgets.QHBoxLayout()
        btn_reset = QtWidgets.QPushButton("Сброс (R)")
        btn_reset.clicked.connect(self.gl.reset_transform)
        btn_row.addWidget(btn_reset)
        layout.addLayout(btn_row)

        instr = QtWidgets.QLabel(
            "Управление:\n"
            "- ЛКМ + перетаскивание: вращение модели (OX/OY)\n"
            "- Shift + ЛКМ: вращение модели вокруг OZ\n"
            "- ПКМ + перетаскивание: перенос модели по XY\n"
            "- Shift + ПКМ: перенос модели вдоль OZ\n"
            "- Колесо мыши: масштаб модели\n"
            "- Ctrl + ЛКМ: вращение камеры вокруг модели\n"
            "- P: переключить проекцию (перспектива / ортография)"
        )
        instr.setWordWrap(True)
        layout.addWidget(instr)

        layout.addStretch()
        panel.setLayout(layout)
        scroll.setWidget(panel)
        dock.setWidget(scroll)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

    def on_proj_changed(self):
        if self.radio_persp.isChecked():
            self.gl.projection_mode = 'perspective'
        else:
            self.gl.projection_mode = 'orthographic'
        self.gl.update()

# ------------------------ Запуск ----------------------------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.resize(1200, 760)
    w.show()
    sys.exit(app.exec_())
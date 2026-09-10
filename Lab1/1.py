import sys                    
import math                    
from PyQt5 import QtCore, QtGui, QtWidgets 
import numpy as np           

MIN_SCALE = 0.2   # минимальный масштаб (объект не может стать слишком мелким)
MAX_SCALE = 4.0   # максимальный масштаб (объект не может стать слишком большим)

# Все матрицы имеют размер 4x4, потому что мы используем однородные координаты (x,y,z,1)

# Матрица переноса в однородных координатах
def translation_matrix(tx: float, ty: float, tz: float) -> np.ndarray:
    # Матрица переноса в однородных координатах
    M = np.eye(4, dtype=float)   # создаём единичную матрицу 4x4
    M[0, 3] = tx   
    M[1, 3] = ty   
    M[2, 3] = tz   
    return M

# Матрица масштабирования по осям
def scale_matrix(sx: float, sy: float, sz: float) -> np.ndarray:
    M = np.eye(4, dtype=float)
    M[0, 0] = sx   
    M[1, 1] = sy   
    M[2, 2] = sz   
    return M

# Матрица поворота вокруг OX.
def rotation_x(theta: float) -> np.ndarray:
    # Формула:
    #  [1   0      0    0]
    #  [0  cos  -sin    0]
    #  [0  sin   cos    0]
    #  [0   0     0     1]
    c, s = math.cos(theta), math.sin(theta)
    M = np.eye(4, dtype=float)
    M[1, 1], M[1, 2] = c, -s   
    M[2, 1], M[2, 2] = s, c   
    return M

# Матрица поворота вокруг OY.
def rotation_y(theta: float) -> np.ndarray:
    # Формула:
    #  [ cos   0   sin  0]
    #  [  0    1    0   0]
    #  [-sin   0   cos  0]
    #  [  0    0    0   1]
    c, s = math.cos(theta), math.sin(theta)
    M = np.eye(4, dtype=float)
    M[0, 0], M[0, 2] = c, s     
    M[2, 0], M[2, 2] = -s, c   
    return M

# Матрица поворота вокруг OZ.
def rotation_z(theta: float) -> np.ndarray:
    # Формула:
    #  [cos -sin  0  0]
    #  [sin  cos  0  0]
    #  [ 0    0   1  0]
    #  [ 0    0   0  1]
    c, s = math.cos(theta), math.sin(theta)
    M = np.eye(4, dtype=float)
    M[0, 0], M[0, 1] = c, -s    
    M[1, 0], M[1, 1] = s, c     
    return M

# Строим матрицу вида (мировые -> камера)
def look_at_matrix(camera_pos: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    # camera_pos: положение камеры в мировых координатах
    # target: точка, на которую смотрит камера
    # up: вектор "вверх" (обычно (0,1,0))
    #
    # Алгоритм:
    # 1) zc = normalize(target - camera) — направление взгляда камеры
    # 2) xc = normalize(cross(zc, up)) — правая ось камеры
    # 3) yc = cross(xc, zc) — вверх камеры (ортогонален)
    # 4) Формируем матрицу из базиса (xc, yc, zc) и переводим так, чтобы камера была в начале координат

    C = np.array(camera_pos, dtype=float)
    T = np.array(target, dtype=float)
    UP = np.array(up, dtype=float)

    # Направление взгляда (от камеры к цели)
    zc = T - C
    zn = np.linalg.norm(zc)          # вычисляем длину вектора
    if zn < 1e-9:                    # если длина почти нулевая (камера совпадает с целью)
        zc = np.array([0.0, 0.0, 1.0])  # задаём направление по умолчанию (вдоль Z)
    else:
        zc = zc / zn                 # нормализуем (делаем единичной длины)

    # Правая ось камеры (перпендикуляр к направлению взгляда и вектору "вверх")
    xc = np.cross(zc, UP)
    xn = np.linalg.norm(xc)
    if xn < 1e-9:                    # если вектор "вверх" параллелен направлению взгляда
        xc = np.array([1.0, 0.0, 0.0])  # берём ось X по умолчанию
    else:
        xc = xc / xn

    # Восстанавливаем "истинный" вектор вверх (перпендикулярный к двум предыдущим)
    yc = np.cross(xc, zc)

    # Строим матрицу вида 4x4
    V = np.eye(4, dtype=float)
    # Первые три строки содержат базисные векторы камеры (в мировых координатах)
    V[0, 0:3] = xc   
    V[1, 0:3] = yc
    V[2, 0:3] = zc

    # R = V[0:3, 0:3] — поворотная часть (3x3)
    # t = -R @ C     — перенос, чтобы камера попала в (0,0,0)
    V[0:3, 3] = -V[0:3, 0:3] @ C  # Результат (вектор из 3 элементов) записываем в четвёртый столбец первых трёх строк
    return V

# Проецирует 3D-точки (в мировых координатах) на плоскость экрана (2D).
def project_with_camera(points4, camera_pos, target, up, focal, mode='perspective', ortho_dist=1.0):
    # points4 – массив вершин в однородных координатах (x,y,z,1).
    # camera_pos, target, up – параметры камеры.
    # focal – фокусное расстояние (для перспективы).
    # mode – 'perspective' или 'orthographic'.
    # ortho_dist – расстояние до объекта, используется для масштабирования в ортографии.
    # Возвращает массив из N элементов, каждый либо None (если точка позади камеры),
    # либо кортеж (x_экран, y_экран) в логических координатах (не пиксели).

    # Преобразуем точки в массив float
    pts = np.array(points4, dtype=float)
    # Получаем матрицу вида (преобразование мировые -> камера)
    V = look_at_matrix(camera_pos, target, up)
    # Умножаем все точки на матрицу вида: (V @ pts.T).T
    pts_cam = (V @ pts.T).T   # теперь точки в системе координат камеры 

    res = []          # список результатов
    eps = 1e-9        # маленькое число для проверки
    for p in pts_cam:
        x, y, z, w = p   # извлекаем координаты (w обычно = 1, но может меняться после перспективного деления)
        # Если точка лежит на плоскости камеры или позади неё (z <= 0), мы её не видим
        if z <= eps:
            res.append(None)
            continue
        # В зависимости от режима проекции вычисляем коэффициент масштабирования t
        if mode == 'perspective':
            # Перспектива: чем дальше точка (z больше), тем меньше масштаб (t = focal / z)
            t = focal / z
        else:  # orthographic
            # Ортографическая проекция: мы просто берём координаты x,y без деления на z.
            # Чтобы размер объекта не скакал при переключении, масштабируем так,
            # чтобы на расстоянии ortho_dist размер совпадал с перспективой.
            t = focal / ortho_dist   # подбираем такой коэффициент, чтобы на расстоянии ortho_dist объект был того же размера
        res.append((x * t, y * t))   # умножаем координаты на коэффициент, получаем логические экранные координаты
    return np.array(res, dtype=object)   # возвращаем как массив объектов (может содержать None)

# Модель буквы M
def make_letter_M(size=1.0, depth=0.2):
    # Создаёт проволочную 3D-букву M в виде набора вершин и рёбер.
    # size – размер буквы (и высота, и ширина), depth – толщина буквы по оси Z.
    # Возвращает кортеж (вершины, рёбра), где вершины – массив (N x 4) в однородных координатах,
    # рёбра – список пар индексов вершин, которые соединяются линией.
    h = size          # высота буквы
    w = size * 0.5    # половина ширины
    z0, z1 = depth / 2, -depth / 2   # передняя и задняя координаты Z

    # Определяем 5 вершин передней грани (индексы 0..4)
    verts = [
        (-w, -h/2, z0, 1.0),   # 0 – левая нижняя
        (-w,  h/2, z0, 1.0),   # 1 – левая верхняя
        (0,  0,   z0, 1.0),    # 2 – средняя точка (нижняя часть буквы M)
        ( w,  h/2, z0, 1.0),   # 3 – правая верхняя
        ( w, -h/2, z0, 1.0),   # 4 – правая нижняя
    ]
    # Добавляем 5 вершин задней грани (индексы 5..9) – такие же координаты, но z = z1
    verts += [
        (-w, -h/2, z1, 1.0),   
        (-w,  h/2, z1, 1.0),   
        (0,  0,   z1, 1.0),    
        ( w,  h/2, z1, 1.0),   
        ( w, -h/2, z1, 1.0),   
    ]

    # Рёбра: соединяем вершины, чтобы получилась проволочная модель.
    # Ребро задаётся парой индексов вершин.
    edges = [
        # Передняя грань (по контуру M)
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Задняя грань
        (5, 6), (6, 7), (7, 8), (8, 9),
        # Боковые рёбра (толщина)
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
    ]

    return np.array(verts, dtype=float), edges

# Создаем класс GLWidget, который наследуется от класса QWidget из модуля QtWidgets 
class GLWidget(QtWidgets.QWidget):
    # Основной виджет, на котором рисуется 3D-сцена.
    # Он обрабатывает события мыши и клавиатуры, а также перерисовывает изображение

    # Сигнал, который передается при изменении положения камеры (для обновления полей ввода)
    cameraChanged = QtCore.pyqtSignal(float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)   # вызываем конструктор родительского класса
        # Устанавливаем минимальный размер виджета
        self.setMinimumSize(800, 600)
        # Разрешаем виджету получать фокус клавиатуры (чтобы обрабатывать нажатия клавиш)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

        # Создаём модель буквы M (вершины и рёбра)
        self.model_vertices, self.model_edges = make_letter_M(size=1.0, depth=0.25)

        # Параметры модели в мировом пространстве (аффинные преобразования)
        self.tx, self.ty, self.tz = 0.0, 0.0, 0.0   # перенос
        self.rx, self.ry, self.rz = 0.0, 0.0, 0.0   # углы поворота (радианы)
        self.s = 1.0   # общий масштаб

        # Для обработки мыши: запоминаем последнюю позицию и режим
        self.last_mouse = None        # последняя позиция мыши (хранятся координаты курсора мыши)
        self.mouse_mode = None        # текущий режим: 'rotate', 'pan', 'orbit_cam' или None

        # Параметры камеры
        self.camera_pos = np.array([4.0, 4.0, 4.0], dtype=float)   # положение камеры в мире
        self.camera_target = np.array([0.0, 0.0, 0.0], dtype=float) # точка, на которую смотрит камера
        self.camera_up = np.array([0.0, 1.0, 0.0], dtype=float)    # вектор "вверх"
        self.focal = 4.0   # фокусное расстояние для перспективной проекции

        # Режим проекции: 'perspective' или 'orthographic'
        self.projection_mode = 'perspective'

        # Пересчитываем сферические координаты камеры (для вращения вокруг цели)
        self._recalc_camera_spherical()

        # Настройка кистей для рисования линий
        self.line_pen = QtGui.QPen(QtGui.QColor(200, 30, 30), 2)   # красные линии для модели
        self.axis_pen_x = QtGui.QPen(QtGui.QColor(200, 30, 30), 2) # красная ось X
        self.axis_pen_y = QtGui.QPen(QtGui.QColor(30, 160, 30), 2) # зелёная ось Y
        self.axis_pen_z = QtGui.QPen(QtGui.QColor(30, 120, 200), 2)# синяя ось Z

    def _recalc_camera_spherical(self):
        # Вычисляет сферические координаты камеры (расстояние, азимут, угол места)
        # по текущему положению камеры и цели. Это нужно для удобного вращения камеры мышью

        v = self.camera_pos - self.camera_target   # вектор от цели к камере
        r = np.linalg.norm(v)   # радиус (расстояние до камеры)
        if r < 1e-6:            # если камера почти в центре цели, задаём расстояние по умолчанию
            r = 1e-6
            v = np.array([r, 0.0, 0.0])

        # Угол места (elevation) – угол между вектором и горизонтальной плоскостью
        el = math.asin(v[1] / r)
        # Азимут (azimuth) – угол в горизонтальной плоскости (от оси X к оси Z)
        az = math.atan2(v[2], v[0])
        # Сохраняем значения
        self.cam_radius = r
        self.cam_azimuth = az
        self.cam_elevation = el

    def _update_camera_from_spherical(self):
        # Обновляет положение камеры (self.camera_pos) на основе сферических координат.
        # Также передает сигнал, чтобы обновить поля ввода на панели
        r = max(1e-6, self.cam_radius)   # берём расстояние, но не меньше минимума
        el = self.cam_elevation
        az = self.cam_azimuth
        # Переводим сферические в декартовы координаты
        x = r * math.cos(el) * math.cos(az)
        y = r * math.sin(el)
        z = r * math.cos(el) * math.sin(az)
        # Позиция камеры = смещение от цели
        self.camera_pos = np.array([x, y, z], dtype=float) + self.camera_target
        # Посылаем сигнал об изменении
        self.cameraChanged.emit(float(self.camera_pos[0]), float(self.camera_pos[1]), float(self.camera_pos[2]))

    def sizeHint(self):
        # Рекомендуемый размер виджета
        return QtCore.QSize(900, 700)

    def model_matrix(self):
        # Формирует матрицу модели (Model Matrix), которая преобразует локальные координаты
        # буквы в мировые. Сначала масштабирование, затем повороты (по X, Y, Z), затем перенос
        T = translation_matrix(self.tx, self.ty, self.tz)
        Rx = rotation_x(self.rx)
        Ry = rotation_y(self.ry)
        Rz = rotation_z(self.rz)
        S = scale_matrix(self.s, self.s, self.s)
        # Порядок: сначала масштаб, потом повороты (Rx, Ry, Rz), потом перенос.
        # В матричной алгебре умножение справа налево: сначала S, потом Rx, потом Ry, потом Rz, потом T
        return T @ (Rz @ (Ry @ (Rx @ S)))

    def paintEvent(self, event):
        # Основной метод рисования. Вызывается автоматически при необходимости обновить виджет.
        # Здесь мы рисуем модель, оси координат и текстовую информацию.
        # Создаём объект QPainter для рисования
        qp = QtGui.QPainter(self)
        qp.setRenderHint(QtGui.QPainter.Antialiasing)  # включаем сглаживание
        # Заливаем фон бежевым цветом
        qp.fillRect(self.rect(), QtGui.QColor(245, 245, 245))

        # Получаем размеры виджета
        w, h = self.width(), self.height()
        cx, cy = w / 2.0, h / 2.0   # центр экрана

        # Вычисляем матрицу модели для текущих параметров
        M = self.model_matrix()
        # Применяем матрицу модели к вершинам буквы, получаем мировые координаты
        verts4_world = (M @ self.model_vertices.T).T

        # Расстояние от камеры до цели (используется для ортографического масштабирования)
        ortho_dist = self.cam_radius

        # Проецируем вершины модели на экран (логические координаты)
        pts2d_model = project_with_camera(
            verts4_world,
            self.camera_pos,
            self.camera_target,
            self.camera_up,
            self.focal,
            mode=self.projection_mode,
            ortho_dist=ortho_dist
        )

        # Масштабный коэффициент для перевода логических координат в пиксели
        scale_screen = min(w, h) * 0.5

        # Рисуем рёбра модели
        qp.setPen(self.line_pen)   # устанавливаем синюю кисть
        for (i, j) in self.model_edges:
            p1 = pts2d_model[i]
            p2 = pts2d_model[j]
            if p1 is None or p2 is None:
                continue   # если вершина позади камеры, не рисуем ребро
            # Преобразуем логические координаты в пиксельные (с учётом центра и масштаба)
            sx1, sy1 = cx + p1[0] * scale_screen, cy - p1[1] * scale_screen
            sx2, sy2 = cx + p2[0] * scale_screen, cy - p2[1] * scale_screen
            qp.drawLine(int(sx1), int(sy1), int(sx2), int(sy2))

        # Рисуем мировые оси координат (чтобы видеть ориентацию)
        axis_len = 1.2   # длина осей в мировых единицах
        origin = np.array([0.0, 0.0, 0.0, 1.0])   # начало координат (однородные)
        # Определяем концы осей (X, Y, Z)
        axis_points = [
            (origin, np.array([axis_len, 0.0, 0.0, 1.0])),
            (origin, np.array([0.0, axis_len, 0.0, 1.0])),
            (origin, np.array([0.0, 0.0, axis_len, 1.0])),
        ]
        # Превращаем в плоский массив для проекции
        pts_for_axes = np.array([p for pair in axis_points for p in pair], dtype=float)
        # Проецируем точки осей
        proj_axes = project_with_camera(
            pts_for_axes,
            self.camera_pos,
            self.camera_target,
            self.camera_up,
            self.focal,
            mode=self.projection_mode,
            ortho_dist=ortho_dist
        )
        # Для каждой оси (X,Y,Z) рисуем линию и стрелку
        for idx, pen in enumerate((self.axis_pen_x, self.axis_pen_y, self.axis_pen_z)):
            p_origin = proj_axes[2 * idx]
            p_end = proj_axes[2 * idx + 1]
            if p_origin is None or p_end is None:
                continue
            sx1, sy1 = cx + p_origin[0] * scale_screen, cy - p_origin[1] * scale_screen
            sx2, sy2 = cx + p_end[0] * scale_screen, cy - p_end[1] * scale_screen
            qp.setPen(pen)
            qp.drawLine(int(sx1), int(sy1), int(sx2), int(sy2))
            # Рисуем стрелку на конце оси (треугольник)
            dx = sx2 - sx1
            dy = sy2 - sy1
            L = math.hypot(dx, dy)   # длина линии
            if L > 1.0:   # если линия не слишком короткая
                ux, uy = dx / L, dy / L   # единичный вектор направления
                vx, vy = -uy, ux          # перпендикулярный вектор
                arrow_len = min(12, L * 0.15)   # длина стрелки
                # Вычисляем координаты двух точек основания стрелки
                ax1 = sx2 - ux * arrow_len + vx * arrow_len * 0.4
                ay1 = sy2 - uy * arrow_len + vy * arrow_len * 0.4
                ax2 = sx2 - ux * arrow_len - vx * arrow_len * 0.4
                ay2 = sy2 - uy * arrow_len - vy * arrow_len * 0.4
                qp.drawLine(int(sx2), int(sy2), int(ax1), int(ay1))
                qp.drawLine(int(sx2), int(sy2), int(ax2), int(ay2))

        # Устанавливаем чёрный цвет и шрифт
        qp.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0)))
        qp.setFont(QtGui.QFont("Consolas", 10))
        x0, y0 = 10, 20          # начальная позиция текста (с отступом от края)
        line_height = 18         # высота строки

        # Каждая строка выводится с новой строки
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

    # Обработка мыши и клавиатуры
    def mousePressEvent(self, ev):
        # Обработка нажатия кнопки мыши.
        # ЛКМ – вращение модели (или камеры, если зажат Ctrl).
        # ПКМ – перемещение модели (pan)

        if ev.button() == QtCore.Qt.LeftButton:
            mods = QtWidgets.QApplication.keyboardModifiers()   # узнаём, зажаты ли модификаторы (Ctrl, Shift)
            if mods & QtCore.Qt.ControlModifier:
                # Режим вращения камеры вокруг объекта
                self.mouse_mode = "orbit_cam"
                # Запоминаем начальные азимут и угол места, чтобы отслеживать изменение
                self._orbit_start_az = self.cam_azimuth
                self._orbit_start_el = self.cam_elevation
                self._orbit_start_mouse = ev.pos()   # позиция мыши в момент нажатия
            else:
                # Режим вращения модели
                self.mouse_mode = "rotate"
            self.last_mouse = ev.pos()
        elif ev.button() == QtCore.Qt.RightButton:
            # Режим перемещения модели (pan)
            self.mouse_mode = "pan"
            self.last_mouse = ev.pos()

    def mouseMoveEvent(self, ev):
        # Обработка движения мыши с зажатой кнопкой.
        # В зависимости от режима (rotate, pan, orbit_cam) изменяем параметры модели или камеры     
        if self.last_mouse is None:
            return   # если кнопка не была нажата, игнорируем
        pos = ev.pos()
        dx = pos.x() - self.last_mouse.x()   # смещение по X
        dy = pos.y() - self.last_mouse.y()   # смещение по Y
        mods = QtWidgets.QApplication.keyboardModifiers()

        if self.mouse_mode == "rotate":
            # Вращение модели: по умолчанию крутим вокруг OY (dx) и OX (dy)
            if mods & QtCore.Qt.ShiftModifier:
                # С зажатым Shift – вращаем вокруг OZ
                self.rz += dx * 0.01
            else:
                self.ry += dx * 0.01
                self.rx += dy * 0.01

        elif self.mouse_mode == "pan":
            # Перемещение модели: по XY (без Shift) или по Z (с Shift)
            if mods & QtCore.Qt.ShiftModifier:
                self.tz += dy * 0.005
            else:
                self.tx += dx * 0.005
                self.ty -= dy * 0.005   # знак минус, потому что Y на экране направлен вниз

        elif self.mouse_mode == "orbit_cam":
            # Вращение камеры вокруг цели: изменяем азимут и угол места
            start = self._orbit_start_mouse
            ddx = pos.x() - start.x()
            ddy = pos.y() - start.y()
            az = self._orbit_start_az + ddx * 0.01
            el = self._orbit_start_el + ddy * 0.01
            # Ограничиваем угол места, чтобы камера не переворачивалась
            max_el = math.radians(89.0)   # почти 90 градусов
            el = max(-max_el, min(max_el, el))
            self.cam_azimuth = az
            self.cam_elevation = el
            # Пересчитываем положение камеры
            self._update_camera_from_spherical()

        # Запоминаем текущую позицию мыши для следующего движения
        self.last_mouse = pos
        self.update()   # вызываем перерисовку

    def mouseReleaseEvent(self, ev):
        # Отпускание кнопки мыши – сбрасываем режим
        self.mouse_mode = None
        self.last_mouse = None

    def wheelEvent(self, ev):
        # Обработка колёсика мыши – масштабирование модели.
        delta = ev.angleDelta().y() / 120.0   # стандартное значение: один "щёлчок" = 120
        factor = 1.1 ** delta   # каждый шаг умножает масштаб на 1.1 (или делит)
        self.s *= factor
        # Ограничиваем масштаб заданными пределами
        self.s = max(MIN_SCALE, min(MAX_SCALE, self.s))
        self.update()

    def keyPressEvent(self, ev):
        # Обработка нажатий клавиш
        k = ev.key()
        if k == QtCore.Qt.Key_R:
            # Сброс всех преобразований
            self.reset_transform()
        elif k == QtCore.Qt.Key_P:
            # Переключение проекции
            self.toggle_projection()
        self.update()   # обновляем экран

    def toggle_projection(self):
        # Переключает режим проекции между перспективой и ортографией
        if self.projection_mode == 'perspective':
            self.projection_mode = 'orthographic'
        else:
            self.projection_mode = 'perspective'
        self.update()

    def reset_transform(self):
        # Сбрасывает все модельные преобразования и камеру в начальное состояние
        self.tx, self.ty, self.tz = 0.0, 0.0, 0.0
        self.rx, self.ry, self.rz = 0.0, 0.0, 0.0
        self.s = 1.0
        self.camera_pos = np.array([4.0, 4.0, 4.0], dtype=float)
        self.camera_target = np.array([0.0, 0.0, 0.0], dtype=float)
        self._recalc_camera_spherical()
        self.cameraChanged.emit(float(self.camera_pos[0]), float(self.camera_pos[1]), float(self.camera_pos[2]))
        self.update()

# Главное окно
class MainWindow(QtWidgets.QMainWindow):
    # Главное окно приложения. Содержит центральный виджет (GLWidget) и панель управления (QDockWidget)
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Лабораторная работа 1 – 3D проволочная буква M")
        # Создаём экземпляр нашего виджета с 3D-сценой
        self.gl = GLWidget(self)
        self.setCentralWidget(self.gl)   # размещаем его в центре окна
        self.create_controls()          # создаём панель управления

    def create_controls(self):
        # Создаёт панель управления справа (QDockWidget) с элементами для выбора проекции, параметров камеры, кнопкой сброса и подсказками
        # Создаём док-виджет (плавающую панель)
        dock = QtWidgets.QDockWidget("Управление", self)
        # Добавляем прокрутку, чтобы все элементы помещались
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)   # вертикальное расположение

        # Кнопки для выбора проекции 
        layout.addWidget(QtWidgets.QLabel("<b>Проекция</b>"))
        # Создаём группу, чтобы кнопки были взаимоисключающими (одна активна)
        self.proj_group = QtWidgets.QButtonGroup(self)
        self.radio_persp = QtWidgets.QRadioButton("Перспективная")
        self.radio_ortho = QtWidgets.QRadioButton("Ортографическая")
        self.radio_persp.setChecked(True)   # по умолчанию выбрана перспектива
        # Добавляем кнопки в группу
        self.proj_group.addButton(self.radio_persp)
        self.proj_group.addButton(self.radio_ortho)
        # Подключаем сигнал изменения состояния к обработчику
        self.radio_persp.toggled.connect(self.on_proj_changed)
        self.radio_ortho.toggled.connect(self.on_proj_changed)
        layout.addWidget(self.radio_persp)
        layout.addWidget(self.radio_ortho)

        # Параметры камеры (поля ввода X,Y,Z) -
        layout.addWidget(QtWidgets.QLabel("<b>Параметры камеры</b>"))
        cam_layout = QtWidgets.QGridLayout()   # табличная раскладка для компактности
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

        # Функция, вызываемая при изменении любого из полей камеры
        def cam_changed(_):
            # Берём значения из полей и устанавливаем позицию камеры
            self.gl.camera_pos = np.array([self.cam_x.value(), self.cam_y.value(), self.cam_z.value()], dtype=float)
            self.gl._recalc_camera_spherical()   # пересчитываем сферические параметры
            self.gl.cameraChanged.emit(float(self.gl.camera_pos[0]), float(self.gl.camera_pos[1]), float(self.gl.camera_pos[2]))
            self.gl.update()

        # Подключаем сигналы изменения к этой функции
        self.cam_x.valueChanged.connect(cam_changed)
        self.cam_y.valueChanged.connect(cam_changed)
        self.cam_z.valueChanged.connect(cam_changed)

        # Функция, обновляющая поля ввода при изменении камеры из других источников (например, вращение мышью)
        def on_camera_changed(x, y, z):
            # Блокируем сигналы, чтобы не вызвать бесконечный цикл
            self.cam_x.blockSignals(True)
            self.cam_y.blockSignals(True)
            self.cam_z.blockSignals(True)
            self.cam_x.setValue(x)
            self.cam_y.setValue(y)
            self.cam_z.setValue(z)
            # Разблокируем сигналы
            self.cam_x.blockSignals(False)
            self.cam_y.blockSignals(False)
            self.cam_z.blockSignals(False)

        # Подключаем сигнал от виджета 3D к этой функции
        self.gl.cameraChanged.connect(on_camera_changed)
        layout.addLayout(cam_layout)

        # Кнопка сброса
        btn_row = QtWidgets.QHBoxLayout()
        btn_reset = QtWidgets.QPushButton("Сброс (R)")
        btn_reset.clicked.connect(self.gl.reset_transform)
        btn_row.addWidget(btn_reset)
        layout.addLayout(btn_row)

        # Текст с подсказками по управлению
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
        instr.setWordWrap(True)   # разрешаем перенос текста
        layout.addWidget(instr)

        layout.addStretch()   # добавляем "растяжку", чтобы элементы прижимались к верху
        panel.setLayout(layout)
        scroll.setWidget(panel)
        dock.setWidget(scroll)
        # Размещаем панель справа
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

    def on_proj_changed(self):
        # Обработчик изменения состояния радиокнопок.
        # Устанавливает соответствующий режим проекции в виджете 3D

        if self.radio_persp.isChecked():
            self.gl.projection_mode = 'perspective'
        else:
            self.gl.projection_mode = 'orthographic'
        self.gl.update()   # перерисовываем сцену

if __name__ == "__main__":
    # Создаём экземпляр приложения Qt
    app = QtWidgets.QApplication(sys.argv)
    # Создаём главное окно
    w = MainWindow()
    # Устанавливаем начальный размер
    w.resize(1200, 760)
    # Показываем окно
    w.show()
    # Запускаем цикл обработки событий; при выходе завершаем программу
    sys.exit(app.exec_())
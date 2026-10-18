import sys, math, time
import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QSurfaceFormat
from PyQt5.QtWidgets import QOpenGLWidget, QColorDialog

from OpenGL import GL
from OpenGL.GL import shaders

# шейдеры для модели освещения Гуро
# Вершинный шейдер вычисляет освещение для каждой вершины
VERT_SHADER = """
#version 330 core // версия языка GLSL

// Эти данные передаются из VBO (Vertex Buffer Object) для каждой вершины
layout(location = 0) in vec3 aPos; // позиция вершины
layout(location = 1) in vec3 aNormal; // нормаль вершины

// Эти данные одинаковы для всех вершин и передаются из программы 
uniform mat4 uMVP; // матрица модель-камера-проекция
uniform vec3 uLightPosWorld; // позиция света в мировых координатах (вектор L)
uniform float uKa; // коэффициент фонового освещения
uniform float uKd; // коэффициент диффузного отражения  
uniform float uIa; // интенсивность фонового освещения
uniform float uIl; // интенсивность источника света

// Передается в следующий этап (фрагментный шейдер) для интерполяции
out float vIntensity; // выходная интенсивность освещения

void main() {
    vec4 posWorld = vec4(aPos, 1.0); // позиция в мировых координатах
    vec3 N = normalize(aNormal); // нормализуем нормаль
    vec3 L = normalize(uLightPosWorld - posWorld.xyz); // направление к свету
    
    float diff = max(0.0, dot(N, L)); // расчет диффузной составляющей
    float I = uIa * uKa + uIl * uKd * diff; // общая интенсивность
    vIntensity = clamp(I, 0.0, 1.0); // ограничиваем диапазон

    // Умножаем 3D координаты вершины на матрицу MVP, 
    // чтобы видеокарта поняла, куда спроецировать эту точку на 2D-экране
    gl_Position = uMVP * vec4(aPos, 1.0); // преобразуем позицию
}
"""

# Фрагментный шейдер - использует вычисленную интенсивность и цвет объекта
FRAG_SHADER = """
#version 330 core
in float vIntensity; // интенсивность из вершинного шейдера
out vec4 FragColor; // Финальный цвет пикселя в формате RGBA (Красный, Зеленый, Синий, Альфа/прозрачность)

uniform vec3 uColor; // цвет объекта

void main() {
    float v = clamp(vIntensity, 0.0, 1.0); // ограничиваем интенсивность
    // Умножаем базовый цвет на интенсивность света
    FragColor = vec4(uColor.r * v, uColor.g * v, uColor.b * v, 1.0); 
}
"""

# Вершинный шейдер для осей 
AXIS_VERT = """
#version 330 core 
layout(location=0) in vec3 aPos; // координаты вершин линий (начало и конец оси)
uniform mat4 uMVP; // Матрица проекции
uniform vec3 uColor; // Цвет для текущей оси
out vec3 vColor; // Выходная переменная

void main() { 
    vColor = uColor; 
    gl_Position = uMVP * vec4(aPos, 1.0); // Преобразуем 3D координаты в 2D экран
}
"""

# Фргаментный шейдер для осей 
AXIS_FRAG = """
#version 330 core
// Цвет, пришедший от вершинного шейдера
in vec3 vColor;
out vec4 FragColor; // Итоговый цвет пикселя

void main() { 
    // Формируем итоговый цвет
    FragColor = vec4(vColor,1.0); 
}
"""

# Создает матрицу перспективной проекции
def perspective(fovy, aspect, znear, zfar):
    # fovy - угол обзора по вертикали (60 градусов)
    # aspect - соотношение сторон экрана (ширина / высота), чтобы объекты не растягивались
    # znear - ближняя плоскость отсечения (всё, что ближе, не рисуется)
    # zfar - дальняя плоскость отсечения (всё, что дальше, не рисуется)

    f = 1.0 / math.tan(fovy / 2.0) # фокусное расстояние
    M = np.zeros((4,4), dtype=np.float32) # тип float32, так как видеокарты работают с 32-битными числами с плавающей точкой
    # Заполняем масштаб по оси X
    M[0,0] = f / aspect
    # Заполняем масштаб по оси Y
    M[1,1] = f 
    # Масштабирование по оси Z (глубине)
    M[2,2] = (zfar + znear) / (znear - zfar) 
    # Смещение по оси Z
    M[2,3] = (2.0 * zfar * znear) / (znear - zfar)
    # Четвертая координата (w) нового вектора становится равной -z (глубине).
    # Видеокарта потом автоматически делит X и Y на эту w, из-за чего 
    # дальние объекты (где W большое) становятся визуально меньше
    M[3,2] = -1.0
    return M

# Создает матрицу камеры
def look_at(eye, center, up):
    # eye - координаты, где стоит камера (x, y, z)
    # center - координаты точки, на которую смотрит камера (x, y, z)
    # up - вектор, указывающий направление "вверх" для камеры (0, 1, 0)

    eye = np.array(eye, dtype=np.float32)
    center = np.array(center, dtype=np.float32)
    up = np.array(up, dtype=np.float32)

    # Вычисляем вектор направления взгляда
    f = center - eye
    f = f / np.linalg.norm(f)          
    u = up / np.linalg.norm(up)    
    # Вычисляем вектор "вправо"        
    s = np.cross(f, u)                  
    s = s / np.linalg.norm(s)
    # Пересчитываем истинный вектор "вверх" для камеры
    u = np.cross(s, f)    
    
    M = np.eye(4, dtype=np.float32)
    M[0,0:3] = s
    M[1,0:3] = u  
    M[2,0:3] = -f # В OpenGL камера смотрит в -Z
    
    T = np.eye(4, dtype=np.float32)
    T[0:3,3] = -eye # Переводим координаты из мирового пространства в пространство камеры
    
    return M @ T

# Генерирует сферу как треугольную сетку
def make_sphere(radius=1.0, lat_segments=20, lon_segments=36):
    # radius - радиус сферы
    # lat_segments - количество сегментов по широте 
    # lon_segments - количество сегментов по долготе 

    # Создаем пустые массивы для хранения вершин и индексов треугольников
    positions = []
    indices = []

    lat_segments = max(2, int(lat_segments))
    lon_segments = max(3, int(lon_segments))
    
    # Генерируем вершины
    for i in range(lat_segments + 1):
        phi = math.pi * i / lat_segments  # угол от 0 до pi
        y = radius * math.cos(phi) # координата Y
        r = radius * math.sin(phi) # радиус на этой широте
        for j in range(lon_segments):
            theta = 2.0 * math.pi * j / lon_segments  # угол от 0 до 2pi
            x = r * math.cos(theta) # координата X
            z = r * math.sin(theta) # координата Z
            positions.append((x,y,z))
    
    # Генерируем индексы треугольников
    L = lon_segments
    for i in range(lat_segments):
        for j in range(lon_segments):
            a = i * L + j # текущая вершина
            b = (i + 1) * L + j # вершина ниже
            c = (i + 1) * L + ((j + 1) % L) # вершина ниже и справа
            d = i * L + ((j + 1) % L) # вершина справа
            
            # Два треугольника образуют прямоугольник
            indices.extend((a,b,c))
            indices.extend((a,c,d))

    # Возвращаем кортеж из двух массивов, приведенных к типам для OpenGL
    return np.array(positions, dtype=np.float32), np.array(indices, dtype=np.uint32)

# Генерирует эллипсоид как треугольную сетку
def make_ellipsoid(rx=1.4, ry=0.8, rz=1.0, lat_segments=18, lon_segments=32):
    # rx, ry, rz - радиусы по осям X, Y и Z соответственно

    # Создаем массивы для вершин и индексов
    pos = []
    idx = []

    lat_segments = max(2, int(lat_segments))
    lon_segments = max(3, int(lon_segments))
    
    for i in range(lat_segments + 1):
        phi = math.pi * i / lat_segments
        # Заранее вычисляем косинус и синус угла
        cy = math.cos(phi)
        sy = math.sin(phi)
        for j in range(lon_segments):
            theta = 2.0 * math.pi * j / lon_segments
            cx = math.cos(theta)
            sx = math.sin(theta)
            # Координаты с учетом разных радиусов по осям
            x = rx * sy * cx
            y = ry * cy  
            z = rz * sy * sx
            pos.append((x,y,z))
    
    # Триангуляция аналогична сфере
    L = lon_segments
    for i in range(lat_segments):
        for j in range(lon_segments):
            a = i * L + j
            b = (i + 1) * L + j
            c = (i + 1) * L + ((j + 1) % L)
            d = i * L + ((j + 1) % L)
            idx.extend((a,b,c))
            idx.extend((a,c,d))
            
    return np.array(pos,dtype=np.float32), np.array(idx,dtype=np.uint32)

# Генерируем перевернутый параболоид (купол) с закрытым дном
def make_paraboloid(R=1.2, H=1.5, radial_segments=24, lon_segments=40):
    # R - радиус основания, H - общая высота фигуры

    pos = []
    idx = []
    radial_segments = max(2, int(radial_segments))
    lon_segments = max(3, int(lon_segments))
    
    # 1. Боковая поверхность (перевернутый параболоид)
    for i in range(radial_segments + 1):
        # u - параметр от 0 до 1
        u = i / radial_segments
        # Радиус
        r = u * R
        y = H * 0.5 - H * (u * u)
        
        for j in range(lon_segments):
            theta = 2.0 * math.pi * j / lon_segments
            x = r * math.cos(theta)
            z = r * math.sin(theta)
            pos.append((x, y, z))
    
    # Индексы боковой поверхности
    L = lon_segments
    for i in range(radial_segments):
        for j in range(lon_segments):
            a = i * L + j
            b = (i + 1) * L + j
            c = (i + 1) * L + ((j + 1) % L)
            d = i * L + ((j + 1) % L)
            idx.extend((a, b, c))
            idx.extend((a, c, d))
    
    # 2. Дно. Используем вершины последнего ряда боковой поверхности.
    # Это обеспечивает сглаживание нормалей и правильный переход света

    # Добавляем всего одну новую вершину - центр дна
    center_idx = len(pos)
    pos.append((0.0, -H * 0.5, 0.0))  # Центральная вершина дна
    
    # Вычисляем индекс, с которого начинается последний ряд боковой поверхности
    # (radial_segments * L - это номер первой вершины нижнего кольца)
    bottom_start = radial_segments * L 
    
    # Треугольники дна
    for j in range(lon_segments):
        a = center_idx
        b = bottom_start + j
        c = bottom_start + ((j + 1) % lon_segments)
        idx.extend((a, b, c)) 
    
    return np.array(pos, dtype=np.float32), np.array(idx, dtype=np.uint32)

# Вычисление нормалей вершин и ориентация
def compute_vertex_normals(positions: np.ndarray, indices: np.ndarray):
    # positions - массив координат всех вершин (x, y, z)
    # indices - массив индексов

    # Получаем общее количество вершин
    N = positions.shape[0]
    normals = np.zeros((N,3), dtype=np.float64)  # Используем float64 для накопления суммы (чтобы избежать ошибок округления)
    
    # Вычисляем, сколько всего треугольников (каждые 3 индекса = 1 треугольник)
    tri_count = indices.size // 3
    for t in range(tri_count):
        # Достаем индексы трех вершин текущего треугольника
        i0 = int(indices[3*t+0])
        i1 = int(indices[3*t+1]) 
        i2 = int(indices[3*t+2])

        # Получаем координаты этих трех вершин
        p0 = positions[i0]
        p1 = positions[i1]
        p2 = positions[i2]
        
        # Нормаль треугольника
        v1 = p1 - p0
        v2 = p2 - p0
        fn = np.cross(v1, v2)
        nlen = np.linalg.norm(fn)
        if nlen > 1e-12:
            fn = fn / nlen # нормализуем
        else:
            fn = np.array([0.0,0.0,0.0]) # вырожденный треугольник
            
        # Суммируем
        normals[i0] += fn 
        normals[i1] += fn  
        normals[i2] += fn
    
    # Нормализуем результирующие нормали
    for i in range(N):
        l = np.linalg.norm(normals[i])
        if l < 1e-12:
            normals[i] = np.array([0.0,0.0,1.0]) # нормаль по умолчанию
        else:
            normals[i] /= l
            
    # Для центрально-симметричных объектов: нормали должны быть направлены наружу
    for i in range(N):
        # Скалярное произведение нормали и вектора позиции вершины
        if np.dot(normals[i], positions[i]) < 0.0:
            normals[i] = -normals[i] # инвертируем если направлена внутрь
            
    return normals.astype(np.float32) # возвращаем как float32 для OpenGL

# Ориентирует треугольники наружу (иначе OpenGL нормально не работает)
def orient_triangles_outward(positions: np.ndarray, indices: np.ndarray):
    # Копируем массив индексов и приводим его к типу int64
    idx = indices.copy().astype(np.int64)
    
    for t in range(idx.size // 3):
        # Достаем индексы трех вершин текущего треугольника
        i0 = idx[3*t+0]
        i1 = idx[3*t+1] 
        i2 = idx[3*t+2]
        # Получаем координаты этих вершин
        p0 = positions[i0]
        p1 = positions[i1]
        p2 = positions[i2]
        
        # Нормаль треугольника
        fn = np.cross(p1 - p0, p2 - p0)
        if np.linalg.norm(fn) < 1e-12:
            continue  # пропускаем вырожденные
            
        centroid = (p0 + p1 + p2) / 3.0  # центр треугольника
        
        # Если нормаль направлена к центру объекта - меняем порядок вершин
        if np.dot(fn, centroid) < 0.0:
            idx[3*t+1], idx[3*t+2] = idx[3*t+2], idx[3*t+1]  # меняем порядок
            
    return idx.astype(np.uint32)


# Виджет OpenGL
class GLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        fmt = QSurfaceFormat()
        fmt.setDepthBufferSize(24) # буфер глубины 
        fmt.setProfile(QSurfaceFormat.CoreProfile)
        fmt.setVersion(3, 3)
        self.setFormat(fmt)

        # Параметры модели
        self.shape = 'sphere' # текущая форма
        self.lat_segments = 24 # сегменты по широте (радиальные для параболоида)
        self.lon_segments = 40 # сегменты по долготе
        self.object_color = np.array([1.0, 1.0, 1.0], dtype=np.float32) # Белый (дает серый при умножении)

        # Камера (сферические координаты)
        self.camera_target = np.array([0.0,0.0,0.0], dtype=np.float32)
        self.cam_radius = 4.5 # расстояние до цели
        self.cam_az = 0.0 # азимут
        self.cam_el = math.radians(10.0) # угол возвышения
        self.camera_up = np.array([0.0,1.0,0.0], dtype=np.float32)

        # Освещение
        self.light_world = np.array([2.5,3.0,2.5], dtype=np.float32)
        self.light_on = True
        self.Ka = 0.35 # коэффициент фонового света
        self.Kd = 1.0 # коэффициент диффузии
        self.Ia = 0.25 # интенсивность фонового света
        self.Il = 1.2 # интенсивность источника

        # Переключатели отображения
        self.show_wireframe = True # показывать триангуляцию
        self.show_normals = False # показывать нормали
        self.show_axes = True # показывать оси

        # Анимация движения камеры
        self.cam_move_active = False # активно ли движение
        self.cam_move_start = 0.0 # время начала
        self.cam_move_dur = 1500 # длительность в мс
        self.cam_move_from_az = 0.0 # начальный азимут
        self.cam_move_from_el = 0.0 # начальный угол возвышения
        self.cam_move_from_radius = 0.0 # начальный радиус

        self.program = None # основная программа шейдеров
        self.axis_program = None # программа для осей
        self.use_vao = True # использовать VAO
        self.vao = None # ID Vertex Array Object (хранит настройки привязки данных)
        self.vbo = None # ID Vertex Buffer Object (хранит координаты вершин и нормали)
        self.ebo = None # # ID Element Buffer Object (хранит индексы треугольников)
        self.index_count = 0 # количество индексов
        
        self.axes_vao = None # VAO для осей
        self.axes_vbo = None # VBO для осей
        
        self.light_vao = None # VAO для маркера света
        self.light_vbo = None # VBO для маркера света
        
        self.normals_vao = None # VAO для нормалей
        self.normals_vbo = None # VBO для нормалей
        self.normals_count = 0 # количество вершин нормалей

        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.mouse_mode = None # режим мыши
        self.last_mouse = None # последняя позиция мыши

        # Таймер для анимации
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.on_timer)
        self.timer.start(16)   

    # Инициализация OpenGL
    def initializeGL(self):
        try:
            # Компилируем шейдеры
            vert = shaders.compileShader(VERT_SHADER, GL.GL_VERTEX_SHADER)
            frag = shaders.compileShader(FRAG_SHADER, GL.GL_FRAGMENT_SHADER)
            self.program = shaders.compileProgram(vert, frag)
            
            vert2 = shaders.compileShader(AXIS_VERT, GL.GL_VERTEX_SHADER)
            frag2 = shaders.compileShader(AXIS_FRAG, GL.GL_FRAGMENT_SHADER)
            self.axis_program = shaders.compileProgram(vert2, frag2)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка шейдеров", str(e))
            self.program = None
            return

        # Проверяем поддержку VAO
        try:
            vao_test = GL.glGenVertexArrays(1)
            try: 
                GL.glDeleteVertexArrays(1, [vao_test])
            except Exception: 
                pass
            self.use_vao = True
        except Exception:
            self.use_vao = False

        # Создаем геометрию
        self.create_mesh()
        self.create_axes()
        self.create_light_marker()

        # Настройки OpenGL
        GL.glEnable(GL.GL_DEPTH_TEST) # тест глубины
        try:
            GL.glEnable(GL.GL_CULL_FACE) # отсечение задних граней
            GL.glCullFace(GL.GL_BACK) # отсекать задние
            GL.glFrontFace(GL.GL_CCW) # лицевые - против часовой
        except Exception:
            pass
            
        GL.glClearColor(0.96,0.96,0.96,1.0) # цвет фона

    # Создает модель в зависимости от выбранной формы
    def create_mesh(self):
        if self.shape == 'sphere':
            pos, idx = make_sphere(radius=1.0, lat_segments=self.lat_segments, lon_segments=self.lon_segments)
            normals = np.zeros_like(pos, dtype=np.float32)
            for i in range(pos.shape[0]):
                p = pos[i]
                n = p / (np.linalg.norm(p) + 1e-12)
                normals[i] = n
                
        elif self.shape == 'ellipsoid':
            pos, idx = make_ellipsoid(rx=1.4, ry=0.8, rz=1.0, 
                                        lat_segments=max(4,self.lat_segments//1), 
                                        lon_segments=self.lon_segments)
            # Аналитические нормали для эллипсоида
            rx, ry, rz = 1.4, 0.8, 1.0
            normals = np.zeros_like(pos, dtype=np.float32)
            for i in range(pos.shape[0]):
                x,y,z = pos[i]
                nx = x/(rx*rx)  # нормаль к эллипсоиду
                ny = y/(ry*ry)
                nz = z/(rz*rz)
                n = np.array((nx,ny,nz), dtype=np.float32)
                n /= (np.linalg.norm(n) + 1e-12)
                normals[i] = n
                
        elif self.shape == 'paraboloid':
            pos, idx = make_paraboloid(R=1.2, H=1.5, 
                                       radial_segments=self.lat_segments, 
                                       lon_segments=self.lon_segments)
            # Нормали будут вычислены ниже через compute_vertex_normals
        else:
            pos, idx = make_sphere()
            normals = compute_vertex_normals(pos, idx)

        # Ориентируем треугольники наружу
        try:
            idx = orient_triangles_outward(pos, idx)
            
            try:
                p0 = pos[idx[0::3]] 
                p1 = pos[idx[1::3]] 
                p2 = pos[idx[2::3]] 
                
                fn = np.cross(p1 - p0, p2 - p0) 
                cent = (p0 + p1 + p2) * (1.0/3.0) 
                dots = (fn * cent).sum(axis=1) 
                score = np.mean(np.sign(dots)) 
                
                if score < 0.0:
                    GL.glFrontFace(GL.GL_CW) 
                else:
                    GL.glFrontFace(GL.GL_CCW) 
            except Exception:
                GL.glFrontFace(GL.GL_CCW) 
        except Exception:
            pass

        # Пересчитываем нормали для гарантии сглаживания
        normals = compute_vertex_normals(pos, idx)

        # Подготавливаем массивы для OpenGL
        pos = np.ascontiguousarray(pos, dtype=np.float32)
        normals = np.ascontiguousarray(normals, dtype=np.float32)
        idx = np.ascontiguousarray(idx, dtype=np.uint32)

        self.index_count = idx.size
        
        verts_interleaved = np.ascontiguousarray(np.hstack((pos, normals)).astype(np.float32), dtype=np.float32)

        try:
            if getattr(self, 'vao', None) and self.use_vao:
                GL.glDeleteVertexArrays(1, [self.vao])
        except Exception: 
            pass
        try:
            if getattr(self, 'vbo', None): 
                GL.glDeleteBuffers(1, [self.vbo])
        except Exception: 
            pass
        try:
            if getattr(self, 'ebo', None): 
                GL.glDeleteBuffers(1, [self.ebo])
        except Exception: 
            pass

        self.vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, verts_interleaved.nbytes, verts_interleaved, GL.GL_STATIC_DRAW)

        self.ebo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, GL.GL_STATIC_DRAW)

        self.vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)
                
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
                
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 6*4, GL.ctypes.c_void_p(0))
                
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 6*4, GL.ctypes.c_void_p(3*4))
                
        GL.glBindVertexArray(0)

        self._build_normals_lines(pos, normals)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)

    # Создает линии для визуализации нормалей
    def _build_normals_lines(self, pos, normals, scale=0.12):
        lines = []
        last_p = None 
        
        for i in range(pos.shape[0]):
            p = pos[i] 
            
            # Пропускаем дубликаты (например, в вершине параболоида или на полюсах сферы)
            if last_p is not None and np.allclose(p, last_p, atol=1e-5):
                continue
                
            n = normals[i] 
            lines.append(p.tolist()) 
            lines.append((p + n * scale).tolist()) 
            
            last_p = p.copy() 
            
        arr = np.array(lines, dtype=np.float32)
        self.normals_count = arr.shape[0] 
        
        try:
            if getattr(self, 'normals_vao', None): 
                GL.glDeleteVertexArrays(1, [self.normals_vao])
            if getattr(self, 'normals_vbo', None): 
                GL.glDeleteBuffers(1, [self.normals_vbo])
        except Exception:
            pass
            
        self.normals_vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.normals_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, arr.nbytes, arr, GL.GL_STATIC_DRAW)
        
        self.normals_vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.normals_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.normals_vbo)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, GL.ctypes.c_void_p(0))
        GL.glBindVertexArray(0)
            
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    # Создает оси координат
    def create_axes(self):
        axes = np.array([
            0,0,0, 1.5,0,0, 
            0,0,0, 0,1.5,0, 
            0,0,0, 0,0,1.5  
        ], dtype=np.float32)
        
        try:
            if getattr(self, 'axes_vao', None): 
                GL.glDeleteVertexArrays(1, [self.axes_vao])
            if getattr(self, 'axes_vbo', None): 
                GL.glDeleteBuffers(1, [self.axes_vbo])
        except Exception:
            pass
            
        self.axes_vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.axes_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, axes.nbytes, axes, GL.GL_STATIC_DRAW)
        
        if self.use_vao:
            self.axes_vao = GL.glGenVertexArrays(1)
            GL.glBindVertexArray(self.axes_vao)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.axes_vbo)
            GL.glEnableVertexAttribArray(0)
            GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, GL.ctypes.c_void_p(0))
            GL.glBindVertexArray(0)
        else:
            self.axes_vao = None
            
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    # Создает маркер для источника света
    def create_light_marker(self):
        try:
            if getattr(self, 'light_vao', None): 
                GL.glDeleteVertexArrays(1, [self.light_vao])
            if getattr(self, 'light_vbo', None): 
                GL.glDeleteBuffers(1, [self.light_vbo])
        except Exception:
            pass
            
        self.light_vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.light_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, 3*4, None, GL.GL_DYNAMIC_DRAW) 
        
        if self.use_vao:
            self.light_vao = GL.glGenVertexArrays(1)
            GL.glBindVertexArray(self.light_vao)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.light_vbo)
            GL.glEnableVertexAttribArray(0)
            GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, GL.ctypes.c_void_p(0))
            GL.glBindVertexArray(0)
        else:
            self.light_vao = None
            
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def resizeGL(self, w, h):
        GL.glViewport(0, 0, w, h)

    def paintGL(self):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        if not self.program:
            return

        w = max(1, self.width())
        h = max(1, self.height())
        aspect = w / float(h)
        
        proj = perspective(math.radians(60.0), aspect, 0.1, 100.0) 
        
        eye = self._sph_to_cart(self.cam_az, self.cam_el, self.cam_radius) + self.camera_target
        view = look_at(eye, self.camera_target, self.camera_up) 
        mv = view @ np.eye(4, dtype=np.float32)
        mvp = proj @ mv  

        # Отрисовка основной модели
        GL.glUseProgram(self.program)
        
        loc = GL.glGetUniformLocation(self.program, 'uMVP')
        GL.glUniformMatrix4fv(loc, 1, GL.GL_FALSE, mvp.T) 
        
        loc = GL.glGetUniformLocation(self.program, 'uLightPosWorld')
        GL.glUniform3f(loc, float(self.light_world[0]), float(self.light_world[1]), float(self.light_world[2]))
        
        GL.glUniform1f(GL.glGetUniformLocation(self.program, 'uKa'), float(self.Ka))
        GL.glUniform1f(GL.glGetUniformLocation(self.program, 'uKd'), float(self.Kd))
        GL.glUniform1f(GL.glGetUniformLocation(self.program, 'uIa'), float(self.Ia))
        GL.glUniform1f(GL.glGetUniformLocation(self.program, 'uIl'), float(self.Il) if self.light_on else 0.0)
        
        # Передаем цвет объекта
        loc = GL.glGetUniformLocation(self.program, 'uColor')
        GL.glUniform3f(loc, float(self.object_color[0]), float(self.object_color[1]), float(self.object_color[2]))

        # Включаем CULL_FACE для всех фигур (параболоид теперь замкнут)
        cull_was_enabled = GL.glIsEnabled(GL.GL_CULL_FACE)
        if not cull_was_enabled:
            GL.glEnable(GL.GL_CULL_FACE)

        GL.glEnable(GL.GL_POLYGON_OFFSET_FILL)
        GL.glPolygonOffset(1.0, 1.0)
        
        if self.use_vao and self.vao is not None:
            GL.glBindVertexArray(self.vao)
            GL.glDrawElements(GL.GL_TRIANGLES, self.index_count, GL.GL_UNSIGNED_INT, None)
            GL.glBindVertexArray(0)
        else:
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
            GL.glEnableVertexAttribArray(0)
            GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 6*4, GL.ctypes.c_void_p(0))
            GL.glEnableVertexAttribArray(1)
            GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 6*4, GL.ctypes.c_void_p(3*4))
            GL.glDrawElements(GL.GL_TRIANGLES, self.index_count, GL.GL_UNSIGNED_INT, None)
            GL.glDisableVertexAttribArray(0)
            GL.glDisableVertexAttribArray(1)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)

        GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)
        
        if not cull_was_enabled:
            GL.glDisable(GL.GL_CULL_FACE)
            
        GL.glUseProgram(0)

        # Триангуляция поверхностей
        if self.show_wireframe:
            GL.glUseProgram(self.axis_program)
            loc = GL.glGetUniformLocation(self.axis_program, 'uMVP')
            GL.glUniformMatrix4fv(loc, 1, GL.GL_FALSE, mvp.T)
            locc = GL.glGetUniformLocation(self.axis_program, 'uColor')
            GL.glUniform3f(locc, 0.08, 0.08, 0.08) 
            
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
            GL.glLineWidth(1.0)
            
            if self.use_vao and self.vao is not None:
                GL.glBindVertexArray(self.vao)
                GL.glDrawElements(GL.GL_TRIANGLES, self.index_count, GL.GL_UNSIGNED_INT, None)
                GL.glBindVertexArray(0)
            else:
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
                GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
                GL.glEnableVertexAttribArray(0)
                GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 6*4, GL.ctypes.c_void_p(0))
                GL.glDrawElements(GL.GL_TRIANGLES, self.index_count, GL.GL_UNSIGNED_INT, None)
                GL.glDisableVertexAttribArray(0)
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
                GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)
                
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
            GL.glUseProgram(0)

        # Визуализация нормалей
        if self.show_normals and getattr(self, 'normals_count', 0) > 0:
            GL.glUseProgram(self.axis_program)
            loc = GL.glGetUniformLocation(self.axis_program, 'uMVP')
            GL.glUniformMatrix4fv(loc, 1, GL.GL_FALSE, mvp.T)
            locc = GL.glGetUniformLocation(self.axis_program, 'uColor')
            GL.glUniform3f(locc, 0.0, 0.45, 0.8) 
            
            if self.use_vao and self.normals_vao is not None:
                GL.glBindVertexArray(self.normals_vao)
                GL.glDrawArrays(GL.GL_LINES, 0, self.normals_count)
                GL.glBindVertexArray(0)
            else:
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.normals_vbo)
                GL.glEnableVertexAttribArray(0)
                GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, GL.ctypes.c_void_p(0))
                GL.glDrawArrays(GL.GL_LINES, 0, self.normals_count)
                GL.glDisableVertexAttribArray(0)
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
                
            GL.glUseProgram(0)

        # Оси координат
        if self.show_axes:
            GL.glUseProgram(self.axis_program)
            loc = GL.glGetUniformLocation(self.axis_program, 'uMVP')
            GL.glUniformMatrix4fv(loc, 1, GL.GL_FALSE, mvp.T)
            locc = GL.glGetUniformLocation(self.axis_program, 'uColor')
            
            GL.glUniform3f(locc, 1.0, 0.0, 0.0)
            if self.use_vao and self.axes_vao is not None:
                GL.glBindVertexArray(self.axes_vao)
                GL.glDrawArrays(GL.GL_LINES, 0, 2)
            else:
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.axes_vbo)
                GL.glEnableVertexAttribArray(0)
                GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, GL.ctypes.c_void_p(0))
                GL.glDrawArrays(GL.GL_LINES, 0, 2)
                GL.glDisableVertexAttribArray(0)
                
            GL.glUniform3f(locc, 0.0, 1.0, 0.0)
            if self.use_vao and self.axes_vao is not None:
                GL.glDrawArrays(GL.GL_LINES, 2, 2)
            else:
                GL.glEnableVertexAttribArray(0)
                GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, GL.ctypes.c_void_p(2*3*4))
                GL.glDrawArrays(GL.GL_LINES, 0, 2) 
                
            GL.glUniform3f(locc, 0.0, 0.0, 1.0)
            if self.use_vao and self.axes_vao is not None:
                GL.glDrawArrays(GL.GL_LINES, 4, 2)
                GL.glBindVertexArray(0)
            else:
                GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, GL.ctypes.c_void_p(4*3*4))
                GL.glDrawArrays(GL.GL_LINES, 0, 2)
                GL.glDisableVertexAttribArray(0)
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
                
            GL.glUseProgram(0)

        # Маркер источника света
        if self.light_vbo is not None:
            GL.glUseProgram(self.axis_program)
            loc = GL.glGetUniformLocation(self.axis_program, 'uMVP')
            GL.glUniformMatrix4fv(loc, 1, GL.GL_FALSE, mvp.T)
            locc = GL.glGetUniformLocation(self.axis_program, 'uColor')
            GL.glUniform3f(locc, 1.0, 0.85, 0.0)  
            
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.light_vbo)
            data = np.array(self.light_world, dtype=np.float32)
            GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, data.nbytes, data) 
            
            if self.use_vao and self.light_vao is not None:
                GL.glBindVertexArray(self.light_vao)
                GL.glPointSize(8.0)  
                GL.glDrawArrays(GL.GL_POINTS, 0, 1)
                GL.glBindVertexArray(0)
            else:
                GL.glEnableVertexAttribArray(0)
                GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, GL.ctypes.c_void_p(0))
                GL.glPointSize(8.0)
                GL.glDrawArrays(GL.GL_POINTS, 0, 1)
                GL.glDisableVertexAttribArray(0)
                
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
            GL.glUseProgram(0)

    def _sph_to_cart(self, az, el, r):
        x = r * math.cos(el) * math.cos(az)
        y = r * math.sin(el)
        z = r * math.cos(el) * math.sin(az)
        return np.array([x, y, z], dtype=np.float32)

    def mousePressEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            self.mouse_mode = 'orbit'
            self._orbit_start = (self.cam_az, self.cam_el, self.cam_radius)  
            self._orbit_mouse = ev.pos()
        self.last_mouse = ev.pos()

    def mouseMoveEvent(self, ev):
        if self.last_mouse is None:
            return
            
        if self.mouse_mode == 'orbit':
            az0, el0, r0 = self._orbit_start
            ddx = ev.x() - self._orbit_mouse.x()
            ddy = ev.y() - self._orbit_mouse.y()
            
            az = az0 + ddx * 0.01  
            el = el0 + ddy * 0.01
            
            max_el = math.radians(89.0)
            el = max(-max_el, min(max_el, el))
            
            self.cam_az = az
            self.cam_el = el
            self.update()
            
        self.last_mouse = ev.pos()

    def mouseReleaseEvent(self, ev):
        self.mouse_mode = None
        self.last_mouse = None

# Функция запускает анимацию движения камеры
    def start_camera_move_via_light(self, duration_ms=1500):

        # Сохраняем текущие координаты камеры (азимут, угол возвышения, радиус)
        # в отдельные переменные, так как они нам понадобятся для интерполяции
        self.cam_move_from_az = self.cam_az
        self.cam_move_from_el = self.cam_el
        self.cam_move_from_radius = self.cam_radius

        # Вычисляем вектор из центра сцены (0,0,0) к источнику света
        light_vec = self.light_world - self.camera_target
        # Находим длину этого вектора
        light_r = np.linalg.norm(light_vec)
        # Вычисляем угол возвышения света (Elevation).
        # asin(y / r) находит угол вверх/вниз
        light_el = math.asin(light_vec[1] / light_r) if light_r > 1e-6 else 0.0
        # Вычисляем азимут света (Azimuth - угол вокруг оси Y).
        # atan2(z, x) находит угол в горизонтальной плоскости.
        light_az = math.atan2(light_vec[2], light_vec[0])

        # Чтобы оказаться напротив света, мы прибавляем 180 градусов (Пи) к азимуту
        opposite_az = light_az + math.pi
        # По углу возвышения просто зеркалим (переворачиваем) его.
        # Если свет светит сверху (+Y), то камера должна смотреть снизу (-Y) 
        opposite_el = -light_el 
        
        self.cam_move_to_az = opposite_az
        self.cam_move_to_el = opposite_el
        self.cam_move_to_radius = light_r  # Оставляем то же расстояние до центра, что и у света

        # Запоминаем время старта в миллисекундах
        self.cam_move_start = time.time() * 1000.0
        # Устанавливаем длительность (защита от 0, чтобы не было деления на 0)
        self.cam_move_dur = max(50, int(duration_ms))
        self.cam_move_active = True

    def on_timer(self):
        now = time.time() * 1000.0
        
        if getattr(self, 'anim_running', False):
            self.cam_az += 0.01
            self.update()
            
        if getattr(self, 'cam_move_active', False):
            elapsed = now - self.cam_move_start
            t = min(1.0, elapsed / float(self.cam_move_dur))
            
            smooth_t = self._ease_in_out(t)
            
            self.cam_az = self._lerp_angle(self.cam_move_from_az, self.cam_move_to_az, smooth_t)
            self.cam_el = self._lerp(self.cam_move_from_el, self.cam_move_to_el, smooth_t)
            self.cam_radius = self._lerp(self.cam_move_from_radius, self.cam_move_to_radius, smooth_t)
            
            if t >= 1.0:
                self.cam_move_active = False 
                
            self.update()

    def _ease_in_out(self, t):
        return 0.5 * (1 - math.cos(t * math.pi))

    def _lerp(self, a, b, t):
        return a + (b - a) * t

    def _lerp_angle(self, a, b, t):
        a = self._normalize_angle(a)
        b = self._normalize_angle(b)
        
        diff = b - a
        if diff > math.pi:
            diff -= 2 * math.pi
        elif diff < -math.pi:
            diff += 2 * math.pi
            
        return a + diff * t

    def _normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle


# Главное окно
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Лабораторная работа 3')
        self.gl = GLWidget()
        self.setCentralWidget(self.gl)
        
        self.create_controls()
        self.resize(1300, 820)
        
        # Таймер для обновления UI
        self.ui_timer = QtCore.QTimer()
        self.ui_timer.timeout.connect(self.update_ui)
        self.ui_timer.start(100)

    def create_controls(self):
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.addWidget(QtWidgets.QLabel('<b>Управление сценой</b>'))

        # Выбор формы
        shape_h = QtWidgets.QHBoxLayout()
        btn_sphere = QtWidgets.QPushButton('Сфера')
        btn_sphere.clicked.connect(lambda: self.set_shape('sphere'))
        btn_ell = QtWidgets.QPushButton('Эллипсоид')
        btn_ell.clicked.connect(lambda: self.set_shape('ellipsoid'))
        btn_parab = QtWidgets.QPushButton('Параболоид')
        btn_parab.clicked.connect(lambda: self.set_shape('paraboloid'))
        
        shape_h.addWidget(btn_sphere)
        shape_h.addWidget(btn_ell)
        shape_h.addWidget(btn_parab)
        layout.addLayout(shape_h)

        # Настройки сетки
        layout.addWidget(QtWidgets.QLabel('Настройки сетки'))
        grid_h = QtWidgets.QHBoxLayout()
        self.lat_spin = QtWidgets.QSpinBox()
        self.lat_spin.setRange(3, 160)
        self.lat_spin.setValue(self.gl.lat_segments)
        self.lon_spin = QtWidgets.QSpinBox()
        self.lon_spin.setRange(3, 320)
        self.lon_spin.setValue(self.gl.lon_segments)
        
        self.lat_spin.valueChanged.connect(self.on_mesh_changed)
        self.lon_spin.valueChanged.connect(self.on_mesh_changed)
        
        grid_h.addWidget(self.lat_spin)
        grid_h.addWidget(self.lon_spin)
        layout.addLayout(grid_h)

        # Переключатели отображения
        self.tri_check = QtWidgets.QCheckBox('Показывать триангуляцию')
        self.tri_check.setChecked(self.gl.show_wireframe)
        self.tri_check.stateChanged.connect(lambda s: self.on_wire_toggle(s))
        
        self.norm_check = QtWidgets.QCheckBox('Показывать векторы нормалей (по вершинам)')
        self.norm_check.setChecked(self.gl.show_normals)
        self.norm_check.stateChanged.connect(lambda s: self.on_normals_toggle(s))
        
        self.axes_check = QtWidgets.QCheckBox('Показывать оси')
        self.axes_check.setChecked(self.gl.show_axes)
        self.axes_check.stateChanged.connect(lambda s: self.on_axes_toggle(s))
        
        layout.addWidget(self.tri_check)
        layout.addWidget(self.norm_check)
        layout.addWidget(self.axes_check)

        # ВЫБОР ЦВЕТА
        layout.addWidget(QtWidgets.QLabel('<b>Цвет фигуры</b>'))
        color_h = QtWidgets.QHBoxLayout()
        
        btn_custom = QtWidgets.QPushButton('Выбрать цвет...')
        btn_custom.clicked.connect(self.choose_color_dialog)
        
        btn_reset = QtWidgets.QPushButton('Сброс (Серый)')
        btn_reset.clicked.connect(self.reset_color)
        
        color_h.addWidget(btn_custom)
        color_h.addWidget(btn_reset)
        layout.addLayout(color_h)

        # Освещение
        layout.addWidget(QtWidgets.QLabel('<b>Свет</b>'))
        self.light_check = QtWidgets.QCheckBox('Включить свет')
        self.light_check.setChecked(True)
        self.light_check.stateChanged.connect(self.on_light_toggle)
        layout.addWidget(self.light_check)
        
        # Координаты света
        coords_h = QtWidgets.QHBoxLayout()
        self.lx = QtWidgets.QDoubleSpinBox()
        self.lx.setRange(-20.0, 20.0)
        self.lx.setSingleStep(0.1)
        self.lx.setValue(float(self.gl.light_world[0]))
        self.lx.valueChanged.connect(self.on_light_changed)
        
        self.ly = QtWidgets.QDoubleSpinBox()
        self.ly.setRange(-20.0, 20.0)
        self.ly.setSingleStep(0.1)
        self.ly.setValue(float(self.gl.light_world[1]))
        self.ly.valueChanged.connect(self.on_light_changed)
        
        self.lz = QtWidgets.QDoubleSpinBox()
        self.lz.setRange(-20.0, 20.0)
        self.lz.setSingleStep(0.1)
        self.lz.setValue(float(self.gl.light_world[2]))
        self.lz.valueChanged.connect(self.on_light_changed)
        
        coords_h.addWidget(QtWidgets.QLabel('x'))
        coords_h.addWidget(self.lx)
        coords_h.addWidget(QtWidgets.QLabel('y'))
        coords_h.addWidget(self.ly)
        coords_h.addWidget(QtWidgets.QLabel('z'))
        coords_h.addWidget(self.lz)
        layout.addLayout(coords_h)

        # Параметры освещения
        params_h = QtWidgets.QGridLayout()
        params_h.addWidget(QtWidgets.QLabel('Ka (коэффициент фонового света)'), 0, 0)
        self.ka_spin = QtWidgets.QDoubleSpinBox()
        self.ka_spin.setRange(0.0, 2.0)
        self.ka_spin.setSingleStep(0.05)
        self.ka_spin.setValue(self.gl.Ka)
        self.ka_spin.valueChanged.connect(self.on_light_params_changed)
        params_h.addWidget(self.ka_spin, 0, 1)
        
        params_h.addWidget(QtWidgets.QLabel('Kd (коэффициент диффузного отражения)'), 1, 0)
        self.kd_spin = QtWidgets.QDoubleSpinBox()
        self.kd_spin.setRange(0.0, 2.0)
        self.kd_spin.setSingleStep(0.05)
        self.kd_spin.setValue(self.gl.Kd)
        self.kd_spin.valueChanged.connect(self.on_light_params_changed)
        params_h.addWidget(self.kd_spin, 1, 1)
        
        params_h.addWidget(QtWidgets.QLabel('Ia (интенсивность фонового света)'), 2, 0)
        self.ia_spin = QtWidgets.QDoubleSpinBox()
        self.ia_spin.setRange(0.0, 2.0)
        self.ia_spin.setSingleStep(0.05)
        self.ia_spin.setValue(self.gl.Ia)
        self.ia_spin.valueChanged.connect(self.on_light_params_changed)
        params_h.addWidget(self.ia_spin, 2, 1)
        
        params_h.addWidget(QtWidgets.QLabel('Il (интенсивность источника света)'), 3, 0)
        self.il_spin = QtWidgets.QDoubleSpinBox()
        self.il_spin.setRange(0.0, 5.0)
        self.il_spin.setSingleStep(0.05)
        self.il_spin.setValue(self.gl.Il)
        self.il_spin.valueChanged.connect(self.on_light_params_changed)
        params_h.addWidget(self.il_spin, 3, 1)
        
        layout.addLayout(params_h)

        # Кнопка движения камеры
        move_h = QtWidgets.QHBoxLayout()
        self.btn_move = QtWidgets.QPushButton('Движение: исходная точка -> противоположная свету')
        self.btn_move.clicked.connect(self.on_start_move)
        move_h.addWidget(self.btn_move)
        layout.addLayout(move_h)

        # Скорость анимации
        sp_h = QtWidgets.QHBoxLayout()
        sp_h.addWidget(QtWidgets.QLabel('Продолжительность движения (мс)'))
        self.speed_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.speed_slider.setRange(200, 5000)
        self.speed_slider.setValue(1500)
        self.speed_label = QtWidgets.QLabel('1500')
        self.speed_slider.valueChanged.connect(lambda v: self.speed_label.setText(str(v)))
        
        sp_h.addWidget(self.speed_slider)
        sp_h.addWidget(self.speed_label)
        layout.addLayout(sp_h)

        # Инструкция
        instr = QtWidgets.QLabel('Вращение вокруг модели: ЛКМ + перетаскивание')
        instr.setWordWrap(True)
        layout.addWidget(instr)

        # Отображение координат
        layout.addWidget(QtWidgets.QLabel('<b>Камера</b>'))
        self.camera_label = QtWidgets.QLabel('x: 0.00, y: 0.00, z: 0.00')
        layout.addWidget(self.camera_label)
        
        layout.addStretch()
        panel.setLayout(layout)
        
        # Создаем виджет для панели управления
        dock = QtWidgets.QDockWidget('Панель управления', self)
        dock.setWidget(panel)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

    # Обработчики UI
    def set_shape(self, s):
        self.gl.makeCurrent()
        self.gl.shape = s
        self.gl.create_mesh()
        self.gl.doneCurrent()
        self.gl.update()

    # Обработчик изменения параметров сетки
    def on_mesh_changed(self, val=None):
        try:
            self.gl.makeCurrent()
            self.gl.lat_segments = int(self.lat_spin.value())
            self.gl.lon_segments = int(self.lon_spin.value())
            self.gl.create_mesh() 
            self.gl.doneCurrent()
            self.gl.update()
        except Exception:
            pass

    # Открытие диалога выбора цвета
    def choose_color_dialog(self):
        current_qcolor = QtCore.Qt.white 
        color = QColorDialog.getColor(current_qcolor, self, "Выберите цвет фигуры")
        
        if color.isValid():
            self.gl.object_color = np.array([color.redF(), color.greenF(), color.blueF()], dtype=np.float32)
            self.gl.update()

    # Сброс цвета на серый (белый)
    def reset_color(self):
        self.gl.object_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.gl.update()

    # Переключатель триангуляции        
    def on_wire_toggle(self, state):
        self.gl.show_wireframe = (state == QtCore.Qt.Checked)
        self.gl.update()

    # Переключатель отображения нормалей
    def on_normals_toggle(self, state):
        self.gl.show_normals = (state == QtCore.Qt.Checked)
        self.gl.update()

    # Переключатель отображения осей
    def on_axes_toggle(self, state):
        self.gl.show_axes = (state == QtCore.Qt.Checked)
        self.gl.update()

    # Переключатель освещения
    def on_light_toggle(self, state):
        self.gl.light_on = (state == QtCore.Qt.Checked)
        self.gl.update()

    # Обработчик изменения позиции света
    def on_light_changed(self, val=None):
        try:
            self.gl.makeCurrent()
            self.gl.light_world = np.array([float(self.lx.value()), float(self.ly.value()), float(self.lz.value())], dtype=np.float32)
            self.gl.create_light_marker() 
            self.gl.doneCurrent()
            self.gl.update()
        except Exception:
            pass

    # Обработчик изменения параметров освещения
    def on_light_params_changed(self, val=None):
        self.gl.Ka = float(self.ka_spin.value())
        self.gl.Kd = float(self.kd_spin.value())
        self.gl.Ia = float(self.ia_spin.value())
        self.gl.Il = float(self.il_spin.value())
        self.gl.update()

    # Запуск анимации движения камеры
    def on_start_move(self):
        dur = int(self.speed_slider.value())
        self.gl.start_camera_move_via_light(duration_ms=dur)
        self.gl.update()

    # Обновление UI
    def update_ui(self):
        # Обновляем координаты камеры
        eye = self.gl._sph_to_cart(self.gl.cam_az, self.gl.cam_el, self.gl.cam_radius) + self.gl.camera_target
        self.camera_label.setText(f'x: {eye[0]:.2f}, y: {eye[1]:.2f}, z: {eye[2]:.2f}')
        
        # Синхронизируем состояния переключателей UI
        if self.tri_check.isChecked() != self.gl.show_wireframe:
            self.tri_check.setChecked(self.gl.show_wireframe)
        if self.norm_check.isChecked() != self.gl.show_normals:
            self.norm_check.setChecked(self.gl.show_normals)
        if self.axes_check.isChecked() != self.gl.show_axes:
            self.axes_check.setChecked(self.gl.show_axes)



app = QtWidgets.QApplication(sys.argv)
w = MainWindow()
w.show()
sys.exit(app.exec_())
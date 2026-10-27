from __future__ import annotations

import ctypes
import math
from typing import Iterable, Sequence

from OpenGL.GL import *


# V3 — трёхмерный вектор (кортеж из трёх float).
# M4 — матрица 4×4 в виде одномерного массива
V3 = tuple[float, float, float]
M4 = Sequence[float]


# Создаёт трёхмерный вектор
def v3(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> V3:
    return (float(x), float(y), float(z))


# Складывает два вектора
def vadd(a: V3, b: V3) -> V3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


# Вычитает один вектор из другого.
def vsub(a: V3, b: V3) -> V3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


# Умножает вектор на число
def vmul(a: V3, s: float) -> V3:
    return (a[0] * s, a[1] * s, a[2] * s)


# Вычисляет скалярное произведение
def vdot(a: V3, b: V3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


# Вычисляет векторное произведение
def vcross(a: V3, b: V3) -> V3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


# Вычисляет длину вектора
def vlength(a: V3) -> float:
    return math.sqrt(vdot(a, a))


# Возвращает единичный вектор
def vnorm(a: V3) -> V3:
    l = vlength(a)
    # Нулевой вектор нельзя нормировать, поэтому возвращаем его без изменений.
    if l < 1e-12:
        return (0.0, 0.0, 0.0)
    return (a[0] / l, a[1] / l, a[2] / l)


# Создаёт единичную матрицу.
def mat4_identity() -> list[float]:
    return [1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0]


# Перемножает две матрицы.
def mat4_mul(a: M4, b: M4) -> list[float]:
    out = [0.0] * 16
    # Матрица 4×4 хранится как одномерный массив из 16 float в порядке
    # столбцов (column-major): сначала весь столбец 0, затем столбец 1 и т.д.
    # Значит, элемент в строке row и столбце col — это arr[col*4 + row]:
    # col*4 — смещение до начала столбца, + row — позиция внутри него
    for col in range(4):
        for row in range(4):
            out[col * 4 + row] = sum(a[k * 4 + row] * b[col * 4 + k] for k in range(4))
    return out


# Создаёт матрицу переноса
def mat4_translate(x: float, y: float, z: float) -> list[float]:
    m = mat4_identity()
    m[12], m[13], m[14] = x, y, z
    return m


# Создаёт матрицу масштаба
def mat4_scale(x: float, y: float, z: float) -> list[float]:
    m = mat4_identity()
    m[0], m[5], m[10] = x, y, z
    return m


# Создаёт поворот вокруг оси X
def mat4_rotate_x(a: float) -> list[float]:
    c, s = math.cos(a), math.sin(a)
    return [1.0, 0.0, 0.0, 0.0,
            0.0, c, s, 0.0,
            0.0, -s, c, 0.0,
            0.0, 0.0, 0.0, 1.0]


# Создаёт поворот вокруг оси Y
def mat4_rotate_y(a: float) -> list[float]:
    c, s = math.cos(a), math.sin(a)
    return [c, 0.0, -s, 0.0,
            0.0, 1.0, 0.0, 0.0,
            s, 0.0, c, 0.0,
            0.0, 0.0, 0.0, 1.0]


# Создаёт поворот вокруг оси Z
def mat4_rotate_z(a: float) -> list[float]:
    c, s = math.cos(a), math.sin(a)
    return [c, s, 0.0, 0.0,
            -s, c, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0]


# Строит видовую матрицу по точке взгляда
def mat4_look_at(eye: V3, target: V3, up: V3 = (0.0, 1.0, 0.0)) -> list[float]:
    # Направление взгляда и правый вектор образуют базис камеры.
    f = vnorm(vsub(target, eye))
    s = vnorm(vcross(f, up))
    u = vcross(s, f)
    return [s[0], u[0], -f[0], 0.0,
            s[1], u[1], -f[1], 0.0,
            s[2], u[2], -f[2], 0.0,
            -vdot(s, eye), -vdot(u, eye), vdot(f, eye), 1.0]


# Строит перспективную матрицу
def mat4_perspective(fovy_deg: float, aspect: float, near: float, far: float) -> list[float]:
    # f — масштаб по вертикали, nf — вспомогательная величина для глубин
    f = 1.0 / math.tan(math.radians(fovy_deg) * 0.5)
    nf = 1.0 / (near - far)
    return [f / aspect, 0.0, 0.0, 0.0,
            0.0, f, 0.0, 0.0,
            0.0, 0.0, (far + near) * nf, -1.0,
            0.0, 0.0, 2.0 * far * near * nf, 0.0]


# Строит ортографическую матрицу, используется для теней от солнца
# Параметры задают границы параллелепипеда видимости в видовой системе
# координат источника света (камера в начале координат, направление
# взгляда совпадает с отрицательным направлением оси Z):
# left, right — границы объёма по оси X (левая и правая плоскости отсечения);
# bottom, top — границы объёма по оси Y (нижняя и верхняя плоскости отсечения);
# near, far — границы объёма по оси Z (ближняя и дальняя плоскости отсечения)
def mat4_ortho(left: float, right: float, bottom: float, top: float,
               near: float, far: float) -> list[float]:
    # Переводим размеры объёма в единичные коэффициенты масштаба
    rl = 1.0 / (right - left)
    tb = 1.0 / (top - bottom)
    fn = 1.0 / (far - near)
    return [2.0 * rl, 0.0, 0.0, 0.0,
            0.0, 2.0 * tb, 0.0, 0.0,
            0.0, 0.0, -2.0 * fn, 0.0,
            -(right + left) * rl, -(top + bottom) * tb, -(far + near) * fn, 1.0]


# Применяет матрицу 4×4 к точке: p` = M * (x, y, z, 1)
def xform_point(m: M4, p: V3) -> V3:
    # Для точки учитываем и поворот, и перенос из последнего столбца.
    x, y, z = p
    return (m[0] * x + m[4] * y + m[8] * z + m[12],
            m[1] * x + m[5] * y + m[9] * z + m[13],
            m[2] * x + m[6] * y + m[10] * z + m[14])


# Применяет матрицу 4×4 к направлению (вектору)
def xform_dir(m: M4, d: V3) -> V3:
    # Для направления перенос не применяется, иначе вектор сместился бы.
    x, y, z = d
    return (m[0] * x + m[4] * y + m[8] * z,
            m[1] * x + m[5] * y + m[9] * z,
            m[2] * x + m[6] * y + m[10] * z)


# Сетка на GPU: набор треугольников, загруженный в видеопамять
# Формат одной вершины:
# [0..2] позиция (x, y, z)
# [3..5] нормаль (nx, ny, nz)
# [6..7] текстурные координаты (u, v)
# Индексы идут по три на каждый треугольник
class Mesh:
    # Создаёт буферы в памяти GPU и настраивает, как читать из них
    # вершины. После __init__ данные лежат в видеопамяти
    def __init__(self, vertices: Sequence[float], indices: Sequence[int]):
        # Сохраняем исходные массивы и создаём объекты VAO, VBO и EBO.
        self.vertices = list(vertices)
        self.indices = list(indices)
        # Создаём три объекта OpenGL. glGen* только резервирует идентификаторы
        self.vao = glGenVertexArrays(1) # vao хранит привязки VBO/EBO и настройки атрибутов вершин
        self.vbo = glGenBuffers(1) # vbo — массив вершин (Vertex Buffer Object)
        self.ebo = glGenBuffers(1) # ebo — массив индексов (Element Buffer Object)
        self.count = len(indices)

        # Привязываем VAO: все последующие настройки (привязки VBO/EBO
        # и glVertexAttribPointer) запишутся именно в него. При
        # следующем glBindVertexArray(self.vao) они восстановятся
        # одним вызовом
        glBindVertexArray(self.vao)
        # Делаем буфер активным
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        # Переводим список float в одномерный массив C 
        arr = (ctypes.c_float * len(vertices))(*vertices)
        # Загружаем его в GPU 
        # GL_ARRAY_BUFFER — целевой буфер (VBO),
        # len(vertices) * 4 — размер в байтах массива вершин,
        # arr — откуда копировать,
        # GL_STATIC_DRAW — данные не меняются, читать часто, писать редко (оптимизация драйвера)
        glBufferData(GL_ARRAY_BUFFER, len(vertices) * 4, arr, GL_STATIC_DRAW)
        # Делаем буфер EBO активным
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        # Индексы — беззнаковые 32-битные (c_uint)
        iarr = (ctypes.c_uint * len(indices))(*indices)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(indices) * 4, iarr, GL_STATIC_DRAW)

        # Описание атрибутов вершины 
        # Одна вершина занимает 8 float = 32 байта; stride — расстояние
        # от начала одной вершины до начала следующей
        stride = 8 * 4
        # Атрибут 0 - позиция вершины
        glEnableVertexAttribArray(0)
        # Аргументы glVertexAttribPointer:
        # 0 — номер атрибута,
        # 3 — сколько компонент (x, y, z),
        # GL_FLOAT — тип компоненты,
        # GL_FALSE — не нормализовать значения,
        # stride — шаг между вершинами в байтах,
        # 0 — смещение от начала вершины
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        # Атрибут 1 — нормаль
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        # Атрибут 2 — UV
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(24))
        # Отвязываем VAO: дальше по коду состояние привязок не должно
        # случайно изменить настройки этой сетки
        glBindVertexArray(0)

    # Рисует сетку
    def draw(self) -> None:
        # Привязываем VAO — вместе с ним восстанавливаются привязки
        # VBO/EBO и все glVertexAttribPointer, сделанные в __init__
        glBindVertexArray(self.vao)
        # glDrawElements читает self.count индексов из EBO, по ним
        # выбирает вершины из VBO и рисует треугольники:
        # GL_TRIANGLES — режим: каждые три индекса = один треугольник;
        # self.count — сколько индексов прочитать;
        # GL_UNSIGNED_INT — тип индекса;
        # None — смещение внутри EBO: начинаем с начала буфера
        glDrawElements(GL_TRIANGLES, self.count, GL_UNSIGNED_INT, None)
        # Отвязываем VAO, чтобы следующая команда не унаследовала
        # чужое состояние
        glBindVertexArray(0)


# Текстура — это изображение (набор пикселей RGBA), которое накладывается
# на поверхность треугольника. В этой сцене текстуры создаются процедурно
# и один раз загружаются в память GPU. Потом шейдер по UV-координатам
# каждой вершины решает, какой пиксель текстуры взять для каждого фрагмента
class Texture:
    # Инициализирует объект
    # width, height — размеры изображения в пикселях
    # rgba — сами пиксели подряд, по 4 байта на пиксель
    # repeat — True: текстура повторяется
    # mipmap — True: вместе с полным изображением строятся его
    # уменьшенные копии (мип-уровни) — для далёких объектов
    def __init__(self, width: int, height: int, rgba: bytes | bytearray,
                 repeat: bool = True, mipmap: bool = True):
        self.width = width
        self.height = height
        # glGenTextures создаёт идентификатор текстуры — число, по которому
        # OpenGL будет узнавать эту текстуру
        self.id = glGenTextures(1)
        # Делаем текстуру активной
        glBindTexture(GL_TEXTURE_2D, self.id)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        # Загружаем пиксели в GPU.
        # GL_TEXTURE_2D — тип текстуры (двумерная),
        # 0 — уровень мипмапа: 0 = полное изображение,
        # GL_RGBA8 — внутренний формат в GPU: 8 бит на канал (R, G, B, A), итого 32 бита на пиксель,
        # width, height — размеры в пикселях,
        # 0 — толщина границы (не используется),
        # GL_RGBA — формат входных данных: RGBA подряд,
        # GL_UNSIGNED_BYTE — тип компоненты: 1 байт на канал,
        # bytes(rgba) — сами байты
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, bytes(rgba))
        # Для обоев и металла текстура повторяется, для надписей — нет.
        wrap = GL_REPEAT if repeat else GL_CLAMP_TO_EDGE
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap)
        # Фильтр при увеличении (когда текстура на экране крупнее, чем
        # её пиксели). GL_LINEAR — линейная интерполяция между соседними
        # пикселями
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        # Фильтр при уменьшении (когда текстура на экране мельче, чем
        # её пиксели — например, стена вагона вдали). Тут нужен мипмап:
        # без него вдали появляется мерцание
        if mipmap:
            # GL_LINEAR_MIPMAP_LINEAR — выбирается подходящий мип-уровень и между его пикселями
            # делается линейная интерполяция
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
             # Строим цепочку мип-уровней. Драйвер сам решает, сколько уровней 
            glGenerateMipmap(GL_TEXTURE_2D)
        else:
            # Минимальный фильтр (при уменьшении объекта на экране) — GL_LINEAR (мип-уровни). 
            # не построены. Просто выполняется линейная интерполяция
            # между соседними пикселями полного изображения
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        # Отвязываем текстуру, чтобы случайный код дальше не изменил её
        # параметры. Сама текстура и её данные остаются в памяти GPU
        glBindTexture(GL_TEXTURE_2D, 0)

    # Делает текстуру доступной шейдеру в указанном слоте
    def bind(self, unit: int = 0) -> None:
        # Слот 0 обычно занят текстурой материала, слот 1 — картой теней.
        glActiveTexture(GL_TEXTURE0 + unit)
        glBindTexture(GL_TEXTURE_2D, self.id)


# Программа GLSL.
class Shader:
    # Компилирует оба шейдера, линкует их в программу
    # vertex_src — исходник вершинного шейдера (строка GLSL);
    # fragment_src — исходник фрагментного шейдера;
    # name — имя программы для сообщений об ошибках (в GLSL и OpenGL его нет — это чисто для отладки)
    def __init__(self, vertex_src: str, fragment_src: str, name: str = "shader"):
        # Создаём программу из вершинного и фрагментного шейдеров
        self.name = name
        self.program = glCreateProgram()
        # Компилируем каждый шейдер отдельно. Возвращается идентификатор
        # скомпилированного шейдера, ещё не привязанного к программе
        vs = self._compile(GL_VERTEX_SHADER, vertex_src)
        fs = self._compile(GL_FRAGMENT_SHADER, fragment_src)
        # Прикрепляем оба шейдера к программе
        glAttachShader(self.program, vs)
        glAttachShader(self.program, fs)
        # Линковка: OpenGL проверяет, что выходы вершинного шейдера
        # (out-переменные) совпадают по имени и типу со входами
        # фрагментного (in-переменные), что используются только
        # допустимые комбинации и т.д.
        glLinkProgram(self.program)
        # Если линковка не удалась, выводим журнал драйвера
        if not glGetProgramiv(self.program, GL_LINK_STATUS):
            raise RuntimeError(f"{name} link error:\n{glGetProgramInfoLog(self.program)}")
        # После успешной линковки отдельные шейдеры больше не нужны:
        # вся информация уже внутри программы. Удаляем, чтобы освободить
        # память GPU
        glDeleteShader(vs)
        glDeleteShader(fs)
        # Кеш «имя uniform-переменной -> её номер в программе»
        self._uniforms: dict[str, int] = {}

    # Компилирует один шейдер (вершинный или фрагментный) и возвращает
    # его идентификатор
    @staticmethod
    # kind — GL_VERTEX_SHADER или GL_FRAGMENT_SHADER
    def _compile(kind: int, source: str) -> int:
        # Компилируем один шейдер и проверяем статус компиляции, source - текст шейдера
        shader = glCreateShader(kind)
        # Передаём исходник
        glShaderSource(shader, source)
        glCompileShader(shader)
        # Если компиляция провалилась — журнал содержит номер строки
        # и текст ошибки
        if not glGetShaderiv(shader, GL_COMPILE_STATUS):
            raise RuntimeError(glGetShaderInfoLog(shader).decode("utf-8", "replace"))
        return shader

    # Делает эту программу активной
    def use(self) -> None:
        glUseProgram(self.program)

    # Возвращает расположение uniform-переменной.
    def loc(self, name: str) -> int:
        # Расположение uniform-переменной запрашивается один раз и кешируется.
        if name not in self._uniforms:
            self._uniforms[name] = glGetUniformLocation(self.program, name)
        return self._uniforms[name]

    # Передаёт целое значение в шейдер
    def set_i(self, name: str, value: int) -> None:
        glUniform1i(self.loc(name), value)

    # Передаёт число с плавающей точкой
    def set_f(self, name: str, value: float) -> None:
        glUniform1f(self.loc(name), value)

    # Передаёт трёхмерный вектор
    def set_3f(self, name: str, value: Sequence[float]) -> None:
        glUniform3f(self.loc(name), float(value[0]), float(value[1]), float(value[2]))

    # Передаёт четырёхмерный вектор
    def set_4f(self, name: str, value: Sequence[float]) -> None:
        glUniform4f(self.loc(name), float(value[0]), float(value[1]),
                    float(value[2]), float(value[3]))

    # Передаёт матрицу
    def set_mat4(self, name: str, value: M4) -> None:
        # Передаём матрицу без транспонирования, так как она уже хранится по столбцам.
        arr = (ctypes.c_float * 16)(*[float(v) for v in value])
        glUniformMatrix4fv(self.loc(name), 1, GL_FALSE, arr)


# Передаёт массив векторов
def set_uniform_array3(shader: Shader, name: str, values: Iterable[Sequence[float]]) -> None:
    # Разворачиваем список векторов в один плоский массив для glUniform3fv.
    arr = [float(c) for v in values for c in v]
    if arr:
        glUniform3fv(shader.loc(name), len(arr) // 3,
                     (ctypes.c_float * len(arr))(*arr))


# Передаёт массив чисел
def set_uniform_array1(shader: Shader, name: str, values: Iterable[float]) -> None:
    # Передаём массив радиусов точечных источников как один вызов.
    arr = [float(v) for v in values]
    if arr:
        glUniform1fv(shader.loc(name), len(arr), (ctypes.c_float * len(arr))(*arr))

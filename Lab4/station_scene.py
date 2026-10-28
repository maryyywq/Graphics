from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass

from OpenGL.GL import *

import procedural as proc
from gfx import (
    M4, Mesh, Shader, Texture, V3, mat4_look_at, mat4_mul, mat4_ortho,
    mat4_perspective, set_uniform_array1, set_uniform_array3, v3, vadd,
    vcross, vdot, vlength, vnorm, vsub,
)

# Вся сцена увеличена в 1.6 раза, включая двери и коридор
WORLD_SCALE = 1.6

# main.py включает этот режим, чтобы не засорять терминал
QUIET = False


# Материал — набор параметров, которые передаются в шейдер.
# Параметры:
# diffuse — основной цвет (как материал выглядит при полном свете);
# specular — цвет блика (для металлов выше, для дерева/ткани ниже);
# shininess — «острота» блика: больше — меньше пятно, ярче;
# texture — текстура (или None, если материал однотонный);
# alpha — общая прозрачность (1.0 — непрозрачный);
# alpha_test — если True, шейдер отбрасывает фрагменты с alpha < 0.28
# (для надписей: прозрачный фон не рисуется);
# unlit — если True, освещение не применяется (лампы, флаг, вывеска);
# emissive — собственное свечение (добавляется к итоговому цвету).
@dataclass
class Material:
    name: str
    diffuse: tuple[float, float, float] = (0.8, 0.8, 0.8)
    specular: tuple[float, float, float] = (0.12, 0.12, 0.12)
    shininess: float = 24.0
    texture: Texture | None = None
    alpha: float = 1.0
    alpha_test: bool = False
    unlit: bool = False
    emissive: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def transparent(self) -> bool:
        # Материал считается прозрачным при частичной прозрачности
        return self.alpha < 1.0 or self.alpha_test

# Batch — «накопитель» треугольников для одного материала и одной
# группы видимости. Все объекты одного материала (например, все
# синие части вагона) собираются в один Batch, а в finish()
# превращаются в один Mesh.

# Группа видимости (group) — это "near", "default", "building":
# нужна, чтобы в режиме разреза (cutaway) скрыть ближнюю стену
# вагона и здание целиком, не разбирая их по треугольникам
class Batch:

    def __init__(self, material: Material, group: str = "default"):
        # Для одного материала и группы храним один общий список вершин.
        self.material = material
        self.group = group
        # vertices — список float: по 8 чисел на вершину (x,y,z, nx,ny,nz, u,v)
        self.vertices: list[float] = []
        # indices — список int: индексы вершин, по 3 на треугольник
        self.indices: list[int] = []
        # center — будущий центр Batch (среднее арифметическое всех
        # позиций вершин). Нужен для сортировки прозрачных пакетов по
        # расстоянию до камеры (в Renderer.render)
        self.center = [0.0, 0.0, 0.0]
        # vertex_count — сколько вершин уже добавлено. Используется как
        # «следующий свободный индекс» и как делитель для среднего
        self.vertex_count = 0
        self.mesh: Mesh | None = None

    @property
    def transparent(self) -> bool:
        return self.material.transparent

    # Добавляет одну вершину в пакет. Возвращает её индекс — чтобы
    # tri/quad могли собрать из индексов треугольник
    def _vert(self, p: V3, n: V3, uv: tuple[float, float]) -> int:
        # Записываем 8 float подряд: позиция (3), нормаль (3), UV (2) (какая точка текстуры соответствует каждой вершине)
        self.vertices.extend((p[0], p[1], p[2], n[0], n[1], n[2], uv[0], uv[1]))
        # idx — индекс только что добавленной вершины
        idx = self.vertex_count
        self.vertex_count += 1
        # Накапливаем сумму координат для будущего центра
        self.center[0] += p[0]
        self.center[1] += p[1]
        self.center[2] += p[2]
        return idx

    # Добавляет треугольник: три вершины + три индекса
    def tri(self, p0: V3, p1: V3, p2: V3,
            uv0=(0.0, 0.0), uv1=(1.0, 0.0), uv2=(1.0, 1.0),
            normal: V3 | None = None) -> None:
        # Если нормаль не задана — считаем её по правилу правой руки:
        # (p1-p0) * (p2-p0) перпендикулярно плоскости треугольника и
        # смотрит в сторону, определяемую порядком вершин. vnorm
        # нормирует до единичной длины — это важно для освещения
        n = normal if normal is not None else vnorm(vcross(vsub(p1, p0), vsub(p2, p0)))
        # Добавляем три вершины, получаем их индексы
        i0 = self._vert(p0, n, uv0) 
        i1 = self._vert(p1, n, uv1)
        i2 = self._vert(p2, n, uv2)
        # Записываем треугольник как три индекса подряд
        self.indices.extend((i0, i1, i2))

    # Добавляет четырёхугольник — как 2 треугольника: (0,1,2) и (0,2,3)
    def quad(self, p0: V3, p1: V3, p2: V3, p3: V3,
             uv0=(0.0, 0.0), uv1=(1.0, 0.0), uv2=(1.0, 1.0), uv3=(0.0, 1.0),
             normal: V3 | None = None) -> None:
        # Нормаль считается по правилу правой руки, а четырёхугольник делится на два треугольника
        n = normal if normal is not None else vnorm(vcross(vsub(p1, p0), vsub(p2, p0)))
        i0 = self._vert(p0, n, uv0)
        i1 = self._vert(p1, n, uv1)
        i2 = self._vert(p2, n, uv2)
        i3 = self._vert(p3, n, uv3)
        self.indices.extend((i0, i1, i2, i0, i2, i3))

    # Прямоугольный блок (параллелепипед): 8 углов, 6 граней, каждая
    # грань — quad (2 треугольника), итого 12 треугольников
    # center — центр блока, size — полные размеры по X, Y, Z.
    # uv_scale — во сколько раз растянуть текстуру по U и V
    def box(self, center: V3, size: V3, uv_scale=(1.0, 1.0)) -> None:
        # По центру и размеру вычисляем восемь углов параллелепипеда
        cx, cy, cz = center
        sx, sy, sz = size
        x0, x1 = cx - sx * 0.5, cx + sx * 0.5
        y0, y1 = cy - sy * 0.5, cy + sy * 0.5
        z0, z1 = cz - sz * 0.5, cz + sz * 0.5
        # u, v — множители для UV
        u, v = uv_scale
        # Грань +Z (передняя): порядок вершин против часовой, если
        # смотреть снаружи (со стороны +Z). Нормаль смотрит в +Z
        self.quad((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
                  (0, 0), (u, 0), (u, v), (0, v))
        # Грань -Z (задняя): порядок обратный, чтобы нормаль смотрела
        # в -Z (наружу от блока).
        self.quad((x1, y0, z0), (x0, y0, z0), (x0, y1, z0), (x1, y1, z0),
                  (0, 0), (u, 0), (u, v), (0, v))
        # Нормали для граней +X и -X.
        self.quad((x1, y0, z1), (x1, y0, z0), (x1, y1, z0), (x1, y1, z1),
                  (0, 0), (u, 0), (u, v), (0, v))
        self.quad((x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0),
                  (0, 0), (u, 0), (u, v), (0, v))
        # Нормали для граней +Y и -Y.
        self.quad((x0, y1, z1), (x1, y1, z1), (x1, y1, z0), (x0, y1, z0),
                  (0, 0), (u, 0), (u, v), (0, v))
        self.quad((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1),
                  (0, 0), (u, 0), (u, v), (0, v))

    # Вспомогательный метод для cylinder/basin. Переводит точку из
    # локальной системы цилиндра (где ось цилиндра — Y) в мировую
    # Если axis="y" — просто сдвиг на center. Если axis="x" — локальная
    # Y-ось совпадает с мировой X (цилиндр «лежит»), поэтому x и y
    # меняются местами. Аналогично для axis="z"
    def _local_to_world(self, axis: str, center: V3, x: float, y: float, z: float):
        # Локальная ось Y цилиндра может быть переставлена на X или Z.
        cx, cy, cz = center
        if axis == "y":
            return (cx + x, cy + y, cz + z)
        if axis == "x":
            return (cx + y, cy + x, cz + z)
        return (cx + x, cy + z, cz + y)

    # То же для направлений (нормалей): без сдвига на center, только
    # перестановка осей. Нормаль — это направление, её нельзя «сдвигать»
    def _local_dir(self, axis: str, x: float, y: float, z: float):
        if axis == "y":
            return (x, y, z)
        if axis == "x":
            return (y, x, z)
        return (x, z, y)

    # Цилиндр: боковая поверхность + два основания.
    # Итого: segments*2 + segments*2 = 4*segments треугольников.
    # Используется для: колёс, фонарных столбов, стволов деревьев,
    # урны, проволоки, буферов
    # Параметры:
    # center — центр цилиндра;
    # radius — радиус;
    # height — полная высота (от нижнего основания до верхнего);
    # axis — вдоль какой оси лежит цилиндр: "y" (стоит вертикально),
    # "x" (лежит поперёк X), "z" (лежит поперёк Z);
    # segments — на сколько кусочков разбить окружность. Чем больше,
    # тем глаже цилиндр
    # caps — рисовать ли основания (верхнее и нижнее)
    def cylinder(self, center: V3, radius: float, height: float,
                 axis: str = "y", segments: int = 18, caps: bool = True) -> None:
        # Половина высоты нужна для верхнего и нижнего оснований
        h = height * 0.5
        # 1. Боковая поверхность
        for i in range(segments):
            # a0 и a1 — углы двух соседних точек на окружности
            # math.tau = 2pi, поэтому segments шагов покрывают полный круг
            a0 = math.tau * i / segments
            a1 = math.tau * (i + 1) / segments
            c0, s0 = math.cos(a0), math.sin(a0)
            c1, s1 = math.cos(a1), math.sin(a1)
            # Четыре точки образуют боковой четырёхугольник между двумя углами
            p0 = self._local_to_world(axis, center, c0 * radius, -h, s0 * radius)
            p1 = self._local_to_world(axis, center, c0 * radius, h, s0 * radius)
            p2 = self._local_to_world(axis, center, c1 * radius, h, s1 * radius)
            p3 = self._local_to_world(axis, center, c1 * radius, -h, s1 * radius)
            # Нормали боковой поверхности
            n0 = vnorm(self._local_dir(axis, c0, 0.0, s0))
            n1 = vnorm(self._local_dir(axis, c1, 0.0, s1))
            # UV
            i0 = self._vert(p0, n0, (i / segments, 0.0))
            i1 = self._vert(p1, n0, (i / segments, 1.0))
            i2 = self._vert(p2, n1, ((i + 1) / segments, 1.0))
            i3 = self._vert(p3, n1, ((i + 1) / segments, 0.0))
            # Полоска — четырёхугольник p0-p1-p2-p3, разбиваем на два треугольника
            self.indices.extend((i0, i1, i2, i0, i2, i3))
        # 2. Основания
        if caps:
            # Торцы собираются треугольниками от центра крышки к окружности.
            ntop = self._local_dir(axis, 0.0, 1.0, 0.0) # нормаль верхнего основания
            nbot = self._local_dir(axis, 0.0, -1.0, 0.0) # нормаль нижнего основания
            for i in range(segments):
                a0 = math.tau * i / segments
                a1 = math.tau * (i + 1) / segments
                # Точки на верхней окружности (y = +h) под углами a0 и a1
                q0 = self._local_to_world(axis, center, math.cos(a0) * radius, h, math.sin(a0) * radius)
                q1 = self._local_to_world(axis, center, math.cos(a1) * radius, h, math.sin(a1) * radius)
                # То же на нижней окружности (y = -h)
                b0 = self._local_to_world(axis, center, math.cos(a0) * radius, -h, math.sin(a0) * radius)
                b1 = self._local_to_world(axis, center, math.cos(a1) * radius, -h, math.sin(a1) * radius)
                # Центры оснований — ровно на оси цилиндра, вверху и внизу
                ctop = self._local_to_world(axis, center, 0, h, 0)
                cbot = self._local_to_world(axis, center, 0, -h, 0)
                # Треугольник верхнего основания
                self.tri(ctop, q1, q0, normal=ntop)
                # Треугольник нижнего основания
                self.tri(cbot, b0, b1, normal=nbot)

    # Добавляет чашу раковины
    # Параметры:
    # outer_r — внешний радиус (внешняя стенка);
    # inner_top — внутренний радиус у обода;
    # inner_bottom — внутренний радиус у дна;
    # depth — глубина чаши;
    # segments — на сколько секторов разбить окружность
    def basin(self, center: V3, outer_r: float, inner_top: float,
              inner_bottom: float, depth: float, segments: int = 24) -> None:
        
        # cx, cy, cz — три числа из center.
        # cx, cz — где раковина стоит на полу (по горизонтали);
        # cy — на какой высоте её верхний край
        cx, cy, cz = center
         # bottom — на какой высоте дно раковины
        bottom = cy - depth
        # 1. Внешняя стенка — то, что видно снаружи, если смотреть
        # на раковину со стороны. Это обычный цилиндр (труба) радиуса
        # outer_r и высотой depth, без верхней и нижней крышек
        # Центр цилиндра сдвинут на -depth/2 по Y, чтобы его верх
        # оказался на высоте cy (верхний край), а низ — на высоте
        # bottom (дно)
        self.cylinder((cx, cy - depth * 0.5, cz), outer_r, depth, "y", segments, caps=False)
        # 2. Верхний край (плоское кольцо сверху) и внутренняя
        # стенка
        for i in range(segments):
            # Углы двух соседних точек на окружности (a0 и a1)
            a0 = math.tau * i / segments
            a1 = math.tau * (i + 1) / segments
            # Координаты этих точек на окружности
            c0, s0 = math.cos(a0), math.sin(a0)
            c1, s1 = math.cos(a1), math.sin(a1)
            # Верхний край: плоское кольцо на высоте cy
            self.quad((cx + outer_r * c0, cy, cz + outer_r * s0),
                      (cx + inner_top * c0, cy, cz + inner_top * s0),
                      (cx + inner_top * c1, cy, cz + inner_top * s1),
                      (cx + outer_r * c1, cy, cz + outer_r * s1),
                      normal=(0.0, 1.0, 0.0))
            # Внутренняя стенка чаши с наклоном к центру
            self.quad((cx + inner_top * c0, cy, cz + inner_top * s0),
                      (cx + inner_bottom * c0, bottom, cz + inner_bottom * s0),
                      (cx + inner_bottom * c1, bottom, cz + inner_bottom * s1),
                      (cx + inner_top * c1, cy, cz + inner_top * s1))
        # Углублённое дно чаши
        for i in range(segments):
            a0 = math.tau * i / segments
            a1 = math.tau * (i + 1) / segments
            # Две соседние точки на нижней окружности (на высоте bottom)
            b0 = (cx + inner_bottom * math.cos(a0), bottom, cz + inner_bottom * math.sin(a0))
            b1 = (cx + inner_bottom * math.cos(a1), bottom, cz + inner_bottom * math.sin(a1))
            self.tri((cx, bottom, cz), b1, b0, normal=(0.0, 1.0, 0.0))

    # Добавляет эллипсоид. Используется для отрисовки крон деревьев, кустов
    # плафонов ламп, сидений унитаза 
    # Параметры:
    # center — центр эллипсоида (cx, cy, cz);
    # radii — три радиуса: rx (по X), ry (по Y), rz (по Z).
    # segments — на сколько долек разбить по «долготе»
    # rings — на сколько слоёв разбить по «широте» 
    def ellipsoid(self, center: V3, radii: V3, segments: int = 18, rings: int = 12) -> None:
        rx, ry, rz = radii
        cx, cy, cz = center
        # Сетка эллипсоида строится по широте и долготе.
        for r in range(rings):
            t0 = math.pi * r / rings
            t1 = math.pi * (r + 1) / rings
            # sin и cos для этих углов 
            st0, ct0 = math.sin(t0), math.cos(t0)
            st1, ct1 = math.sin(t1), math.cos(t1)
            for s in range(segments):
                p0 = math.tau * s / segments
                p1 = math.tau * (s + 1) / segments
                # Четыре угла одной ячейки на поверхности
                q00 = (cx + rx * st0 * math.cos(p0), cy + ry * ct0, cz + rz * st0 * math.sin(p0))
                q10 = (cx + rx * st1 * math.cos(p0), cy + ry * ct1, cz + rz * st1 * math.sin(p0))
                q11 = (cx + rx * st1 * math.cos(p1), cy + ry * ct1, cz + rz * st1 * math.sin(p1))
                q01 = (cx + rx * st0 * math.cos(p1), cy + ry * ct0, cz + rz * st0 * math.sin(p1))
                # Нормаль эллипсоида — нормированный градиент его уравнения.
                n00 = vnorm(((q00[0] - cx) / (rx * rx), (q00[1] - cy) / (ry * ry), (q00[2] - cz) / (rz * rz)))
                n10 = vnorm(((q10[0] - cx) / (rx * rx), (q10[1] - cy) / (ry * ry), (q10[2] - cz) / (rz * rz)))
                n11 = vnorm(((q11[0] - cx) / (rx * rx), (q11[1] - cy) / (ry * ry), (q11[2] - cz) / (rz * rz)))
                n01 = vnorm(((q01[0] - cx) / (rx * rx), (q01[1] - cy) / (ry * ry), (q01[2] - cz) / (rz * rz)))
                # Добавляем 4 вершины и получаем их индексы
                i00 = self._vert(q00, n00, (s / segments, 1 - r / rings))
                i10 = self._vert(q10, n10, (s / segments, 1 - (r + 1) / rings))
                i11 = self._vert(q11, n11, ((s + 1) / segments, 1 - (r + 1) / rings))
                i01 = self._vert(q01, n01, ((s + 1) / segments, 1 - r / rings))
                # Ячейка — четырёхугольник q00-q01-q11-q10, разбиваем
                # на два треугольника
                self.indices.extend((i00, i11, i10, i00, i01, i11))

    # finish() делает три вещи:
    # 1. Делит сумму на число вершин — получает настоящий центр
    # (среднее арифметическое). Он нужен Renderer'у для
    # сортировки прозрачных пакетов по расстоянию до камеры.
    # 2. Создаёт Mesh — тот загружает вершины в VBO, индексы в EBO,
    # настраивает атрибуты в VAO. После этого данные уже в GPU,
    # и Renderer может вызывать mesh.draw().
    # 3. Очищает Python-списки vertices и indices 
    def finish(self) -> "Batch":
        if self.vertex_count:
            self.center = tuple(c / self.vertex_count for c in self.center)
            self.mesh = Mesh(self.vertices, self.indices)
        self.vertices.clear()
        self.indices.clear()
        return self

# Builder — хранилище всех пакетов (Batch) сцены. Сам геометрию не
# создаёт, только раскладывает её по пакетам: каждому треугольнику
# находит «свою коробку» по материалу и группе видимости
class Builder:
    def __init__(self):
        # Словарь всех пакетов. Ключ — пара (имя_материала, группа (для отрисовки в разрезе)).
        # Значение — Batch, в который складываются вершины
        self.batches: dict[tuple[str, str], Batch] = {}

    # Возвращает пакет для материала и группы. Если такого ещё нет —
    # создаёт пустой Batch и запоминает. Если есть — возвращает его
    def get(self, material: Material, group: str = "default") -> Batch:
        key = (material.name, group)
        # Ищем существующий пакет по ключу
        b = self.batches.get(key)
        # Если такого пакета ещё нет — создаём и запоминаем в словаре
        if b is None:
            b = Batch(material, group)
            self.batches[key] = b
        return b

    # Масштабирует всю геометрию на общий коэффициент
    def scale_all(self, factor: float) -> None:
         # Идём по всем пакетам (неважно, какая группа)
        for batch in self.batches.values():
            v = batch.vertices
            for i in range(0, len(v), 8):
                v[i] *= factor
                v[i + 1] *= factor
                v[i + 2] *= factor
            batch.center[0] *= factor
            batch.center[1] *= factor
            batch.center[2] *= factor

    # Сдвигает все пакеты указанной группы
    def translate_group(self, group: str, dx: float, dy: float, dz: float) -> None:
        # Сдвиг применяется только к пакетам указанной группы, например к зданию.
        for batch in self.batches.values():
            if batch.group != group:
                continue
            v = batch.vertices
            for i in range(0, len(v), 8):
                v[i] += dx
                v[i + 1] += dy
                v[i + 2] += dz
            batch.center[0] += dx
            batch.center[1] += dy
            batch.center[2] += dz

    # Завершает пакет и создаёт сетку
    def finish(self) -> list[Batch]:
        # Списковое включение: для каждого непустого пакета вызываем
        # b.finish() — он посчитает средний центр, создаст Mesh
        # (загрузит вершины в VBO, индексы в EBO) и очистит
        # Python-списки
        return [b.finish() for b in self.batches.values() if b.vertex_count]


# Шейдеры
# MAIN_VS — вершинный шейдер основной геометрии
MAIN_VS = """
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec2 aUV;
uniform mat4 uViewProj;
uniform mat4 uLightVP; // матрица типа вид-проекция источника света
out vec3 vWorld; // мировая позиция (для освещения и тумана)
out vec3 vNormal; // нормаль (для освещения)
out vec2 vUV; // UV (для текстуры)
out vec4 vLightPos; // позиция в системе координат света (для теней)
void main() {
    vWorld = aPos;
    vNormal = normalize(aNormal);
    vUV = aUV;
    vLightPos = uLightVP * vec4(aPos, 1.0);
    gl_Position = uViewProj * vec4(aPos, 1.0); // где вершина окажется на экране
}
"""

# MAIN_FS - фрагментный шейдер основной геометрии
# Выполняется для каждого пикселя каждого треугольника. Берёт
# интерполированные данные (vWorld, vNormal, vUV, vLightPos),
# текстуру и параметры освещения — и возвращает итоговый цвет
# пикселя
MAIN_FS = """
#version 330 core
// входные интерполированные данные
in vec3 vWorld;
in vec3 vNormal;
in vec2 vUV;
in vec4 vLightPos;

// выходные данные: цвет пикселя RGBA
out vec4 FragColor;

// текстура материала 
uniform sampler2D uTexture;
// карта теней: глубина от источника света (по ней определяем, в тени ли пиксель)
uniform sampler2D uShadowMap;
// флаги материала
uniform int uUseTexture; // 1 — умножать на текстуру, 0 — чистый цвет
uniform int uAlphaTest; // 1 — отбрасывать прозрачные фрагменты
uniform int uUnlit; // 1 — не применять освещение (лампы, флаг)
// Параметры материала
uniform vec3 uDiffuse; // основной цвет
uniform vec3 uSpecular; // цвет блика
uniform vec3 uEmissive; // собственное свечение (лампы, надписи)
uniform float uShininess; // «острота» блика: больше — ярче
uniform float uAlpha; // общая прозрачность материала

// Параметры сцены
uniform vec3 uCameraPos; // позиция камеры (для блика)
uniform vec3 uLightDir; // направление НА солнце
uniform vec3 uLightColor; // цвет солнечного света
uniform vec3 uAmbientSky; // цвет подсветки сверху (небо)
uniform vec3 uAmbientGround; // цвет подсветки снизу (земля)
uniform vec3 uFogColor; // цвет тумана
uniform float uFogDensity; // плотность тумана
uniform float uShadowTexel; // размер текселя карты теней
uniform float uPointFalloff; // затухание точечных источников
uniform int uShadowsEnabled; // 1 — рисовать тени, 0 — без них

// Точечные источники (лампы в вагоне, светильники в туалете)
uniform vec3 uPointPos[8]; // позиция i-й лампы в мире (x, y, z)
uniform vec3 uPointColor[8]; // (r, g, b)
uniform float uPointRange[8]; // радиус действия: дальше него свет не достаёт
uniform int uPointCount; // сколько ламп реально используется сейчас

// shadowFactor — проверяет, в тени ли пиксель, и возвращает
// коэффициент от 0.0 (полная тень) до 1.0 (полностью освещён)
// lightPos — позиция пикселя в системе координат источника света
// (пришла из вершинного шейдера как vLightPos);
// ndl — dot(N, L)
float shadowFactor(vec4 lightPos, float ndl) {
    // используется ортопроекция для света, w = 1, деление ничего не меняет (для универсальности кода)
    vec3 proj = lightPos.xyz / lightPos.w;
    // texture() ждёт UV в [0, 1], а не в [-1, 1]
    proj = proj * 0.5 + 0.5;
    // Пиксель вне карты теней — считаем освещённым, иначе чёрные полосы по краям
    if (proj.x < 0.0 || proj.x > 1.0 || proj.y < 0.0 || proj.y > 1.0 || proj.z > 1.0)
        return 1.0;
    // Bias — сдвиг глубины, чтобы поверхность не затеняла сама себя
    float bias = max(0.0045 * (1.0 - ndl), 0.0014);
    // tx, ty — сдвиг на 1 пиксель карты теней по X и по Y
    vec2 tx = vec2(uShadowTexel, 0.0);
    vec2 ty = vec2(0.0, uShadowTexel);
    // Читаем карту теней 5 раз: в центре и у 4 соседей.
    // proj.z — глубина нашего пикселя; если она больше записанной в карте —
    // значит, между пикселем и светом что-то есть -> тень (0), иначе свет (1)
    float sum = (proj.z - bias > texture(uShadowMap, proj.xy).r) ? 0.0 : 1.0; // центр
    sum += (proj.z - bias > texture(uShadowMap, proj.xy + tx).r) ? 0.0 : 1.0; // справа
    sum += (proj.z - bias > texture(uShadowMap, proj.xy - tx).r) ? 0.0 : 1.0; // слева
    sum += (proj.z - bias > texture(uShadowMap, proj.xy + ty).r) ? 0.0 : 1.0; // сверху
    sum += (proj.z - bias > texture(uShadowMap, proj.xy - ty).r) ? 0.0 : 1.0; // снизу
    return sum / 5.0;
}

void main() {
    // Берём цвет из текстуры материала по UV
    vec4 texel = texture(uTexture, vUV);
    vec3 base = uDiffuse;
    float alpha = uAlpha;
    // Если у материала есть текстура — умножаем цвет и прозрачность
    if (uUseTexture == 1) {
        base *= texel.rgb;
        alpha *= texel.a;
    }
    // Прозрачные фрагменты (например, фон надписи) выбрасываем
    if (uAlphaTest == 1 && alpha < 0.28) discard;

    // N — нормаль (куда смотрит грань), V — вектор от пикселя к камере
    vec3 N = normalize(vNormal);
    vec3 V = normalize(uCameraPos - vWorld);
    vec3 color;
    // Unlit — без освещения (лампы, флаг, вывеска)
    if (uUnlit == 1) {
        color = base + uEmissive; // просто цвет + собственное свечение
    } else {
        // L — направление на солнце; ndl — насколько грань повёрнута к свету (0..1)
        vec3 L = normalize(uLightDir); // единичный вектор на солнце
        float ndl = max(dot(N, L), 0.0); // 1 = прямо на свет, 0 = отвёрнута
        // Тень от солнца (0 = тень, 1 = свет)
        float shadow = 1.0; // по умолчанию без тени
        if (uShadowsEnabled == 1) { // если тени включены 
            shadow = shadowFactor(vLightPos, ndl); // проверяем по карте теней
        }
        vec3 H = normalize(L + V); // вектор для блика
        float spec = pow(max(dot(N, H), 0.0), uShininess) * (0.18 + 0.82 * ndl); // сила блика

        vec3 ambient = mix(uAmbientGround, uAmbientSky, N.y * 0.5 + 0.5) * base; // подсветка (небо/земля)
        vec3 direct = uLightColor * base * ndl * shadow; // прямой свет от солнца
        vec3 specular = uLightColor * uSpecular * spec * shadow; // блик от солнца
        color = ambient + direct + specular + uEmissive; // суммируем всё

        for (int i = 0; i < 8; ++i) { // цикл по лампам (до 8)
            if (i >= uPointCount) break;
            vec3 toLight = uPointPos[i] - vWorld; // вектор от пикселя к лампе
            float dist = length(toLight); // расстояние до лампы
            vec3 Pl = toLight / max(dist, 0.0001);  // единичный вектор на лампу (защита от /0)
            float att = clamp(1.0 - dist / uPointRange[i], 0.0, 1.0); // линейное затухание до радиуса
            att = att * att / (1.0 + uPointFalloff * dist * dist); // квадратичное затухание
            float pndl = max(dot(N, Pl), 0.0); // насколько грань повёрнута к лампе
            color += uPointColor[i] * base * pndl * att; // добавляем свет от лампы
        }
    }

    float dist = length(vWorld - uCameraPos); // расстояние от пикселя до камеры
    float fog = 1.0 - exp(-pow(dist * uFogDensity, 2.0)); // сила тумана: 0 (близко) .. 1 (далеко)
    color = mix(color, uFogColor, clamp(fog, 0.0, 0.9)); // смешиваем с цветом тумана (не более 0.9)
    FragColor = vec4(color, alpha); // итоговый цвет пикселя
}
"""

DEPTH_VS = """
#version 330 core
layout(location=0) in vec3 aPos; // вход: позиция вершины (x, y, z)
uniform mat4 uLightVP; // матрица света (вид * проекция камеры солнца)
void main() { gl_Position = uLightVP * vec4(aPos, 1.0); } // куда вершина попадёт «с точки зрения солнца»
"""

DEPTH_FS = """
#version 330 core
void main() {} // пусто: цвет не нужен, важна только глубина (её пишет GPU сам)
"""

SKY_VS = """
#version 330 core
layout(location=0) in vec3 aPos; // вход: позиция вершины сферы неба
layout(location=1) in vec3 aNormal; // вход: нормаль (не используется, но Mesh передаёт — надо объявить)
layout(location=2) in vec2 aUV; // вход: UV для текстуры неба
uniform mat4 uViewProj; // матрица «вид * проекция» камеры
out vec2 vUV; // выход: UV передаём во фрагментный шейдер
void main() {
    vUV = aUV; // передаём UV дальше
    gl_Position = uViewProj * vec4(aPos, 1.0); // куда вершина попадёт на экране
}
"""

SKY_FS = """
#version 330 core                     
in vec2 vUV;                          // вход: UV от вершинного шейдера (интерполированные)
out vec4 FragColor;                   // выход: цвет пикселя
uniform sampler2D uSkyTexture;        // панорамная текстура неба
void main() {
    vec3 c = texture(uSkyTexture, vUV).rgb;  // берём цвет из текстуры по UV
    FragColor = vec4(c, 1.0);                // отдаём цвет, альфа = 1 (небо непрозрачное)
}
"""


# Scene — строит всю статическую сцену станции и хранит её пакеты геометрии.
# Класс отвечает за текстуры, материалы, примитивы, группы видимости,
# точечные источники света и сферу неба.
# После создания объекта нужно вызвать build(); он заполнит self.batches,
# self.point_lights, self.sky_texture и self.sky_mesh
class Scene:
    # STATION_DOORS — два проёма входных дверей со стороны станции.
    # Формат каждой пары: (x0, x1), где x0 и x1 — границы проёма по длине вагона
    STATION_DOORS = ((-11.42, -10.48), (10.48, 11.42))

    # Вычисляет окна и двери вагона
    # Параметры:
    # side — +1 для ближнего борта (к станции; там где двери), -1 для дальнего (из купе выходят окна).
    # Возвращает: (windows, doors), где windows — список (x0, x1, тип),
    # а doors — список (x0, x1).
    def _wall_openings(self, side: int):
        # xs — девять границ, которые делят вагон на восемь купе шириной 2,1 м.
        # Каждое окно ставится в центр своего купе.
        xs = [-7.6 + i * 2.1 for i in range(9)]
        windows = []
        for i in range(8):
            cx = (xs[i] + xs[i + 1]) * 0.5 # центр купе по оси X
            windows.append((cx - 0.50, cx + 0.50, "coupe")) # строим окно шириной 1 м
        # Окно проводника расположено отдельно, слева от восьми пассажирских купе.
        windows.append((-8.85, -7.85, "conductor"))
        # Сортировка нужна, чтобы окна и двери шли слева направо при построении стен.
        windows.sort()
        # Двери есть только на борту со стороны станции
        doors = list(self.STATION_DOORS) if side > 0 else []
        return windows, doors

    # Делит отрезок [start, end] на части, исключая intervals.
    # Параметры:
    # start, end — границы исходного отрезка по оси;
    # intervals — список вырезаемых интервалов (a, b).
    # Возвращает: список оставшихся отрезков (a, b) длиной больше 0.02.
    # возвращает оставшиеся куски стены, которые надо нарисовать
    @staticmethod
    def _segments_excluding(start: float, end: float, intervals):
        # result — список участков, которые останутся после вырезания проёмов.
        # cursor — начало ещё не обработанной части исходного отрезка.
        result = []
        cursor = start
        # Интервалы сортируются, чтобы обрабатывать вырезы слева направо.
        for a, b in sorted(intervals):
            # Вырез обрезается по границам исходного отрезка: за его пределы не выходим
            a, b = max(start, a), min(end, b)
            if b <= cursor:
                continue
            # Если между курсором и началом выреза есть место, добавляем этот участок.
            if a > cursor:
                result.append((cursor, a))
            # Курсор переходит за конец выреза; вложенные и пересекающиеся вырезы не ломают порядок.
            cursor = max(cursor, b)
        # После последнего выреза добавляем хвост исходного отрезка.
        if cursor < end:
            result.append((cursor, end))
        # Совсем короткие обрезки отбрасываем
        return [(a, b) for a, b in result if b - a > 0.02]

    # Создаёт пустую сцену и задаёт общие параметры освещения и неба
    def __init__(self):
        # batches — готовые пакеты геометрии, по одному на материал и группу.
        # point_lights — список (позиция, цвет, радиус) для ламп в вагоне.
        self.batches: list[Batch] = []
        self.point_lights: list[tuple[V3, V3, float]] = []
        # Направление на солнце и его цвет задают основной направленный свет.
        self.sun_dir = vnorm((-0.36, 0.78, -0.51))
        self.sun_color = (1.0, 0.92, 0.78)
        # Текстура и Mesh неба создаются позже, в _make_textures и _build_sky.
        self.sky_texture: Texture | None = None
        self.sky_mesh: Mesh | None = None
        # ambient_sky и ambient_ground — цвета полусферической подсветки сверху и снизу.
        # fog_color — цвет, с которым смешиваются дальние объекты.
        self.ambient_sky = (0.34, 0.40, 0.50)
        self.ambient_ground = (0.13, 0.13, 0.12)
        self.fog_color = (0.66, 0.75, 0.84)

    # build — собирает всю сцену: текстуры, материалы, землю, путь, платформы,
    # здание КПП, три забора, окружение и пассажирский вагон
    def build(self) -> None:
        # Сначала создаются текстуры и материалы: без них нельзя строить пакеты.
        self._make_textures()
        self._make_materials()
        b = Builder()
        # Дальше объекты добавляются в Builder в порядке от дальних к ближним.
        self._build_ground(b) # Земля/трава: один большой четырёхугольник
        self._build_track(b) # Железнодорожный путь
        self._build_platform(b) # Две платформы (станционная и противоположная)
        self._build_building(b) # Здание КПП
        # Здание строится вокруг локального начала координат, а затем сдвигается к платформе.
        b.translate_group("building", -9.0, 0.0, 0.0)
        self._build_fences(b) # Три вложенных забора с воротами для поезда
        self._build_environment(b) # Деревья и кусты
        self._build_train(b) # Вагон целиком
        # Позиции и радиусы ламп переводятся в мировые единицы вместе со всей сценой.
        self.point_lights = [
            (tuple(c * WORLD_SCALE for c in pos), color, radius * WORLD_SCALE)
            for pos, color, radius in self.point_lights
        ]
        # Вся геометрия увеличивается в WORLD_SCALE раз; это затрагивает только позиции.
        b.scale_all(WORLD_SCALE)
        # finish() превращает накопленные списки вершин и индексов в Mesh (VAO/VBO/EBO).
        self.batches = b.finish()
        # Сфера неба строится отдельно: она не входит в общий Builder и рисуется первой.
        self._build_sky()

    # TEXTURE_CACHE_VERSION — версия формата кеша
    TEXTURE_CACHE_VERSION = "v2"

    # Создаёт все текстуры сцены и загружает их в OpenGL.
    # Для каждой текстуры сначала проверяется кеш на диске.
    def _make_textures(self) -> None:
        # t — словарь текстур: по имени потом выбирается нужный объект Texture.
        t = self.tex = {}
        # Кеш лежит в .cache/textures рядом с программой; имя файла содержит имя и размер.
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 ".cache", "textures")
        try:
            # Пытаемся создать папку кеша; при ошибке cache_dir станет пустым и кеш отключится.
            os.makedirs(cache_dir, exist_ok=True)
        except OSError:
            cache_dir = ""
        # Счётчики нужны только для итогового сообщения о времени генерации.
        generated = 0
        cached = 0
        started = time.perf_counter()

        # tex — вложенная функция: проверяет кеш, при промахе вызывает генератор
        # и сохраняет готовые RGBA-байты обратно на диск.
        # Параметры: name — имя текстуры; w, h — размеры; generator — функция
        # генерации; repeat — нужно ли повторять текстуру по краям.
        # Возвращает: готовый объект Texture.
        def tex(name: str, w: int, h: int, generator, repeat: bool = True):
            nonlocal generated, cached
            # path — путь к файлу кеша; пустая строка означает, что кеширование недоступно.
            path = ""
            # Сначала пробуем прочитать файл и проверяем, что его размер совпадает с w*h*4.
            if cache_dir:
                path = os.path.join(cache_dir, f"{self.TEXTURE_CACHE_VERSION}_{name}_{w}x{h}.rgba")
                try:
                    with open(path, "rb") as fh:
                        data = fh.read()
                    # Размер проверяется обязательно: старый или обрезанный файл нельзя загружать в GPU.
                    if len(data) == w * h * 4:
                        cached += 1
                        return Texture(w, h, data, repeat=repeat)
                except OSError:
                    pass
            if not QUIET:
                print(f"  [текстуры] {name} {w}x{h} ...", flush=True)
            t0 = time.perf_counter()
            # Кеша нет или он повреждён — вызываем процедурный генератор текстуры.
            data = generator()
            generated += 1
            # Свежие байты сохраняем в кеш, чтобы следующий запуск не считал заново.
            if path:
                try:
                    with open(path, "wb") as fh:
                        fh.write(bytes(data))
                except OSError:
                    pass
            if not QUIET:
                print(f"  [текстуры] {name}: {time.perf_counter() - t0:.1f} c", flush=True)
            return Texture(w, h, data, repeat=repeat)

        # Ниже перечислены все текстуры сцены: трава, балласт, бетон, кирпич, дерево,
        # обои, пол, ковролин, металл, одеяла, надписи, потолок и небо.
        t["grass"] = tex("grass", 512, 512, lambda: proc.tex_grass(512))
        t["ballast"] = tex("ballast", 256, 256, lambda: proc.tex_ballast(256))
        t["concrete"] = tex("concrete", 512, 512, lambda: proc.tex_concrete(512))
        t["platform"] = tex("platform", 512, 512,
                            lambda: proc.tex_concrete(512, tiles=8, base=0.56))
        t["brick"] = tex("brick", 512, 512, lambda: proc.tex_brick(512))
        t["wood"] = tex("wood", 512, 512, lambda: proc.tex_wood(512, light=True))
        t["wood_dark"] = tex("wood_dark", 512, 512, lambda: proc.tex_wood(512, light=False))
        t["wallpaper"] = tex("wallpaper", 256, 256, lambda: proc.tex_wallpaper(256))
        t["floor"] = tex("floor", 512, 512, lambda: proc.tex_floor(512))
        t["carpet"] = tex("carpet", 256, 256, lambda: proc.tex_carpet(256))
        t["metal_blue"] = tex("metal_blue", 256, 256, lambda: proc.tex_metal(256, blue=True))
        t["metal"] = tex("metal", 256, 256, lambda: proc.tex_metal(256, blue=False))
        t["blanket_blue"] = tex("blanket_blue", 256, 256,
                                lambda: proc.tex_fabric(256, (42, 78, 138)))
        t["blanket_teal"] = tex("blanket_teal", 256, 256,
                                lambda: proc.tex_fabric(256, (28, 104, 101)))
        t["blanket_red"] = tex("blanket_red", 256, 256,
                               lambda: proc.tex_fabric(256, (112, 39, 42)))
        # Герб Казахстана — единственное внешнее изображение; он читается из assets.
        emblem = None
        emblem_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "assets", "kazakhstan_emblem_256.rgba")
        try:
            with open(emblem_path, "rb") as fh:
                emblem_data = fh.read()
            # Проверяем размер RGBA-файла герба, чтобы не передать в текстуру мусор.
            if len(emblem_data) == 256 * 256 * 4:
                emblem = (256, 256, emblem_data)
        except OSError:
            pass
        # Надпись маршрута и герб собираются в одну текстуру 3072×256 без повторения.
        t["label"] = tex("label_emblem", 3072, 256,
                         lambda: proc.make_train_label(3072, 256, emblem), repeat=False)
        # Вывеска КПП — отдельная текстура 2560×400, тоже без повторения.
        t["sign"] = tex("sign", 2560, 400,
                        lambda: proc.make_building_sign(2560, 400), repeat=False)
        t["ceiling"] = tex("ceiling", 256, 256, lambda: proc.tex_ceiling(256))
        # Панорама неба 1024×512 генерируется с учётом направления солнца.
        t["sky"] = tex("sky", 1024, 512,
                       lambda: proc.tex_sky(1024, 512, self.sun_dir), repeat=False)
        if not QUIET:
            # В конце выводится, сколько текстур взято из кеша и сколько сгенерировано.
            print(f"[сцена] текстуры: {cached} из кеша, {generated} сгенерировано "
                  f"за {time.perf_counter() - started:.1f} c", flush=True)

    # Создаёт словарь материалов для всех объектов сцены.
    # Параметры: self — объект Scene, у которого уже заполнен self.tex.
    # Материалы потом выбираются по строковому ключу.
    def _make_materials(self) -> None:
        # t — уже готовый словарь текстур; материалы ссылаются на них по имени.
        t = self.tex
        self.m = m = {}
        # Материалы подвижного состава: синий и жёлтый металл, крыша, сталь, тёмный металл.
        m["train_blue"] = Material("train_blue", (1, 1, 1), (0.52, 0.60, 0.78), 72, t["metal_blue"])
        m["train_yellow"] = Material("train_yellow", (0.94, 0.62, 0.06), (0.55, 0.42, 0.12), 48)
        m["train_yellow_dark"] = Material("train_yellow_dark", (0.72, 0.44, 0.03), (0.35, 0.28, 0.08), 30)
        m["train_roof"] = Material("train_roof", (0.48, 0.55, 0.61), (0.42, 0.45, 0.48), 40, t["metal"])
        m["steel"] = Material("steel", (0.48, 0.50, 0.53), (0.8, 0.82, 0.86), 90, t["metal"])
        m["dark_metal"] = Material("dark_metal", (0.13, 0.15, 0.17), (0.5, 0.5, 0.55), 70)
        m["rust"] = Material("rust", (0.30, 0.16, 0.09), (0.12, 0.1, 0.08), 18)
        m["rubber"] = Material("rubber", (0.055, 0.06, 0.065), (0.16, 0.16, 0.16), 24)
        # Материалы интерьера: дерево, стол, плинтус, обои, потолок, пол и ковролин.
        m["wood"] = Material("wood", (1, 1, 1), (0.16, 0.12, 0.08), 22, t["wood"])
        m["table_wood"] = Material("table_wood", (0.52, 0.34, 0.18),
                                   (0.22, 0.17, 0.10), 30, None,
                                   emissive=(0.045, 0.030, 0.016))
        m["skirting"] = Material("skirting", (0.56, 0.47, 0.36),
                                 (0.10, 0.08, 0.06), 18, None)
        m["wood_dark"] = Material("wood_dark", (1, 1, 1), (0.22, 0.16, 0.1), 30, t["wood_dark"])
        m["wallpaper"] = Material("wallpaper", (1, 1, 1), (0.08, 0.07, 0.05), 12, t["wallpaper"])
        m["ceiling"] = Material("ceiling", (0.94, 0.93, 0.89), (0.12, 0.12, 0.12), 18, t["ceiling"],
                                emissive=(0.16, 0.145, 0.115))
        m["floor"] = Material("floor", (1, 1, 1), (0.32, 0.36, 0.38), 58, t["floor"])
        m["carpet"] = Material("carpet", (1, 1, 1), (0.08, 0.05, 0.05), 12, t["carpet"])
        m["mattress"] = Material("mattress", (0.92, 0.90, 0.82), (0.08, 0.08, 0.07), 12)
        m["sheet"] = Material("sheet", (0.97, 0.96, 0.91), (0.1, 0.1, 0.1), 16)
        m["blanket_blue"] = Material("blanket_blue", (1, 1, 1), (0.06, 0.06, 0.07), 10, t["blanket_blue"])
        m["blanket_teal"] = Material("blanket_teal", (1, 1, 1), (0.06, 0.06, 0.07), 10, t["blanket_teal"])
        m["blanket_red"] = Material("blanket_red", (1, 1, 1), (0.07, 0.05, 0.05), 10, t["blanket_red"])
        m["pillow"] = Material("pillow", (0.96, 0.95, 0.90), (0.12, 0.12, 0.11), 16)
        m["curtain"] = Material("curtain", (0.42, 0.11, 0.09), (0.16, 0.08, 0.06), 14)
        m["porcelain"] = Material("porcelain", (0.92, 0.94, 0.92), (0.9, 0.92, 0.9), 110)
        m["mirror"] = Material("mirror", (0.42, 0.43, 0.44),
                               (1.0, 1.0, 1.0), 200, None,
                               emissive=(0.020, 0.021, 0.023))
        m["chrome"] = Material("chrome", (0.78, 0.80, 0.82), (1.0, 1.0, 1.0), 150)
        # Стекло полупрозрачное: alpha меньше единицы, поэтому рисуется отдельным проходом.
        m["glass"] = Material("glass", (0.23, 0.36, 0.46), (1.0, 1.0, 1.0), 150, alpha=0.30)
        m["glass_dark"] = Material("glass_dark", (0.10, 0.17, 0.22), (0.9, 0.95, 1.0), 120, alpha=0.48)
        m["label"] = Material("label", (1, 1, 1), (0.25, 0.25, 0.22), 20, t["label"],
                              alpha_test=True)
        m["sign"] = Material("sign", (1, 1, 1), (0.0, 0.0, 0.0), 1, t["sign"], unlit=True,
                             emissive=(0.22, 0.22, 0.22))
        # Лампы, вывеска и флаг помечены unlit/emissive: они не зависят от освещения.
        m["lamp"] = Material("lamp", (0.98, 0.73, 0.38), (0.2, 0.2, 0.2), 20,
                             unlit=True, emissive=(1.2, 0.72, 0.27))
        m["lamp_cool"] = Material("lamp_cool", (0.9, 0.95, 1.0), (0.2, 0.2, 0.2), 20,
                                  unlit=True, emissive=(0.72, 0.84, 1.0))
        # Материалы окружения: трава, балласт, бетон, платформа, кирпич, крыша, заборы.
        m["grass"] = Material("grass", (0.96, 1.0, 0.94), (0.02, 0.02, 0.02), 4, t["grass"])
        m["ballast"] = Material("ballast", (1, 1, 1), (0.04, 0.04, 0.04), 6, t["ballast"])
        m["concrete"] = Material("concrete", (1, 1, 1), (0.12, 0.12, 0.12), 18, t["concrete"])
        m["platform"] = Material("platform", (1, 1, 1), (0.20, 0.20, 0.20), 28, t["platform"])
        m["brick"] = Material("brick", (0.96, 0.91, 0.88), (0.08, 0.06, 0.05), 14, t["brick"])
        m["roof_green"] = Material("roof_green", (0.14, 0.28, 0.23), (0.32, 0.36, 0.33), 40, t["metal"])
        m["fence_wood"] = Material("fence_wood", (1, 1, 1), (0.08, 0.06, 0.04), 12, t["wood_dark"])
        m["fence_white"] = Material("fence_white", (0.84, 0.82, 0.74), (0.08, 0.08, 0.07), 14)
        m["stone"] = Material("stone", (0.44, 0.43, 0.40), (0.10, 0.10, 0.10), 16)
        m["tree_trunk"] = Material("tree_trunk", (0.22, 0.13, 0.07), (0.04, 0.03, 0.02), 8)
        m["tree_a"] = Material("tree_a", (0.12, 0.29, 0.12), (0.04, 0.06, 0.03), 8)
        m["tree_b"] = Material("tree_b", (0.17, 0.36, 0.14), (0.05, 0.07, 0.03), 8)
        m["bush"] = Material("bush", (0.13, 0.30, 0.11), (0.04, 0.06, 0.03), 8)
        m["white"] = Material("white", (0.93, 0.93, 0.90), (0.2, 0.2, 0.2), 25)
        m["red"] = Material("red", (0.70, 0.04, 0.06), (0.2, 0.05, 0.05), 30)
        # Три полосы флага России сделаны unlit, чтобы их цвета не темнели в тени.
        m["flag_white"] = Material("flag_white", (0.95, 0.95, 0.95), (0.1, 0.1, 0.1), 20,
                                   unlit=True, emissive=(0.10, 0.10, 0.10))
        m["flag_blue"] = Material("flag_blue", (0.02, 0.14, 0.45), (0.1, 0.1, 0.1), 20,
                                  unlit=True, emissive=(0.03, 0.10, 0.26))
        m["flag_red"] = Material("flag_red", (0.72, 0.02, 0.04), (0.1, 0.1, 0.1), 20,
                                 unlit=True, emissive=(0.22, 0.01, 0.01))
        m["blue_flag"] = Material("blue_flag", (0.03, 0.16, 0.48), (0.15, 0.18, 0.3), 30)
        m["green_bin"] = Material("green_bin", (0.10, 0.24, 0.16), (0.3, 0.35, 0.32), 30)

    # B — возвращает пакет Batch для материала key и группы group.
    # Параметры:
    # b — Builder, в котором накапливается геометрия;
    # key — имя материала в self.m;
    # group — группа видимости: "default", "near" или "building".
    # Возвращает: готовый или только что созданный Batch.
    def B(self, b: Builder, key: str, group: str = "default") -> Batch:
        # Builder.get ищет пакет по паре (имя материала, группа) и создаёт его при необходимости.
        return b.get(self.m[key], group)

    # Строит вертикальный четырёхугольник на боковой стене вагона.
    # Параметры:
    # batch — пакет, куда добавляется грань;
    # side — +1 или -1, задаёт направление нормали;
    # x0, x1, y0, y1 — границы прямоугольника;
    # z — плоскость стены;
    # uv — координаты текстуры (u0, v0, u1, v1);
    # material_normal — оставлен для совместимости
    def _quad_side(self, batch: Batch, side: int, x0: float, x1: float,
                   y0: float, y1: float, z: float, uv=(0, 0, 1, 1),
                   material_normal: bool = False) -> None:
        # u0,v0 — левый нижний угол текстуры, u1,v1 — правый верхний.
        u0, v0, u1, v1 = uv
        # Для ближнего борта вершины идут в одном порядке, для дальнего — в обратном,
        # чтобы нормаль всегда смотрела наружу от вагона.
        if side > 0:
            batch.quad((x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z),
                       (u0, v0), (u1, v0), (u1, v1), (u0, v1))
        else:
            batch.quad((x1, y0, z), (x0, y0, z), (x0, y1, z), (x1, y1, z),
                       (u0, v0), (u1, v0), (u1, v1), (u0, v1))

    # Cтроит землю вокруг станции: одну большую травяную плоскость.
    # Параметры: b — Builder для накопления геометрии.
    # Возвращает: None.
    def _build_ground(self, b: Builder) -> None:
        g = self.B(b, "grass")
        # Один большой четырёхугольник лежит чуть ниже нуля и повторяет текстуру травы 30×26 раз.
        g.quad((-135, -0.03, -115), (-135, -0.03, 120), (135, -0.03, 120), (135, -0.03, -115),
               (0, 0), (0, 26), (30, 26), (30, 0))

    # Строит железнодорожный путь: балласт, шпалы, два рельса
    # и металлические крепления у станции.
    # Параметры: b — Builder для накопления геометрии.
    def _build_track(self, b: Builder) -> None:
        # Длина пути 210 м; балласт — вытянутый блок с повторением текстуры вдоль оси X.
        track_length = 210.0
        ballast = self.B(b, "ballast")
        # Балласт — длинный блок 210×0.13×4 м (щебень под шпалами).
        # uv_scale=(48, 1) — текстура повторяется 48 раз вдоль X
        ballast.box((0, 0.055, 0), (track_length, 0.13, 4.0), uv_scale=(48, 1))
        # Шпалы — деревянные бруски поперёк пути.
        sleepers = self.B(b, "wood_dark")
        # Количество шпал считается по шагу 0,8 м, чтобы они равномерно заполнили путь.
        count = int(track_length / 0.8)
        # Центр каждой шпалы сдвигается от левого конца пути на 0,4 + i*0,8 м.
        for i in range(count):
            x = -track_length * 0.5 + 0.4 + i * 0.8
            sleepers.box((x, 0.135, 0), (0.24, 0.12, 2.45), uv_scale=(1, 1))
        # Каждый рельс состоит из широкой подошвы и узкой головки, стоящих друг на друге.
        rails = self.B(b, "steel")
        for z in (-0.755, 0.755):
            rails.box((0, 0.235, z), (track_length, 0.09, 0.075)) # подошва рельса
            rails.box((0, 0.295, z), (track_length, 0.045, 0.10)) # головка рельса.
        fast = self.B(b, "dark_metal") # пакет креплений (чтоб поезд не съезжал на станции), 
                                       # видны только в ее пределах
        for i in range(0, 72, 2):
            x = -28.0 + i * 0.8
            for z in (-0.755, 0.755):
                fast.box((x, 0.31, z), (0.10, 0.035, 0.22)) # пластина крепления

    # Строит две пассажирские платформы:
    # станционную с мебелью и противоположную без мебели.
    # Параметры: b — Builder для накопления геометрии.
    def _build_platform(self, b: Builder) -> None:
        # Четыре материала, которые нужны для платформы:
        # conc — бетон основания (серый),
        # top — плитка верхнего покрытия,
        # dark — тёмный металл (швы, столбики, опоры скамеек, фонари),
        # yellow — жёлтая краска (полоса безопасности у края).
        conc = self.B(b, "concrete")
        top = self.B(b, "platform")
        dark = self.B(b, "dark_metal")
        yellow = self.B(b, "train_yellow")
        # Станционная платформа: бетонное основание высотой 1,15 м.
        conc.box((0, 0.575, 5.0), (40, 1.15, 6.0))
        # Верх платформы — горизонтальный четырёхугольник с UV, растянутыми на 12×7 плиток.
        top.quad((-20, 1.155, 2.0), (-20, 1.155, 8.0), (20, 1.155, 8.0), (20, 1.155, 2.0),
                 (0, 0), (0, 7), (12, 7), (12, 0))
        # Жёлтая полоса безопасности и тёмная линия отделяют край платформы от пути.
        yellow.box((0, 1.158, 2.32), (39.4, 0.018, 0.26))
        dark.box((0, 1.157, 2.02), (40, 0.015, 0.07))
        # Три декоративных шва на плитке (имитируют стыки плит):
        # на z = 3.5, 5.0, 6.5, по всей длине, высотой 1.2 см, шириной
        # 3.5 см, высота чуть больше плитки (1.159), чтобы были видны
        for z in (3.5, 5.0, 6.5):
            dark.box((0, 1.159, z), (39.7, 0.012, 0.035))
        # Противоположная платформа — зеркальная копия основания без мебели и фонарей.
        # Бетонное основание
        conc.box((0, 0.575, -5.0), (40, 1.15, 6.0))
        # Верхний слой плитки
        top.quad((-20, 1.155, -8.0), (-20, 1.155, -2.0), (20, 1.155, -2.0), (20, 1.155, -8.0),
                 (0, 0), (0, 7), (12, 7), (12, 0))
        # Жёлтая полоса у края противоположной платформы
        yellow.box((0, 1.158, -2.32), (39.4, 0.018, 0.26))
        # Тёмная линия-бордюр на краю
        dark.box((0, 1.157, -2.02), (40, 0.015, 0.07))
        # Декоративные швы
        for z in (-3.5, -5.0, -6.5):
            dark.box((0, 1.159, z), (39.7, 0.012, 0.035))
        # Скамейки: деревянное сиденье, спинка и две металлические опоры.
        for bx in (-17.0, 10.0):
            wood = self.B(b, "wood") # дерево для сиденья и спинки
            metal = self.B(b, "dark_metal") # металл для опор
            # Сиденье
            wood.box((bx, 1.63, 6.55), (1.50, 0.09, 0.46))
            # Спинка
            wood.box((bx, 1.96, 6.77), (1.50, 0.42, 0.07))
            # Две металлические опоры (ножки) по краям сиденья
            for dx in (-0.60, 0.60):
                metal.box((bx + dx, 1.36, 6.55), (0.07, 0.48, 0.42))
        green = self.B(b, "green_bin")
        # Урна — зелёный цилиндр, сверху тёмный ободок-крышка.
        # Корпус урны
        green.cylinder((-13.0, 1.55, 6.7), 0.23, 0.75, "y", 16)
        dark = self.B(b, "dark_metal")
        # Тёмная крышка сверху (диск-обод)
        dark.cylinder((-13.0, 1.96, 6.7), 0.25, 0.07, "y", 16)
        # Фонарные столбы
        for lx in (-16.0, -5.0, 7.0, 16.0):
            pole = self.B(b, "dark_metal")
            # Столб: цилиндр
            pole.cylinder((lx, 2.85, 6.9), 0.055, 3.4, "y", 12)
            # Горизонтальный кронштейн (перекладина), на которой висит лампа
            pole.box((lx, 4.53, 6.62), (0.075, 0.075, 0.62))
            # Сама лампа — светящийся эллипсоид
            lamp = self.B(b, "lamp")
            lamp.ellipsoid((lx, 4.46, 6.32), (0.14, 0.12, 0.14), 12, 8)
        # Девять низких столбиков отмечают край платформы со стороны пути.
        for i in range(9):
            x = -18 + i * 4.5
            dark.cylinder((x, 1.35, 2.05), 0.055, 0.38, "y", 10)

    # Cтроит здание КПП: кирпичные стены, фундамент, цоколь,
    # двускатную крышу, окна, дверь, вывеску, фонарь и флаг России.
    # Параметры: b — Builder; все части добавляются в группу "building".
    def _build_building(self, b: Builder) -> None:
        # Все части здания добавляются в группу "building", чтобы в разрезе скрыть его целиком.
        B = lambda key: self.B(b, key, "building")
        # Кирпичный корпус
        brick = B("brick")
        # Основной объём — кирпичный параллелепипед 14×5×4,5 м с центром около платформы.
        brick.box((1.0, 3.65, 10.35), (14.0, 5.0, 4.5))
        # Три дополнительные кирпичные грани закрывают переднюю и боковые стены.
        # Передняя стена
        brick.quad((8.0, 1.15, 8.09), (-6.0, 1.15, 8.09), (-6.0, 6.15, 8.09), (8.0, 6.15, 8.09),
                   (0, 0), (5, 0), (5, 3.0), (0, 3.0))
        # Левая боковая стена 
        brick.quad((-6.09, 1.15, 8.1), (-6.09, 1.15, 12.6), (-6.09, 6.15, 12.6), (-6.09, 6.15, 8.1),
                   (0, 0), (2.4, 0), (2.4, 3.0), (0, 3.0))
        # Правая боковая стена
        brick.quad((8.09, 1.15, 12.6), (8.09, 1.15, 8.1), (8.09, 6.15, 8.1), (8.09, 6.15, 12.6),
                   (0, 0), (2.4, 0), (2.4, 3.0), (0, 3.0))
        # Фундамент до земли и каменный цоколь идут по периметру здания.
        stone = B("stone")
        # Фундамент: блок чуть шире корпуса,
        # чтобы выступал за кирпич
        stone.box((1.0, 0.575, 10.35), (14.28, 1.15, 4.78))
        # Каменный цоколь над фундаментом
        stone.box((1.0, 1.38, 10.35), (14.15, 0.46, 4.62))
        # Передняя грань цоколя
        stone.quad((-6.095, 1.15, 8.1), (-6.095, 1.15, 12.6),
                   (-6.095, 1.61, 12.6), (-6.095, 1.61, 8.1))
        # Задняя грань цоколя
        stone.quad((8.095, 1.15, 12.6), (8.095, 1.15, 8.1),
                   (8.095, 1.61, 8.1), (8.095, 1.61, 12.6))
        # Двускатная крыша: два наклонных ската и два торцевых треугольника.
        roof = B("roof_green")
        # Передний скат
        roof.quad((-6.55, 6.14, 7.82), (-6.55, 7.15, 10.35), (8.55, 7.15, 10.35), (8.55, 6.14, 7.82),
                  (0, 0), (0, 2.4), (6, 2.4), (6, 0))
        # Задний скат
        roof.quad((8.55, 6.14, 12.88), (8.55, 7.15, 10.35), (-6.55, 7.15, 10.35), (-6.55, 6.14, 12.88),
                  (0, 0), (0, 2.4), (6, 2.4), (6, 0))
        # Левый торцевой треугольник: от (z=7.82) до (z=12.88) с
        # вершиной в коньке (z=10.35, y=7.15). «Треугольник» задан
        # четырьмя точками, две из которых совпадают (конёк), чтобы
        # получить именно треугольник через quad.
        roof.quad((-6.55, 6.14, 7.82), (-6.55, 6.14, 12.88), (-6.55, 7.15, 10.35), (-6.55, 7.15, 10.35),
                  (0, 0), (2.4, 0), (1.2, 1.8), (1.2, 1.8))
        # Правый торцевой треугольник
        roof.quad((8.55, 6.14, 12.88), (8.55, 6.14, 7.82), (8.55, 7.15, 10.35), (8.55, 7.15, 10.35),
                  (0, 0), (2.4, 0), (1.2, 1.8), (1.2, 1.8))
        # Свес крыши (софит) закрывает щель между стеной и скатами.
        soffit = B("ceiling")
        # Плоский блок 15.1×0.13×5.06 над стенами
        soffit.box((1.0, 6.075, 10.35), (15.10, 0.13, 5.06))
        # Окна фасада (4 штуки)
        win_glass = B("glass_dark") # тёмное стекло
        frame = B("white") # белые рамы и перекладины
        # Четыре окна фасада: белая рама, тёмное стекло и две перекладины.
        for wx in (-4.7, -2.45, 2.55, 4.8):
            # Белая рама
            frame.box((wx, 3.88, 8.03), (1.32, 1.58, 0.10))
            # Само стекло
            win_glass.quad((wx + 0.56, 3.18, 7.97), (wx - 0.56, 3.18, 7.97),
                           (wx - 0.56, 4.55, 7.97), (wx + 0.56, 4.55, 7.97))
            # Вертикальная перекладина
            frame.box((wx, 3.86, 7.965), (0.055, 1.44, 0.05))
            # Горизонтальная перекладина
            frame.box((wx, 3.86, 7.965), (1.20, 0.055, 0.05))
        # Дверь прямоугольная, деревянная, с белым наличником и ручкой.
        # Деревянное полотно двери
        door = B("wood_dark")
        door.box((1.0, 2.30, 8.03), (1.20, 2.26, 0.13))
        door_frame = B("white")
        # Левая стойка наличника
        door_frame.box((0.35, 2.27, 8.035), (0.10, 2.30, 0.17))
        # Правая стойка наличника
        door_frame.box((1.65, 2.27, 8.035), (0.10, 2.30, 0.17))
        # Верхняя перекладина наличника
        door_frame.box((1.0, 3.47, 8.035), (1.42, 0.10, 0.17))
         # Хромированная ручка
        chrome = B("chrome")
        chrome.box((1.48, 2.15, 7.945), (0.055, 0.28, 0.035))
        # Каменный порог под дверью
        stone = B("stone")
        stone.box((1.0, 1.175, 8.06), (1.34, 0.055, 0.18))
        # Над дверью висит светящийся прямоугольный плафон.
        lamp = B("lamp")
        lamp.box((1.0, 3.72, 7.94), (0.64, 0.16, 0.20))
        # Вывеска «КПП Локоть» — текстурированный четырёхугольник на передней стене.
        sign = B("sign")
        sign.quad((6.25, 4.80, 7.90), (-4.25, 4.80, 7.90),
                  (-4.25, 5.92, 7.90), (6.25, 5.92, 7.90),
                  (0, 0), (1, 0), (1, 1), (0, 1))
        # Золотая рамка вокруг вывески
        sign_frame = B("train_yellow_dark")
        sign_frame.box((1.0, 5.36, 7.96), (10.7, 1.32, 0.10))
        # Боковые окна строятся с разным порядком вершин, чтобы стёкла смотрели наружу.
        # sx = -6.02 / 8.02 — плоскости боковых стен.
        # sign_x = -1 / +1 — в какую сторону смотрит нормаль стекла:
        # для левой стены наружу = -X, для правой = +X.
        for sx, sign_x in ((-6.02, -1), (8.02, 1)):
            for wz in (9.25, 11.25):
                # Белая рама
                frame.box((sx + sign_x * 0.07, 4.05, wz), (0.12, 1.30, 1.05))
                # Стекло
                if sign_x < 0:
                    win_glass.quad((sx + sign_x * 0.145, 3.45, wz - 0.44),
                                   (sx + sign_x * 0.145, 3.45, wz + 0.44),
                                   (sx + sign_x * 0.145, 4.68, wz + 0.44),
                                   (sx + sign_x * 0.145, 4.68, wz - 0.44))
                else:
                    win_glass.quad((sx + sign_x * 0.145, 3.45, wz + 0.44),
                                   (sx + sign_x * 0.145, 3.45, wz - 0.44),
                                   (sx + sign_x * 0.145, 4.68, wz - 0.44),
                                   (sx + sign_x * 0.145, 4.68, wz + 0.44))
                frame.box((sx + sign_x * 0.15, 4.065, wz), (0.05, 1.22, 0.055))
                frame.box((sx + sign_x * 0.15, 4.065, wz), (0.05, 0.055, 0.98))
        # Флагшток и три горизонтальные полосы флага России.
        pole = B("chrome")
        # Флагшток — хромированный цилиндр
        pole.cylinder((6.5, 7.75, 9.0), 0.035, 3.2, "y", 10)
        # Три горизонтальные полосы флага
        for i, key in enumerate(("flag_white", "flag_blue", "flag_red")):
            flag = B(key)
            flag.box((7.30, 9.02 - i * 0.22, 9.0), (1.55, 0.22, 0.035))

    # Строит один прямой участок деревянного забора между точками
    # (x0, z0) и (x1, z1). Состоит из трёх частей:
    # 1. Столбы — вертикальные брусья через каждые 2.6 м.
    # 2. Прожилины — горизонтальные доски между столбами (2 или 3).
    # 3. Штакетник — вертикальные доски, заполняющие пролёт.
    # Параметры:
    # b — Builder для накопления геометрии;
    # x0, z0, x1, z1 — начало и конец участка (в метрах, по земле);
    # height — высота забора;
    # pickets — True: со сплошным штакетником;
    # False: редкий забор (только прожилины).
    def _wood_fence_line(self, b: Builder, x0: float, z0: float,
                         x1: float, z1: float, height: float,
                         pickets: bool = True) -> None:
        # dx,dz — вектор от начала к концу участка.
        dx, dz = x1 - x0, z1 - z0
        # Длина участка в метрах — по теореме Пифагора.
        length = math.hypot(dx, dz)
        # Слишком короткий участок пропускаем: если столбы окажутся
        # ближе 5 см друг к другу, забор будет выглядеть как один столб
        if length < 0.05:
            return
        # Пакет дерева для забора
        wood = self.B(b, "fence_wood")
        # along_x — вдоль какой оси тянуть доски:
        # True  — участок длиннее по X (доски вдоль X);
        # False — участок длиннее по Z (доски вдоль Z)
        along_x = abs(dx) >= abs(dz)
        # Один столб на каждые 2.6 м длины, минимум 1
        posts = max(1, int(length / 2.6))
        for i in range(posts + 1):
            # t — доля длины от начала: 0 (первый столб) .. 1 (последний)
            t = i / posts
            # Точка на прямой (x0, z0) -> (x1, z1) при параметре t:
            # px, pz = начало + направление * t
            px, pz = x0 + dx * t, z0 + dz * t
            # Столб — вертикальный брусок
            wood.box((px, height * 0.5 + 0.02, pz), (0.14, height, 0.14))
        # Для штакетника нужны две прожилины, для редкого забора — три.
        rail_levels = (height * 0.34, height * 0.72) if pickets else (0.50, 1.15, 1.75)
        # Проходим по всем высотам, на которых должны быть горизонтальные доски
        for ry in rail_levels:
            # Середина забора по X и по Z
            cx, cz = (x0 + x1) * 0.5, (z0 + z1) * 0.5
            # Теперь рисуем доску
            # Нужно, чтобы длинная сторона доски шла вдоль забора, а не поперёк.
            # Если забор идёт вдоль X — длина должна быть по X.
            # Если вдоль Z — длина должна быть по Z.
            if along_x:
                wood.box((cx, ry, cz), (length, 0.10, 0.07))
            else:
                wood.box((cx, ry, cz), (0.07, 0.10, length))
        # Штакетник
        if pickets:
            # Доска через каждые 0.34 м. count — число досок.
            count = max(1, int(length / 0.34))
            for i in range(count):
                # t = (i + 0.5) / count — центр i-й доски, распределённой
                # равномерно по длине. +0.5 чтобы доски не наезжали
                # ровно на столбы по краям.
                t = (i + 0.5) / count
                px, pz = x0 + dx * t, z0 + dz * t
                if along_x:
                    wood.box((px, height * 0.47, pz), (0.065, height * 0.82, 0.18))
                else:
                    wood.box((px, height * 0.47, pz), (0.18, height * 0.82, 0.065))

    # Cтроит участок проволочного забора:
    # бетонные столбы, четыре ряда проволоки и колючки.
    # Параметры:
    # b — Builder; x0,z0,x1,z1 — концы участка; height — высота столбов.
    def _wire_fence_line(self, b: Builder, x0: float, z0: float,
                         x1: float, z1: float, height: float) -> None:
        # Участок задаётся концами; length и along_x нужны для расстановки столбов и проволоки.
        dx, dz = x1 - x0, z1 - z0
        length = math.hypot(dx, dz)
        # Слишком короткий участок пропускаем
        if length < 0.05:
            return
        # along_x — вдоль какой оси тянуть проволоку
        along_x = abs(dx) >= abs(dz)
        # Середина участка — центр для проволок и колючек
        cx, cz = (x0 + x1) * 0.5, (z0 + z1) * 0.5
        # Бетонные столбы проволочного забора ставятся через 3,8 м.
        posts = max(1, int(length / 3.8))
        concrete = self.B(b, "concrete")
        for i in range(posts + 1):
            # t — доля длины от начала: 0 (первый столб) .. 1 (последний)
            t = i / posts
            # Точка на прямой (x0, z0) -> (x1, z1) при параметре t
            px, pz = x0 + dx * t, z0 + dz * t
            # Столб — бетонный брусок
            concrete.box((px, height * 0.5, pz), (0.18, height, 0.18))
            # Крышка на столбе
            concrete.box((px, height + 0.04, pz), (0.24, 0.09, 0.24))
        # Четыре горизонтальные проволоки — тонкие цилиндры вдоль участка.
        wires = self.B(b, "steel")
        for wy in (1.10, 1.55, 2.00, 2.38):
            # Участок вдоль X: проволока лежит вдоль X
            if along_x:
                wires.cylinder((cx, wy, cz), 0.013, length, "x", 6)
            # Участок вдоль Z
            else:
                wires.cylinder((cx, wy, cz), 0.013, length, "z", 6)
        # Колючки. 
        barbs = self.B(b, "steel")
        # random.Random(77) — отдельный генератор с фиксированным зерном,
        # чтобы шаг колючек был одинаковым при каждом запуске.
        rng = random.Random(77)
        # Колючки только на трёх верхних проволоках (нижняя 1.10 без
        # колючек — она низко, там и так не перелезут).
        for wy in (1.55, 2.00, 2.38):
            # pos — текущее расстояние от начала участка (в метрах)
            pos = 0.4
            # Идём вдоль проволоки, пока не дойдём до конца
            while pos < length - 0.3:
                # t — доля длины: какую точку проволоки мы сейчас обрабатываем
                t = pos / length
                # Точка на прямой (x0, z0) -> (x1, z1) при параметре t
                px, pz = x0 + dx * t, z0 + dz * t
                # Колючка — три пересекающихся бруска.
                # Все три имеют центр в одной точке (px, wy, pz), но
                # вытянуты по разным осям:
                barbs.box((px, wy, pz), (0.105, 0.014, 0.014))
                barbs.box((px, wy, pz), (0.014, 0.105, 0.014))
                barbs.box((px, wy, pz), (0.014, 0.014, 0.105))
                # Следующая колючка — через 0.8 м + небольшой случайный сдвиг (0..0.15 м)
                pos += 0.8 + rng.random() * 0.15

    # Cтроит бетонный столб ворот с жёлтыми маркировочными полосами.
    # Параметры: b — Builder; x,z — положение столба; height — высота.
    # Возвращает: None.
    def _gate_post(self, b: Builder, x: float, z: float, height: float) -> None:
        # Столб ворот — бетонный блок, сверху тёмный металлический оголовок.
        post = self.B(b, "concrete")
        post.box((x, height * 0.5, z), (0.24, height, 0.24))
        cap = self.B(b, "dark_metal")
        cap.box((x, height + 0.04, z), (0.30, 0.08, 0.30))
        # Жёлтые полосы наносятся только ниже верха столба, чтобы не выходить за него.
        yellow = self.B(b, "train_yellow")
        for y in (0.55, 1.35, 2.15):
            if y < height - 0.15:
                yellow.box((x, y, z), (0.27, 0.16, 0.27))

    # Строит замкнутый деревянный периметр с разрывами
    # для ворот поезда на боковых сторонах.
    # Параметры:
    # b — Builder для геометрии;
    # xmin, xmax, zmin, zmax — границы периметра;
    # gap — ширина ворот (разрыва) в метрах;
    # height — высота забора;
    # pickets — True: со штакетником (частокол);
    # False: только прожилины (редкий забор).
    def _wood_perimeter(self, b: Builder, xmin: float, xmax: float,
                        zmin: float, zmax: float, gap: float,
                        height: float, pickets: bool) -> None:
        # На каждой боковой стороне — разрыв gap посередине, чтобы
        # поезд мог проехать
        # для левой и правой стороны
        for x in (xmin, xmax):
            self._wood_fence_line(b, x, zmin, x, -gap * 0.5, height, pickets) # кусок до разрыва
            self._wood_fence_line(b, x, gap * 0.5, x, zmax, height, pickets) # кусок после разрыва
            self._gate_post(b, x, -gap * 0.5, height + 0.25) # столб ворот
            self._gate_post(b, x, gap * 0.5, height + 0.25) # столб ворот
        # Передняя и задняя стороны периметра строятся целиком, без ворот.
        self._wood_fence_line(b, xmin, zmin, xmax, zmin, height, pickets)
        self._wood_fence_line(b, xmin, zmax, xmax, zmax, height, pickets)

    # Строит замкнутый проволочный периметр с воротами для проезда поезда.
    # Параметры:
    # b — Builder; xmin,xmax,zmin,zmax — границы; gap — ширина ворот;
    # height — высота столбов.
    def _wire_perimeter(self, b: Builder, xmin: float, xmax: float,
                        zmin: float, zmax: float, gap: float,
                        height: float) -> None:
        # Проволочный периметр также разрывается на двух боковых сторонах под ворота.
        for x in (xmin, xmax):
            # Первая половина боковой стороны
            self._wire_fence_line(b, x, zmin, x, -gap * 0.5, height)
            # Вторая половина
            self._wire_fence_line(b, x, gap * 0.5, x, zmax, height)
            # Столбы ворот по краям разрыва
            self._gate_post(b, x, -gap * 0.5, height + 0.25)
            self._gate_post(b, x, gap * 0.5, height + 0.25)
        # Передняя сторона
        self._wire_fence_line(b, xmin, zmin, xmax, zmin, height)
        # Задняя сторона
        self._wire_fence_line(b, xmin, zmax, xmax, zmax, height)

    # Строит три вложенных забора вокруг станции и ворота
    # для проезда железнодорожного состава.
    # Параметры: b — Builder для накопления геометрии.
    def _build_fences(self, b: Builder) -> None:
        # Три забора вложены друг в друга; ширина ворот 5,6 м достаточна для поезда.
        gap = 5.6
        # Внутренний забор — деревянный со штакетником, средний — проволочный,
        # внешний — редкий деревянный без штакетника.
        self._wood_perimeter(b, -24.0, 24.0, -20.0, 18.0, gap, 2.15, True)
        self._wire_perimeter(b, -26.0, 26.0, -22.0, 20.0, gap, 2.60)
        self._wood_perimeter(b, -28.0, 28.0, -24.0, 22.0, gap, 2.05, False)

    # Строит окружение станции: деревья с кронами и кусты.
    # Параметры: b — Builder для накопления геометрии.
    def _build_environment(self, b: Builder) -> None:
        # tree_specs — вручную заданные крупные деревья: (x, z, масштаб).
        tree_specs = [
            (-40.0, -14.0, 1.15), (-46.0, 4.0, 1.35), (-42.0, 23.0, 1.05),
            (-38.0, -30.0, 1.20), (-28.0, -38.0, 1.00), (-12.0, -44.0, 1.20),
            (6.0, -47.0, 0.90), (24.0, -42.0, 1.10), (40.0, -32.0, 1.25),
            (47.0, -15.0, 1.00), (49.0, 5.0, 1.35), (45.0, 25.0, 1.10),
            (36.0, 36.0, 1.00), (18.0, 44.0, 1.20), (-2.0, 46.0, 0.90),
            (-22.0, 42.0, 1.15), (-38.0, 32.0, 1.00), (-56.0, 18.0, 1.25),
            (-60.0, -16.0, 1.10), (60.0, -24.0, 1.05), (58.0, 35.0, 1.25),
            (-30.0, -52.0, 1.00), (32.0, -54.0, 1.20), (66.0, 8.0, 0.90),
            (-66.0, 5.0, 1.05), (0.0, -62.0, 1.15), (14.0, -58.0, 0.90),
        ]
        # Фиксированный seed делает расположение деревьев и кустов одинаковым при каждом запуске.
        rng = random.Random(5)
        # Для каждого дерева строим ствол-цилиндр и три эллипсоида кроны.
        for i, (x, z, scale) in enumerate(tree_specs):
            # Высота ствола
            h = (2.4 + rng.random() * 1.6) * scale
            trunk = self.B(b, "tree_trunk")
            # Цилиндр-ствол
            trunk.cylinder((x, h * 0.5, z), 0.16 * scale + rng.random() * 0.04, h, "y", 8)
            # Крона
            foliage = self.B(b, "tree_a" if i % 2 else "tree_b")
            # 1. Большой сверху — основная масса листвы.
            foliage.ellipsoid((x, h + 0.55 * scale, z),
                              (1.25 * scale, 1.40 * scale, 1.15 * scale), 12, 8)
            # 2. Меньший слева-снизу
            foliage.ellipsoid((x - 0.55 * scale, h + 0.15 * scale, z + 0.20 * scale),
                              (0.82 * scale, 0.88 * scale, 0.80 * scale), 10, 7)
            # 3. Меньший справа-снизу
            foliage.ellipsoid((x + 0.60 * scale, h + 0.25 * scale, z - 0.25 * scale),
                              (0.78 * scale, 0.92 * scale, 0.85 * scale), 10, 7)
        # placed — сколько уже поставили
        placed = 0
        # Добавляем ещё 14 случайных деревьев, но не внутри станции и не на путях.
        while placed < 14:
            # Случайная точка в квадрате
            x = rng.uniform(-82.0, 82.0)
            z = rng.uniform(-68.0, 68.0)
            # Не ставим деревья в центре (станция и ближние окрестности)
            if -33.0 < x < 33.0 and -29.0 < z < 26.0:
                continue
            # Не ставим на путях (полоса z от -5 до 5 — рельсы)
            if abs(z) < 5.0:
                continue
            # Масштаб 0.75..1.4 — случайный размер
            scale = 0.75 + rng.random() * 0.65
            # Высота 2.2..3.9 м
            h = 2.2 + rng.random() * 1.7
            # Ствол
            trunk = self.B(b, "tree_trunk")
            # Крона из двух эллипсоидов
            trunk.cylinder((x, h * 0.5, z), 0.13 + rng.random() * 0.05, h, "y", 8)
            foliage = self.B(b, "tree_a" if placed % 2 else "tree_b")
            foliage.ellipsoid((x, h + 0.45, z), (1.05 * scale, 1.2 * scale, 1.0 * scale), 10, 7)
            foliage.ellipsoid((x + 0.45, h + 0.15, z - 0.25),
                              (0.75 * scale, 0.80 * scale, 0.72 * scale), 9, 6)
            placed += 1
        # Кусты расставляются случайно, но также обходят платформы и железнодорожный путь.
        bush_positions = []
        # Сначала собираем 48 подходящих точек в список
        while len(bush_positions) < 48:
            x = rng.uniform(-75.0, 75.0)
            z = rng.uniform(-62.0, 62.0)
            # Не на станции
            if -31.0 < x < 31.0 and -27.0 < z < 24.0:
                continue
            # Не на путях
            if abs(z) < 5.0:
                continue
            bush_positions.append((x, z))
        # Каждый куст — один или два сплюснутых эллипсоида.
        bush = self.B(b, "bush")
        for x, z in bush_positions:
            r = 0.45 + rng.random() * 0.55
            bush.ellipsoid((x, 0.32 * r + 0.10, z), (r, r * 0.55, r * 0.85), 9, 6)
            if rng.random() > 0.45:
                bush.ellipsoid((x + r * 0.8, 0.25 * r + 0.10, z + r * 0.35),
                               (r * 0.65, r * 0.38, r * 0.60), 8, 6)

    # Строит пассажирский вагон целиком: раму, пол, стены,
    # потолок, крышу, восемь купе, купе проводника, два туалета, двери,
    # тележки, ступени и внешние надписи.
    # Параметры: b — Builder для накопления геометрии.
    def _build_train(self, b: Builder) -> None:
        # blue и yellow — быстрый доступ к пакетам синего и жёлтого металла для нужной группы.
        blue = lambda grp="default": self.B(b, "train_blue", grp)
        yellow = lambda grp="default": self.B(b, "train_yellow", grp)
        # Пакеты, которые всегда в группе "default" (не скрываются в разрезе)
        dark = self.B(b, "dark_metal")
        steel = self.B(b, "steel")
        # Рама вагона — тёмный блок, на котором держится кузов.
        dark.box((0, 1.04, 0), (24.0, 0.22, 2.95))
        # Второй блок — основание пола; поверх него лежит текстура пола.
        dark.box((0, 1.22, 0), (23.4, 0.16, 2.88))
        # Пол — горизонтальный четырёхугольник, заходящий под стены, чтобы не было щелей.
        floor = self.B(b, "floor")
        floor.quad((-11.98, 1.334, -1.45), (-11.98, 1.334, 1.45),
                   (11.98, 1.334, 1.45), (11.98, 1.334, -1.45),
                   (0, 0), (0, 2), (8, 2), (8, 0))
        # Ковровая дорожка идёт вдоль коридора от одного тамбура до другого.
        carpet = self.B(b, "carpet")
        carpet.box((0, 1.337, 0.95), (22.5, 0.012, 0.76))
        # Торцевые стены закрывают вагон с двух концов; сверху идёт жёлтый пояс.
        for x, grp in ((-12.0, "default"), (12.0, "default")):
            blue(grp).box((x, 2.52, 0), (0.12, 2.40, 3.10))
            yellow(grp).box((x, 3.58, 0), (0.13, 0.13, 3.14))
        # Ближний борт входит в группу near и скрывается в режиме разреза; дальний остаётся.
        self._build_side_wall(b, +1, "near")
        self._build_side_wall(b, -1, "default")
        # Внутренняя обшивка обоями повторяет проёмы окон и дверей, не закрывая их.
        for side, grp in ((+1, "near"), (-1, "default")):
             # z = ±1.435 — плоскость обоев (чуть внутри от внешней стены z=±1.5).
            z = side * 1.435
            wall = self.B(b, "wallpaper", grp)
            windows, doors = self._wall_openings(side)
            # Объединяем окна и двери в один список проёмов для вырезания
            all_openings = [(a, b) for a, b, _ in windows] + doors
            # Ближняя стена
            if side > 0:
                # нижняя часть обоев: от y=1.32 до y=2.50 (центр 1.91,
                # высота 1.18). Режется на куски, исключая ДВЕРИ
                # (окна начинаются выше, здесь их нет).
                for a, bnd in self._segments_excluding(-11.95, 11.95, doors):
                    wall.box(((a + bnd) * 0.5, 1.91, z), (bnd - a, 1.18, 0.025))
                # верхняя часть обоев: от y=3.39 до y=3.63 (центр 3.51,
                # высота 0.24). Сплошная, потому что двери не доходят
                # до этой высоты.
                wall.box((0, 3.51, z), (23.9, 0.24, 0.025))
            else:
                # Нижняя часть обоев: целиком, без вырезов (дверей нет).
                wall.box((0, 1.91, z), (23.9, 1.18, 0.025))
                # Верхняя часть обоев: тоже целиком.
                wall.box((0, 3.51, z), (23.9, 0.24, 0.025))
            # Средняя часть обоев (на уровне окон)
            for a, bnd in self._segments_excluding(-11.95, 11.95, all_openings):
                wall.box(((a + bnd) * 0.5, 2.94, z), (bnd - a, 0.90, 0.025))
            # Узкие полоски обоев над каждой дверью
            for x0, x1 in doors:
                wall.box(((x0 + x1) * 0.5, 3.36, z), (x1 - x0, 0.10, 0.025))
        # Торцевые стены внутри также получают слой обоев.
        for x in (-11.91, 11.91):
            wall = self.B(b, "wallpaper")
            wall.box((x, 2.48, 0), (0.025, 2.36, 2.98))
        # Потолок закрывает вагон сверху; шесть ламп добавляют точечный свет в коридор.
        ceil = self.B(b, "ceiling")
        ceil.box((0, 3.585, 0), (24.24, 0.055, 3.30))
        # Каждая лампа — светящийся прямоугольник и точечный источник с радиусом 4,6 м.
        for x in (-8.5, -4.2, 0.0, 4.2, 8.5, 10.95):
            lamp = self.B(b, "lamp")
            lamp.box((x, 3.545, 0.98), (0.62, 0.055, 0.24))
            self.point_lights.append(((x, 3.32, 0.98), (1.25, 0.82, 0.47), 4.6))
        # Снаружи крыша имеет полуцилиндрическую форму с рёбрами жёсткости.
        self._build_roof(b)
        # xs — девять границ, образующих восемь купе шириной 2,1 м.
        xs = [-7.6 + i * 2.1 for i in range(9)]
        # Между купе ставятся поперечные перегородки, не доходящие до коридора.
        for x in xs:
            wall = self.B(b, "wallpaper")
            wall.box((x, 2.45, -0.45), (0.055, 2.26, 1.91))
        # Перегородка между левым туалетом и купе проводника.
        # Тоже обои, но чуть толще (6.5 см) — потому что нет двери
        wall = self.B(b, "wallpaper")
        wall.box((-9.10, 2.45, -0.45), (0.065, 2.26, 1.91))
        # Плинтусы идут по дальней стене, торцам и перегородкам, закрывая стык пола и стен.
        skirt = self.B(b, "skirting")
        # Плинтус вдоль дальних стен
        skirt.box((0, 1.370, -1.415), (23.90, 0.09, 0.04))
        # Плинтус вдоль ближней стены (z = +1.415) — с вырезом под
        # входные двери (там, где проём, плинтуса нет).
        for a, bnd in self._segments_excluding(-11.95, 11.95, self.STATION_DOORS):
            skirt.box(((a + bnd) * 0.5, 1.370, 1.415),
                      (bnd - a, 0.09, 0.04))
        # Плинтусы на торцевых стенах
        for x in (-11.90, 11.90):
            skirt.box((x, 1.370, 0.0), (0.04, 0.09, 2.86))
        # Плинтусы вдоль всех перегородок между купе (xs) и купе проводника и туалетом
        for x in xs + [-9.10]:
            skirt.box((x, 1.370, -0.45), (0.09, 0.09, 1.86))
        # Для каждого купе строится стена коридора с дверным проёмом по центру.
        for i in range(8):
            # Границы i-го купе и его центр
            x0, x1 = xs[i], xs[i + 1]
            xmid = (x0 + x1) * 0.5
            # Дверной проём по центру купе
            door_x0, door_x1 = xmid - 0.475, xmid + 0.475
            # Стена коридора с вырезом под дверь
            wall = self.B(b, "wallpaper", "near")
            for a, bnd in self._segments_excluding(x0, x1, [(door_x0, door_x1)]):
                wall.box(((a + bnd) * 0.5, 2.44, 0.50),
                         (bnd - a, 2.24, 0.055))
            # Наличник вокруг двери (3 блока) 
            # Левая стойка.
            doorframe = self.B(b, "wood_dark", "near")
            doorframe.box((door_x0 - 0.035, 2.44, 0.50), (0.09, 2.24, 0.07))
            # Правая стойка
            doorframe.box((door_x1 + 0.035, 2.44, 0.50), (0.09, 2.24, 0.07))
            # Верхняя перекладина (над дверью)
            doorframe.box((xmid, 3.48, 0.50),
                          (door_x1 - door_x0 + 0.12, 0.10, 0.07))
            # Дверь открыта, створки прижаты к стене слева и справа от
            # проёма, поэтому сам проём свободен
            comp_door = self.B(b, "wood_dark", "near")
            comp_handle = self.B(b, "chrome", "near")
            for a, bnd in self._segments_excluding(x0, x1, [(door_x0, door_x1)]):
                # Центр и ширина створки
                px = (a + bnd) * 0.5
                pw = max(0.30, bnd - a - 0.055)
                # Полотно створки
                comp_door.box((px, 2.335, 0.440), (pw, 2.05, 0.040))
                # Ручка: если створка левее центра — ручка у правого края
                # (у проёма), если правее — у левого.
                hx = door_x0 - 0.075 if px < xmid else door_x1 + 0.075
                comp_handle.box((hx, 2.18, 0.412), (0.05, 0.08, 0.035))
                comp_handle.box((hx, 1.62, 0.412), (0.12, 0.14, 0.035))
            # Плинтус вокруг стены коридора (с двух сторон)
            for a, bnd in self._segments_excluding(x0, x1, [(door_x0, door_x1)]):
                skirt.box(((a + bnd) * 0.5, 1.370, 0.475),
                          (bnd - a, 0.09, 0.035))
                skirt.box(((a + bnd) * 0.5, 1.370, 0.525),
                          (bnd - a, 0.09, 0.035))
            # Мебель внутри купе (полки, столик, лампы)
            self._build_compartment(b, x0, x1, i)
        # Купе проводника, два туалета и торцевые перегородки добавляются после купе.
        self._build_conductor(b)
        self._build_toilet(b, -9.80, -1)
        self._build_toilet(b, 9.90, +1)
        self._partition_with_door_x(b, -10.50, -1.40, 0.50, -0.50, 0.40)
        self._partition_with_door_x(b, 10.60, -1.40, 0.50, -0.50, 0.40)
        self._partition_with_door_z(b, -9.10, -7.60, 0.50, -8.85, -7.85)
        # Входные двери открыты наружу, к платформе ведут три ступени.
        self._build_doors(b)
        self._build_bogies(b)
        # Под вагоном размещены ящики оборудования; на крыше — вентиляционные короба.
        # Три ящика оборудования под рамой 
        for x, sx, sz in ((-4.2, 2.0, 1.05), (4.4, 2.4, 1.15), (9.4, 1.2, 0.9)):
            box = self.B(b, "dark_metal")
            box.box((x, 0.82, 0.0), (sx, 0.30, sz))
        # Три вентиляционных короба на крыше
        for x in (-5.4, 0.2, 5.7):
            vent = self.B(b, "train_roof")
            vent.box((x, 4.26, 0.0), (1.1, 0.16, 0.75))
        # Буферы и сцепки завершают оба торца вагона.
        # На каждом торце (x = ±12.15) — по 2 буфера
        for x in (-12.15, 12.15):
            for z in (-0.82, 0.82):
                # Буфер: цилиндр радиуса 20 см
                steel.cylinder((x, 0.89, z), 0.20, 0.24, "x", 14)
            # Сцепка: тёмный блок между буферами
            dark.box((x, 0.81, 0.0), (0.30, 0.26, 0.35))

    # Cтроит один борт вагона: синие и жёлтые полосы,
    # оконные проёмы, стёкла, рамы, занавески, дверные обрамления и надпись.
    # Параметры:
    # b — Builder; side — +1 для ближнего борта, -1 для дальнего;
    # group — группа видимости ("near" или "default").
    def _build_side_wall(self, b: Builder, side: int, group: str) -> None:
        blue = self.B(b, "train_blue", group)
        yellow = self.B(b, "train_yellow", group)
        frame = self.B(b, "dark_metal", group)
        inner_frame = self.B(b, "wood", group)
        wall_fill = self.B(b, "wallpaper", group)
        glass = self.B(b, "glass", group)
        # Для ближнего +1.50, для дальнего -1.50.
        # От знака side зависит, в какую сторону смотрят наружные детали (рамы, стёкла)
        zc = side * 1.50
        # Получаем список окон [(x0, x1, тип)] и дверей [(x0, x1)] для
        # этого борта. У дальнего дверей нет (пустой список).
        windows, doors = self._wall_openings(side)
        # Объединённый список проёмов (окна + двери) — для вырезания
        # верхней полосы, где есть и окна, и двери
        all_openings = [(x0, x1) for x0, x1, _ in windows] + doors

        # Синий низ и жёлтый пояс разбиваются на отрезки вокруг дверных проёмов.
        # Идёт по всей длине, КРОМЕ дверных проёмов (там вход в вагон)
        for a, bnd in self._segments_excluding(-12.0, 12.0, doors):
            blue.box(((a + bnd) * 0.5, 1.90, zc), (bnd - a, 1.17, 0.095))
        # Верхняя синяя полоса
        blue.box((0, 3.53, zc), (24.0, 0.30, 0.095))
        # Желтая полоса 
        for a, bnd in self._segments_excluding(-12.0, 12.0, doors):
            yellow.box(((a + bnd) * 0.5, 2.465, zc), (bnd - a, 0.105, 0.105))
        # Верхняя синяя полоса вырезается и вокруг окон, и вокруг дверей.
        for a, bnd in self._segments_excluding(-12.0, 12.0, all_openings):
            blue.box(((a + bnd) * 0.5, 2.93, zc), (bnd - a, 1.00, 0.095))

        # wy0..wy1 — вертикальные границы проёма окна.
        wy0, wy1 = 2.64, 3.36
        # Проходим по всем окнам. kind = "coupe" или "conductor",
        for x0, x1, kind in windows:
            # Ширина окна и его центр по X.
            w = x1 - x0
            mid = (x0 + x1) * 0.5
            # Заполнение ПОД окном (от 2.485 до 2.64) 
            # Высота заполнения и его центр.
            fill_h = wy0 - 2.485
            # Синяя панель снаружи
            blue.box((mid, 2.485 + fill_h * 0.5, zc), (w, fill_h, 0.095))
            # Обои внутри
            wall_fill.box((mid, 2.485 + fill_h * 0.5, zc - side * 0.065),
                          (w, fill_h, 0.025))
            # Заполнение НАД окном
            blue.box((mid, (wy1 + 3.43) * 0.5, zc), (w, 3.43 - wy1, 0.095))
            wall_fill.box((mid, (wy1 + 3.43) * 0.5, zc - side * 0.065),
                          (w, 3.43 - wy1, 0.025))
            # Наружная тёмная рамка обрамляет окно с внешней стороны борта.
            # Верхняя перекладина
            frame.box((mid, 2.60, zc + side * 0.055), (w + 0.12, 0.10, 0.04))
            # Нижняя перекладина
            frame.box((mid, 3.40, zc + side * 0.055), (w + 0.12, 0.08, 0.04))
            # Левая стойка
            frame.box((x0 - 0.035, 3.005, zc + side * 0.055), (0.07, 0.86, 0.04))
            # Правая стойка
            frame.box((x1 + 0.035, 3.005, zc + side * 0.055), (0.07, 0.86, 0.04))
            # Внутренняя деревянная рамка
            iz = zc - side * 0.068
            rz = zc - side * 0.020
            # Верхняя и нижняя перекладины внутренней рамы.
            inner_frame.box((mid, 3.40, iz), (w + 0.18, 0.09, 0.045))
            inner_frame.box((mid, 2.60, iz), (w + 0.18, 0.08, 0.045))
            # Левая и правая стойки внутренней рамы.
            inner_frame.box((x0 - 0.015, 3.005, iz), (0.12, 0.82, 0.045))
            inner_frame.box((x1 + 0.015, 3.005, iz), (0.12, 0.82, 0.045))
            # Откосы (боковые грани проёма, уходящие вглубь стены).
            inner_frame.box((x0, 3.005, rz), (0.065, 0.76, 0.14))
            inner_frame.box((x1, 3.005, rz), (0.065, 0.76, 0.14))
            inner_frame.box((mid, 3.375, rz), (w, 0.065, 0.14))
            inner_frame.box((mid, 2.635, rz), (w, 0.065, 0.14))
            # Стекло 
            gz = zc + side * 0.070
            gy0, gy1 = 2.66, 3.36 # границы стекла по высоте
            # Порядок вершин разный для ближнего и дальнего борта, чтобы
            # нормаль стекла смотрела наружу
            if side > 0:
                glass.quad((x0, gy0, gz), (x1, gy0, gz), (x1, gy1, gz), (x0, gy1, gz))
            else:
                glass.quad((x1, gy0, gz), (x0, gy0, gz), (x0, gy1, gz), (x1, gy1, gz))
            # Две красные занавески висят по краям каждого окна.
            cur = self.B(b, "curtain", group)
            izc = zc - side * 0.075
            cur.box((x0 + 0.14, 3.02, izc), (0.22, 0.80, 0.035)) # левая
            cur.box((x1 - 0.14, 3.02, izc), (0.22, 0.80, 0.035)) # правая
            # Под окном закреплена деревянная полка.
            shelf = self.B(b, "wood", group)
            shelf.box((mid, 2.60, izc), (w + 0.18, 0.055, 0.20))

        # Дверные проёмы (ближний борт) получают металлическое обрамление, порог и деревянные откосы.
        for x0, x1 in doors:
            # Наружное металлическое обрамление двери 
            # Левая стойка, сдвинута наружу.
            frame.box((x0 - 0.02, 2.325, zc + side * 0.06), (0.08, 2.01, 0.04))
            # Правая стойка
            frame.box((x1 + 0.02, 2.325, zc + side * 0.06), (0.08, 2.01, 0.04))
            # Верхняя перекладина
            frame.box(((x0 + x1) * 0.5, 3.31, zc + side * 0.06), (x1 - x0 + 0.14, 0.09, 0.04))
            # Порог двери
            # Тёмный металлический блок
            dark = self.B(b, "dark_metal", group)
            dark.box(((x0 + x1) * 0.5, 1.315, zc + side * 0.055), (x1 - x0, 0.035, 0.13))
            # Внутренние деревянные откосы двери 
            jamb = self.B(b, "wood", group)
            jz = zc - side * 0.015
            jamb.box((x0, 2.325, jz), (0.08, 2.01, 0.20)) # левый
            jamb.box((x1, 2.325, jz), (0.08, 2.01, 0.20)) # правый
            jamb.box(((x0 + x1) * 0.5, 3.32, jz), (x1 - x0 + 0.16, 0.08, 0.20)) # верхний

        # Надпись маршрута рисуется на внешней стороне обоих бортов.
        # Текстура "label" (создана в _make_textures через make_train_label,
        # там же герб). Клеится как quad на внешнюю сторону борта.
        label = self.B(b, "label", group)
        # lz — плоскость надписи, чуть снаружи стены 
        lz = side * 1.556
        # Порядок вершин разный для ближнего и дальнего борта, чтобы
        # надпись не была зеркальной и нормаль смотрела наружу
        if side > 0:
            label.quad((-6.6, 1.42, lz), (6.6, 1.42, lz), (6.6, 2.38, lz), (-6.6, 2.38, lz),
                       (0, 0), (1, 0), (1, 1), (0, 1))
        else:
            label.quad((6.6, 1.42, lz), (-6.6, 1.42, lz), (-6.6, 2.38, lz), (6.6, 2.38, lz),
                       (0, 0), (1, 0), (1, 1), (0, 1))
        # Тонкая желтая полоса под надписью
        for a, bnd in self._segments_excluding(-12.0, 12.0, doors):
            # С вырезом под двери, снаружи борта 
            yellow.box(((a + bnd) * 0.5, 1.375, zc + side * 0.052), (bnd - a, 0.035, 0.02))

    # Строит мебель одного купе: две нижние и две верхние
    # полки, матрасы, бельё, одеяла, подушки, столик и светильники.
    # Параметры:
    # b — Builder; x0,x1 — границы купе по длине вагона;
    # index — номер купе, по нему выбирается цвет одеяла.
    def _build_compartment(self, b: Builder, x0: float, x1: float, index: int) -> None:
        # cx — центр купе; index выбирает цвет одеяла из трёх вариантов.
        cx = (x0 + x1) * 0.5
        # Две койки стоят слева и справа от центра; у каждой нижняя и верхняя полки.
        for sx in (-0.67, 0.67):
            # bx — центр этой койки по X
            bx = cx + sx
            # Каркас полки — деревянный блок; матрас, простыня, одеяло и подушка лежат сверху.
            frame = self.B(b, "wood")
            # Горизонтальная доска каркаса
            frame.box((bx, 1.49, -0.70), (0.76, 0.14, 1.32))
            # Вертикальная «спинка» каркаса у окна. Закрывает торец
            # полки со стороны окна, чтобы матрас не «съезжал» в окно.
            frame.box((bx, 1.40, -1.28), (0.76, 0.24, 0.08))
            # Матрас — лежит на каркасе
            mattress = self.B(b, "mattress")
            mattress.box((bx, 1.635, -0.70), (0.72, 0.15, 1.22))
            # Простыня — тонкий слой поверх матраса
            sheet = self.B(b, "sheet")
            sheet.box((bx, 1.723, -0.55), (0.73, 0.045, 0.88))
            # Одеяло — поверх простыни. Цвет выбирается по номеру купе
            blanket = self.B(b, ("blanket_blue", "blanket_teal", "blanket_red")[index % 3])
            blanket.box((bx, 1.755, -0.60), (0.75, 0.055, 0.92))
            # Подушка 
            pillow = self.B(b, "pillow")
            pillow.box((bx, 1.79, -1.16), (0.58, 0.11, 0.25))
            # Верхняя полка повторяет нижнюю, но поднята на высоту 2,37 м.
            frame.box((bx, 2.37, -0.70), (0.76, 0.12, 1.32))
            mattress.box((bx, 2.50, -0.70), (0.72, 0.14, 1.22))
            sheet.box((bx, 2.585, -0.55), (0.73, 0.04, 0.88))
            blanket.box((bx, 2.615, -0.60), (0.75, 0.05, 0.92))
            pillow.box((bx, 2.64, -1.16), (0.58, 0.10, 0.25))
        # Столик стоит у окна между нижними полками и не перекрывает проход.
        table_top = self.B(b, "table_wood")
        # Столешница
        table_top.box((cx, 1.90, -1.14), (0.44, 0.055, 0.44))
        # Вертикальная опора столика
        table_top.box((cx, 1.76, -1.14), (0.07, 0.24, 0.30))
        # Нижняя полка столика
        table_top.box((cx, 1.70, -1.14), (0.30, 0.04, 0.32))
        # Над каждой верхней полкой висит потолочный светильник, у нижней — маленькая лампа.
        lamp = self.B(b, "lamp")
        base = self.B(b, "dark_metal")
        for lx in (cx - 0.67, cx + 0.67):
            # Потолочный светильник над верхней полкой
            lamp.ellipsoid((lx, 3.50, -0.55), (0.12, 0.05, 0.10), 8, 6)
            # Основание плафона — тёмный металлический блок
            base.box((lx, 3.555, -0.55), (0.20, 0.030, 0.12))
            # Маленькая лампа у нижней полки
            lamp.ellipsoid((lx, 2.14, -1.30), (0.08, 0.08, 0.06), 8, 6)

    # Cтроит перегородку поперёк оси X с дверным
    # проёмом, наличниками, порогом и плинтусами.
    # Параметры:
    # b — Builder; x — плоскость перегородки;
    # z0,z1 — границы по ширине вагона; door_z0,door_z1 — границы проёма.
    # Возвращает: None.
    # Строит внутреннюю поперёк оси X (то есть плоскость
    # стены перпендикулярна оси X, стена идёт по Z). В перегородке
    # есть дверной проём с дверью, наличниками, порогом и плинтусами.
    # Используется для тамбуров и туалетов.
    def _partition_with_door_x(self, b: Builder, x: float,
                               z0: float, z1: float,
                               door_z0: float, door_z1: float) -> None:
        # Стена строится двумя отрезками по ширине вагона, между ними остаётся дверной проём.
        wall = self.B(b, "wallpaper")
        for a, bnd in self._segments_excluding(z0, z1, [(door_z0, door_z1)]):
            wall.box((x, 2.45, (a + bnd) * 0.5), (0.065, 2.26, bnd - a))
        # Верхняя перемычка НАД дверным проёмом: от верха двери (3.14) до потолка (3.58).
        wall.box((x, 3.36, (door_z0 + door_z1) * 0.5),
                 (0.065, 0.44, door_z1 - door_z0))
        # Ширина створки чуть меньше проёма; side задаёт сторону, куда она открывается.
        width = door_z1 - door_z0 - 0.02
        side = -1 if x < 0 else 1
        leaf_z0, leaf_z1 = door_z0 - 0.02 - width, door_z0 - 0.02
        # Сама створка повёрнута к тамбуру и не закрывает проход.
        leaf = self.B(b, "wood_dark")
        # Сама створка: 4.5 см толщиной, 1.80 м высотой (центр 2.22),
        # шириной width по Z.
        leaf.box((x + side * 0.035, 2.22, (leaf_z0 + leaf_z1) * 0.5),
                 (0.045, 1.80, width))
        # Плинтусы ставятся с обеих сторон перегородки.
        skirt = self.B(b, "skirting")
        for a, bnd in self._segments_excluding(z0, z1, [(door_z0, door_z1)]):
            skirt.box((x - 0.030, 1.370, (a + bnd) * 0.5),
                      (0.035, 0.09, bnd - a))
            skirt.box((x + 0.030, 1.370, (a + bnd) * 0.5),
                      (0.035, 0.09, bnd - a))
        # Наличники обрамляют проём сверху и по бокам.
        frame = self.B(b, "wood_dark")
        frame.box((x, 3.135, door_z0 - 0.025), (0.075, 0.06, 0.05)) # Левая стойка наличника
        frame.box((x, 3.135, door_z1 + 0.025), (0.075, 0.06, 0.05)) # Правая стойка наличника
        frame.box((x, 1.325, (door_z0 + door_z1) * 0.5), # Нижняя планка
                  (0.075, 0.04, door_z1 - door_z0 + 0.10))
        # Ручка и тёмная нижняя накладка завершают дверь.
        chrome = self.B(b, "chrome")
        chrome.box((x + side * 0.085, 2.18, leaf_z1 - 0.10),
                   (0.035, 0.08, 0.07))
        dark = self.B(b, "dark_metal")
        dark.box((x + side * 0.062, 1.66, (leaf_z0 + leaf_z1) * 0.5),
                 (0.02, 0.26, width - 0.04))

    # Строит перегородку поперёк оси Z (стену купе
    # проводника со стороны коридора) с открытой створкой и наличниками.
    # Параметры:
    # b — Builder для геометрии;
    # x0, x1 — границы стены по длине (по X);
    # z — плоскость стены (Z-координата, где стоит стена);
    # door_x0, door_x1 — границы дверного проёма по X.
    def _partition_with_door_z(self, b: Builder, x0: float, x1: float,
                               z: float, door_x0: float, door_x1: float) -> None:
        # Стена купе проводника строится вдоль оси Z с проёмом со стороны коридора.
        wall = self.B(b, "wallpaper")
        for a, bnd in self._segments_excluding(x0, x1, [(door_x0, door_x1)]):
            wall.box(((a + bnd) * 0.5, 2.45, z), (bnd - a, 2.26, 0.055))
        # Верхняя перемычка НАД дверным проёмом
        wall.box(((door_x0 + door_x1) * 0.5, 3.36, z),
                 (door_x1 - door_x0, 0.44, 0.055))
        # Створка открыта внутрь комнаты и повёрнута у правого наличника.
        width = door_x1 - door_x0 - 0.02
        # Дверь крепится к правой стойке проёма.
        leaf = self.B(b, "wood_dark")
        leaf.box((door_x1 - 0.03, 2.22, z - width * 0.5),
                 (0.045, 1.80, width))
        # Плинтусы идут по обеим сторонам перегородки.
        skirt = self.B(b, "skirting")
        for a, bnd in self._segments_excluding(x0, x1, [(door_x0, door_x1)]):
            skirt.box(((a + bnd) * 0.5, 1.370, z - 0.025),
                      (bnd - a, 0.09, 0.035))
            skirt.box(((a + bnd) * 0.5, 1.370, z + 0.025),
                      (bnd - a, 0.09, 0.035))
        # Наличники окрашены как обои, чтобы слева не было коричневой полосы.
        frame = self.B(b, "wallpaper")
        frame.box((door_x0 - 0.025, 3.135, z), (0.05, 0.06, 0.075))
        frame.box((door_x1 + 0.025, 3.135, z), (0.05, 0.06, 0.075))
        frame.box(((door_x0 + door_x1) * 0.5, 1.325, z),
                  (door_x1 - door_x0 + 0.10, 0.04, 0.075))
        # Ручка и нижняя накладка завершают створку.
        chrome = self.B(b, "chrome")
        chrome.box((door_x1 - 0.075, 2.18, z - width + 0.13),
                   (0.035, 0.08, 0.07))
        dark = self.B(b, "dark_metal")
        dark.box((door_x1 - 0.03, 1.66, z - width * 0.5),
                 (0.02, 0.26, width - 0.05))

    # Строит купе проводника: нижнюю полку, стол, стены,
    # дверь и потолочный светильник.
    # Параметры: b — Builder для накопления геометрии.
    def _build_conductor(self, b: Builder) -> None:
        # В купе проводника одна нижняя полка у дальней стены.
        frame = self.B(b, "wood")
        frame.box((-8.62, 1.49, -0.55), (0.72, 0.14, 1.68))
        mattress = self.B(b, "mattress")
        mattress.box((-8.62, 1.635, -0.55), (0.70, 0.15, 1.58))
        sheet = self.B(b, "sheet")
        sheet.box((-8.62, 1.723, -0.38), (0.71, 0.045, 0.94))
        blanket = self.B(b, "blanket_red")
        blanket.box((-8.62, 1.755, -0.24), (0.70, 0.055, 0.90))
        pillow = self.B(b, "pillow")
        pillow.box((-8.62, 1.79, -1.15), (0.58, 0.11, 0.27))
        # Небольшой стол стоит у окна
        table = self.B(b, "table_wood")
        table.box((-7.96, 1.86, -0.92), (0.38, 0.055, 0.62))
        table.box((-7.96, 1.70, -0.92), (0.06, 0.28, 0.50))
        # Потолочный светильник и тёмное основание освещают купе.
        lamp = self.B(b, "lamp")
        lamp.box((-8.35, 3.52, -0.55), (0.52, 0.045, 0.24))
        base = self.B(b, "dark_metal")
        base.box((-8.35, 3.555, -0.55), (0.60, 0.030, 0.30))

    # Строит туалет: унитаз, бачок, раковину,
    # смеситель, зеркало в хромированной раме и холодный светильник.
    # Параметры:
    # b — Builder; cx — центр туалета по длине вагона;
    # direction — +1 или -1, зеркально располагает сантехнику.
    def _build_toilet(self, b: Builder, cx: float, direction: int) -> None:
       # Стена отделяет туалет от коридора. Идёт вдоль X (по длине
       # вагона), на z = 0.50. Ширина 1.42 м (чуть уже туалета),
       # высота 2.24 м, толщина 5.5 см.
        wall = self.B(b, "wallpaper")
        wall.box((cx, 2.45, 0.50), (1.42, 2.24, 0.055))
        # Плинтусы (с двух сторон)
        skirt = self.B(b, "skirting")
        skirt.box((cx, 1.370, 0.475), (1.42, 0.09, 0.035))
        skirt.box((cx, 1.370, 0.525), (1.42, 0.09, 0.035))
        # Унитаз: цилиндрическая чаша, эллипсоидное сиденье и прямоугольный бачок.
        toilet = self.B(b, "porcelain")
        toilet.cylinder((cx + 0.25 * direction, 1.53, -0.90), 0.22, 0.35, "y", 18)
        toilet.ellipsoid((cx + 0.25 * direction, 1.73, -0.87), (0.24, 0.08, 0.29), 16, 8)
        toilet.box((cx + 0.25 * direction, 1.93, -1.22), (0.48, 0.42, 0.20))
        toilet.box((cx + 0.25 * direction, 1.72, -1.12), (0.38, 0.08, 0.28))
        dark = self.B(b, "dark_metal")
        dark.cylinder((cx + 0.25 * direction, 2.155, -1.22), 0.035, 0.03, "y", 8)
        # Раковина использует полую чашу basin, а не закрытый цилиндр.
        sink = self.B(b, "porcelain")
        # Позиция раковины по X — сдвинута в ПРОТИВОПОЛОЖНУЮ сторону
        # от унитаза (унитаз на +0.25, раковина на -0.32). Так они
        # не наезжают друг на друга
        sink_x = cx - 0.32 * direction
        # Сама чаша раковины 
        # basin рисует полую чашу (а не сплошной цилиндр): внешняя
        # стенка + обод + наклонная внутренняя стенка + дно.
        sink.basin((sink_x, 2.14, -1.00), 0.26, 0.19, 0.10, 0.16, 24)
        # Ножка раковины 
        sink.box((sink_x, 1.75, -1.00), (0.16, 0.44, 0.15))
        # Сливное отверстие
        drain = self.B(b, "dark_metal")
        drain.cylinder((sink_x, 1.985, -1.00), 0.026, 0.012, "y", 10)
        # Кран
        chrome = self.B(b, "chrome")
        chrome.cylinder((sink_x, 2.27, -1.14), 0.018, 0.26, "y", 8)
        chrome.cylinder((sink_x, 2.38, -1.07), 0.018, 0.18, "z", 8)
        # Зеркало закреплено на дальней стене ровно над раковиной.
        mirror = self.B(b, "mirror")
        mx0, mx1 = sink_x - 0.34, sink_x + 0.34
        mz = -1.418
        mirror.quad((mx0, 2.34, mz), (mx1, 2.34, mz),
                    (mx1, 3.10, mz), (mx0, 3.10, mz),
                    (0, 0), (1, 0), (1, 1), (0, 1))
        # Хромированная рамка вокруг зеркала
        fr = self.B(b, "chrome")
        fr.box((sink_x, 3.13, mz), (0.74, 0.035, 0.035))
        fr.box((sink_x, 2.31, mz), (0.74, 0.035, 0.035))
        fr.box((mx0 - 0.02, 2.72, mz), (0.035, 0.80, 0.035))
        fr.box((mx1 + 0.02, 2.72, mz), (0.035, 0.80, 0.035))
        # Светильник добавляет точечный источник света в туалет.
        lamp = self.B(b, "lamp_cool")
        lamp.box((cx, 3.47, -0.15), (0.50, 0.06, 0.24))
        self.point_lights.append(((cx, 3.28, -0.15), (0.95, 0.98, 1.0), 2.4))

    # Cтроит две открытые входные двери вагона, их деревянные
    # створки, ручки, обрамления и три ступени к платформе.
    # Параметры: b — Builder; геометрия добавляется в группу "near".
    def _build_doors(self, b: Builder) -> None:
        blue = self.B(b, "train_blue", "near")
        yellow = self.B(b, "train_yellow", "near")
        dark = self.B(b, "dark_metal", "near")
        chrome = self.B(b, "chrome", "near")
        glass = self.B(b, "glass_dark", "near")
        stepmat = self.B(b, "platform", "near")

        # Две двери
        for index, (x0, x1) in enumerate(self.STATION_DOORS):
            # cx — центр проёма
            cx = (x0 + x1) * 0.5
            # hinge_x — где находится петля двери:
            # для левой двери (index=0) петля слева (x0);
            # для правой двери (index=1) петля справа (x1).
            # Двери открыты НАРУЖУ, поэтому петля с внешней стороны.
            hinge_x = x0 if index == 0 else x1
            outside = -1 if index == 0 else 1
            # Створка повёрнута наружу и целиком сделана из дерева.
            leaf_core = self.B(b, "wood_dark", "near")
            leaf_core.box((hinge_x, 2.335, 2.045), (0.075, 2.01, 0.92))
            # Деревянные накладки закрывают торцы и внутреннюю сторону створки.
            leaf_trim = self.B(b, "wood", "near")
            leaf_trim.box((hinge_x, 3.325, 2.045), (0.095, 0.065, 0.94))
            leaf_trim.box((hinge_x, 1.365, 2.045), (0.095, 0.065, 0.94))
            leaf_trim.box((hinge_x, 2.34, 2.505), (0.095, 1.99, 0.06))
            chrome.box((hinge_x + outside * 0.07, 2.15, 2.38), (0.035, 0.07, 0.11))
            # Внутренняя ручка
            inner_x = hinge_x - outside * 0.045
            chrome.box((inner_x - outside * 0.02, 2.15, 2.36), (0.03, 0.07, 0.10))
            # Три ступени с уменьшающейся высотой образуют спуск от двери к платформе.
            for j, top in enumerate((1.285, 1.220, 1.158)):
                z = 1.61 + j * 0.145
                depth = 0.18
                # Сама ступень
                stepmat.box((cx, top - 0.05, z), (1.02, 0.10, depth))
                # Жёлтая полоса 
                yellow.box((cx, top + 0.006, z + 0.075), (0.96, 0.018, 0.035))
                 # Тёмные боковины ступени (слева и справа)
                dark.box((cx - 0.52, top - 0.09, z), (0.04, 0.18, depth))
                dark.box((cx + 0.52, top - 0.09, z), (0.04, 0.18, depth))

    # Строит две тележки вагона: рамы, оси, колёсные пары,
    # диски, буксы, пружины и коробки подвески.
    # Параметры: b — Builder для накопления геометрии.
    def _build_bogies(self, b: Builder) -> None:
        steel = self.B(b, "steel")
        dark = self.B(b, "dark_metal")
        # Две тележки стоят симметрично относительно центра вагона.
        for bc in (-7.7, 7.7):
            dark.box((bc, 0.78, 0.0), (2.9, 0.44, 1.55))
            dark.box((bc, 0.98, 0.0), (2.4, 0.16, 1.75))
            # В каждой тележке две оси; на каждой оси по два колеса.
            for ax in (-0.92, 0.92):
                x = bc + ax
                steel.cylinder((x, 0.78, 0.0), 0.10, 1.60, "z", 12)
                for z in (-0.77, 0.77):
                    steel.cylinder((x, 0.78, z), 0.46, 0.14, "z", 22)
                    dark.cylinder((x, 0.78, z + (0.075 if z > 0 else -0.075)), 0.27, 0.035, "z", 18)
                    steel.cylinder((x, 0.78, z), 0.10, 0.17, "z", 12)
                for z in (-0.62, 0.62):
                    dark.box((x, 0.72, z), (0.14, 0.52, 0.10))
            # Пружины и коробки подвески соединяют тележку с рамой вагона.
            for z in (-0.55, 0.55):
                steel.cylinder((bc, 0.96, z), 0.12, 0.24, "y", 10)
                dark.box((bc, 1.12, z), (0.38, 0.13, 0.30))

    # Строит полуцилиндрическую крышу вагона, торцевые заглушки
    # и продольные рёбра жёсткости.
    # Параметры: b — Builder для накопления геометрии.
    def _build_roof(self, b: Builder) -> None:
        # Пакет материала крыши
        roof = self.B(b, "train_roof")
        # half_w — половина ширины крыши, rise — высота подъёма дуги над стеной.
        half_w, rise = 1.66, 0.58
        # yc — высота, с которой начинается арка (уровень стен).
        yc = 3.66
        # yc — высота, с которой начинается арка (уровень стен).
        x0, x1 = -12.12, 12.12
        # segs — на сколько сегментов разбить дугу
        segs = 18
        # Боковая поверхность крыши (полуэллипс)
        # Разбиваем дугу на segs кусочков. Каждый кусочек — это
        # «полоска» крыши от одного угла a0 до соседнего a1.
        for i in range(segs):
            # Два соседних угла на дуге.
            a0 = math.pi * i / segs
            a1 = math.pi * (i + 1) / segs
            # Точки на дуге в плоскости YZ
            z0, y0 = math.cos(a0) * half_w, yc + math.sin(a0) * rise
            z1, y1 = math.cos(a1) * half_w, yc + math.sin(a1) * rise
            p0 = (x0, y0, z0)
            p1 = (x1, y0, z0)
            p2 = (x1, y1, z1)
            p3 = (x0, y1, z1)
            roof.quad(p0, p1, p2, p3)
        # Торцевые заглушки используют разный порядок вершин, чтобы нормали смотрели наружу.
        for end_index, x in enumerate((x0, x1)):
            for i in range(segs):
                a0 = math.pi * i / segs
                a1 = math.pi * (i + 1) / segs
                # Центр заглушки — на уровне стен (yc), посередине
                # по Z (z = 0)
                p0 = (x, yc, 0.0)
                # Две соседние точки на дуге (на этом торце x)
                p1 = (x, yc + math.sin(a0) * rise, math.cos(a0) * half_w)
                p2 = (x, yc + math.sin(a1) * rise, math.cos(a1) * half_w)
                # Порядок вершин разный для двух торцов, чтобы нормали
                # смотрели НАРУЖУ от крыши
                if end_index == 0:
                    roof.tri(p0, p1, p2)
                else:
                    roof.tri(p0, p2, p1)
        # Продольные рёбра жёсткости идут по крыше и по её краям.
        rib = self.B(b, "steel")
        # 9 поперечных рёбер
        for x in (-10.5, -8.0, -5.5, -3.0, -0.5, 2.0, 4.5, 7.0, 9.5):
            rib.box((x, 4.20, 0.0), (0.08, 0.035, 2.4))
        # 2 длинных ребра по краям
        for z in (-1.63, 1.63):
            rib.box((0, 3.73, z), (24.3, 0.06, 0.07))

    # Cтроит сферу неба вокруг сцены и сохраняет её текстуру и Mesh.
    # Параметры: self — объект Scene с уже созданной текстурой self.tex["sky"].
    def _build_sky(self) -> None:
        self.sky_texture = self.tex["sky"]
        # Сфера неба делится на 48 сегментов по долготе и 28 колец по широте.
        segments, rings = 48, 28
        verts: list[float] = []
        inds: list[int] = []
        radius = 180.0 * WORLD_SCALE
        # theta — широта, phi — долгота; UV совпадают с панорамной текстурой.
        for r in range(rings + 1):
            theta = math.pi * r / rings
            st, ct = math.sin(theta), math.cos(theta)
            for s in range(segments + 1):
                phi = math.tau * s / segments
                x = radius * st * math.cos(phi)
                y = radius * ct
                z = radius * st * math.sin(phi)
                verts.extend((x, y, z, -x, -y, -z, s / segments, 1.0 - r / rings))
        # Соседние кольца соединяются в два треугольника на каждую ячейку.
        for r in range(rings):
            for s in range(segments):
                i0 = r * (segments + 1) + s
                i1 = i0 + 1
                i2 = i0 + segments + 1
                i3 = i2 + 1
                inds.extend((i0, i2, i1, i1, i2, i3))
        # Готовая сфера превращается в Mesh и будет рисоваться первой, без теста глубины.
        self.sky_mesh = Mesh(verts, inds)


PRESETS = {
    "1": ("Общий вид", dict(target=(-1.0, 2.3, 1.5), yaw=125.0, pitch=12.0, dist=24.0, cutaway=False)),
    "2": ("Вагон сбоку", dict(target=(0.0, 2.4, 0.0), yaw=180.0, pitch=4.0, dist=19.5, cutaway=False)),
    "3": ("Разрез вагона", dict(target=(0.5, 2.15, -0.2), yaw=30.0, pitch=16.0, dist=16.5, cutaway=True)),
    "4": ("Купе", dict(target=(-2.35, 1.95, -1.05), yaw=0.0, pitch=15.0, dist=1.45, fov=68.0, cutaway=False)),
    "5": ("КПП Локоть", dict(target=(-8.0, 4.60, 8.0), yaw=215.0, pitch=1.5, dist=9.1, cutaway=False)),
    "6": ("Туалет", dict(target=(9.58, 2.15, -1.0), yaw=0.0, pitch=8.0, dist=1.35, fov=78.0, cutaway=False)),
    "7": ("Купе проводника", dict(target=(-8.5, 1.9, -0.7), yaw=0.0, pitch=14.0, dist=2.0, fov=65.0, cutaway=False)),
    "8": ("Заборы и проезд", dict(target=(0.0, 2.2, 0.0), yaw=120.0, pitch=22.0, dist=55.0, cutaway=False)),
    "9": ("Открытые двери", dict(target=(10.9, 1.9, 1.2), yaw=-29.0, pitch=6.0, dist=5.0, cutaway=False)),
    "0": ("Коридор вагона", dict(target=(-5.0, 2.0, 1.0), yaw=90.0, pitch=2.0, dist=14.0, cutaway=False)),
}


# Camera — свободная камера с обзором мышью и перемещением по шести направлениям.
# Хранит position, yaw (поворот), pitch (наклон), fov (угол обзора) и speed.
# Камера не привязана к объекту: она может летать вокруг станции и заходить внутрь вагона.
class Camera:

    # Cоздаёт камеру и сразу ставит её в общий ракурс «1».
    # Параметры: self — будущий объект Camera.
    def __init__(self):
        # position — точка, из которой смотрит камера; она будет заменена ракурсом «1».
        self.position = [18.2, 7.3, -12.0]
        # yaw — поворот вокруг вертикальной оси, pitch — наклон вверх/вниз.
        self.yaw = 305.0
        self.pitch = -11.0
        # fov — угол обзора по вертикали, speed — базовая скорость полёта.
        self.fov = 43.0
        self.speed = 10.0
        # auto_rotate — флаг автоматического вращения вокруг текущей точки.
        self.auto_rotate = False
        # При создании сразу применяем общий ракурс, чтобы камера не начинала с пустоты.
        self.snap_preset("1")

    # inside_train — проверяет, находится ли камера внутри вагона.
    # Параметры: self — текущая камера.
    # Возвращает: True, если точка находится внутри увеличенного корпуса вагона.
    def inside_train(self) -> bool:
        # Берём мировые координаты и сравниваем их с размерами корпуса.
        x, y, z = self.position
        # abs(x) ограничивает длину, abs(z) — ширину, y — высоту помещения.
        return (abs(x) < 12.4 * WORLD_SCALE
                and abs(z) < 1.95 * WORLD_SCALE
                and 0.2 < y < 4.55 * WORLD_SCALE)

    # near_train — проверяет, находится ли камера рядом с вагоном или на платформе.
    # Параметры: self — текущая камера.
    # Возвращает: True, если камера входит в зону замедления.
    def near_train(self) -> bool:
        # Зона рядом с вагоном шире корпуса: в неё попадает платформа и подходы.
        x, y, z = self.position
        # Если камера ниже 6,5WORLD_SCALE и близко по X/Z, включается замедление.
        return (abs(x) < 15.0 * WORLD_SCALE
                and abs(z) < 10.0 * WORLD_SCALE
                and y < 6.5 * WORLD_SCALE)

    # navigation_factor — возвращает множитель скорости для текущего положения.
    # Параметры: self — текущая камера.
    # Возвращает: 0.25 внутри вагона, 0.45 рядом с ним и 1.0 снаружи.
    def navigation_factor(self) -> float:
        # Внутри вагона шаг уменьшается до четверти, чтобы не проскакивать двери.
        if self.inside_train():
            return 0.25
        # Рядом с вагоном скорость примерно вдвое ниже, чем на открытой станции.
        if self.near_train():
            return 0.45
        # Снаружи камера движется с полной базовой скоростью.
        return 1.0

    # snap_preset — ставит камеру в готовый ракурс из словаря PRESETS.
    # Параметры:
    # key — строка "1"…"9" или "0"; значение должно быть ключом PRESETS.
    # Возвращает: None. Меняет position, yaw, pitch и fov камеры.
    def snap_preset(self, key: str) -> None:
        # p — словарь с target, yaw, pitch, dist и, возможно, fov.
        _, p = PRESETS[key]
        # Точка взгляда из PRESETS задана в локальных единицах и умножается на WORLD_SCALE.
        target = tuple(c * WORLD_SCALE for c in p["target"])
        # Углы переводятся в радианы, расстояние — в мировые единицы.
        yaw = math.radians(p["yaw"])
        pitch = math.radians(p["pitch"])
        dist = p["dist"] * WORLD_SCALE
        # cp — косинус наклона; он сжимает горизонтальный радиус при взгляде сверху/снизу.
        cp = math.cos(pitch)
        # Положение камеры — точка на сфере вокруг target: X и Z через sin/cos yaw, Y через pitch.
        eye = (target[0] + dist * math.sin(yaw) * cp,
               target[1] + dist * math.sin(pitch),
               target[2] + dist * math.cos(yaw) * cp)
        self.position = list(eye)
        # Из вектора на цель обратно вычисляем yaw через atan2 и pitch через asin.
        direction = vsub(target, eye)
        self.yaw = math.degrees(math.atan2(direction[0], direction[2])) % 360.0
        length = max(1e-9, vlength(direction))
        # Ограничение [-1,1] защищает asin от ошибок округления.
        self.pitch = math.degrees(math.asin(max(-1.0, min(1.0, direction[1] / length))))
        # Если для ракурса задан свой угол обзора, используем его, иначе 43°.
        self.fov = float(p.get("fov", 43.0))

    # forward — вычисляет единичный вектор направления взгляда.
    # Параметры: self — текущая камера.
    # Возвращает: (x, y, z) — направление, в котором смотрит камера.
    def forward(self) -> V3:
        # Переводим углы в радианы и считаем горизонтальный коэффициент cp.
        yaw = math.radians(self.yaw)
        pitch = math.radians(self.pitch)
        cp = math.cos(pitch)
        # X и Z задают поворот вокруг вертикали, Y — наклон камеры.
        return (math.sin(yaw) * cp, math.sin(pitch), math.cos(yaw) * cp)

    # right — вычисляет горизонтальный правый вектор камеры.
    # Параметры: self — текущая камера.
    # Возвращает: (x, y, z), перпендикулярный взгляду в плоскости XZ.
    def right(self) -> V3:
        # Правый вектор не зависит от pitch: он всегда лежит в горизонтальной плоскости.
        yaw = math.radians(self.yaw)
        # При yaw=0 камера смотрит в +Z, а правый вектор направлен в -X.
        return (-math.cos(yaw), 0.0, math.sin(yaw))

    # eye — возвращает точку положения камеры.
    # Параметры: self — текущая камера.
    # Возвращает: (x, y, z) в мировых координатах.
    def eye(self) -> V3:
        # position хранится списком, а функции нужен кортеж из трёх чисел.
        return (self.position[0], self.position[1], self.position[2])

    # view_matrix — строит видовую матрицу для текущего положения камеры.
    # Параметры: self — текущая камера.
    # Возвращает: матрицу 4×4, переводящую мир в систему координат камеры.
    def view_matrix(self) -> M4:
        # Точка взгляда находится на единичном векторе forward перед камерой.
        eye = self.eye()
        # lookAt строит базис камеры и переносит мир в её систему координат.
        target = vadd(eye, self.forward())
        return mat4_look_at(eye, target, (0.0, 1.0, 0.0))

    # look — поворачивает камеру по смещению мыши.
    # Параметры: dx, dy — смещение курсора в пикселях по X и Y.
    # Возвращает: None. Углы yaw и pitch ограничиваются допустимыми значениями.
    def look(self, dx: float, dy: float) -> None:
        # Чувствительность 0,18° на пиксель; внутри вагона она снижается до 70%.
        sens = 0.18 * (0.70 if self.inside_train() else 1.0)
        # Движение мыши вправо уменьшает yaw, чтобы камера поворачивалась вправо.
        self.yaw = (self.yaw - dx * sens) % 360.0
        # Наклон ограничен ±89°, иначе камера перевернётся через полюс.
        self.pitch = max(-89.0, min(89.0, self.pitch - dy * sens))

    # turn — поворачивает камеру стрелками клавиатуры.
    # Параметры: d_yaw, d_pitch — приращения углов в градусах.
    # Возвращает: None. Внутри вагона приращения уменьшаются.
    def turn(self, d_yaw: float, d_pitch: float) -> None:
        # Стрелки внутри вагона поворачивают камеру на 65% обычного шага.
        factor = 0.65 if self.inside_train() else 1.0
        # yaw сворачивается в диапазон [0,360), pitch ограничен ±89°.
        self.yaw = (self.yaw + d_yaw * factor) % 360.0
        self.pitch = max(-89.0, min(89.0, self.pitch + d_pitch * factor))

    # dolly — двигает камеру вдоль направления взгляда (колесо мыши).
    # Параметры: distance — желаемое смещение; знак задаёт направление.
    # Возвращает: None. Итоговое смещение ограничено границами сцены.
    def dolly(self, distance: float) -> None:
        # Двигаемся вдоль взгляда; distance умножается на адаптивный множитель скорости.
        f = self.forward()
        distance *= self.navigation_factor()
        # Границы не дают камере улететь далеко от станции или уйти под землю.
        lim_x, lim_z, lim_y = 150.0 * WORLD_SCALE, 140.0 * WORLD_SCALE, 85.0 * WORLD_SCALE
        # Каждая координата независимо ограничивается своим диапазоном.
        self.position[0] = max(-lim_x, min(lim_x, self.position[0] + f[0] * distance))
        self.position[1] = max(0.12, min(lim_y, self.position[1] + f[1] * distance))
        self.position[2] = max(-lim_z, min(lim_z, self.position[2] + f[2] * distance))

    # update — обновляет положение камеры по нажатым клавишам за время dt.
    # Параметры:
    # dt — время с предыдущего кадра в секундах;
    # keys — множество нажатых латинских клавиш.
    # Возвращает: None. Учитывает автоповорот, W/S/A/D, Q/E и границы сцены.
    def update(self, dt: float, keys: set[str]) -> None:
        # Автоповорот добавляет 8° в секунду к yaw, не меняя положение камеры.
        if self.auto_rotate:
            self.yaw = (self.yaw + dt * 8.0) % 360.0
        # move — суммарный вектор перемещения в мировых координатах.
        move = [0.0, 0.0, 0.0]
        f = self.forward()
        r = self.right()
        # W/S добавляют forward, A/D добавляют right, E/Q меняют только высоту.
        if "w" in keys:
            move[0] += f[0]; move[1] += f[1]; move[2] += f[2]
        if "s" in keys:
            move[0] -= f[0]; move[1] -= f[1]; move[2] -= f[2]
        if "d" in keys:
            move[0] += r[0]; move[2] += r[2]
        if "a" in keys:
            move[0] -= r[0]; move[2] -= r[2]
        if "e" in keys:
            move[1] += 1.0
        if "q" in keys:
            move[1] -= 1.0
        # Длина вектора нужна, чтобы диагональное движение не было быстрее осевого.
        length = math.sqrt(move[0] ** 2 + move[1] ** 2 + move[2] ** 2)
        # Если клавиши не нажаты, ничего не меняем; иначе нормируем шаг.
        if length > 1e-8:
            step = self.speed * self.navigation_factor() * dt / length
            # После сдвига координаты ограничиваются теми же границами, что и в dolly.
            lim_x, lim_z, lim_y = 150.0 * WORLD_SCALE, 140.0 * WORLD_SCALE, 85.0 * WORLD_SCALE
            self.position[0] = max(-lim_x, min(lim_x, self.position[0] + move[0] * step))
            self.position[1] = max(0.12, min(lim_y, self.position[1] + move[1] * step))
            self.position[2] = max(-lim_z, min(lim_z, self.position[2] + move[2] * step))


# Renderer — отвечает за всю отрисовку сцены на OpenGL.
# Хранит Scene, три шейдерные программы (main, depth, sky), FBO карты теней,
# матрицу света light_vp и признаки готовности карты теней.
# Параметры конструктора: нет.
# Главный метод render() вызывается из display() на каждый кадр.
class Renderer:
    # SHADOW_SIZE — сторона квадратной карты теней в пикселях.
    # 2048×2048 достаточно для станции и даёт мягкие края при пяти отсчётах PCF.
    SHADOW_SIZE = 2048

    # __init__ — создаёт рендерер: сцену, три шейдера, карту теней и матрицу света.
    # Параметры: self — будущий объект Renderer.
    # Возвращает: None. После создания объект готов рисовать кадры.
    def __init__(self):
        # Читаем имя видеодрайвера, чтобы отличить аппаратный рендер от программного.
        renderer_name = glGetString(GL_RENDERER)
        self.renderer_name = renderer_name.decode("utf-8", "replace") if renderer_name else "unknown"
        # llvmpipe, softpipe и swrast — программные реализации; для них окно уменьшается.
        low = self.renderer_name.lower()
        self.software = any(k in low for k in ("llvmpipe", "softpipe", "swrast", "software"))
        # _shadow_valid показывает, построена ли карта теней; _shadow_cutaway хранит режим разреза.
        self._shadow_valid = False
        self._shadow_cutaway = None
        # Сцена создаётся и сразу строится: вся геометрия попадает в self.scene.batches.
        self.scene = Scene()
        self.scene.build()
        # main_shader рисует материалы, depth_shader — только глубину, sky_shader — небо.
        self.main_shader = Shader(MAIN_VS, MAIN_FS, "main")
        self.depth_shader = Shader(DEPTH_VS, DEPTH_FS, "depth")
        self.sky_shader = Shader(SKY_VS, SKY_FS, "sky")
        # Создаём FBO тени и матрицу света; размер карты задан в SHADOW_SIZE.
        self.shadow_fbo, self.shadow_tex, _ = self._make_shadow_fbo()
        self.light_vp = self._make_light_matrix()
        # Последний шаг — включить глубину, отсечение граней и мультисэмплинг.
        self._prepare_gl()

    # _make_shadow_fbo — создаёт FBO и глубинную текстуру для карты теней.
    # Параметры: self — рендерер, размер задаётся полем SHADOW_SIZE.
    # Возвращает: (fbo, texture, None) — идентификаторы буфера и текстуры.
    def _make_shadow_fbo(self):
        # Создаём глубинную текстуру 2048×2048 формата GL_DEPTH_COMPONENT24.
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, self.SHADOW_SIZE,
                     self.SHADOW_SIZE, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
        # Фильтры LINEAR нужны для пяти отсчётов PCF: сравнение идёт по глубине.
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        # За границей карты используется белая граница: такие фрагменты считаются освещёнными.
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, [1.0, 1.0, 1.0, 1.0])
        glBindTexture(GL_TEXTURE_2D, 0)
        # FBO подключает глубинную текстуру как depth attachment; цветовой буфер не нужен.
        fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, tex, 0)
        # Отключаем запись цвета и чтения цвета, оставляя только глубину.
        glDrawBuffer(GL_NONE)
        glReadBuffer(GL_NONE)
        # Проверяем полноту FBO; при ошибке сообщаем код, а не рисуем молча.
        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        if status != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"shadow FBO incomplete: {status:#x}")
        return fbo, tex, None

    # _make_light_matrix — строит видовую и ортографическую матрицы камеры света.
    # Параметры: self — рендерер, используется направление солнца из Scene.
    # Возвращает: матрицу 4×4 light_vp для перевода мира в пространство света.
    def _make_light_matrix(self) -> M4:
        # target — центр станции; eye отодвинут от цели вдоль направления солнца.
        s = self.scene.sun_dir
        target = (0.0, 2.0 * WORLD_SCALE, 0.0)
        eye = (target[0] + s[0] * 48.0 * WORLD_SCALE,
               target[1] + s[1] * 48.0 * WORLD_SCALE,
               target[2] + s[2] * 48.0 * WORLD_SCALE)
        # Видовая матрица света смотрит из eye в target, верх — мировой +Y.
        view = mat4_look_at(eye, target, (0.0, 1.0, 0.0))
        # Ортографический объём покрывает станцию по X и Z и глубину от 1 до 176.
        proj = mat4_ortho(-30.0 * WORLD_SCALE, 30.0 * WORLD_SCALE,
                          -26.0 * WORLD_SCALE, 26.0 * WORLD_SCALE,
                          1.0, 110.0 * WORLD_SCALE)
        # Порядок как в OpenGL: сначала view, затем projection; итог — light_vp.
        return mat4_mul(proj, view)

    # _prepare_gl — задаёт базовое состояние OpenGL для всей сцены.
    # Параметры: self — рендерер.
    # Возвращает: None. Включает глубину, отсечение задних граней и MSAA.
    def _prepare_gl(self):
        # Глубина с LEQUAL: ближние фрагменты перекрывают дальние.
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        # Задние грани отсекаются, лицевыми считаются вершины против часовой стрелки.
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glFrontFace(GL_CCW)
        # MSAA сглаживает края геометрии; на программном рендере его эффективность ниже.
        glEnable(GL_MULTISAMPLE)

    # _active_batches — выбирает пакеты для текущего прохода и режима разреза.
    # Параметры:
    # cutaway — True, если включён разрез вагона;
    # transparent — True для прозрачных пакетов, False для непрозрачных.
    # Возвращает: список Batch, которые нужно нарисовать.
    def _active_batches(self, cutaway: bool, transparent: bool):
        # Проходим по всем пакетам сцены и оставляем только подходящие.
        out = []
        for b in self.scene.batches:
            # Разделяем непрозрачный и прозрачный проходы: у них разный режим blending.
            if b.transparent != transparent:
                continue
            # В разрезе скрываем группу near (ближняя стена) и building (здание).
            if cutaway and b.group in ("near", "building"):
                continue
            # Пакет без Mesh пропускаем: он не был завершён или не содержит вершин.
            if b.mesh is None:
                continue
            out.append(b)
        return out

    # _set_material — передаёт цвета, блик, прозрачность и текстуру материала в шейдер.
    # Параметры: batch — пакет, материал которого нужно применить.
    # Возвращает: None. Все значения уходят в uniform-переменные main_shader.
    def _set_material(self, batch: Batch) -> None:
        s = self.main_shader
        m = batch.material
        # Передаём диффузный цвет, цвет блика и собственное свечение материала.
        s.set_3f("uDiffuse", m.diffuse)
        s.set_3f("uSpecular", m.specular)
        s.set_3f("uEmissive", m.emissive)
        # Shininess ограничен снизу: при значении меньше 2 блик становится слишком широким.
        s.set_f("uShininess", max(2.0, m.shininess))
        s.set_f("uAlpha", m.alpha)
        # Флаги unlit и alpha_test переводятся в целые 0/1 для шейдера.
        s.set_i("uUnlit", 1 if m.unlit else 0)
        s.set_i("uAlphaTest", 1 if m.alpha_test else 0)
        # Текстура материала занимает слот 0; карта теней постоянно находится в слоте 1.
        if m.texture is not None:
            m.texture.bind(0)
            s.set_i("uUseTexture", 1)
        else:
            s.set_i("uUseTexture", 0)

    # render — рисует один полный кадр: карту теней, небо, непрозрачную
    # и прозрачную геометрию.
    # Параметры:
    # camera — текущая камера; width, height — размер кадрового буфера;
    # cutaway — скрывать ли ближнюю стену и здание;
    # shadows — включать ли тени; target_fbo — целевой FBO (0 — экран);
    # time_s — текущее время (параметр интерфейса, сейчас не используется).
    # Возвращает: None. Результат остаётся в target_fbo или в заднем буфере окна.
    def render(self, camera: Camera, width: int, height: int, cutaway: bool,
               shadows: bool = True, target_fbo: int = 0, time_s: float = 0.0) -> None:
        # Сначала считаем видовую и перспективную матрицы, затем их произведение vp.
        view = camera.view_matrix()
        # Ближняя плоскость 0,05, дальняя 220·WORLD_SCALE; aspect берётся из окна.
        proj = mat4_perspective(camera.fov, width / max(1, height), 0.05,
                                220.0 * WORLD_SCALE)
        # uViewProj переводит мировые координаты сразу в clip-space.
        vp = mat4_mul(proj, view)
        eye = camera.eye()

        # Тени пересчитываются только если карты ещё нет или изменился режим разреза.
        if shadows and (not self._shadow_valid or cutaway != self._shadow_cutaway):
            # Первый проход рисует сцену в FBO теней, не выводя цвет на экран.
            glBindFramebuffer(GL_FRAMEBUFFER, self.shadow_fbo)
            glViewport(0, 0, self.SHADOW_SIZE, self.SHADOW_SIZE)
            # Очищаем только глубину: цвет в этом проходе всё равно отключён.
            glClear(GL_DEPTH_BUFFER_BIT)
            # Запись цвета выключена; нужна только глубина от источника света.
            glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE)
            glEnable(GL_DEPTH_TEST)
            # Включаем отсечение граней; ниже передние грани будут отброшены.
            glEnable(GL_CULL_FACE)
            # Отсекаем передние грани: тонкие стены не затеняют сами себя.
            glCullFace(GL_FRONT)
            glEnable(GL_POLYGON_OFFSET_FILL)
            # Смещение полигонов по глубине борется с полосами самозатенения.
            glPolygonOffset(2.0, 4.0)
            # Глубинный шейдер использует только uLightVP и не считает цвет.
            self.depth_shader.use()
            self.depth_shader.set_mat4("uLightVP", self.light_vp)
            # В карту теней попадают только непрозрачные пакеты; прозрачные тени не дают.
            for b in self._active_batches(cutaway, False):
                b.mesh.draw()
            glDisable(GL_POLYGON_OFFSET_FILL)
            # После прохода возвращаем запись цвета, задние грани и обычное состояние.
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE)
            glCullFace(GL_BACK)
            self._shadow_valid = True
            self._shadow_cutaway = cutaway

        # Второй проход рисует обычную сцену в target_fbo (0 — экран).
        glBindFramebuffer(GL_FRAMEBUFFER, target_fbo)
        glViewport(0, 0, width, height)
        # Цвет очистки совпадает с горизонтом неба, чтобы не было резкой рамки.
        glClearColor(0.66, 0.75, 0.84, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)
        # Перед непрозрачной геометрией blending выключен, глубина пишется.
        glDisable(GL_BLEND)
        glCullFace(GL_BACK)

        # Небо рисуется без теста и без записи глубины — оно всегда фон.
        glDisable(GL_DEPTH_TEST)
        glDepthMask(GL_FALSE)
        glDisable(GL_CULL_FACE)
        # Небо использует отдельную программу и панорамную текстуру.
        self.sky_shader.use()
        self.sky_shader.set_mat4("uViewProj", vp)
        if self.scene.sky_texture:
            self.scene.sky_texture.bind(0)
            self.sky_shader.set_i("uSkyTexture", 0)
        if self.scene.sky_mesh:
            self.scene.sky_mesh.draw()
        # После неба возвращаем отсечение граней, запись глубины и тест глубины.
        glEnable(GL_CULL_FACE)
        glDepthMask(GL_TRUE)
        glEnable(GL_DEPTH_TEST)

        # Непрозрачная геометрия рисуется с обычным отсечением задних граней.
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        # Основной шейдер получает матрицы, параметры света, туман и карту теней.
        self.main_shader.use()
        s = self.main_shader
        s.set_mat4("uViewProj", vp)
        s.set_mat4("uLightVP", self.light_vp)
        # Позиция камеры нужна для блика; направление и цвет солнца — для прямого света.
        s.set_3f("uCameraPos", eye)
        s.set_3f("uLightDir", self.scene.sun_dir)
        s.set_3f("uLightColor", self.scene.sun_color)
        s.set_3f("uAmbientSky", self.scene.ambient_sky)
        s.set_3f("uAmbientGround", self.scene.ambient_ground)
        s.set_3f("uFogColor", self.scene.fog_color)
        # Плотность тумана делится на WORLD_SCALE, чтобы масштаб сцены не менял вид.
        s.set_f("uFogDensity", 0.0048 / WORLD_SCALE)
        s.set_f("uShadowTexel", 1.0 / self.SHADOW_SIZE)
        s.set_f("uPointFalloff", 0.30 / (WORLD_SCALE * WORLD_SCALE))
        s.set_i("uTexture", 0)
        # Карта теней привязывается к слоту 1, материал — к слоту 0.
        s.set_i("uShadowMap", 1)
        s.set_i("uShadowsEnabled", 1 if shadows else 0)
        # Активируем первый слот и привязываем глубинную текстуру тени.
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.shadow_tex)
        # Шейдер поддерживает максимум восемь точечных источников; лишние отбрасываются.
        lights = self.scene.point_lights[:8]
        if lights:
            # Позиции, цвета и радиусы ламп передаются тремя массивами.
            set_uniform_array3(s, "uPointPos[0]", [l[0] for l in lights])
            set_uniform_array3(s, "uPointColor[0]", [l[1] for l in lights])
            set_uniform_array1(s, "uPointRange[0]", [l[2] for l in lights])
        s.set_i("uPointCount", len(lights))

        # Каждый непрозрачный пакет получает свой материал и рисуется одним вызовом.
        for b in self._active_batches(cutaway, False):
            self._set_material(b)
            b.mesh.draw()

        # Прозрачные пакеты собираются отдельно и сортируются по расстоянию.
        transparent = self._active_batches(cutaway, True)
        # Сортировка по квадрату расстояния: дальние рисуются первыми, ближние последними.
        transparent.sort(key=lambda b: (b.center[0] - eye[0]) ** 2 + (b.center[1] - eye[1]) ** 2 +
                                       (b.center[2] - eye[2]) ** 2, reverse=True)
        # Для стекла включаем стандартное смешивание source-over.
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthMask(GL_FALSE)
        for b in transparent:
            self._set_material(b)
            b.mesh.draw()
        glDepthMask(GL_TRUE)
        glDisable(GL_BLEND)

        # Возвращаем стандартный кадровый буфер, чтобы GLUT показал готовый кадр.
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

# Текстуры, надписи и герб Казахстана

from __future__ import annotations

import math
import random
from typing import Iterable, Sequence

# RGBA-цвет в диапазоне 0..255
RGBA = tuple[int, int, int, int]


# Шум и вспомогательные функции.

# Ограничивает значение отрезком [0, 1]. Везде используется в шейдерах
# и в шуме, чтобы значения не выходили за пределы цветового диапазона
def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


# Линейно интерполирует два значения
def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


# Сглаживающая интерполяция (нулевая производная на концах)
def smooth(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


# Детерминированный хеш пары координат.
# Псевдослучайное число в [0, 1] по (ix, iy, seed), но всегда одно
# и то же для одних и тех же входов

# Умножения на простые числа смешивают ix, iy, seed в одно число,
# XOR-сдвиги перемешивают биты (чтобы соседние ix не давали «соседние»
# результаты — иначе в текстуре были бы полосы), & 0xFFFFFFFF обрезает
# до 32 бит, деление на 2^32 - 1 нормирует к [0, 1]
def _hash2(ix: int, iy: int, seed: int) -> float:
    n = (ix * 374761393 + iy * 668265263 + seed * 1442695041) & 0xFFFFFFFF
    n = (n ^ (n >> 13)) * 1274126177 & 0xFFFFFFFF
    n = (n ^ (n >> 16)) & 0xFFFFFFFF
    return n / 4294967295.0


# Детерминированный хеш трёх координат
def _hash3(ix: int, iy: int, iz: int, seed: int) -> float:
    n = (ix * 374761393 + iy * 668265263 + iz * 2147483647 + seed * 1442695041) & 0xFFFFFFFF
    n = (n ^ (n >> 13)) * 1274126177 & 0xFFFFFFFF
    n = (n ^ (n >> 16)) & 0xFFFFFFFF
    return n / 4294967295.0


# Двумерный шум (value noise): плавная псевдослучайная функция,
# определённая на всей плоскости. Возвращает число в [0, 1].
# Принцип:
# 1. Разбиваем плоскость на целочисленную сетку (ячейки 1*1).
# 2. В каждом углу ячейки — «случайное» значение (из _hash2).
# 3. Внутри ячейки интерполируем эти четыре значения по fx, fy.
# smooth() на дробной части нужен, чтобы переход через границу ячейки
# был гладким, а не угловатым.
# Результат: крупные «пятна» шума, из которых потом собираются
# текстуры (трава, бетон, облака).
def noise2(x: float, y: float, seed: int = 1) -> float:
    # ix, iy — целые координаты ячейки; fx, fy — дробное положение
    # внутри неё (0..1). smooth() делает переход между ячейками гладким
    ix, iy = math.floor(x), math.floor(y)
    fx, fy = smooth(x - ix), smooth(y - iy)
    # Четыре угла ячейки дают значения шума, которые затем интерполируются.
    a = _hash2(ix, iy, seed)
    b = _hash2(ix + 1, iy, seed)
    c = _hash2(ix, iy + 1, seed)
    d = _hash2(ix + 1, iy + 1, seed)
    return lerp(lerp(a, b, fx), lerp(c, d, fx), fy) # Сначала интерполируем по X, потом по Y


# Фрактальный шум: сумма нескольких
# октав noise2 с разной частотой и убывающей амплитудой.
# Зачем: один слой noise2 — это однородные «пятна». Чтобы получить
# естественную текстуру (трава, бетон, дерево), нужно наложить
# несколько слоёв: крупные пятна + мелкие детали поверх них.
# Параметры:
# octaves — число слоёв. 3-4 обычно достаточно: больше — мельче детали, но дольше считает.
# Каждая следующая октава:
# * имеет вдвое меньшую амплитуду (amp *= 0.5) — не «перекрикивает» крупный слой;
# * имеет вдвое большую частоту (x *= 2.03) — добавляет мелкие детали;
# * сдвигается константой (11.7, -7.3), чтобы октавы не коррелировали
# между собой и не давали видимых артефактов
def fbm2(x: float, y: float, seed: int = 1, octaves: int = 4) -> float:
    # total — накопленная сумма; norm — суммарная амплитуда для
    # нормировки к [0, 1] в конце
    total = 0.0
    amp = 0.5
    norm = 0.0
    for o in range(octaves):
        total += amp * noise2(x, y, seed + o * 17)
        norm += amp
        # Каждая следующая октава имеет вдвое более высокую частоту
        x = x * 2.03 + 11.7
        y = y * 2.01 - 7.3
        amp *= 0.5
    return total / norm


# Трёхмерный шум. Нужен для облаков на небе
# Принцип тот же, что в noise2, только интерполируются 8 углов куба
# по X, потом по Y, потом по Z — трилинейная интерполяция
def noise3(x: float, y: float, z: float, seed: int = 1) -> float:
    # Трёхмерный шум использует восемь углов куба и трилинейную интерполяцию.
    ix, iy, iz = math.floor(x), math.floor(y), math.floor(z)
    fx, fy, fz = smooth(x - ix), smooth(y - iy), smooth(z - iz)
     # Обёртка над _hash3 — чтобы не писать +dx, +dy, +dz каждый раз
    def h(dx, dy, dz):
        return _hash3(ix + dx, iy + dy, iz + dz, seed)
    x00 = lerp(h(0, 0, 0), h(1, 0, 0), fx)
    x10 = lerp(h(0, 1, 0), h(1, 1, 0), fx)
    x01 = lerp(h(0, 0, 1), h(1, 0, 1), fx)
    x11 = lerp(h(0, 1, 1), h(1, 1, 1), fx)
    return lerp(lerp(x00, x10, fy), lerp(x01, x11, fy), fz)


# Полотно — это плоский bytearray, где 4 байта на пиксель: R, G, B, A.
# Создаёт пустое полотно заданного размера, залитое одним цветом.
# По умолчанию — полностью прозрачное (0, 0, 0, 0)
def new_canvas(w: int, h: int, color: RGBA = (0, 0, 0, 0)) -> bytearray:
    return bytearray(color * (w * h))


# Параметры:
# buf       — полотно (bytearray);
# w, h      — его размеры (чтобы проверять выход за границы);
# x, y      — координаты пикселя;
# color     — цвет в float [0, 1] (R, G, B);
# alpha     — прозрачность
def _blend(buf: bytearray, w: int, h: int, x: int, y: int,
           color: Sequence[float], alpha: float = 1.0) -> None:
    # Выход за границы полотна и нулевая прозрачность ничего не меняют
    if x < 0 or y < 0 or x >= w or y >= h or alpha <= 0.0:
        return
    alpha = min(1.0, alpha)
    # Смещение пикселя в bytearray:
    i = (y * w + x) * 4
    # da — текущая непрозрачность пикселя в полотне, в долях [0, 1].
    # Делим байт 0..255 на 255, чтобы перевести в долю.
    # sa — сколько накладываем
    da = buf[i + 3] / 255.0
    sa = alpha
    # Итоговая непрозрачность: новый слой делает пиксель непрозрачным
    # на sa, плюс старый вклад da — но только в той части, которая
    # не перекрыта новым (1 - sa)
    oa = sa + da * (1.0 - sa)
    if oa <= 0.0:
        # Оба слоя полностью прозрачны — смешивать нечего
        return
    for c in range(3):
        # Старый цвет канала
        dst = buf[i + c] / 255.0
        # Новый цвет канала: ограничиваем в [0, 1] на случай, если вызывающий
        # код передал значение чуть за границей (ошибка float в lerp и т.п.)
        src = max(0.0, min(1.0, color[c]))
        # src * sa — вклад нового слоя с учётом его прозрачности;
        # dst * da * (1-sa) — остаток старого цвета в неперекрытой части;
        # / oa — делим на итоговую непрозрачность
        out = (src * sa + dst * da * (1.0 - sa)) / oa
        # Запись обратно в байт:
        # * 255 — из доли [0, 1] в 0..255;
        # + 0.5 — округление к ближайшему;
        # max/min — clamp от накопленной ошибки float
        buf[i + c] = int(max(0.0, min(255.0, out * 255.0 + 0.5)))
    # И записываем итоговую альфу
    buf[i + 3] = int(max(0.0, min(255.0, oa * 255.0 + 0.5)))


# Заливает всё полотно цветом
def fill_canvas(buf: bytearray, w: int, h: int, color: RGBA) -> None:
    buf[:] = bytearray(color) * (w * h)


# Заливает прямоугольник с учётом прозрачности (использует _blend
# для каждого пикселя).
# Параметры (x0, y0) и (x1, y1) — углы прямоугольника в координатах
# полотна; не обязательно целые. Целые координаты получаются через
# floor/ceil, чтобы охватить все затронутые пиксели
def fill_rect(buf: bytearray, w: int, h: int, x0: float, y0: float,
              x1: float, y1: float, color: Sequence[float], alpha: float = 1.0) -> None:
    # Обрезаем прямоугольник по границам полотна
    ix0, iy0 = max(0, int(math.floor(x0))), max(0, int(math.floor(y0)))
    ix1, iy1 = min(w, int(math.ceil(x1))), min(h, int(math.ceil(y1)))
    for y in range(iy0, iy1):
        for x in range(ix0, ix1):
            _blend(buf, w, h, x, y, color, alpha)

# Рисует сглаженную линию заданной толщины.
# buf — полотно (bytearray, 4 байта на пиксель), куда пишем результат;
# w, h — размеры полотна в пикселях (для проверки границ и индекса);
# p0, p1 — начало и конец отрезка (x, y), могут быть дробными;
# color — цвет (R, G, B) в долях [0, 1];
# width — толщина линии в пикселях;
# alpha — общая прозрачность линии [0, 1];

# Алгоритм:
#   1. Для каждого пикселя в прямоугольнике вокруг отрезка находим
#      ближайшую точку на самом отрезке (проекция).
#   2. Расстояние до этой точки d сравниваем с половиной толщины r.
#   3. Если d < r — пиксель внутри линии, рисуем с полной силой.
#      Если d чуть больше r — частичное покрытие (сглаживание)
def draw_line(buf: bytearray, w: int, h: int, p0: Sequence[float], p1: Sequence[float],
              color: Sequence[float], width: float = 1.0, alpha: float = 1.0) -> None:
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    length2 = dx * dx + dy * dy
    # Отрезок нулевой длины — просто точка (рисуем как круг)
    if length2 < 1e-9:
        _disc(buf, w, h, x0, y0, width * 0.5, color, alpha)
        return
    # r — половина толщины
    r = width * 0.5
    # Прямоугольник, в котором будем перебирать пиксели
    minx = max(0, int(math.floor(min(x0, x1) - r - 1)))
    maxx = min(w - 1, int(math.ceil(max(x0, x1) + r + 1)))
    miny = max(0, int(math.floor(min(y0, y1) - r - 1)))
    maxy = min(h - 1, int(math.ceil(max(y0, y1) + r + 1)))
    for y in range(miny, maxy + 1):
        py = y + 0.5
        for x in range(minx, maxx + 1):
            px = x + 0.5
            # Проекция точки (px, py) на прямую
            t = ((px - x0) * dx + (py - y0) * dy) / length2
            # Ограничиваем t отрезком [0, 1]
            t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
            # Ближайшая к (px, py) точка на самом отрезке
            qx, qy = x0 + t * dx, y0 + t * dy
            # Расстояние от центра пикселя до этой ближайшей точки
            d = math.hypot(px - qx, py - qy)
            # «Покрытие» cov — доля пикселя, которую занимает линия
            cov = clamp01(r + 0.75 - d)
            # cov = 0 — пиксель полностью снаружи, писать нечего
            if cov > 0.0:
                # Смешиваем цвет пикселя с учётом покрытия и общей
                # прозрачности
                _blend(buf, w, h, x, y, color, cov * alpha)


# Рисует закрашенный круг
# Параметры:
# buf — полотно (bytearray, 4 байта на пиксель: R, G, B, A), куда пишется результат;
# w, h — размеры полотна в пикселях (для проверки границ и индекса);
# cx, cy — координаты центра круга, могут быть дробными;
# radius — радиус в пикселях;
# color  — цвет (R, G, B) в долях [0, 1];
# alpha  — прозрачность круга [0, 1]
def _disc(buf: bytearray, w: int, h: int, cx: float, cy: float, radius: float,
          color: Sequence[float], alpha: float = 1.0) -> None:
    # Ограничиваем перебор только квадратом вокруг круга
    minx = max(0, int(math.floor(cx - radius - 1)))
    maxx = min(w - 1, int(math.ceil(cx + radius + 1)))
    miny = max(0, int(math.floor(cy - radius - 1)))
    maxy = min(h - 1, int(math.ceil(cy + radius + 1)))
    for y in range(miny, maxy + 1):
        for x in range(minx, maxx + 1):
            # Расстояние от центра пикселя до центра круга
            d = math.hypot(x + 0.5 - cx, y + 0.5 - cy)
            # Покрытие
            cov = clamp01(radius + 0.75 - d)
            if cov > 0.0:
                _blend(buf, w, h, x, y, color, cov * alpha)

# Рисует окружность (контур)
# Разбивает окружность на отрезки и рисует каждый как
# линию через draw_line. Чем больше radius, тем больше шагов, чтобы
# контур оставался гладким
def draw_circle(buf: bytearray, w: int, h: int, cx: float, cy: float,
                radius: float, color: Sequence[float], width: float = 1.0,
                alpha: float = 1.0) -> None:
    # Сколько отрезков использовать для аппроксимации окружности.
    # min 24 — даже у маленькой окружности не меньше 24 отрезков,
    # иначе она выглядит как квадрат/шестиугольник
    steps = max(24, int(radius * 1.2))
    # Считаем точки на окружности
    pts = [(cx + math.cos(i * math.tau / steps) * radius,
            cy + math.sin(i * math.tau / steps) * radius) for i in range(steps + 1)]
    # Проходим по соседним парам точек и рисуем между ними отрезок.
    # zip(pts, pts[1:]) даёт пары (pts[0], pts[1]), (pts[1], pts[2]), ...
    for a, b in zip(pts, pts[1:]):
        draw_line(buf, w, h, a, b, color, width, alpha)


# Штриховой шрифт 5 на 7: заглавные буквы, латиница и цифры.
# Строит точки на эллиптической дуге — используется для букв
# с закруглениями (О, С, Э, Ю, 8, 9 и т.д.).
# Параметры:
# cx, cy  — центр эллипса;
# rx, ry — радиусы эллипса по X и Y. Если rx == ry — дуга окружности; если разные — вытянутый/сжатый овал;
# a0, a1 — начальный и конечный углы дуги в радианах
# steps — число отрезков, на которые разбивается дуга
# Возвращает список точек [(x0, y0), (x1, y1), ..., (xN, yN)] длиной
# steps + 1. Этот список потом передаётся в
# FONT как одна полилиния
def _arc(cx: float, cy: float, rx: float, ry: float, a0: float, a1: float,
         steps: int = 20) -> list[tuple[float, float]]:
    return [(cx + math.cos(a0 + (a1 - a0) * i / steps) * rx,
             cy + math.sin(a0 + (a1 - a0) * i / steps) * ry)
            for i in range(steps + 1)]

# Готовый овал для «О»
_OVAL = _arc(2.5, 3.5, 2.5, 3.5, 0.0, math.tau, 30)

# Сама таблица глифов. Ключ — символ, значение — список полилиний
FONT: dict[str, list[list[tuple[float, float]]]] = {
    "А": [[(0, 0), (2.5, 7), (5, 0)], [(1, 2.4), (4, 2.4)]],
    "Б": [[(0, 0), (0, 7), (5, 7)], [(0, 4), (4, 4), (5, 3), (5, 1), (4, 0), (0, 0)]],
    "В": [[(0, 0), (0, 7), (4, 7), (5, 6), (5, 4.5), (4, 4), (0, 4)],
          [(4, 4), (5, 3.5), (5, 1), (4, 0), (0, 0)]],
    "Г": [[(0, 0), (0, 7), (5, 7)]],
    "Д": [[(0, 0), (1, 2), (1, 7), (4, 7), (4, 2), (5, 0)], [(0, 0), (5, 0)]],
    "Е": [[(0, 0), (0, 7), (5, 7)], [(0, 3.5), (4, 3.5)], [(0, 0), (5, 0)]],
    "Ё": [[(0, 0), (0, 7), (5, 7)], [(0, 3.5), (4, 3.5)], [(0, 0), (5, 0)],
          [(1.2, 8.1), (1.2, 8.2)], [(3.8, 8.1), (3.8, 8.2)]],
    "Ж": [[(2.5, 0), (2.5, 7)], [(0, 7), (2.5, 3.5), (0, 0)],
          [(5, 7), (2.5, 3.5), (5, 0)]],
    "З": [[(0, 7), (4, 7), (5, 6), (5, 4.5), (4, 4), (1, 4)],
          [(4, 4), (5, 3), (5, 1), (4, 0), (0, 0)]],
    "И": [[(0, 0), (0, 7)], [(4.7, 0), (4.7, 7)], [(0.3, 0), (4.4, 7)]],
    "Й": [[(0, 0), (0, 7)], [(4.7, 0), (4.7, 7)], [(0.3, 0), (4.4, 7)],
          [(0.7, 8.0), (1.5, 8.5), (3.3, 8.5), (4.1, 8.0)]],
    "К": [[(0, 0), (0, 7)], [(5, 7), (0, 3.2)], [(1.8, 4.4), (5, 0)]],
    "Л": [[(0, 0), (1.5, 7), (5, 7), (5, 0)]],
    "М": [[(0, 0), (0, 7), (2.5, 3.2), (5, 7), (5, 0)]],
    "Н": [[(0, 0), (0, 7)], [(5, 0), (5, 7)], [(0, 3.5), (5, 3.5)]],
    "О": [_OVAL],
    "П": [[(0, 0), (0, 7), (5, 7), (5, 0)]],
    "Р": [[(0, 0), (0, 7), (4, 7), (5, 6), (5, 4), (4, 3), (0, 3)]],
    "С": [_arc(2.6, 3.5, 2.5, 3.5, math.radians(42), math.radians(318), 24)],
    "Т": [[(0, 7), (5, 7)], [(2.5, 7), (2.5, 0)]],
    "У": [[(0, 7), (2.5, 3.5), (5, 7)], [(2.5, 3.5), (1.1, 0)]],
    "Ф": [[(2.5, 0), (2.5, 7)], _arc(2.5, 3.5, 2.05, 1.65, 0, math.tau, 24)],
    "Х": [[(0, 0), (5, 7)], [(0, 7), (5, 0)]],
    "Ц": [[(0, 7), (0, 0), (4.7, 0), (4.7, 7)], [(4.7, 0), (5.3, -1.0)]],
    "Ч": [[(0, 7), (0, 3.5), (4, 3.5), (4, 7)], [(4, 0), (4, 7)]],
    "Ш": [[(0, 0), (0, 7)], [(2.5, 0), (2.5, 7)], [(5, 0), (5, 7)], [(0, 0), (5, 0)]],
    "Щ": [[(0, 0), (0, 7)], [(2.1, 0), (2.1, 7)], [(4.2, 0), (4.2, 7)],
          [(0, 0), (4.2, 0)], [(4.2, 0), (4.8, -1.0)]],
    "Ъ": [[(0, 7), (2.8, 7)], [(2.8, 7), (2.8, 0)],
          [(2.8, 4), (4.4, 4), (5, 3), (5, 1), (4.4, 0), (2.8, 0)]],
    "Ы": [[(0, 0), (0, 7)], [(0, 4), (2.2, 4), (3, 3), (3, 1), (2.2, 0), (0, 0)],
          [(5, 0), (5, 7)]],
    "Ь": [[(0, 0), (0, 7)], [(0, 4), (3.2, 4), (4, 3), (4, 1), (3.2, 0), (0, 0)]],
    "Э": [_arc(2.4, 3.5, 2.5, 3.5, math.radians(222), math.radians(138), 24),
          [(2.1, 3.5), (5, 3.5)]],
    "Ю": [[(0, 0), (0, 7)], [(0, 3.5), (2.0, 3.5)],
          _arc(3.5, 3.5, 1.8, 3.5, 0, math.tau, 26)],
    "Я": [[(5, 0), (5, 7), (1, 7), (0, 6), (0, 4.5), (1, 3.5), (5, 3.5)],
          [(3.0, 3.5), (0, 0)]],
    # Латинские буквы нужны для служебных подписей.
    "A": [[(0, 0), (2.5, 7), (5, 0)], [(1, 2.4), (4, 2.4)]],
    "B": [[(0, 0), (0, 7), (4, 7), (5, 6), (5, 4.5), (4, 4), (0, 4)],
          [(4, 4), (5, 3.5), (5, 1), (4, 0), (0, 0)]],
    "C": [_arc(2.6, 3.5, 2.5, 3.5, math.radians(42), math.radians(318), 24)],
    "D": [[(0, 0), (0, 7), (3.4, 7), (5, 5.5), (5, 1.5), (3.4, 0), (0, 0)]],
    "E": [[(0, 0), (0, 7), (5, 7)], [(0, 3.5), (4, 3.5)], [(0, 0), (5, 0)]],
    "F": [[(0, 0), (0, 7), (5, 7)], [(0, 3.6), (4, 3.6)]],
    "G": [_arc(2.6, 3.5, 2.5, 3.5, math.radians(42), math.radians(330), 24),
          [(2.7, 3.4), (5, 3.4)]],
    "H": [[(0, 0), (0, 7)], [(5, 0), (5, 7)], [(0, 3.5), (5, 3.5)]],
    "I": [[(2.5, 0), (2.5, 7)], [(1.2, 7), (3.8, 7)], [(1.2, 0), (3.8, 0)]],
    "K": [[(0, 0), (0, 7)], [(5, 7), (0, 3.2)], [(1.8, 4.4), (5, 0)]],
    "L": [[(0, 7), (0, 0), (5, 0)]],
    "M": [[(0, 0), (0, 7), (2.5, 3.2), (5, 7), (5, 0)]],
    "N": [[(0, 0), (0, 7), (5, 0), (5, 7)]],
    "O": [_OVAL],
    "P": [[(0, 0), (0, 7), (4, 7), (5, 6), (5, 4), (4, 3), (0, 3)]],
    "R": [[(0, 0), (0, 7), (4, 7), (5, 6), (5, 4), (4, 3), (0, 3)],
          [(2.7, 3), (5, 0)]],
    "S": [[(5, 6), (4, 7), (1, 7), (0, 6), (0, 4.5), (1, 3.6), (4, 3.4),
           (5, 2.5), (5, 1), (4, 0), (1, 0), (0, 1)]],
    "T": [[(0, 7), (5, 7)], [(2.5, 7), (2.5, 0)]],
    "U": [[(0, 7), (0, 1), (1, 0), (4, 0), (5, 1), (5, 7)]],
    "V": [[(0, 7), (2.5, 0), (5, 7)]],
    "W": [[(0, 7), (1.3, 0), (2.5, 4.2), (3.7, 0), (5, 7)]],
    "Y": [[(0, 7), (2.5, 3.5), (5, 7)], [(2.5, 3.5), (2.5, 0)]],
    "0": [_OVAL],
    "1": [[(1, 5.8), (2.5, 7), (2.5, 0)], [(1, 0), (4, 0)]],
    "2": [[(0, 6), (1, 7), (4, 7), (5, 6), (5, 4), (0, 0), (5, 0)]],
    "3": [[(0, 7), (4, 7), (5, 6), (5, 4.2), (4, 3.5), (1, 3.5)],
          [(4, 3.5), (5, 2.8), (5, 1), (4, 0), (0, 0)]],
    "4": [[(4, 0), (4, 7), (0, 3), (5, 3)]],
    "5": [[(5, 7), (0, 7), (0, 4), (4, 4), (5, 3), (5, 1), (4, 0), (0, 0)]],
    "6": [[(4.5, 7), (1, 7), (0, 5), (0, 1), (1, 0), (4, 0), (5, 1), (5, 2.5),
           (4, 3.5), (0, 3.5)]],
    "7": [[(0, 7), (5, 7), (2, 0)]],
    "8": [_arc(2.5, 5.25, 1.9, 1.75, 0, math.tau, 24),
          _arc(2.5, 1.75, 2.2, 1.75, 0, math.tau, 24)],
    "9": [[(0.5, 0), (4, 0), (5, 2), (5, 6), (4, 7), (1, 7), (0, 6),
           (0, 4.5), (1, 3.5), (5, 3.5)]],
    "-": [[(1, 3.5), (4, 3.5)]],
    "—": [[(0.2, 3.5), (4.8, 3.5)]],
    "–": [[(0.5, 3.5), (4.5, 3.5)]],
    ".": [[(2.5, 0.2), (2.5, 0.3)]],
    ",": [[(2.8, 0.2), (2.2, -1.0)]],
    ":": [[(2.5, 1.3), (2.5, 1.4)], [(2.5, 4.2), (2.5, 4.3)]],
    "/": [[(0, 0), (5, 7)]],
    "№": [[(0, 0), (0, 7), (3, 7), (3, 4), (0, 4)],
          [(4.2, 0), (4.2, 2.5), (5, 2.5), (5, 0)]],
    " ": [],
}


# Вычисляет ширину строки в пикселях, не рисуя её
# advance — ширина одной буквы;
# tracking — дополнительный зазор между буквами
def text_width(text: str, size: float, tracking: float = 0.18) -> float:
    advance = size * (5.0 / 7.0)
    return sum((advance * (0.55 if ch == " " else 1.0)) + size * tracking
               for ch in text) - size * tracking

# Рисует строку штриховым шрифтом
# Параметры:
# text — что рисовать;
# x — X-координата опорной точки (левого края, центра или правого края — зависит от align);
# baseline — Y-координата базовой линии (низа заглавных букв);
# size — высота заглавных букв в пикселях;
# align— "left" (по умолчанию), "center", "right";
# tracking — расстояние между буквами в долях size;
# stroke — толщина штриха в долях size;
# shadow — цвет тени (или None); тень — тот же текст со смещением;
# shadow_offset — сдвиг тени в долях size.
# Возвращает ширину нарисованной строки — чтобы можно было поставить
# несколько строк друг под другом или в ряд
def draw_text(buf: bytearray, w: int, h: int, text: str, x: float, baseline: float,
              size: float, color: Sequence[float], align: str = "left",
              tracking: float = 0.18, stroke: float = 0.085,
              alpha: float = 1.0, shadow: Sequence[float] | None = None,
              shadow_offset: float = 0.05) -> float:
    # Сначала считаем ширину строки, чтобы правильно применить выравнивание
    width = text_width(text, size, tracking)
    if align == "center":
        x -= width * 0.5
    elif align == "right":
        x -= width
    cursor = x # текущая позиция «каретки» по X
    # advance — ширина одной буквы в пикселях
    # Это не видимая ширина конкретной буквы, а шаг, на который
    # сдвигается каретка после любой букв
    advance = size * (5.0 / 7.0) 
    linew = max(1.0, size * stroke) # толщина штриха минимум 1 пиксель
    if shadow is not None:
        # Тень рисуется тем же текстом со смещением и меньшей прозрачностью.
        draw_text(buf, w, h, text, x + size * shadow_offset, baseline - size * shadow_offset,
                  size, shadow, "left", tracking, stroke, alpha * 0.55, None)
    for ch in text:
        glyph = FONT.get(ch)
        if glyph:
            # Координаты глифа из сетки 5*7 переводятся в пиксели полотна.
            # px/5.0*advance — доля ширины буквы;
            # py/7.0*size — доля высоты
            for poly in glyph:
                pts = [(cursor + px / 5.0 * advance, baseline + py / 7.0 * size)
                       for px, py in poly]
                # Полилиния из одной точки (как «.» или «:») — рисуем кружком
                if len(pts) == 1:
                    _disc(buf, w, h, pts[0][0], pts[0][1], linew * 0.55, color, alpha)
                else:
                     # Иначе — соединяем соседние точки отрезками.
                     # zip(pts, pts[1:]) даёт пары (pts[0], pts[1]),
                     # (pts[1], pts[2]), ...
                    for a, b in zip(pts, pts[1:]):
                        draw_line(buf, w, h, a, b, color, linew, alpha)
        if ch == " ":
             # Сдвигаем каретку на следующую позицию
            cursor += advance * 0.55 + size * tracking
        else:
            cursor += advance + size * tracking
    return width


# Рисует несколько строк друг под другом с заданным шагом (gap)
# Параметры:
# buf — полотно
# w, h — размеры полотна в пикселях;
# lines — список строк для рисования. Каждый элемент — кортеж (текст, размер, цвет):
# cx — X-координата опорной точки. По умолчанию align="center",
# поэтому cx — это центр всех строк: каждая строка
# центрируется относительно cx;
# y0 — Y-координата базовой линии ПЕРВОЙ строки (низ еёт аглавных букв);
# gap — расстояние между базовыми линиями соседних строк в пикселях.
# align  — выравнивание всех строк: "center" (по умолчанию),
# "left" или "right". Применяется к каждой строке относительно cx;
# stroke — толщина штриха в долях size (общая для всех строк)
def draw_multiline(buf: bytearray, w: int, h: int, lines: Iterable[tuple[str, float, Sequence[float]]],
                   cx: float, y0: float, gap: float, align: str = "center",
                   stroke: float = 0.085) -> None:
    # Каждая строка рисуется со своим размером и цветом, затем базовая линия сдвигается
    y = y0
    for text, size, color in lines:
        draw_text(buf, w, h, text, cx, y, size, color, align=align, stroke=stroke)
        y += gap


# Герб
def draw_sun(buf: bytearray, w: int, h: int, cx: float, cy: float, radius: float,
             color: Sequence[float] = (1.0, 0.76, 0.12, 1.0)) -> None:
    # Чётные лучи длиннее нечётных, поэтому солнце выглядит объёмным
    rays = 32
    for i in range(rays):
        a = i * math.tau / rays
        long_ray = (i % 2 == 0)
        r0 = radius * 0.95
        r1 = radius * (1.58 if long_ray else 1.36)
        half = radius * (0.025 if long_ray else 0.018)
        p0 = (cx + math.cos(a - half) * r0, cy + math.sin(a - half) * r0)
        p1 = (cx + math.cos(a + half) * r0, cy + math.sin(a + half) * r0)
        p2 = (cx + math.cos(a) * r1, cy + math.sin(a) * r1)
        # Треугольный луч рисуется несколькими сужающимися линиями
        for k in range(6):
            t = k / 5.0
            qa = (lerp(p0[0], p2[0], t), lerp(p0[1], p2[1], t))
            qb = (lerp(p1[0], p2[0], t), lerp(p1[1], p2[1], t))
            draw_line(buf, w, h, qa, qb, color, max(1.0, radius * 0.032 * (1.0 - 0.35 * t)))
    _disc(buf, w, h, cx, cy, radius, color)
    # Внутренние орнаментальные кольца
    draw_circle(buf, w, h, cx, cy, radius * 0.78, (0.35, 0.12, 0.02, 1.0), radius * 0.035)
    draw_circle(buf, w, h, cx, cy, radius * 0.52, (1.0, 0.88, 0.42, 1.0), radius * 0.022)
    for i in range(8):
        a = i * math.tau / 8
        draw_line(buf, w, h,
                  (cx + math.cos(a) * radius * 0.17, cy + math.sin(a) * radius * 0.17),
                  (cx + math.cos(a) * radius * 0.43, cy + math.sin(a) * radius * 0.43),
                  (1.0, 0.9, 0.52, 1.0), max(1.5, radius * 0.025))

# Генераторы текстур
# Каждая функция возвращает bytearray с пикселями RGBA, который потом
# попадает в Texture(...) и загружается в GPU. Все текстуры — процедурные

# Трава
def tex_grass(size: int = 512) -> bytearray:
    rng = random.Random(11)
    buf = bytearray(size * size * 4)
    for y in range(size):
        for x in range(size):
            # Крупный шум задаёт пятна травы, мелкий — отдельные травинки
            n = fbm2(x * 0.035, y * 0.035, 3, 4)
            fine = fbm2(x * 0.17, y * 0.17, 9, 3)
            dry = clamp01((fbm2(x * 0.012, y * 0.012, 21, 3) - 0.52) * 4.0)
            r = 43 + n * 35 + dry * 42 + fine * 11
            g = 72 + n * 52 + dry * 28 + fine * 17
            b = 30 + n * 23 + fine * 9
            i = (y * size + x) * 4
            buf[i:i + 4] = bytes((int(r), int(g), int(b), 255))
    # Редкие травинки
    for _ in range(size * 8):
        x, y = rng.randrange(size), rng.randrange(size)
        c = rng.choice([(55, 95, 39), (84, 113, 48), (38, 67, 31)])
        _blend(buf, size, size, x, y, c, 0.25)
    return buf


# Генерирует текстуру щебня под рельсами
def tex_ballast(size: int = 256) -> bytearray:
    rng = random.Random(12)
    buf = bytearray(size * size * 4)
    for y in range(size):
        for x in range(size):
            n = fbm2(x * 0.12, y * 0.12, 4, 3)
            v = 55 + n * 42
            i = (y * size + x) * 4
            buf[i:i + 4] = bytes((int(v * 0.95), int(v * 0.94), int(v * 0.9), 255))
    # Случайные светлые и тёмные камешки
    for _ in range(1100):
        x, y = rng.randrange(size), rng.randrange(size)
        r = rng.choice((1, 1, 2, 2, 3, 4))
        shade = rng.randint(65, 150)
        _disc(buf, size, size, x, y, r, (shade, shade, shade * 0.96, 1.0), 0.7)
    return buf


# Генерирует текстуру бетона (для платформы)
def tex_concrete(size: int = 512, tiles: int = 0, base: float = 0.48) -> bytearray:
    rng = random.Random(14)
    buf = bytearray(size * size * 4)
    for y in range(size):
        for x in range(size):
            n = fbm2(x * 0.02, y * 0.02, 5, 4)
            f = fbm2(x * 0.19, y * 0.19, 7, 2)
            v = base + n * 0.18 + f * 0.045
            # При tiles > 0 добавляются швы и лёгкое чередование плиток
            if tiles:
                tw = size / tiles
                gx = abs((x % tw) - tw * 0.5)
                gy = abs((y % tw) - tw * 0.5)
                if gx > tw * 0.5 - 1.2 or gy > tw * 0.5 - 1.2:
                    v *= 0.72
                if (int(x / tw) + int(y / tw)) % 2:
                    v *= 0.975
            i = (y * size + x) * 4
            buf[i:i + 4] = bytes((int(v * 255), int(v * 254), int(v * 251), 255))
    for _ in range(450):
        x, y = rng.randrange(size), rng.randrange(size)
        v = rng.randint(95, 165) / 255.0
        _disc(buf, size, size, x, y, rng.choice((1, 1, 2)), (v, v, v, 1.0), 0.3)
    # Несколько тонких трещин
    for _ in range(10):
        x = rng.randrange(size)
        y = rng.randrange(size)
        for _ in range(rng.randint(12, 35)):
            nx = x + rng.randint(-4, 4)
            ny = y + rng.randint(2, 7)
            draw_line(buf, size, size, (x, y), (nx, ny), (0.28, 0.28, 0.27, 1.0), 1.0, 0.35)
            x, y = nx, ny
    return buf


# Генерирует текстуру кирпича
def tex_brick(size: int = 512) -> bytearray:
    rng = random.Random(15)
    buf = bytearray(size * size * 4)
    # Кирпичи имеют фиксированный размер, а каждый второй ряд смещён на полкирпича
    bw, bh = 100, 42
    for y in range(size):
        row = y // bh
        for x in range(size):
            offset = (bw // 2) if row % 2 else 0
            bx = (x + offset) % bw
            by = y % bh
            mortar = bx < 4 or by < 4
            if mortar:
                v = 150 + int(fbm2(x * 0.1, y * 0.1, 8, 2) * 30)
                col = (v, v * 0.97, v * 0.9)
            else:
                idx = (row * 37 + (x + offset) // bw * 13 + 3) & 0xffff
                tone = _hash2(idx, row, 2)
                r = 122 + tone * 55
                g = 55 + tone * 32
                b = 39 + tone * 22
                n = fbm2(x * 0.11, y * 0.11, 4, 2)
                col = (r * (0.85 + n * 0.25), g * (0.82 + n * 0.25), b * (0.82 + n * 0.25))
            i = (y * size + x) * 4
            buf[i:i + 4] = bytes((int(min(255, col[0])), int(min(255, col[1])),
                                  int(min(255, col[2])), 255))
    return buf


# Генерирует текстуру дерева
def tex_wood(size: int = 512, light: bool = False) -> bytearray:
    rng = random.Random(16 + int(light))
    buf = bytearray(size * size * 4)
    for y in range(size):
        # Смещение warp делает волокна дерева слегка волнистыми
        warp = fbm2(0.0, y * 0.028, 4, 3) * 33.0
        for x in range(size):
            grain = math.sin((x + warp) * 0.14) * 0.5 + 0.5
            fine = fbm2(x * 0.06, y * 0.45, 7, 2)
            k = grain * 0.22 + fine * 0.28
            if light:
                r, g, b = 178 + k * 45, 141 + k * 38, 92 + k * 28
            else:
                r, g, b = 104 + k * 54, 69 + k * 38, 41 + k * 24
            i = (y * size + x) * 4
            buf[i:i + 4] = bytes((int(r), int(g), int(b), 255))
    for _ in range(12):
        cx, cy = rng.randrange(size), rng.randrange(size)
        for rr in range(rng.randint(2, 7), 0, -1):
            draw_circle(buf, size, size, cx, cy, rr * 2.2, (0.28, 0.16, 0.08, 1.0), 1.0, 0.18)
    return buf


# Генерирует текстуру обоев
def tex_wallpaper(size: int = 256) -> bytearray:
    buf = bytearray(size * size * 4)
    for y in range(size):
        for x in range(size):
            n = fbm2(x * 0.025, y * 0.025, 17, 3)
            stripe = 0.018 if (x // 16) % 2 == 0 else 0.0
            v = 0.68 + n * 0.08 + stripe
            i = (y * size + x) * 4
            buf[i:i + 4] = bytes((int(v * 255), int(v * 0.93 * 255),
                                  int(v * 0.78 * 255), 255))
    return buf


# Генерирует текстуру пола
def tex_floor(size: int = 512) -> bytearray:
    buf = bytearray(size * size * 4)
    for y in range(size):
        for x in range(size):
            n = fbm2(x * 0.05, y * 0.05, 19, 3)
            v = 0.35 + n * 0.09
            i = (y * size + x) * 4
            buf[i:i + 4] = bytes((int(v * 0.94 * 255), int(v * 0.99 * 255),
                                  int(v * 1.05 * 255), 255))
    # Тёмные линии делят пол на крупные плиты
    for x in range(0, size, size // 4):
        fill_rect(buf, size, size, x, 0, x + 2, size, (0.19, 0.22, 0.25, 1.0))
    for y in range(0, size, size // 4):
        fill_rect(buf, size, size, 0, y, size, y + 2, (0.19, 0.22, 0.25, 1.0))
    return buf


# Генерирует текстуру ковровой дорожки
def tex_carpet(size: int = 256) -> bytearray:
    buf = bytearray(size * size * 4)
    for y in range(size):
        for x in range(size):
            n = fbm2(x * 0.15, y * 0.15, 23, 3)
            weave = (1 if (x + y) % 4 < 2 else 0) * 0.035
            v = 0.26 + n * 0.12 + weave
            i = (y * size + x) * 4
            buf[i:i + 4] = bytes((int(v * 1.65 * 255), int(v * 0.55 * 255),
                                  int(v * 0.55 * 255), 255))
    for x in range(0, size, 64):
        fill_rect(buf, size, size, x, 0, x + 4, size, (0.38, 0.16, 0.14, 1.0))
    return buf


# Генерирует текстуру металла (рельсы, крыша, обшивка вагона)
def tex_metal(size: int = 256, blue: bool = False) -> bytearray:
    buf = bytearray(size * size * 4)
    for y in range(size):
        for x in range(size):
            # Анизотропный шум вытянут по горизонтали и даёт следы прокатки.
            n = fbm2(x * 0.02, y * 0.3, 25, 3)
            streak = math.sin(y * 0.9 + n * 8.0) * 0.02
            if blue:
                r, g, b = 34 + n * 22, 64 + n * 26, 116 + n * 38
            else:
                r, g, b = 92 + n * 55, 98 + n * 55, 104 + n * 58
            i = (y * size + x) * 4
            buf[i:i + 4] = bytes((int(r + streak * 255), int(g + streak * 255),
                                  int(b + streak * 255), 255))
    return buf


# Генерирует текстуру ткани (для одеял на кроватях)
def tex_fabric(size: int = 256, color=(46, 84, 142)) -> bytearray:
    buf = bytearray(size * size * 4)
    for y in range(size):
        for x in range(size):
            # Диагональное чередование ярких и тёмных нитей создаёт ткань.
            weave = 0.06 * (1 if (x + y) % 6 < 3 else -0.02)
            n = fbm2(x * 0.08, y * 0.08, 27, 3) * 0.11
            k = 0.76 + weave + n
            i = (y * size + x) * 4
            buf[i:i + 4] = bytes((min(255, int(color[0] * k)), min(255, int(color[1] * k)),
                                  min(255, int(color[2] * k)), 255))
    return buf


# Генерирует панорамное небо
def tex_sky(width: int = 1024, height: int = 512, sun_dir=(0.45, 0.72, 0.53)) -> bytearray:
    """Панорамное небо с мягкими облаками и солнцем."""
    sx, sy, sz = sun_dir
    sl = math.sqrt(sx * sx + sy * sy + sz * sz)
    sx, sy, sz = sx / sl, sy / sl, sz / sl
    buf = bytearray(width * height * 4)
    zenith = (0.20, 0.38, 0.68)
    horizon = (0.69, 0.79, 0.88)
    below = (0.36, 0.40, 0.40)
    # Строки панорамы соответствуют широте, столбцы — долготе.
    for py in range(height):
        v = py / (height - 1)
        lat = (v - 0.5) * math.pi
        clat, slat = math.cos(lat), math.sin(lat)
        for px in range(width):
            u = px / (width - 1)
            lon = u * math.tau
            dx, dz = clat * math.cos(lon), clat * math.sin(lon)
            dy = slat
            if dy >= 0:
                t = pow(dy, 0.55)
                col = [lerp(horizon[c], zenith[c], t) for c in range(3)]
            else:
                t = clamp01(-dy * 3.0)
                col = [lerp(horizon[c], below[c], t) for c in range(3)]
            # Два масштаба шума дают крупные и мелкие облака без видимого шва
            if dy > 0.02:
                n = noise3(dx * 3.2 + 1.7, dy * 5.2 - 0.4, dz * 3.2 + 2.1, 31)
                n2 = noise3(dx * 8.5 - 3.0, dy * 10.0 + 1.0, dz * 8.5, 41)
                cloud = smooth(clamp01((n * 0.75 + n2 * 0.25 - 0.53) * 3.2))
                cloud *= clamp01(dy * 5.5) * (1.0 - clamp01((dy - 0.82) * 4.0))
                col = [lerp(col[c], (0.95, 0.96, 0.97)[c], cloud * 0.92) for c in range(3)]
            # Скалярное произведение с направлением солнца даёт диск и ореол
            dot = dx * sx + dy * sy + dz * sz
            if dot > 0:
                glow = math.pow(dot, 90.0) * 0.65 + math.pow(dot, 18.0) * 0.12
                col = [min(1.0, col[c] + glow * (0.95, 0.82, 0.55)[c]) for c in range(3)]
                if dot > 0.99965:
                    col = [1.0, 0.96, 0.82]
            i = (py * width + px) * 4
            buf[i:i + 4] = bytes((int(max(0, min(255, col[0] * 255))),
                                  int(max(0, min(255, col[1] * 255))),
                                  int(max(0, min(255, col[2] * 255))), 255))
    return buf


# Текстуры конкретных надписей
def blit_rgba(dst: bytearray, dw: int, dh: int,
              src: bytes | bytearray, sw: int, sh: int,
              x0: int, y0: int, tw: int, th: int,
              flip_y: bool = False) -> None:
    
    # Для PNG-подобных данных строки нужно перевернуть по вертикали
    if flip_y:
        row = sw * 4
        flipped = bytearray(len(src))
        for y in range(sh):
            flipped[y * row:(y + 1) * row] = src[(sh - 1 - y) * row:(sh - y) * row]
        src = flipped
    # Для каждого пикселя цели усредняем попавший в него прямоугольник источника
    for oy in range(th):
        sy0 = oy * sh // th
        sy1 = max(sy0 + 1, (oy + 1) * sh // th)
        for ox in range(tw):
            sx0 = ox * sw // tw
            sx1 = max(sx0 + 1, (ox + 1) * sw // tw)
            r = g = b = a = count = 0
            for sy in range(sy0, sy1):
                base = (sy * sw + sx0) * 4
                for sx in range(sx0, sx1):
                    i = base + (sx - sx0) * 4
                    r += src[i]
                    g += src[i + 1]
                    b += src[i + 2]
                    a += src[i + 3]
                    count += 1
            alpha = (a / count) / 255.0
            if alpha <= 0.0:
                continue
            _blend(dst, dw, dh, x0 + ox, y0 + oy,
                   (r / count / 255.0, g / count / 255.0, b / count / 255.0),
                   alpha)


# Создаёт надпись с гербом для вагона
def make_train_label(width: int = 3072, height: int = 256,
                     emblem: tuple[int, int, bytes | bytearray] | None = None) -> bytearray:
    buf = new_canvas(width, height)
    gold = (1.0, 0.78, 0.22, 1.0)
    pale = (1.0, 0.9, 0.53, 1.0)
    # Если герб загружен, он вставляется слева, иначе рисуется процедурное солнце
    if emblem is not None:
        ew, eh, epx = emblem
        blit_rgba(buf, width, height, epx, ew, eh, 52, 13, 230, 230, flip_y=True)
    else:
        draw_sun(buf, width, height, 160.0, height * 0.52, 74.0, gold)
    draw_text(buf, width, height, "НОВОСИБИРСК-АЛМАТЫ 2",
              315.0, 80.0, 128.0, pale, align="left", tracking=0.14,
              stroke=0.072, shadow=(0.01, 0.03, 0.10, 1.0), shadow_offset=0.035)
    return buf


# Создаёт вывеску КПП
def make_building_sign(width: int = 2560, height: int = 400) -> bytearray:
    buf = new_canvas(width, height)
    # Синий фон и золотая рамка вывески рисуются прямоугольниками
    fill_rect(buf, width, height, 0, 0, width, height, (0.025, 0.10, 0.24, 1.0))
    fill_rect(buf, width, height, 8, 8, width - 8, height - 8, (0.07, 0.20, 0.42, 1.0))
    fill_rect(buf, width, height, 8, 8, width - 8, 14, (0.94, 0.72, 0.22, 1.0))
    fill_rect(buf, width, height, 8, height - 14, width - 8, height - 8, (0.94, 0.72, 0.22, 1.0))
    fill_rect(buf, width, height, 8, 8, 14, height - 8, (0.94, 0.72, 0.22, 1.0))
    fill_rect(buf, width, height, width - 14, 8, width - 8, height - 8, (0.94, 0.72, 0.22, 1.0))
    draw_text(buf, width, height, "КПП ЛОКОТЬ", width * 0.5, 155.0, 118.0,
              (1.0, 0.93, 0.73, 1.0), align="center", tracking=0.16, stroke=0.072)
    draw_text(buf, width, height, "РОССИЙСКАЯ ФЕДЕРАЦИЯ", width * 0.5, 38.0, 66.0,
              (0.98, 0.82, 0.38, 1.0), align="center", tracking=0.22, stroke=0.085)
    return buf


# Создаёт табличку туалета
def make_toilet_sign(width: int = 512, height: int = 160) -> bytearray:
    buf = new_canvas(width, height)
    fill_rect(buf, width, height, 0, 0, width, height, (0.02, 0.16, 0.36, 1.0))
    fill_rect(buf, width, height, 4, 4, width - 4, height - 4, (0.06, 0.28, 0.56, 1.0))
    draw_text(buf, width, height, "ТУАЛЕТ", width * 0.5, 38.0, 66.0,
              (1.0, 1.0, 1.0, 1.0), align="center", tracking=0.22, stroke=0.075)
    return buf


# Создаёт табличку платформы
def make_platform_sign(width: int = 1536, height: int = 256) -> bytearray:
    buf = new_canvas(width, height)
    fill_rect(buf, width, height, 0, 0, width, height, (0.04, 0.15, 0.30, 1.0))
    fill_rect(buf, width, height, 5, 5, width - 5, height - 5, (0.12, 0.34, 0.53, 1.0))
    draw_text(buf, width, height, "ЛОКОТЬ  •  ПЛАТФОРМА 1", width * 0.5, 54.0, 72.0,
              (1.0, 0.96, 0.85, 1.0), align="center", tracking=0.18, stroke=0.075)
    return buf


# Создаёт табличку проводника
def make_conductor_sign(width: int = 1024, height: int = 192) -> bytearray:
    buf = new_canvas(width, height)
    fill_rect(buf, width, height, 0, 0, width, height, (0.04, 0.13, 0.29, 0.94))
    fill_rect(buf, width, height, 4, 4, width - 4, height - 4, (0.10, 0.28, 0.48, 1.0))
    draw_text(buf, width, height, "ПРОВОДНИК", width * 0.5, 42.0, 76.0,
              (1.0, 0.94, 0.76, 1.0), align="center", tracking=0.16, stroke=0.075)
    return buf


# Генерирует поверхность зеркала
def tex_mirror(size: int = 512) -> bytearray:
    """Ровная тёмная поверхность зеркала без нарисованных окон."""
    buf = bytearray(size * size * 4)
    for y in range(size):
        for x in range(size):
            # Мягкий шум делает поверхность зеркала слегка неоднородной
            n = fbm2(x * 0.016, y * 0.016, 61, 3)
            v = 0.43 + 0.07 * (1.0 - y / size) + (n - 0.5) * 0.045
            i = (y * size + x) * 4
            buf[i:i + 4] = bytes((int(min(255, v * 0.96 * 255)),
                                  int(min(255, v * 0.98 * 255)),
                                  int(min(255, v * 1.02 * 255)), 255))
    return buf


# Генерирует текстуру потолка
def tex_ceiling(size: int = 256) -> bytearray:
    """Светлые потолочные панели с едва заметными стыками."""
    buf = bytearray(size * size * 4)
    for y in range(size):
        for x in range(size):
            n = fbm2(x * 0.035, y * 0.035, 71, 3)
            v = 0.86 + n * 0.045
            # Тёмные полосы через каждые 64 пикселя обозначают стыки панелей
            if x % 64 < 2 or y % 64 < 2:
                v *= 0.88
            i = (y * size + x) * 4
            buf[i:i + 4] = bytes((int(v * 255), int(v * 0.985 * 255),
                                  int(v * 0.94 * 255), 255))
    return buf

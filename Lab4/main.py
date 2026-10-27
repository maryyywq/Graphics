from __future__ import annotations

import os
import sys
import time

# Локальная папка зависимостей для запуска без установки
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEPS = os.path.join(_HERE, ".deps")
if os.path.isdir(_DEPS) and _DEPS not in sys.path:
    sys.path.insert(0, _DEPS)

from OpenGL.GL import *
from OpenGL.error import NullFunctionError

import station_scene
from station_scene import Camera, PRESETS, Renderer

try:
    from OpenGL.GLUT import *
    _HAS_GLUT = True
except Exception:
    _HAS_GLUT = False

try:
    import OpenGL.contextdata as _gl_contextdata
    _original_get_context = _gl_contextdata.getContext

    def _get_context_safe(context=None):
        if context is not None:
            return context
        try:
            return _original_get_context(None)
        except Exception:
            return "glut-external-context"

    _gl_contextdata.getContext = _get_context_safe
except Exception:
    pass

try:
    import OpenGL.GLUT.special as _gl_glut_special
    _gl_glut_special.CurrentContextIsValid = lambda: True
except Exception:
    pass


# Глобальное состояние окна, сцены и ввода
WIDTH, HEIGHT = 1120, 630
renderer: Renderer | None = None
camera = Camera()
cutaway = False
shadows = True
fullscreen = False
last_time = 0.0
mouse_down = False
mouse_button = 0
mouse_x = 0
mouse_y = 0
keys: set[str] = set()


# Приводит значение ключа GLUT к обычной строке
def _ustr(value) -> str:
    # GLUT может передавать клавишу как bytes или как str
    if isinstance(value, bytes):
        return value.decode("latin-1", "replace")
    return str(value)


current_preset = "1"


# Применяет выбранный готовый ракурс камеры.
def _apply_preset(key: str) -> None:
    # Проверяем, что клавиша действительно соответствует готовому ракурсу.
    global current_preset, cutaway
    if key not in PRESETS:
        return
    current_preset = key
    camera.snap_preset(key)
    cutaway = bool(PRESETS[key][1]["cutaway"])


# Возвращает камеру к общему виду станции.
def _reset() -> None:
    # Сбрасываем автоповорот и возвращаемся к общему виду.
    global cutaway
    camera.auto_rotate = False
    camera.snap_preset("1")
    cutaway = False
    current_preset = "1"


# Обработчики событий GLUT.
def display() -> None:
    # Ограничиваем dt, чтобы после паузы камера не прыгнула слишком далеко.
    global last_time
    now = time.perf_counter()
    dt = min(0.06, now - last_time) if last_time else 0.016
    last_time = now
    camera.update(dt, keys)
    # Рендерер получает текущее состояние камеры, разреза и теней.
    renderer.render(camera, WIDTH, HEIGHT, cutaway, shadows=shadows,
                    target_fbo=0, time_s=now)
    glutSwapBuffers()


# Обновляет размеры окна и область просмотра.
def reshape(w: int, h: int) -> None:
    # Размеры не могут быть нулевыми, иначе нарушится соотношение сторон.
    global WIDTH, HEIGHT
    WIDTH, HEIGHT = max(1, w), max(1, h)
    glViewport(0, 0, WIDTH, HEIGHT)


# Планирует следующий кадр через таймер GLUT.
def timer(value: int) -> None:
    # Просим GLUT перерисовать кадр и снова запускаем таймер.
    glutPostRedisplay()
    glutTimerFunc(16, timer, 0)


# Переводит клавишу русской раскладки в латинское имя.
def _norm_key(key) -> str:
    # Русские буквы на клавиатуре должны работать так же, как латинские.
    k = _ustr(key).lower()
    cyr = {
        "ц": "w", "ф": "a", "ы": "s", "в": "d", "й": "q", "у": "e",
        "с": "c", "к": "r", "а": "f", "щ": "o", "т": "t",
    }
    return cyr.get(k, k)


# Обрабатывает нажатие клавиши.
def keyboard(key, x: int, y: int) -> None:
    # Обрабатываем нажатие один раз, пока клавиша не была отпущена.
    global cutaway, shadows, fullscreen
    k = _norm_key(key)
    # Escape корректно завершает главный цикл GLUT.
    if k == "\x1b":
        try:
            glutLeaveMainLoop()
        except Exception:
            raise SystemExit(0)
        return
    # Повторное событие нажатия игнорируется, чтобы не переключать режим дважды.
    if k in keys:
        return
    keys.add(k)
    if k in PRESETS:
        _apply_preset(k)
    elif k == "c":
        cutaway = not cutaway
    elif k == "t":
        # Тени можно отключить для ускорения программного рендера.
        shadows = not shadows
    elif k == "o":
        camera.auto_rotate = not camera.auto_rotate
    elif k == "r":
        _reset()
    elif k == "f":
        # Полноэкранный режим и возврат к исходному размеру окна.
        fullscreen = not fullscreen
        if fullscreen:
            glutFullScreen()
        else:
            glutReshapeWindow(1120, 630)
            glutPositionWindow(80, 60)
    elif k in ("+", "="):
        # Базовая скорость ограничена сверху, чтобы сцена не пролетала мгновенно.
        camera.speed = min(40.0, camera.speed + 2.0)
    elif k == "-":
        camera.speed = max(1.0, camera.speed - 2.0)
    glutPostRedisplay()


# Обрабатывает отпускание клавиши.
def keyboard_up(key, x: int, y: int) -> None:
    # После отпускания клавиша перестаёт участвовать в векторе движения.
    keys.discard(_norm_key(key))


# Обрабатывает стрелки и служебные клавиши.
def special(key: int, x: int, y: int) -> None:
    # Стрелки поворачивают камеру, PageUp/PageDown меняют скорость.
    if key == GLUT_KEY_LEFT:
        camera.turn(5.0, 0.0)
    elif key == GLUT_KEY_RIGHT:
        camera.turn(-5.0, 0.0)
    elif key == GLUT_KEY_UP:
        camera.turn(0.0, 4.0)
    elif key == GLUT_KEY_DOWN:
        camera.turn(0.0, -4.0)
    elif key == GLUT_KEY_PAGE_UP:
        camera.speed = min(40.0, camera.speed + 2.0)
    elif key == GLUT_KEY_PAGE_DOWN:
        camera.speed = max(1.0, camera.speed - 2.0)
    glutPostRedisplay()


# Обрабатывает кнопки мыши и колесо.
def mouse(button: int, state: int, x: int, y: int) -> None:
    # Запоминаем кнопку и точку нажатия, чтобы затем считать смещение мыши.
    global mouse_down, mouse_button, mouse_x, mouse_y
    if state == GLUT_DOWN:
        if button == 3:
            camera.dolly(1.0)
            return
        if button == 4:
            camera.dolly(-1.0)
            return
        # Любая из двух кнопок включает режим обзора.
        if button in (GLUT_LEFT_BUTTON, GLUT_RIGHT_BUTTON):
            mouse_down = True
            mouse_button = button
            mouse_x, mouse_y = x, y
    elif state == GLUT_UP:
        mouse_down = False
    glutPostRedisplay()


# Обрабатывает движение мыши с зажатой кнопкой.
def motion(x: int, y: int) -> None:
    # При отпущенной кнопке только запоминаем позицию курсора.
    global mouse_x, mouse_y
    if not mouse_down:
        mouse_x, mouse_y = x, y
        return
    dx, dy = x - mouse_x, y - mouse_y
    mouse_x, mouse_y = x, y
    # Смещение передаётся камере в пикселях.
    camera.look(dx, dy)
    glutPostRedisplay()


# Обрабатывает прокрутку колеса мыши.
def wheel(device: int, direction: int, x: int, y: int) -> None:
    # Положительное направление колеса приближает камеру.
    camera.dolly(1.0 if direction > 0 else -1.0)
    glutPostRedisplay()


# Перенаправляет изменение размера окна в reshape.
def resize_window(w: int, h: int) -> None:
    # Дублирует обработчик изменения размера для совместимости с GLUT.
    reshape(w, h)


# Точка входа в программу.
def run_glut() -> int:
    # Инициализируем GLUT и создаём окно с двойной буферизацией и глубиной.
    global renderer, last_time
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH)
    glutInitWindowSize(WIDTH, HEIGHT)
    glutInitWindowPosition(70, 45)
    # Название на английском, чтобы заголовок не зависел от кодировки системы.
    glutCreateWindow(b"Railway station")
    try:
        # При закрытии окна главный цикл должен вернуть управление в Python.
        glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS)
    except Exception:
        pass
    # Единственная строка, которая выводится при обычном запуске.
    print("Свободная камера: WASD + Q/E, мышь — обзор, 1-9/0 — ракурсы, "
          "C — разрез, T — тени, Esc — выход.", flush=True)

    # Готовим сцену и отключаем её служебные сообщения.
    station_scene.QUIET = True
    renderer = Renderer()
    camera.snap_preset("1")

    # Регистрируем все callback-функции до входа в главный цикл.
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    glutKeyboardUpFunc(keyboard_up)
    glutSpecialFunc(special)
    glutMouseFunc(mouse)
    glutMotionFunc(motion)
    try:
        glutMouseWheelFunc(wheel)
    except Exception:
        pass
    # На программном OpenGL уменьшаем окно, чтобы поднять частоту кадров.
    if renderer.software:
        try:
            glutReshapeWindow(960, 540)
        except Exception:
            pass
    # Первый таймер запускает циклическую перерисовку примерно 60 раз в секунду.
    glutTimerFunc(16, timer, 0)
    last_time = time.perf_counter()
    glutMainLoop()
    return 0


# Точка входа приложения.
def main() -> int:
    # Без freeglut интерактивное окно создать нельзя, поэтому сообщаем об ошибке.
    if not _HAS_GLUT:
        print("Не найден GLUT/freeglut.")
        return 1
    try:
        if not bool(glutInit):
            print("Не найден GLUT/freeglut.")
            return 1
        return run_glut()
    except NullFunctionError:
        print("Не найден GLUT/freeglut.")
        return 1
    except Exception as exc:
        print("Ошибка при создании окна GLUT:", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

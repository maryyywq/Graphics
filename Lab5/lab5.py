import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QSlider, QComboBox, QGroupBox,
                             QMessageBox)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.ticker as mticker
import cv2

class ImageViewer(QMainWindow):
    def __init__(self):
        # Вызов конструктора родительского класса QMainWindow (создает базовое окно)
        super().__init__()

        # Переменные для хранения изображений
        self.original_image = None 
        self.current_image = None
        self.original_array = None  # Оригинальное изображение в формате numpy array
        self.current_array = None   # Текущее изображение после преобразований
        # Переменные для выделения области
        self.selection_rect = None  # Координаты выделенной области
        self.drawing_selection = False  # Флаг: сейчас пользователь рисует прямоугольник (зажат ЛКМ)
        self.selection_start = None # Начальная точка выделения (координаты мыши)
        self.selection_active = False # Флаг: включен ли вообще режим выделения кнопкой "Выделить область"
        
        # Параметры преобразований изображения
        self.brightness_value = 0 # Значение яркости (по умолчанию 0)
        self.contrast_value = 100 # Значение контрастности (по умолчанию 1.0)
        
        # Переменные для работы с гистограммой
        self.hist_initialized = False # Флаг инициализации гистограммы
        self.hist_bars_r = None # Ссылки на столбцы гистограммы красного канала
        self.hist_bars_g = None # Ссылки на столбцы гистограммы зеленого канала
        self.hist_bars_b = None # Ссылки на столбцы гистограммы синего канала
        self.hist_bars_y = None # Ссылки на столбцы гистограммы яркости
        
        # Таймер для отложенного обновления (иначе тормозит)
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.apply_all_transforms)

        # Порог бинаризации по умолчанию
        self.binary_threshold = 128

        # Отступ по оси X у гистограмм: чтобы крайние столбцы (яркость 0 и 255)
        # не прятались под рамкой графика
        self.hist_margin = 4

        # Вызываем метод initUI. Он отрисует все кнопки, ползунки и графики на экране.
        # Делаем это в конце, чтобы все атрибуты класса уже существовали,
        # когда Qt начнет присылать события (resizeEvent и т.п.)
        self.initUI()
        
    def initUI(self):
        # Настройка основного окна приложения
        self.setWindowTitle('Лабораторная работа 5')
        self.setGeometry(100, 100, 1400, 900) # Отступ от левого верхнего угла (100,100) и размеры окна 1400x900
        
        # Создание центрального виджета
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # горизонтальное разделение на левую и правую панели
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Левая панель для отображения изображения
        left_panel = QVBoxLayout()
        
        # Метка для отображения изображения
        self.image_label = QLabel('Изображение не загружено')
        self.image_label.setAlignment(Qt.AlignCenter) # Центрируем текст/картинку внутри метки
        self.image_label.setMinimumSize(600, 400) # Минимальный размер, чтобы окно не сжималось слишком сильно
        self.image_label.setStyleSheet("border: 1px solid gray;") # Рисуем серую рамку вокруг картинки

        # Назначение обработчиков событий мыши для выделения области
        self.image_label.mousePressEvent = self.image_mouse_press
        self.image_label.mouseMoveEvent = self.image_mouse_move
        self.image_label.mouseReleaseEvent = self.image_mouse_release
        left_panel.addWidget(self.image_label)
        
        # Панель с кнопками управления
        btn_layout = QHBoxLayout()
        
        # Кнопка загрузки изображения
        load_btn = QPushButton('Загрузить изображение')
        load_btn.clicked.connect(self.load_image) # Привязка клика к методу load_image
        btn_layout.addWidget(load_btn)
        
        # Кнопка сброса изменений
        reset_btn = QPushButton('Сбросить изменения')
        reset_btn.clicked.connect(self.reset_image)
        btn_layout.addWidget(reset_btn)
        
        # Кнопка включения режима выделения
        select_btn = QPushButton('Выделить область')
        select_btn.clicked.connect(self.toggle_selection)
        btn_layout.addWidget(select_btn)
        
        # Кнопка сброса выделения
        clear_select_btn = QPushButton('Сбросить выделение')
        clear_select_btn.clicked.connect(self.clear_selection)
        btn_layout.addWidget(clear_select_btn)

        # Добавляем панель с кнопками под картинку
        left_panel.addLayout(btn_layout)
        
        # Правая панель - элементы управления преобразованиями
        right_panel = QVBoxLayout()
        
        # Группа для гистограмм
        # QGroupBox создает визуальную рамку с заголовком
        hist_group = QGroupBox('Гистограммы')
        hist_layout = QVBoxLayout()

        # Встраиваем Matplotlib график (наш кастомный класс MplCanvas) в интерфейс PyQt
        self.hist_canvas = MplCanvas(self, width=6, height=4, dpi=100)
        hist_layout.addWidget(self.hist_canvas)
        hist_group.setLayout(hist_layout)
        right_panel.addWidget(hist_group)
        
        # Группа для управления яркостью и контрастностью
        bc_group = QGroupBox('Яркость и контрастность')
        bc_layout = QVBoxLayout()
        
        # Слайдер для регулировки яркости
        brightness_layout = QHBoxLayout()
        brightness_layout.addWidget(QLabel('Яркость:'))
        self.brightness_slider = QSlider(Qt.Horizontal) # Горизонтальный ползунок
        self.brightness_slider.setRange(-255, 255) # Диапазон от -255 до +255
        self.brightness_slider.setValue(0) # Начальное положение
        # Сигналы нажатия/отпускания (чтобы фиксировать начало и конец движения)
        self.brightness_slider.sliderPressed.connect(self.slider_pressed)
        self.brightness_slider.sliderReleased.connect(self.slider_released)
        brightness_layout.addWidget(self.brightness_slider)
        self.brightness_value_label = QLabel('0') # Текст, показывающий текущее значение
        brightness_layout.addWidget(self.brightness_value_label)
        bc_layout.addLayout(brightness_layout)
        
        # Слайдер для регулировки контрастности
        contrast_layout = QHBoxLayout()
        contrast_layout.addWidget(QLabel('Контрастность (%):'))
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(10, 10000) # Диапазон 0.1 - 100.0 
        self.contrast_slider.setValue(100)
        self.contrast_slider.sliderPressed.connect(self.slider_pressed)
        self.contrast_slider.sliderReleased.connect(self.slider_released)
        contrast_layout.addWidget(self.contrast_slider)
        self.contrast_value_label = QLabel('100')
        contrast_layout.addWidget(self.contrast_value_label)
        bc_layout.addLayout(contrast_layout)

        self.brightness_slider.valueChanged.connect(self.on_brightness_changed)
        self.contrast_slider.valueChanged.connect(self.on_contrast_changed)

        bc_group.setLayout(bc_layout)
        right_panel.addWidget(bc_group)
        
        # Группа для изменения цветности изображения
        color_group = QGroupBox('Изменение цветности')
        color_layout = QVBoxLayout()
        
        # Элементы управления бинаризацией
        binary_layout = QVBoxLayout()
        binary_method_layout = QHBoxLayout()
        binary_method_layout.addWidget(QLabel('Метод бинаризации:'))
        self.binary_combo = QComboBox()
        self.binary_combo.addItems(['Фиксированный порог', 'Адаптивный порог', 'Метод Оцу'])
        self.binary_combo.currentTextChanged.connect(self.on_binary_method_changed)
        binary_method_layout.addWidget(self.binary_combo)
        
        # Layout для элементов порога бинаризации
        self.threshold_layout = QHBoxLayout()
        self.threshold_label = QLabel('Порог:')
        self.threshold_layout.addWidget(self.threshold_label)
        self.binary_threshold_slider = QSlider(Qt.Horizontal)
        self.binary_threshold_slider.setRange(0, 255)
        self.binary_threshold_slider.setValue(128)
        self.binary_threshold_slider.valueChanged.connect(self.binary_threshold_changed)
        self.threshold_layout.addWidget(self.binary_threshold_slider)
        self.binary_threshold_label = QLabel('128')
        self.threshold_layout.addWidget(self.binary_threshold_label)
        
        binary_layout.addLayout(binary_method_layout)
        binary_layout.addLayout(self.threshold_layout)
        
        # Кнопка применения бинаризации
        apply_binary_btn = QPushButton('Применить бинаризацию')
        apply_binary_btn.clicked.connect(self.apply_binarization)
        binary_layout.addWidget(apply_binary_btn)
        
        color_layout.addLayout(binary_layout)
        
        # Кнопки других преобразований цветности
        transform_layout = QHBoxLayout()
        
        gray_btn = QPushButton('Оттенки серого')
        gray_btn.clicked.connect(self.apply_grayscale)
        transform_layout.addWidget(gray_btn)
        
        negative_btn = QPushButton('Негатив')
        negative_btn.clicked.connect(self.apply_negative)
        transform_layout.addWidget(negative_btn)
        
        # Кнопка для преобразования гистограмм (выравнивание)
        hist_equalize_btn = QPushButton('Выровнять гистограммы')
        hist_equalize_btn.clicked.connect(self.apply_histogram_equalization)
        transform_layout.addWidget(hist_equalize_btn)
        
        color_layout.addLayout(transform_layout)
        color_group.setLayout(color_layout)
        right_panel.addWidget(color_group)
        
        # Добавление левой и правой панелей в основной layout
        # Правой панели даем больше места, чтобы гистограммы не были "сплюснутыми" (256 столбцов)
        main_layout.addLayout(left_panel, 3) 
        main_layout.addLayout(right_panel, 2)
        
        # Изначальная настройка видимости элементов бинаризации
        self.on_binary_method_changed(self.binary_combo.currentText())
    
    # Обработчик изменения метода бинаризации
    def on_binary_method_changed(self, method):
        if method == 'Фиксированный порог':
            # Для фиксированного порога показываем элементы управления порогом
            self.threshold_label.setVisible(True)
            self.binary_threshold_slider.setVisible(True)
            self.binary_threshold_label.setVisible(True)
        else:
            # Для адаптивных методов скрываем элементы управления порогом
            self.threshold_label.setVisible(False)
            self.binary_threshold_slider.setVisible(False)
            self.binary_threshold_label.setVisible(False)

    # обработчики обновления слайдеров
    def on_brightness_changed(self, v):
        self.brightness_value_label.setText(str(v))
        self.brightness_value = v
        self.update_timer.start(150) 

    def on_contrast_changed(self, v):
        self.contrast_value_label.setText(str(v))
        self.contrast_value = v
        self.update_timer.start(150)

    # Загрузка изображения из файла
    def load_image(self):
        # Открываем стандартное диалоговое окно Windows для выбора файла.
        # Возвращает кортеж: (путь_к_файлу, выбранный_фильтр). 
        # Нам нужен только путь, поэтому второй элемент мы записываем в '_'.
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Выберите изображение", "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)")
        
        if file_name:
            # Загружаем изображение через OpenCV. 
            # OpenCV читает картинку в формате BGR (Blue-Green-Red), 
            # а не привычном для нас RGB.
            self.original_array = cv2.imread(file_name)
            if self.original_array is None:
                QMessageBox.warning(self, "Ошибка", "Не удалось загрузить изображение")
                return
            
            # Конвертируем из BGR (формат OpenCV) в RGB (формат для отображения)
            self.original_array = cv2.cvtColor(self.original_array, cv2.COLOR_BGR2RGB)
            self.current_array = self.original_array.copy()

            # Сбрасываем выделение и ползунки, чтобы новое изображение
            # показывалось без преобразований от предыдущей картинки
            self.selection_rect = None
            self.selection_active = False
            self.reset_sliders()
            self.binary_threshold_slider.setValue(128)

            # Гистограммы строим заново: наборы столбцов пересоздаются под новое изображение
            self.init_histogram()

            # Обновляем изображение (гистограммы уже построены выше)
            self.update_display(update_hist=False)
    
    # Настройка столбцов гистограммы.
    # У непустого столбца делаем обводку того же цвета: у гистограммы из 256 столбцов
    # один столбец занимает меньше пикселя, и без обводки "иглы" (например, после
    # бинаризации, когда есть только яркости 0 и 255) становятся невидимыми.
    # У пустых столбцов обводку убираем, иначе они рисуют линию по всей оси X.
    def style_hist_bars(self, patches, color):
        for patch in patches:
            if patch.get_height() > 0:
                patch.set_linewidth(0.9)
                patch.set_edgecolor(color)
            else:
                patch.set_linewidth(0.0)

    # Инициализация гистограмм при загрузке изображения
    def init_histogram(self):
        if self.current_array is None:
            return

        # Получаем каналы изображения
        # .flatten() — превращает двумерную матрицу (Высота * Ширина) в одномерный массив (список пикселей)
        r = self.current_array[:, :, 0].flatten()
        g = self.current_array[:, :, 1].flatten()
        b = self.current_array[:, :, 2].flatten()

        # Вычисляем яркость по формуле из лекции Y = 0.299R + 0.5876G + 0.114B
        brightness = 0.299 * self.current_array[:, :, 0] + 0.5876 * self.current_array[:, :, 1] + 0.114 * self.current_array[:, :, 2]
        brightness = brightness.flatten()

        # размеры шрифтов
        title_font = 10
        label_font = 9
        tick_font = 8

        # Проходимся по всем 4-м осям и очищаем их
        for ax in self.hist_canvas.axes:
            ax.clear()

        # Метод .hist() делает сразу две вещи: считает частоту пикселей и рисует столбики.
        # Он возвращает кортеж из 3 элементов:
        # counts_r — массив с частотами (сколько пикселей попало в каждый интервал).
        # bins_r — границы интервалов (от 0 до 256).
        # patches_r — список объектов-столбиков (прямоугольников)

        # Красный канал (ось 0)
        # r - данные: одномерный массив со значениями красного канала
        # bins - разбиваем на 256 интервалов
        # range - задаем диапазон значений 
        # alhpa - прозрачность столбиков (70%)
        # color - цвет столбиков
        
        counts_r, bins_r, patches_r = self.hist_canvas.axes[0].hist(r, bins=256, range=(0, 256), alpha=0.7, color='red')
        self.style_hist_bars(patches_r, 'red')
        self.hist_canvas.axes[0].set_title('Красный канал', fontsize=title_font)
        self.hist_canvas.axes[0].set_xlabel('Яркость', fontsize=label_font)
        self.hist_canvas.axes[0].set_ylabel('Частота', fontsize=label_font)

        # Зеленый канал (ось 1)
        counts_g, bins_g, patches_g = self.hist_canvas.axes[1].hist(g, bins=256, range=(0, 256), alpha=0.7, color='green')
        self.style_hist_bars(patches_g, 'green')
        self.hist_canvas.axes[1].set_title('Зеленый канал', fontsize=title_font)
        self.hist_canvas.axes[1].set_xlabel('Яркость', fontsize=label_font)
        self.hist_canvas.axes[1].set_ylabel('Частота', fontsize=label_font)

        # Синий канал (ось 2)
        counts_b, bins_b, patches_b = self.hist_canvas.axes[2].hist(b, bins=256, range=(0, 256), alpha=0.7, color='blue')
        self.style_hist_bars(patches_b, 'blue')
        self.hist_canvas.axes[2].set_title('Синий канал', fontsize=title_font)
        self.hist_canvas.axes[2].set_xlabel('Яркость', fontsize=label_font)
        self.hist_canvas.axes[2].set_ylabel('Частота', fontsize=label_font)

        # Вычисленная яркость (ось 3)
        counts_y, bins_y, patches_y = self.hist_canvas.axes[3].hist(brightness, bins=256, range=(0, 256), alpha=0.7, color='gray')
        self.style_hist_bars(patches_y, 'gray')
        self.hist_canvas.axes[3].set_title('Яркость', fontsize=title_font)
        self.hist_canvas.axes[3].set_xlabel('Яркость', fontsize=label_font)
        self.hist_canvas.axes[3].set_ylabel('Частота', fontsize=label_font)

        # Общие настройки: размер тиков и читаемость оси Y
        for ax in self.hist_canvas.axes:
            ax.tick_params(axis='both', which='major', labelsize=tick_font)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
            # чуть отодвинем подпись Y чтобы не цеплялась за левую рамку
            ax.yaxis.labelpad = 6
            # Небольшой отступ по X: иначе столбцы, попавшие в интервалы 0 и 255,
            # рисуются вплотную к рамке и полностью под ней скрываются
            ax.set_xlim(-self.hist_margin, 256 + self.hist_margin)

        # Сохраняем ссылки на столбцы гистограмм для последующего обновления
        self.hist_bars_r = patches_r
        self.hist_bars_g = patches_g
        self.hist_bars_b = patches_b
        self.hist_bars_y = patches_y
        # Ставим флаг, что графики созданы
        self.hist_initialized = True

        self.hist_canvas.draw()

    # Обновление данных гистограмм без изменения масштаба и осей
    # Этот метод вызывается при каждом изменении картинки (яркость, контраст, фильтры и т.д.)
    def update_histogram(self):
        if self.current_array is None or not self.hist_initialized:
            return
            
        # Получаем каналы изображения
        r = self.current_array[:,:,0].flatten()
        g = self.current_array[:,:,1].flatten()
        b = self.current_array[:,:,2].flatten()
        
        # Вычисляем яркость текущего изображения
        brightness = 0.299 * self.current_array[:,:,0] + 0.5876 * self.current_array[:,:,1] + 0.114 * self.current_array[:,:,2]
        brightness = brightness.flatten()
        # Здесь мы используем np.histogram(), а не ax.hist().
        # np.histogram() считает только частоты (сколько пикселей в каком интервале).
        # Он не рисует график. Это работает в разы быстрее.
        # Мы получаем counts (массив с частотами) и bins (границы интервалов).
        counts_r, bins_r = np.histogram(r, bins=256, range=(0, 256))
        counts_g, bins_g = np.histogram(g, bins=256, range=(0, 256))
        counts_b, bins_b = np.histogram(b, bins=256, range=(0, 256))
        counts_y, bins_y = np.histogram(brightness, bins=256, range=(0, 256))
        
        # Обновляем высоты столбцов и их обводку 
        for counts, patches, color in ((counts_r, self.hist_bars_r, 'red'),
                                       (counts_g, self.hist_bars_g, 'green'),
                                       (counts_b, self.hist_bars_b, 'blue'),
                                       (counts_y, self.hist_bars_y, 'gray')):
            for count, patch in zip(counts, patches):
                patch.set_height(count) # Меняем высоту готового столбика на новое значение
            self.style_hist_bars(patches, color)
        
        # Автоматически масштабируем оси Y для лучшего отображения
        # (вручную задаем лимиты, чтобы правый столбик на 255 не пропадал)
        for ax, counts in zip(self.hist_canvas.axes, [counts_r, counts_g, counts_b, counts_y]):
            # Отступ по X, чтобы крайние столбцы (0 и 255) не прятались под рамкой графика
            ax.set_xlim(-self.hist_margin, 256 + self.hist_margin)
            # Берем максимальную высоту столбика и добавляем 10% запаса сверху
            max_h = max(1, np.max(counts))
            ax.set_ylim(0, max_h * 1.1)
        
        # Перерисовываем гистограммы
        self.hist_canvas.draw_idle()

    # преобразование numpy array в QImage
    # OpenCV хранит картинку как массив NumPy, а PyQt5 (для отображения) требует объект QImage
    def array_to_qimage(self, array):
        # QImage требует непрерывный (C-contiguous) буфер в памяти
        array = np.ascontiguousarray(array)
        height, width, channels = array.shape # Получаем размеры: высоту, ширину и количество каналов (3 для RGB)
        # bytes_per_line (шаг сканирования) — это объем памяти (в байтах), который занимает одна строка пикселей.
        bytes_per_line = channels * width # (3 канала * ширина) байт.
        # QImage не копирует данные, поэтому array должен жить, пока картинка используется
        return QImage(array.data, width, height, bytes_per_line, QImage.Format_RGB888)
    
    # Обновление отображения изображения и гистограмм
    def update_display(self, update_hist=True): 
        if self.current_array is not None:
            # Создаем копию для отображения
            display_array = self.current_array.copy()
            
            # Масштабируем изображение для отображения
            h, w = display_array.shape[:2]
            max_display_size = 800
            if max(h, w) > max_display_size:
                scale = max_display_size / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                display_array = cv2.resize(display_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Преобразуем numpy array в QImage и затем в QPixmap для отображения
            qimage = self.array_to_qimage(display_array)
            pixmap = QPixmap.fromImage(qimage)
            
            # Если есть активное выделение, рисуем его поверх изображения
            if self.selection_active and self.selection_rect:
                # Создаем painter для рисования поверх pixmap
                painter = QPainter(pixmap)
                
                # Настраиваем перо для тонкой пунктирной линии
                pen = QPen(Qt.black)
                pen.setWidth(1)
                pen.setStyle(Qt.DashLine)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)  # Прозрачная заливка
                
                # Масштабируем координаты выделения к отображаемому размеру
                scale_x = pixmap.width() / self.current_array.shape[1]
                scale_y = pixmap.height() / self.current_array.shape[0]
                
                x = int(self.selection_rect[0] * scale_x)
                y = int(self.selection_rect[1] * scale_y)
                w = int(self.selection_rect[2] * scale_x)
                h = int(self.selection_rect[3] * scale_y)
                
                # Рисуем прямоугольник выделения
                painter.drawRect(x, y, w, h)
                painter.end()
            
            # Устанавливаем pixmap в метку для отображения
            self.image_label.setPixmap(pixmap)
            
            # Обновляем гистограммы только если нужно
            if update_hist:
                if self.hist_initialized:
                    self.update_histogram()
                else:
                    self.init_histogram()
    
    # Обработчик нажатия на слайдер (ничего не делаем)
    def slider_pressed(self):
        pass
    
    # Обработчик отпускания слайдера - применяем преобразования
    def slider_released(self):
        # Забираем финальные значения с ползунков и сохраняем их в переменные класса.
        self.brightness_value = self.brightness_slider.value()
        self.contrast_value = self.contrast_slider.value()
        # Обновляем метки значений
        self.brightness_value_label.setText(str(self.brightness_value))
        self.contrast_value_label.setText(str(self.contrast_value))
        # Применяем преобразования
        self.apply_all_transforms()
    
    # Обработчик изменения порога бинаризации
    # Просто обновляет текстовую метку и сохраняет значение в переменную 
    def binary_threshold_changed(self, value):
        self.binary_threshold = value
        self.binary_threshold_label.setText(str(value))

    # Есть ли реально выделенная область (а не только включенный режим выделения).
    # Нужна, чтобы преобразования не применялись к "пустому" прямоугольнику 0*0
    def has_selection(self):
        return (self.selection_active and self.selection_rect is not None
                and self.selection_rect[2] > 0 and self.selection_rect[3] > 0)

    # Применение преобразований яркости и контрастности к изображению
    def apply_all_transforms(self):
        # Если картинка не загружена, ничего не делаем
        if self.original_array is None:
            return
        
        # 1. Обработка выделенной области

        if self.has_selection():
            # Распаковываем координаты выделенного прямоугольника
            # x, y — координаты левого верхнего угла прямоугольника 
            # (отступ от левого края картинки и от верхнего края соответственно)
            # w (width), h (height) — ширина и высота выделенного прямоугольника
            x, y, w, h = self.selection_rect 
            # Выделяем область
            roi = self.original_array[y:y+h, x:x+w].copy().astype(np.float32) # переводим в числа с плавающей точкой, 
                                                                              # чтобы при вычитании среднего и умножении на контраст не было переполнения uint8
            
            # Применяем контрастность по формуле из лекции
            if self.contrast_value != 100:
                contrast = self.contrast_value / 100.0
                # Вычисляем средние значения для каждого канала
                mean_r = np.mean(roi[:,:,0])
                mean_g = np.mean(roi[:,:,1])
                mean_b = np.mean(roi[:,:,2])
                
                # Применяем преобразование контрастности
                roi[:,:,0] = contrast * (roi[:,:,0] - mean_r) + mean_r
                roi[:,:,1] = contrast * (roi[:,:,1] - mean_g) + mean_g
                roi[:,:,2] = contrast * (roi[:,:,2] - mean_b) + mean_b
            
            # Применяем яркость (простое сложение/вычитание)
            if self.brightness_value != 0:
                roi += self.brightness_value
            
            # Обрезаем значения до допустимого диапазона [0, 255]
            # возвращаем тип uint8 (его требует QImage для отображения)
            roi = np.clip(roi, 0, 255).astype(np.uint8)
            
            # Обновляем только выделенную область в текущем изображении
            self.current_array = self.original_array.copy()
            self.current_array[y:y+h, x:x+w] = roi

        # 2. Обработка всей картинки
        else:
            # Обрабатываем все изображение
            self.current_array = self.original_array.copy().astype(np.float32)
            
            # Применяем контрастность
            if self.contrast_value != 100:
                contrast = self.contrast_value / 100.0
                mean_r = np.mean(self.current_array[:,:,0])
                mean_g = np.mean(self.current_array[:,:,1])
                mean_b = np.mean(self.current_array[:,:,2])
                
                self.current_array[:,:,0] = contrast * (self.current_array[:,:,0] - mean_r) + mean_r
                self.current_array[:,:,1] = contrast * (self.current_array[:,:,1] - mean_g) + mean_g
                self.current_array[:,:,2] = contrast * (self.current_array[:,:,2] - mean_b) + mean_b
            
            # Применяем яркость
            if self.brightness_value != 0:
                self.current_array += self.brightness_value
            
            # Обрезаем значения и конвертируем обратно в uint8
            self.current_array = np.clip(self.current_array, 0, 255).astype(np.uint8)
        
        # Обновляем отображение
        self.update_display()
    
    # Применение бинаризации к изображению
    def apply_binarization(self):
        # если картинка не загружена, ничего не делаем
        if self.current_array is None:
            return
        # Получаем выбранный метод из выпадающего списка
        method = self.binary_combo.currentText()

        # 1. Обработка выделенной области 
        if self.has_selection():
            # Определяем область для обработки
            x, y, w, h = self.selection_rect

            # Берем область и переводим в float для ручного расчета яркости
            roi_float = self.current_array[y:y+h, x:x+w].astype(np.float32)
            # Ручной перевод в grayscale
            gray = self.rgb_to_gray(roi_float)

            # Применяем выбранный метод
            binary = self.binarize(gray, method)

            # Собираем обратно в RGB
            binary_rgb = np.stack((binary, binary, binary), axis=-1)

            # Обновляем ТОЛЬКО выделенную область
            self.current_array[y:y+h, x:x+w] = binary_rgb
        else:
            # 2. Все изображение

            # Ручной перевод в grayscale
            gray = self.rgb_to_gray(self.current_array.astype(np.float32))

            # Применяем выбранный метод
            binary = self.binarize(gray, method)

            # Собираем обратно в RGB
            self.current_array = np.stack((binary, binary, binary), axis=-1)

        self.update_display()

    # Перевод RGB -> яркость (серый) по формуле из лекции: Y = 0.299R + 0.5876G + 0.114B
    def rgb_to_gray(self, rgb_float):
        gray = 0.299 * rgb_float[:,:,0] + 0.5876 * rgb_float[:,:,1] + 0.114 * rgb_float[:,:,2]
        # Ограничиваем диапазон и переводим обратно в uint8
        return np.clip(gray, 0, 255).astype(np.uint8)

    # Бинаризация серого изображения выбранным методом
    def binarize(self, gray, method):
        if method == 'Фиксированный порог':
            threshold = self.binary_threshold_slider.value()
            # если пиксель > порог -> 255, иначе -> 0
            return np.where(gray > threshold, 255, 0).astype(np.uint8)

        if method == 'Адаптивный порог':
            # Адаптивная бинаризация
            # 1. gray - исходное одноканальное (серое) изображение.
            # 2. 255 - значение (белый цвет), если пиксель > локального порога
            # 3. cv2.ADAPTIVE_THRESH_GAUSSIAN_C - способ вычисления
            # 4. cv2.THRESH_BINARY - тип (каким пиксель станет после
            #    сравнения с порогом): пиксель > T -> 255, иначе -> 0
            # 5. block - размер окрестности (11*11 пикселей), должен быть нечётным
            #    и не больше самого изображения (важно для маленького выделения)
            # 6. 2 - константа, вычитаемая из среднего

            # 1. Защита от слишком маленького выделения (окресность 11*11)
            block = min(11, gray.shape[0], gray.shape[1])
            # 2. OpenCV требует, чтобы размер окрестности был строго нечетный
            if block % 2 == 0:
                block -= 1
            # 3. Минимальный допустимый размер окна в OpenCV для адаптивного порога — это 3
            block = max(3, block)
            return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, block, 2)

        # Метод Оцу
        # 1. gray - исходное одноканальное (серое) изображение.
        # 2. 0 - заглушка (порог). При использовании Оцу это значение игнорируется, 
        # так как алгоритм сам найдет оптимальный порог.
        # 3. 255 - значение (белый цвет), если пиксель > найденного порога.
        # 4. cv2.THRESH_BINARY + cv2.THRESH_OTSU - комбинированный тип:
        # THRESH_BINARY задает правило (пиксель > T -> 255, иначе -> 0),
        # а THRESH_OTSU используется для вычисления порога.
        # Возвращает: (найденный_порог, бинарное_изображение). 
        # Порог пишем в '_', так как для нас он не важен.
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    
    # Преобразование изображения в оттенки серого
    def apply_grayscale(self):
        if self.current_array is None:
            return
            
        # 1. Обработка выделенной области 
        if self.has_selection():
            x, y, w, h = self.selection_rect   

            # Переводим в float32, чтобы избежать переполнения при умножении
            roi = self.current_array[y:y+h, x:x+w].copy().astype(np.float32)
            
            # Ручной расчет по формуле: gray = 0.299R + 0.5876G + 0.114B
            gray = self.rgb_to_gray(roi)
            
            # R' = G' = B' = серый (собираем 3 одинаковых канала в один массив)
            gray_rgb = np.stack((gray, gray, gray), axis=-1)
            
            # Обновляем только выделенную область
            self.current_array[y:y+h, x:x+w] = gray_rgb
            
        else:
            # 2. Обрабатываем все изображение
            roi = self.current_array.copy().astype(np.float32)
            
            # Ручной расчет
            gray = self.rgb_to_gray(roi)
            
            # R' = G' = B' = серый
            self.current_array = np.stack((gray, gray, gray), axis=-1)
        
        self.update_display()
    
    # Преобразование изображения в негатив
    def apply_negative(self):
        if self.current_array is None:
            return
        
        # 1. Обработка выделенной области 
        
        # Определяем область для обработки
        if self.has_selection():
            x, y, w, h = self.selection_rect
            # Инвертируем цвета в выделенной области
            self.current_array[y:y+h, x:x+w] = 255 - self.current_array[y:y+h, x:x+w]
        else:
            # 2. Обрабатываем все изображение

            # Инвертируем цвета во всем изображении
            self.current_array = 255 - self.current_array
        
        self.update_display()
    
    # Выравнивание гистограмм (преобразование гистограмм)
    def apply_histogram_equalization(self):
        if self.current_array is None:
            return
        
        # 1. Обработка выделенной области 

        # Определяем область для обработки
        if self.has_selection():
            x, y, w, h = self.selection_rect
            roi = self.current_array[y:y+h, x:x+w].copy()
            
            # Применяем выравнивание гистограммы к каждому каналу
            # Преобразуем в YUV для работы с яркостью.
            # Если выравнивать каналы R, G, B по отдельности, изменится цветовой баланс
            # (картинка может уйти в синеву или зелень)
            yuv = cv2.cvtColor(roi, cv2.COLOR_RGB2YUV)
            # Выравниваем гистограмму яркостного канала
            yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
            # Преобразуем обратно в RGB
            equalized_roi = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
            
            # Обновляем только выделенную область
            self.current_array[y:y+h, x:x+w] = equalized_roi
        else:
            # 2. Обрабатываем все изображение

            # Обрабатываем все изображение
            # Преобразуем в YUV для работы с яркостью
            yuv = cv2.cvtColor(self.current_array, cv2.COLOR_RGB2YUV)
            # Выравниваем гистограмму яркостного канала
            yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
            # Преобразуем обратно в RGB
            self.current_array = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
        
        self.update_display()
    
    # Сброс изображения к оригинальному состоянию
    def reset_image(self):
        if self.original_array is not None:
            self.current_array = self.original_array.copy()
            self.update_display()
            self.reset_sliders()
    
    # Сброс слайдеров к начальным значениям
    def reset_sliders(self):
        self.brightness_slider.setValue(0)
        self.contrast_slider.setValue(100)
        self.brightness_value = 0
        self.contrast_value = 100
        self.brightness_value_label.setText('0')
        self.contrast_value_label.setText('100')
    
    # Включение/выключение режима выделения области
    def toggle_selection(self):
        self.selection_active = not self.selection_active
        if self.selection_active:
            self.statusBar().showMessage('Режим выделения: рисуйте прямоугольник на изображении')
        else:
            self.statusBar().showMessage('')
            self.clear_selection()
    
    # Очистка выделенной области
    def clear_selection(self):
        self.selection_rect = None
        self.selection_active = False
        self.update_display()
        self.statusBar().showMessage('')
    
    # Обработчик нажатия кнопки мыши на изображении (начало выделения)
    def image_mouse_press(self, event):
        if self.selection_active and self.current_array is not None:
            pos = event.pos()
            label_size = self.image_label.size()
            pixmap_size = self.image_label.pixmap().size() if self.image_label.pixmap() else label_size
            
            # Вычисляем смещение для центрированного изображения
            offset_x = (label_size.width() - pixmap_size.width()) // 2
            offset_y = (label_size.height() - pixmap_size.height()) // 2
            
            # Проверяем, что клик внутри изображения
            if (offset_x <= pos.x() < offset_x + pixmap_size.width() and
                offset_y <= pos.y() < offset_y + pixmap_size.height()):
                
                # Пересчитываем координаты в систему изображения (с учетом масштабирования)
                img_x = int((pos.x() - offset_x) * self.current_array.shape[1] / pixmap_size.width())
                img_y = int((pos.y() - offset_y) * self.current_array.shape[0] / pixmap_size.height())
                
                # Сохраняем начальную точку выделения
                self.selection_start = (img_x, img_y)
                self.selection_rect = (img_x, img_y, 0, 0)
    
    # Обработчик движения мыши при выделении области
    def image_mouse_move(self, event):
        if (self.selection_active and self.selection_start is not None and 
            self.current_array is not None):
            
            pos = event.pos()
            label_size = self.image_label.size()
            pixmap_size = self.image_label.pixmap().size() if self.image_label.pixmap() else label_size
            
            # Вычисляем смещение для центрированного изображения
            offset_x = (label_size.width() - pixmap_size.width()) // 2
            offset_y = (label_size.height() - pixmap_size.height()) // 2
            
            # Проверяем, что движение внутри изображения
            if (offset_x <= pos.x() < offset_x + pixmap_size.width() and
                offset_y <= pos.y() < offset_y + pixmap_size.height()):
                
                # Пересчитываем координаты в систему изображения
                img_x = int((pos.x() - offset_x) * self.current_array.shape[1] / pixmap_size.width())
                img_y = int((pos.y() - offset_y) * self.current_array.shape[0] / pixmap_size.height())
                
                start_x, start_y = self.selection_start
                
                # Вычисляем координаты прямоугольника с учетом направления движения
                x1 = min(start_x, img_x)
                y1 = min(start_y, img_y)
                x2 = max(start_x, img_x)
                y2 = max(start_y, img_y)
                
                # Ограничиваем выделение размерами изображения
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(self.current_array.shape[1], x2)
                y2 = min(self.current_array.shape[0], y2)
                
                width = x2 - x1
                height = y2 - y1
                
                # Обновляем координаты выделения
                self.selection_rect = (x1, y1, width, height)
                self.update_display(update_hist=False)
    
    # Обработчик отпускания кнопки мыши (окончание выделения)
    def image_mouse_release(self, event):
        if self.selection_active and self.selection_start is not None:
            self.selection_start = None
            if self.selection_rect is not None and self.selection_rect[2] > 0 and self.selection_rect[3] > 0:
                # Показываем информацию о размере выделенной области
                self.statusBar().showMessage(f'Выделена область: {self.selection_rect[2]}x{self.selection_rect[3]} пикселей')
            else:
                self.clear_selection()
    
    # Обработчик изменения размера окна - просто перерисовываем изображение
    def resizeEvent(self, event):
        # На случай, если событие пришло до полной инициализации окна
        if getattr(self, 'image_label', None) is not None:
            self.update_display()
        super().resizeEvent(event)


# Класс для встраивания matplotlib графиков в PyQt приложение
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=6, height=4, dpi=100):
        # Включаем constrained_layout, чтобы matplotlib автоматически распределял места
        self.fig = Figure(figsize=(width, height), dpi=dpi, constrained_layout=True)
        # Создаем 4 подграфика для отображения 4 гистограмм
        self.axes = [
            self.fig.add_subplot(2, 2, 1),  # Красный канал
            self.fig.add_subplot(2, 2, 2),  # Зеленый канал
            self.fig.add_subplot(2, 2, 3),  # Синий канал
            self.fig.add_subplot(2, 2, 4)   # Яркость
        ]
        super().__init__(self.fig)
        self.setParent(parent)



app = QApplication(sys.argv)
viewer = ImageViewer()
viewer.show()
sys.exit(app.exec_())
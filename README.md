# Show and Tell

Модель Show and Tell является примером нейронной сети кодер-декодер, работа которой проходит в два этапа: сначала изображение кодируется в векторное представление фиксированной длины, а затем декодируется в описание на естественном языке.

Кодер изображения представляет собой сверточную нейронную сеть. Этот тип сети широко используется для задач распознавания и обнаружения объектов на изображении. В данном случае выбрана сеть распознавания изображений Inception v3, предварительно подготовленная на наборе данных классификации изображений ILSVRC-2012-CLS .

Декодер - это сеть с кратковременной памятью (LSTM). Этот тип сети обычно используется для задач моделирования последовательностей, таких как языковое моделирование и машинный перевод. В модели Show and Tell сеть LSTM обучается как языковая модель, обусловленная кодированием изображения.

Слова в описании изображения представлены с помощью модели вложения. Каждое слово в словаре связано с векторным представлением фиксированной длины, которое подбирается во время обучения сети.

### Аннотирование видео-файла

Для аннотирования видео-файла в модель Show and Tell добавлен алгоритм поиска различающихся кадров, основанный на индексе структурного сходства (SSIM от англ. structure similarity). А также, для более точного результата, на выходе нейронной сети добавлен еще один, уже семантический, фильтр. Он сравнивает аннотации к двум соседним  изображениям и, если аннотации отличаются более чем на 55%, то подписывает и сохраняет изображение в папку out в корне проекта.

### Перевод аннотаций на русский язык

Все аннотации в Show and Tell выполняются на английском языке. Для перевода их на русский язык используется сайт Яндекс.Переводчика и программная библиотека для управления браузерами Selenium WebDriver.

## Использование

### Подготовка
Для работы с программой необходимо установить следующие пакеты:
* Сборщик проекта [Bazel](http://bazel.io/docs/install.html)
* [TensorFlow](https://www.tensorflow.org/install/) 1.9 или новее
* [Чекпоинты](https://github.com/Gharibim/Tensorflow_im2txt_5M_Step) полностью обученной модели (положить в папку model)
* NumPy
* opencv-python
* selenium

> Рекомендуется создать отдельное виртуальное окружение с python 3.6
>```shell
>conda create -n venv pip python=3.6
>conda activate venv
>pip install tensorflow
>pip install opencv-python
>```

### Сборка и запуск
```shell
# Путь к файлу контрольной точки.
CHECKPOINT_PATH="/path/to/model.ckpt-5000000"

# Путь к файлу словаря.
VOCAB_FILE="/path/to/word_counts.txt"

# Путь к видео-файлу.
VIDEO_FILE="/path/to/video.mp4"

# Сборка.
bazel build -c opt im2txt/run_inference

# Запуск.
bazel-bin/im2txt/run_inference \
  --checkpoint_path=${CHECKPOINT_PATH} \
  --vocab_file=${VOCAB_FILE} \
  --input_files=${VIDEO_FILE}
```

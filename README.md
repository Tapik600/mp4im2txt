# Show and Tell
> Исходный репозиторий: [Tensorflow_im2txt](https://github.com/tensorflow/models/tree/master/research/im2txt)
### Подготовка
Для работы с программой необходимо установить следующие пакеты:
* Сборщик проекта [Bazel](http://bazel.io/docs/install.html)
* [TensorFlow](https://www.tensorflow.org/install/) 1.9 или новее
* [Чекпоинты](https://github.com/Gharibim/Tensorflow_im2txt_5M_Step) полностью обученной модели (положить в папку model)
* NumPy
* opencv-python

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


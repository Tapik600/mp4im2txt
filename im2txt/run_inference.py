# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Generate captions for images using default beam search parameters."""
# bazel build -c opt im2txt/run_inference
# bazel-bin/im2txt/run_inference

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

import tensorflow as tf

import cv2
import numpy as np
from typing import List, Iterable
from shutil import rmtree
from difflib import SequenceMatcher

from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("input_files", "",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")

tf.logging.set_verbosity(tf.logging.INFO)

# ======================================================================================================================
# VideoReader
# ----------------------------------------------------------------------------------------------------------------------
class FrameHolder:
    def __init__(self, frame: np.ndarray, index_number: int):
        self.frame = frame
        self.index_number = index_number


class VideoReader:

    def __init__(self, path_to_video_file: str):
        self._path_to_video = path_to_video_file

    def read_all_frames(self) -> List[FrameHolder]:
        return list(self.get_frames_one_by_one_generator())

    def get_frames_one_by_one_generator(self) -> Iterable[FrameHolder]:
        video = cv2.VideoCapture(self._path_to_video)
        index = 0
        while video.grab():
            ret, frame = video.retrieve()

            indexed_frame = FrameHolder(frame, index)
            index += 1
            yield indexed_frame

        video.release()
# ----------------------------------------------------------------------------------------------------------------------
def covariation(matrix_1: np.ndarray, matrix_2: np.ndarray):
    width = matrix_1.shape[1]
    height = matrix_1.shape[0]

    mean_m1 = np.mean(matrix_1)
    mean_m2 = np.mean(matrix_2)

    sum = 0
    for y in range(height):
        for x in range(width):
            sum += (matrix_1[y][x] - mean_m1) * (matrix_2[y][x] - mean_m2)

    return sum / (width * height)
# ----------------------------------------------------------------------------------------------------------------------
def ssim(matrix_1: np.ndarray, matrix_2: np.ndarray):
    """
    SSIM is used for measuring the similarity between two images.
    The resultant SSIM index is a decimal value between -1 and 1,
    and value 1 is only reachable in the case of two identical sets of data.

    :param matrix_1:
    :param matrix_2:
    :return:
    """
    mean_m1 = np.mean(matrix_1)
    mean_m2 = np.mean(matrix_2)
    var_m1 = np.var(matrix_1)
    var_m2 = np.var(matrix_2)
    return (
            (covariation(matrix_1, matrix_2) / (math.sqrt(var_m1) * math.sqrt(var_m2))) *

            ((2 * mean_m1 * mean_m2) / (mean_m1 ** 2 + mean_m2 ** 2)) *

            ((2 * math.sqrt(var_m1) * math.sqrt(var_m2)) / (var_m1 + var_m2))
    )
# ----------------------------------------------------------------------------------------------------------------------
def filter_by_ssim(images: Iterable[FrameHolder], threshold=0.7) -> List[FrameHolder]:
    current = None
    filtered_list = []
    for item in images:
        if current is None:
            current = item
            filtered_list.append(item)
            continue

        if ssim(cv2.resize(cv2.cvtColor(current.frame, cv2.COLOR_BGR2GRAY), (32, 32)),
                cv2.resize(cv2.cvtColor(item.frame, cv2.COLOR_BGR2GRAY), (32, 32))) < threshold:
            current = item
            filtered_list.append(item)

    return filtered_list
# ======================================================================================================================

def main(_):
    # Build the inference graph.
    g = tf.Graph()
    with g.as_default():
        model = inference_wrapper.InferenceWrapper()
        restore_fn = model.build_graph_from_config(configuration.ModelConfig(), "model/model.ckpt-5000000")
    g.finalize()

    # Create the vocabulary.
    vocab = vocabulary.Vocabulary("word_counts.txt")

    rmtree("out", ignore_errors=True)
    vr = VideoReader("resources/dobr.mp4")
    frames = vr.read_all_frames()

    # SSIM sequence
    os.makedirs("out")

    with tf.Session(graph=g) as sess:
        # Load the model from checkpoint.
        restore_fn(sess)

        # Prepare the caption generator. Here we are implicitly using the default
        # beam search parameters. See caption_generator.py for a description of the
        # available beam search parameters.
        generator = caption_generator.CaptionGenerator(model, vocab)

        prev_sentence, count_picIn, count_picOut, threshold = ("", 0, 0, 0.45)

        for image in filter_by_ssim(frames, threshold):
            file_path = os.path.join("out", str(image.index_number) + '.jpg')

            captions = generator.beam_search(sess, cv2.imencode('.jpg', image.frame)[1].tostring())

            sentence = [vocab.id_to_word(w) for w in captions[0].sentence[1:-1]]
            sentence = " ".join(sentence)

            ratio = SequenceMatcher(None, sentence, prev_sentence).ratio()
            print("=========== %f ===========" % ratio)

            count_picIn += 1

            print("Captions for image %s: %s (P=%f)" %
                  (str(image.index_number), sentence, math.exp(captions[0].logprob)))

            prev_sentence = sentence

            if ratio <= 0.55:
                count_picOut += 1
                font = cv2.FONT_HERSHEY_PLAIN
                out_im = cv2.line(image.frame, (0, 350), (640, 350), (10, 10, 10), 30)
                out_im = cv2.putText(out_im, sentence, (10, 350), font, 1.2, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imwrite(file_path, out_im)

        print("----------------------------------------------------")
        print(" count_picIn = %d " % count_picIn)
        print(" count_picOut = %d " % count_picOut)
        print(" picIn - picOut = %d " % (count_picIn-count_picOut))


if __name__ == "__main__":
    tf.app.run()

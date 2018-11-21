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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

import math
import os


import tensorflow as tf

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


def main(_):
  # Build the inference graph.
  g = tf.Graph()
  with g.as_default():
    model = inference_wrapper.InferenceWrapper()
    # restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
    #                                            FLAGS.checkpoint_path)
    restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                               "model/model.ckpt-5000000")
  g.finalize()

  # Create the vocabulary.
  # vocab = vocabulary.Vocabulary(FLAGS.vocab_file)
  vocab = vocabulary.Vocabulary("word_counts.txt")

  filenames = []
  # for file_pattern in FLAGS.input_files.split(","):
  #   filenames.extend(tf.gfile.Glob(file_pattern))
  # tf.logging.info("Running caption generation on %d files matching %s",
  #                 len(filenames), FLAGS.input_files)

  input_files = "images/*"
  for file_pattern in input_files.split(","):
    filenames.extend(tf.gfile.Glob(file_pattern))
  tf.logging.info("Running caption generation on %d files matching %s",
                  len(filenames), input_files)

  with tf.Session(graph=g) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)

    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.
    generator = caption_generator.CaptionGenerator(model, vocab)
    for filename in filenames:
      with tf.gfile.GFile(filename, "rb") as f:
        image = f.read()
      captions = generator.beam_search(sess, image)
      print("Captions for image %s:" % os.path.basename(filename))

      sentence = [vocab.id_to_word(w) for w in captions[0].sentence[1:-2]]
      sentence = " ".join(sentence)
      print(sentence)

      # for i, caption in enumerate(captions):
      #   # Ignore begin and end words.
      #
      #   sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
      #   sentence = " ".join(sentence)
      #   print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))

      img = Image.open(filename)
      draw = ImageDraw.Draw(img)
      # font = ImageFont.truetype(<font-file>, <font-size>)
      font = ImageFont.truetype("Ubuntu-C.ttf", 22)
      # draw.text((x, y),"Sample Text",(r,g,b))
      draw.text((0, 0), sentence, (255, 55, 55), font=font)

      img.save("images/_%s" % (os.path.basename(filename)))


if __name__ == "__main__":
  tf.app.run()

import numpy as np

class History:
  def __init__(self, config):
    self.cnn_format = config.cnn_format

    batch_size, sequence_length, screen_height, screen_width = \
        config.batch_size, config.sequence_length, config.screen_height, config.screen_width

    self.history = np.zeros(
        [sequence_length, 1, screen_height, screen_width], dtype=np.float32)

  def add(self, screen):
    self.history[:-1] = self.history[1:]
    self.history[-1] = screen

  def reset(self):
    self.history *= 0

  def get(self):
    if self.cnn_format == 'NHWC':
      return np.transpose(self.history, (0, 2, 3, 1))
    else:
      return self.history

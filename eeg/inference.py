import sys

import numpy as np
import serial
from sklearn import decomposition
from sklearn import svm

import collect
import train
from util import open_bci

_SAMPLE_FREQ = 250
_ARDUINO_MAX_FREQ = 1

class StreamingInference(object):
  """Handles an input stream -> transform -> outputstream."""
  def __init__(self, model, board, arduino, threshold=25):
    self.model = model
    self.board = board
    self.arduino = arduino
    self.threshold = threshold
    self.num_gos = 0
    self.num_samples = 0
    self.num_samples_per_write = _SAMPLE_FREQ / float(_ARDUINO_MAX_FREQ)

  def start(self):
    self.board.start(self.handle_sample)

  def handle_sample(self, sample):
    y = self.model.predict(np.array(sample.channel_data).reshape(1, -1))[0]
    print("predict", self.model.predict(np.array(sample.channel_data).reshape(1, -1)))
    self.num_gos += 1 if y == 'go' else 0
    self.num_samples += 1
    if self.num_samples >= self.num_samples_per_write:
      msg = bytes('1' if self.num_gos > self.threshold else '0', 'utf8')
      self.num_gos = 0
      self.num_samples = 0
      print("Writing '{}' to serial...".format(msg))
      self.arduino.write(msg)

if __name__ == '__main__':
  assert len(sys.argv) == 2, "Must provide filename as argument."

  # Read in file.
  data = train.read_clean(sys.argv[1], 'go', 'stop')

  # Split into train/test
  Xtr, Xte, Ytr, Yte = train.prepare_train_test(data)

  # Prepare submodels
  ica = decomposition.FastICA(n_components=6)
  nn = svm.SVC()
  model = train.TransformClassifier(ica, nn)

  # Train model
  model.fit(Xtr, Ytr)
  print("\nModel score: {}\n".format(model.score(Xte, Yte)))

  # Set ports.
  # TODO: remove hardcoded values.
  bci_port = '/dev/tty.usbserial-DQ007SU3'
  arduino_port = '/dev/cu.usbmodem1421'

  print("Starting connection with OpenBCI on port={}...".format(bci_port))
  board = open_bci.OpenBCIBoard(port=bci_port)

  print("Starting connection with Arduino...")
  arduino_ser = serial.Serial(arduino_port, baudrate=57600)

  streamer = StreamingInference(model, board, arduino_ser)

  print("Starting stream...")
  streamer.start()

import sys

import numpy as np
import serial
from sklearn import decomposition
from sklearn import svm

import collect
import inference
import train
from util import open_bci
from util import csv_collector

if __name__ == '__main__':
  # TODO: remove hardcoded values.
  bci_port = '/dev/tty.usbserial-DQ007SU3'
  arduino_port = '/dev/cu.usbmodem1421'

  assert len(sys.argv) == 2, "Error: Please provide filename."

  filename = sys.argv[1]

  print("Writing to file: {}".format(filename))

  labels = ['go', 'stop']

  print("Starting connection with OpenBCI on port={}...".format(bci_port))
  board = open_bci.OpenBCIBoard(port=bci_port)
  collector = csv_collector.CSVCollector(filename, board)
  collector_options = collect.CollectionOptions(3, 1, 30, 1, labels)
  print("Collecting data...")
  collect.collect(collector, collector_options)

  data = train.read_clean(filename, labels[0], labels[1])

  Xtr, Xte, Ytr, Yte = train.prepare_train_test(data)

  ica = decomposition.FastICA(n_components=6)
  nn = svm.SVC()
  model = train.TransformClassifier(ica, nn)
  model.fit(Xtr, Ytr)
  print("\nModel score: {}\n".format(model.score(Xte, Yte)))

  print("Starting connection with Arduino...")
  arduino_ser = serial.Serial(arduino_port, baudrate=57600)

  streamer = inference.StreamingInference(model, board, arduino_ser)

  print("Starting stream...")
  streamer.start()

import sys
import time

from util import open_bci
from util import csv_collector


class CollectionOptions:
  def __init__(self, stabilization_period, waiting_period, collection_period, num_cycles, thoughts):
    self.stabilization_period = stabilization_period
    self.waiting_period = waiting_period
    self.collection_period = collection_period
    self.num_cycles = num_cycles
    self.thoughts = thoughts

def collect(collector, options):
  collector.start()
  time.sleep(options.stabilization_period)

  for i in range(options.num_cycles):
    for thought in options.thoughts:
      print("Think '{}'...".format(thought))
      collector.tag('0')
      time.sleep(options.waiting_period)
      collector.tag(thought)
      time.sleep(options.collection_period)

  collector.stop()

if __name__ == '__main__':
  ports = open_bci.find_ports()
  port = ports[0] if ports else None

  assert len(sys.argv) == 2, "Error: Please provide filename."
  assert port, "Error: Could not find serial port in use."

  filename = sys.argv[1]
  print("Writing to file: {}".format(filename))
  print("Using Port: {}".format(port))

  board = open_bci.OpenBCIBoard(port=port)
  collector = csv_collector.CSVCollector(filename, board)
  collector_options = CollectionOptions(3, 1, 30, 3, ['go', 'stop'])
  collect(collector, collector_options)

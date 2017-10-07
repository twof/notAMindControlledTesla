import sys

import serial

if __name__ == '__main__':
  port = sys.argv[1] if sys.argv[1:] else '/dev/tty.usbserial-DQ007SU3'
  print("Using port={}".format(port))
  arduino = serial.Serial(port, baudrate=57600)
  s = '0'
  i = 0
  while True:
    msg = bytes(s, 'utf8')
    print("Writing: {}".format(msg))
    arduino.write(msg)
    time.sleep(4)
    i += 4
    if i % 1000:
      s = '1' if s == '0' else '0'

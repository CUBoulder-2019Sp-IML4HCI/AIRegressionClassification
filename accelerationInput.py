import serial
from pythonosc import udp_client
from pythonosc import osc_message_builder
from time import sleep
#port - /dev/cu.usbmodem1412


PORT = "/dev/cu.usbmodem1412"

BAUD = 115200

s = serial.Serial(PORT)

s.baudrate = BAUD

s.parity = serial.PARITY_NONE
s.databits = serial.EIGHTBITS
s.stopbits = serial.STOPBITS_ONE




send_address = '127.0.0.1', 6448

# OSC basic client
c = udp_client.SimpleUDPClient('127.00.1', 6448)
i = 0
try:
	while True:
		data = str(s.readline().rstrip())

		x_y_z_tuple = data[2:].split(" ")

		x = int(x_y_z_tuple[0])
		y = int(x_y_z_tuple[1])

		#z = x_y_z_tuple[2]

		#data = float(data)

		c.send_message("/wek/inputs", [float(x), float(y)])




finally:
	s.close()

from microbit import *
import radio
radio.on()
radio.config(channel=72)
radio.config(power=7)

# X - tilting from left to right.
# Y - tilting forwards and backwards.
# Z - moving up and down.
while True:
	accel = accelerometer.get_values()

	if (accel is not None):
	    x = str(accel[0])
	    y = str(accel[1])
	    z = str(accel[2])
	    radio.send(str(x+" "+y+" "+z))
	else:
		display.show("No accel")

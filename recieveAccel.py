# Add your Python code here. E.g.
from microbit import *
import radio
radio.on()
radio.config(channel=19)
radio.config(power=5)

while True:
    #If shaken, send data
    temp = radio.receive()
    if temp:
    	print(temp)
   
  

        
        
#server OSC
from pythonosc import dispatcher
from pythonosc import osc_server

controlArray = []

def handle_classification(unused_addr, args, num):
    print(args, num)
    controlArray.append((args,num))


if __name__ == "__main__":
	# try:
	#
	# c = controls.Controls()
	# c.go()
	# while True:
	# 	dispatcher = dispatcher.Dispatcher()
	# 	dispatcher.map("/wek/outputs", handle_classification)
	#
	# 	server = osc_server.ThreadingOSCUDPServer( ("127.0.0.1", 12000), dispatcher)
	#
	# 	server.serve_forever()
	#
	# except KeyboardInterrupt:
	# 	print("\nsClient Ctrl-C\nShutting Down Server")



	dispatcher = dispatcher.Dispatcher()
	dispatcher.map("/wek/outputs", handle_classification)
	server = osc_server.ThreadingOSCUDPServer( ("127.0.0.1", 12000), dispatcher)
	server.serve_forever()

from pythonosc import udp_client
from pythonosc import osc_message_builder


#Take output from wikinator at port 12000
#Send it to port 6449


def handle_classification(unused_addr, args, num):
    print(args, num)
    controlArray.append((args,num))


dispatcher = dispatcher.Dispatcher()
dispatcher.map("/wek/outputs", handle_classification)
server = osc_server.ThreadingOSCUDPServer( ("127.0.0.1", 12000), dispatcher)


if __name__ == "__main__":



	#Have a separate thread to handle the server output that it receives.
    threading.Thread(target=server.serve_forever, daemon=True).start()

    c = udp_client.SimpleUDPClient('127.00.1', 6449)
    while True:
        try:
            c.send_message("/wek/inputs", )
        except:
            pass

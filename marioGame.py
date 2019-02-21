import retro
from pythonosc import dispatcher
from pythonosc import osc_server
import threading
from time import sleep
controlArray = []


def handle_classification(unused_addr, action):
    #1 is left
    #2 is right
    #3 is up
    #4 is down
    #5 is nothing
    #6 is up-right
    #7 is up-left
    #add it to array for pacman to call
    action = int(action)

    if action == 1:
        action = [0,0,0,0,0,0,1,0,0]
        controlArray.append(action)
        return
    if action == 2:
        action = [0,0,0,0,0,0,0,1,0]
        controlArray.append(action)
        return
    if action == 3:
        action = [0,0,0,0,0,0,0,0,1]
        controlArray.append(action)
        return
    if action == 4:
        action = [0,0,0,0,0,1,0,0,0]
        controlArray.append(action)
        return
    if action == 5:
        action = [0,0,0,0,0,0,0,0,0]
        controlArray.append(action)
        return
    if action == 6:
        action = [0,0,0,0,0,0,0,1,1]
        controlArray.append(action)
        return
    if action == 7:
        action = [0,0,0,0,0,0,1,0,1]
        controlArray.append(action)
        return








dispatcher = dispatcher.Dispatcher()
dispatcher.map("/wek/outputs", handle_classification)
server = osc_server.ThreadingOSCUDPServer( ("127.0.0.1", 12001), dispatcher)

#Have a separate thread to handle the server output that it receives.
threading.Thread(target=server.serve_forever, daemon=True).start()

env = retro.make('SuperMarioBros-Nes')
env.reset()

done=False
while not done:
    env.render()
    try:
        action = controlArray.pop(len(controlArray)-1)

        #action = env.action_space.sample()

        _obs, _rew, done, _info = env.step(action)

        if done:
            break
    except:
        pass

from pyOpenBCI import OpenBCICyton
import time


global last_print, count
last_print = time.time()
count = 0

def print_raw(sample):
    # print(sample.channels_data)
    global last_print, count
    count +=1
    if time.time() - last_print >1:
        print(count)
        last_print = time.time()
        count = 0

board = OpenBCICyton(port='COM3', daisy=True)

board.start_stream(print_raw)


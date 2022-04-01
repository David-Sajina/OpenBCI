from pylsl import StreamInlet, resolve_stream
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import style
from collections import deque

# last_print = time.time()
fps_counter = deque(maxlen=150)



print("looking for EEG stream")
streams = resolve_stream('type', 'EEG')
inlet = StreamInlet(streams[0])

channel_data = {}

last_print = time.time()
count = 0

for _ in range(500):
    for i in range(16):
        sample, timestamp = inlet.pull_sample()
        if i not in channel_data:
            channel_data[i] = sample
            # print(type(sample))
        else:
            channel_data[i].extend(sample)
            # print(type(sample))

    # fps_counter.append(time.time() - last_print)
    # last_print = time.time()
    # cur_raw_hz = 1/(sum(fps_counter)/len(fps_counter))
    # print(cur_raw_hz)
    count +=1
    if time.time() - last_print >1:
        print(count)
        last_print = time.time()
        count = 0

# print(channel_data)
print(len(channel_data[0]))
#for chan in channel_data:
#    print(chan, len(channel_data[chan]), channel_data[chan])
#for chan in channel_data:
#    plt.plot(channel_data[chan][:])
#plt.show()

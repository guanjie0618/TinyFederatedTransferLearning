from turtle import down, left
from matplotlib.cbook import sanitize_sequence
import serial
from serial.tools.list_ports import comports

import struct
import time
import numpy as np
import matplotlib.pyplot as plt
import threading
import time
import json
import os
import random
from queue import Queue

from tables import Unknown

random.seed(4321)
np.random.seed(4321)

samples_per_device = 120 # Amount of samples of each word to send to each device
batch_size = 20 # Must be even, hsa to be split into 2 types of samples
experiment = 'train-test' # 'iid', 'no-iid', 'train-test', None
use_threads = True

output_amount = 4
test_samples_amount = 60
size_hidden_nodes = 25
size_hidden_layer = (650+1)*size_hidden_nodes
size_output_layer = (size_hidden_nodes+1)*output_amount
hidden_layer = np.random.uniform(-0.5, 0.5, size_hidden_layer).astype('float32')
output_layer = np.random.uniform(-0.5, 0.5, size_output_layer).astype('float32')
momentum = 0.9
learningRate= 0.01
pauseListen = False # So there are no threads reading the serial input at the same time

one_files = [file for file in os.listdir("datasets/train_json_one") if file.startswith("one")]
two_files = [file for file in os.listdir("datasets/train_json_one") if file.startswith("two")]
three_files = [file for file in os.listdir("datasets/train_json_one") if file.startswith("three")]
# unknown_files = [file for file in os.listdir("datasets/train_json_one") if file.startswith("unknown")]
test_one_files = [file for file in os.listdir("datasets/test_json_one") if file.startswith("one")]
test_two_files = [file for file in os.listdir("datasets/test_json_one") if file.startswith("two")]
test_three_files = [file for file in os.listdir("datasets/test_json_one") if file.startswith("three")]
# test_unknown_files = [file for file in os.listdir("datasets/test_json_one") if file.startswith("unknown")]

random.shuffle(one_files)
random.shuffle(two_files)
random.shuffle(three_files)
# random.shuffle(unknown_files)

numbers = list(sum(zip(one_files, two_files, three_files), ()))
test_numbers = list(sum(zip(test_one_files, test_two_files, test_three_files), ()))

def print_until_keyword(keyword, arduino):
    while True: 
        msg = arduino.readline().decode()
        if msg[:-2] == keyword:
            break
        else:
            print(f'({arduino.port}):',msg, end='')

def init_network(hidden_layer, output_layer, device, deviceIndex):
    device.reset_input_buffer()
    device.write(b's')
    print_until_keyword('start', device)
    print(f"Sending model to {device.port}")

    device.write(struct.pack('f', learningRate))
    device.write(struct.pack('f', momentum))

    for i in range(len(hidden_layer)):
        device.read() # wait until confirmation of float received
        float_num = hidden_layer[i]
        data = struct.pack('f', float_num)
        device.write(data)
    
    for i in range(len(output_layer)):
        device.read() # wait until confirmation of float received
        float_num = output_layer[i]
        data = struct.pack('f', float_num)
        device.write(data)

    print(f"Model sent to {device.port}")
    modelReceivedConfirmation = device.readline().decode()
    # print(f"Model received confirmation: ", modelReceivedConfirmation)
    
# Batch size: The amount of samples to send
def sendSamplesIID(device, deviceIndex, batch_size, batch_index):
    global one_files, two_files, three_files, unknown_files, numbers, four_files, five_files, six_files, yes_files, no_files, up_files

    start = (deviceIndex*samples_per_device) + (batch_index * batch_size)
    end = (deviceIndex*samples_per_device) + (batch_index * batch_size) + batch_size

    print(f"[{device.port}] Sending samples from {start} to {end}")

    files = numbers[start:end]

    for i, filename in enumerate(files):
        if (filename.startswith("one")):
            num_button = 1
        elif (filename.startswith("two")):
            num_button = 2
        elif (filename.startswith("three")):
            num_button = 3
        # elif (filename.startswith("unknown")):
        #     num_button = 0


        print(f"[{device.port}] Sending sample {filename} ({i}/{len(files)}): Button {num_button}")
        sendSample(device, 'datasets/train_json_one/'+filename, num_button, deviceIndex)

def sendSamplesNonIID(device, deviceIndex, batch_size, batch_index):
    global one_files, two_files, three_files, four_files, five_files, six_files, samples_per_device

    start = (batch_index * batch_size)
    end = (batch_index * batch_size) + batch_size

    dir = 'datasets_mine/' # TMP fix
    if (samples_per_device <= 20):
        if (deviceIndex == 0):
            files = three_files[start:end]
            num_button = 3
            dir = 'train_json_one'
        elif  (deviceIndex == 1):
            files = one_files[start:end]
            num_button = 1
            dir = 'train_json_one'
        else:
            exit("Exceeded device index")

        for i, filename in enumerate(files):
            print(f"[{device.port}] Sending sample {filename} ({i}/{len(files)}): Button {num_button}")
            sendSample(device, f"datasets/{dir}/{filename}", num_button, deviceIndex)

    if (samples_per_device <= 40):
        if (deviceIndex == 0):
            files = two_files[start:end]
            num_button = 2
            dir = 'train_json_one'
        elif  (deviceIndex == 1):
            files = two_files[start:end]
            num_button = 2
            dir = 'train_json_one'
        else:
            exit("Exceeded device index")

        for i, filename in enumerate(files):
            print(f"[{device.port}] Sending sample {filename} ({i}/{len(files)}): Button {num_button}")
            sendSample(device, f"datasets/{dir}/{filename}", num_button, deviceIndex)

    if (samples_per_device <= 60):
        if (deviceIndex == 0):
            files = one_files[start:end]
            num_button = 1
            dir = 'train_json_one'
        elif  (deviceIndex == 1):
            files = three_files[start:end]
            num_button = 3
            dir = 'train_json_one'
        else:
            exit("Exceeded device index")

        for i, filename in enumerate(files):
            print(f"[{device.port}] Sending sample {filename} ({i}/{len(files)}): Button {num_button}")
            sendSample(device, f"datasets/{dir}/{filename}", num_button, deviceIndex)

def sendSample(device, samplePath, num_button, deviceIndex, only_forward = False):
    with open(samplePath) as f:
        ini_time = time.time() * 1000
        data = json.load(f)
        device.write(b't')
        startConfirmation = device.readline().decode()
        # print(f"[{device.port}] Train start confirmation:", startConfirmation)

        device.write(struct.pack('B', num_button))
        device.readline().decode() # Button confirmation

        device.write(struct.pack('B', 1 if only_forward else 0))
        print(f"Only forward confirmation: {device.readline().decode()}") # Button confirmation

        for i, value in enumerate(data['payload']['values']):
            device.write(struct.pack('h', value))
    
        print(f"[{device.port}] Sample received confirmation:", device.readline().decode())

        # print(f"Fordward millis received: ", device.readline().decode())
        # print(f"Backward millis received: ", device.readline().decode())
        device.readline().decode() # Accept 'graph' command
        error, num_button_predicted = read_graph(device, deviceIndex)
        if (error > 0.28):
            print(f"[{device.port}] Sample {samplePath} generated an error of {error}")
        print(f'Sample sent in: {(time.time()*1000)-ini_time} milliseconds)')
    
    print(f"{num_button} - {num_button_predicted}")
    return error, num_button == num_button_predicted

train_test = False
accuracy = []
tmp_acc = []
def sendTestSamples(device, deviceIndex, successes_queue):
    global test_numbers, accuracy, test_one_files, test_two_files, test_three_files, test_unknown_files, test_four_files, test_five_files, test_six_files, test_yes_files, test_no_files, test_up_files

    start = deviceIndex*test_samples_amount
    end = (deviceIndex*test_samples_amount) + test_samples_amount
   
    files = test_numbers[start:end]
   
    for i, filename in enumerate(files):
        if (filename.startswith("one")):
            num_button = 1
        elif (filename.startswith("two")):
            num_button = 2
        elif (filename.startswith("three")):
            num_button = 3
        else:
            num_button = 0

        error, success = sendSample(device, 'datasets/test_json_one/'+filename, num_button, deviceIndex, True)
        tmp_acc.append(success)
        successes_queue.put(success)
        accuracy.append([sum(tmp_acc)/len(tmp_acc), deviceIndex])
    

count = 0
nb = b'0'

graph_test = []
def read_graph(device, deviceIndex):
    global repaint_graph, count, nb, experiment, samples_per_device, train_test

    outputs = device.readline().decode().split()
    print(f'Ouptuts: {outputs}')

    bpred = outputs.index(max(outputs))
    if nb == b'5' and float(max(outputs)) < 0.8:
        print(f'Predicted button: 0')
    else:
        print(f'Predicted button: {bpred}')

    # print(f"Outputs: ", outputs)
    
    error = device.readline().decode()
    if nb == b'1' or nb == b'2' or nb == b'3':
        print(f'Error: {error}')

    ne = device.readline()[:-2]
    n_epooch = int(ne)

    n_error = device.read(4)
    [n_error] = struct.unpack('f', n_error)
    nb = device.readline()[:-2]

    if (nb != b'5') and (not train_test):
        graph.append([n_epooch, n_error, deviceIndex])
    
    repaint_graph = True

    if (train_test) and (float(max(outputs)) < 0.8):
        return n_error, 0
    
    return n_error, outputs.index(max(outputs))

def read_number(msg):
    while True:
        try:
            #return 2;
            return int(input(msg))
        except:
            print("ERROR: Not a number")

def read_port(msg):
    while True:
        try:
            port = input(msg)
            #port = "COM3";
            return serial.Serial(port, 9600)
        except:
            print(f"ERROR: Wrong port connection ({port})")

first_paint = True
graph = []
repaint_graph = True
first_inference = True
is_predicted = []
all_predict_error = []
inference = True

def plot_graph():
    global graph, repaint_graph, devices, first_paint, nb, first_inference, is_predicted, all_predict_error, train_test, graph_test, ax1, ax2, accuracy

    if (repaint_graph):
        colors = ['r', 'g', 'b', 'y']
        markers = ['-', '--', ':', '-.']
        epochs = 1

        for device_index, device in enumerate(devices):
            epoch = [x[0] for x in graph if x[2] == device_index]
            error = [x[1] for x in graph if x[2] == device_index]

            epochs = max(len(error), epochs)
            if experiment != None:
                plt.plot(error, colors[device_index] + markers[device_index], label=f"Device {device_index}")
            else:
                if (nb != b'5'):
                    ax1.plot(error, colors[device_index] + markers[device_index], label=f"Device {device_index}")
                else:
                    while True:
                        label_predicted = input("Is predicted right? ")
                        if (label_predicted == "1"):
                            is_predicted.append([1, device_index])
                            break
                        elif (label_predicted == "0"):
                            is_predicted.append([0, device_index])
                            break
                        else:
                            print("Wrong command!!")
                    
                    sum = 0
                    for x, y in is_predicted:
                        if (y == device_index):
                            sum += x
                    all_predict_error.append([1-sum/(len(is_predicted)/2), device_index])
                    predict_error = [y[0] for y in all_predict_error if y[1] == device_index]
                    ax2.plot(predict_error, colors[device_index] + markers[device_index], label=f"Device {device_index}")

        if (first_paint):
            if experiment == None:
                ax1.set_xlim(left=0)
                ax1.set_ylim(bottom=0, top=1)
                ax1.set_ylabel('Loss')
                ax1.set_xlabel('Epoch')

                ax2.set_xlim(left=0)
                ax2.set_ylim(bottom=0, top=1)
                ax2.set_ylabel('Error')
                ax2.set_xlabel('Epoch')

                ax1.autoscale(axis='both')
                ax2.autoscale(axis='both')
            else:
                plt.legend()
                plt.xlim(left=0)
                plt.ylim(bottom=0)
                plt.ylabel('Loss') # or Error
                plt.xlabel('Epoch')
                plt.autoscale(axis='both')
            
            first_paint = False

        if (nb == b'5') and (first_inference):
            first_inference = False

        repaint_graph = False

    plt.pause(2)


def listenDevice(device, deviceIndex):
    global pauseListen, graph, nb
    while True:
        while (pauseListen):
            time.sleep(0.1)

        d.timeout = None
        msg = device.readline().decode()
        if (len(msg) > 0):
            print(f'({device.port}):', msg, end="")
            # Modified to graph
            if msg[:-2] == 'graph':
                read_graph(device, deviceIndex)

            elif msg[:-2] == 'start_fl':
                startFL()

def getDevices():
    global devices, devices_connected
    num_devices = read_number("Number of devices: ")

    available_ports = comports()
    print("Available ports:")
    for available_port in available_ports:
        print(available_port)

    devices = [read_port(f"Port device_{i+1}: ") for i in range(num_devices)]
    devices_connected = devices

def FlGetModel(d, device_index, devices_hidden_layer, devices_output_layer, devices_num_epochs, old_devices_connected):
    global size_hidden_layer, size_output_layer
    d.reset_input_buffer()
    d.reset_output_buffer()
    d.timeout = 5

    print(f'Starting connection to {d.port} ...') # Hanshake
    d.write(b'>') # Python --> SYN --> Arduino
    if d.read() == b'<': # Python <-- SYN ACK <-- Arduino
        d.write(b's') # Python --> ACK --> Arduino
        
        print('Connection accepted.')
        devices_connected.append(d)
        #devices_hidden_layer = np.vstack((devices_hidden_layer, np.empty(size_hidden_layer)))
        #devices_output_layer = np.vstack((devices_output_layer, np.empty(size_output_layer)))
        d.timeout = None

        print_until_keyword('start', d)
        devices_num_epochs.append(int(d.readline()[:-2]))

        print(f'Receiving model from {d.port} ...')
        ini_time = time.time()

        for i in range(size_hidden_layer): # hidden layer
            data = d.read(4)
            [float_num] = struct.unpack('f', data)
            devices_hidden_layer[device_index][i] = float_num

        for i in range(size_output_layer): # output layer
            data = d.read(4)
            [float_num] = struct.unpack('f', data)
            devices_output_layer[device_index][i] = float_num

        print(f'Model received from {d.port} ({time.time()-ini_time} seconds)')

        # if it was not connected before, we dont use the devices' model
        if not d in old_devices_connected:
            devices_num_epochs[device_index] = 0
            print(f'Model not used. The device {d.port} has an outdated model')

    else:
        print(f'Connection timed out. Skipping {d.port}.')

def sendModel(d, hidden_layer, output_layer):
    ini_time = time.time()
    for i in range(size_hidden_layer): # hidden layer
        #d.read() # wait until confirmation
        float_num = hidden_layer[i]
        data = struct.pack('f', float_num)
        d.write(data)

    for i in range(size_output_layer): # output layer
        #d.read() # wait until confirmation
        float_num = output_layer[i]
        data = struct.pack('f', float_num)
        d.write(data)

    print(f'Model sent to {d.port} ({time.time()-ini_time} seconds)')

def startFL():
    global devices_connected, hidden_layer, output_layer, pauseListen

    pauseListen = True

    print('Starting Federated Learning')
    old_devices_connected = devices_connected
    devices_connected = []
    devices_hidden_layer = np.empty((len(devices), size_hidden_layer), dtype='float32')
    devices_output_layer = np.empty((len(devices), size_output_layer), dtype='float32')
    devices_num_epochs = []

    ##################
    # Receiving models
    ##################
    threads = []
    for i, d in enumerate(devices):
        if use_threads:
            thread = threading.Thread(target=FlGetModel, args=(d, i, devices_hidden_layer, devices_output_layer, devices_num_epochs, old_devices_connected))
            thread.daemon = True
            thread.start()
            threads.append(thread)
        else:
            FlGetModel(d, i, devices_hidden_layer, devices_output_layer, devices_num_epochs, old_devices_connected)
 
    for thread in threads: thread.join() # Wait for all the threads to end

    
    ####################
    # Processing models
    ####################

    # if sum == 0, any device made any epoch
    if sum(devices_num_epochs) > 0:
        # We can use weights to change the importance of each device
        # example weights = [1, 0.5] -> giving more importance to the first device...
        # is like percentage of importance :  sum(a * weights) / sum(weights)
        ini_time = time.time() * 1000
        hidden_layer = np.average(devices_hidden_layer, axis=0, weights=devices_num_epochs)
        output_layer = np.average(devices_output_layer, axis=0, weights=devices_num_epochs)
        print(f'Average millis: {(time.time()*1000)-ini_time} milliseconds)')


    #################
    # Sending models
    #################
    threads = []
    for d in devices_connected:
        print(f'Sending model to {d.port} ...')

        if use_threads:
            thread = threading.Thread(target=sendModel, args=(d, hidden_layer, output_layer))
            thread.daemon = True
            thread.start()
            threads.append(thread)
        else:
            sendModel(d, hidden_layer, output_layer)
    for thread in threads: thread.join() # Wait for all the threads to end
    pauseListen = False


getDevices()

# To load a Pre-trained model
# hidden_layer = np.load("./hidden_one_two_three_2.npy")
# output_layer = np.load("./output_one_two_three_2.npy")


# Send the blank model to all the devices
threads = []
for i, d in enumerate(devices):
    if use_threads:
        thread = threading.Thread(target=init_network, args=(hidden_layer, output_layer, d, i))
        thread.daemon = True
        thread.start()
        threads.append(thread)
    else:
        init_network(hidden_layer, output_layer, d, i)
for thread in threads: thread.join() # Wait for all the threads to end

# startFL()

test_result = []
# for i in range(10):
if experiment != None:
    train_ini_time = time.time()

    # Train the device
    # for times in range(3):
    for batch in range(int(samples_per_device/batch_size)):
        # for batch in range(4):
        batch_ini_time = time.time()
        for deviceIndex, device in enumerate(devices):
            if experiment == 'iid' or experiment == 'train-test':
                method = sendSamplesIID            
            elif experiment == 'no-iid':
                method = sendSamplesNonIID

            if use_threads:
                thread = threading.Thread(target=method, args=(device, deviceIndex, batch_size, batch))
                thread.daemon = True
                thread.start()
                threads.append(thread)
            else:
                method(device, deviceIndex, batch_size, batch)
        for thread in threads: thread.join() # Wait for all the threads to end
        print(f'Batch time: {time.time() - batch_ini_time} seconds)')
        fl_ini_time = time.time()
        startFL()
        print(f'FL time: {time.time() - fl_ini_time} seconds)')

        # samples_per_device += 20

    train_time = time.time()-train_ini_time
        # print(f'Trained in ({train_time} seconds)')

    # np.save('hidden_one_two_three_31.npy', hidden_layer)
    # np.save('output_one_two_three_31.npy', output_layer)

    if experiment == 'train-test' or experiment == 'no-iid':
        train_test = True
        successes_queue = Queue()
        for deviceIndex, device in enumerate(devices):
            if use_threads:
                thread = threading.Thread(target=sendTestSamples, args=(device, deviceIndex, successes_queue))
                thread.daemon = True
                thread.start()
                threads.append(thread)
            else:
                sendTestSamples(device, deviceIndex, successes_queue)
        for thread in threads: thread.join() # Wait for all the threads to end

        test_accuracy = sum(successes_queue.queue)/len(successes_queue.queue)
        test_result.append(test_accuracy)
        print(f"Testing accuracy: {test_accuracy}")
        print(f"{test_accuracy}, ", end = '')
        print(f"All test accuracy: {test_result}")


plt.ion()
# plt.title(f"Loss vs Epoch")
plt.show()

font_sm = 13
font_md = 16
font_xl = 18
plt.rc('font', size=font_sm)          # controls default text sizes
plt.rc('axes', titlesize=font_sm)     # fontsize of the axes title
plt.rc('axes', labelsize=font_md)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=font_sm)    # fontsize of the tick labels
plt.rc('ytick', labelsize=font_sm)    # fontsize of the tick labels
plt.rc('legend', fontsize=font_sm)    # legend fontsize
plt.rc('figure', titlesize=font_xl)   # fontsize of the figure title

if experiment == None:
    fig, (ax1, ax2) = plt.subplots(2, 1)

plot_graph()

if experiment != None:
    figname = f"newplots/BS{batch_size}-LR{learningRate}-M{momentum}-HL{size_hidden_nodes}-TT{train_time}-{experiment}.png"
    plt.savefig(figname, format='png')
    print(f"Generated {figname}")

# Listen their updates
for i, d in enumerate(devices):
    thread = threading.Thread(target=listenDevice, args=(d, i))
    thread.daemon = True
    thread.start()

while True:
    plot_graph()
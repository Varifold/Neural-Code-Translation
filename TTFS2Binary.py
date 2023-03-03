import nest
nest.set_verbosity("M_WARNING")
nest.ResetKernel()
import numpy
import random
import math
import matplotlib.pyplot as plt


def connect_layers(self_layer, weight_matrix, delay_matrix):
    for i, pre in enumerate(self_layer):
        # Extract the weights column.

        weights = weight_matrix[:, i]
        delay = delay_matrix[:, i]

        # To only connect pairs with a nonzero weight, we use array indexing
        # to extract the weights and post-synaptic neurons.
        nonzero_indices = numpy.where(weights != 0)[0]
        weights = weights[nonzero_indices]
        delay = delay[nonzero_indices]
        post = self_layer[nonzero_indices]

        # Generate an array of node IDs for the column of the weight
        # matrix, with length based on the number of nonzero
        # elements. The array's dtype must be an integer.
        pre_array = numpy.ones(len(nonzero_indices), dtype=numpy.int64) * pre.get('global_id')
        # nest.Connect() automatically converts post to a NumPy array
        # because pre_array contains multiple identical node IDs. When
        # also specifying a one_to_one connection rule, the arrays of
        # node IDs can then be connected.

        if list(post)!=[]:
            nest.Connect(pre_array, post, conn_spec='one_to_one', syn_spec={'weight': weights, 'delay': delay})



def create_connections_TTFS2Binary(Bits):

    N=Bits+2 #Number of neurons 
    W_arr = numpy.array([[0 for j in range(N)] for i in range(N)], dtype=numpy.float32) #Weight array. Array type must be float due non-integer weights!
    del_arr = numpy.array([[1 for j in range(N)] for i in range(N)]) #Delay array

    W_arr_2 = numpy.array([[0 for j in range(N)] for i in range(N)], dtype=numpy.float32) #Contains the weights of negative duplicate connections
    del_arr_2 = numpy.array([[1 for j in range(N)] for i in range(N)]) #Delay array

    #Timing neuron connections
    W_arr[Bits,0] = 10
    for i in range(Bits):
        W_arr[i+1,0] = 10
        del_arr[i+1,0] = 1

    #Bit connections
    for i in range(1,Bits):
        W_arr[i+1,i] = 10
        del_arr[i+1,i] = 2**(Bits-i)

        W_arr_2[i+1,i] = -10
    for i in range(1,Bits+1):
        if Bits>=i+2:
            for j in range(i+2,Bits+1):
                W_arr[j,i] = 10
                del_arr[j,i] = 2**(Bits-i) #2**(j-i)
                W_arr_2[j,i] = -10

    #Activation neuron connections
    for i in range(Bits):
        W_arr[1+i, N-1] = 10

    for i in range(1, Bits):
        del_arr[1+i, N-1] = del_arr[i+1,i] + del_arr[i, N-1]

    return W_arr, del_arr, W_arr_2, del_arr_2
    


def create_TTFS2Binary_Circuit(Bits, spike_times):

    index=3

    ndict = {'V_min':-70, "tau_m": [10]+[-(2**(Bits-i-1)-1+0.05)/math.log(0.5) for i in range(Bits)]+[10]} #-t/log(0.5) is equal to tau_m which is such that the voltage halves in t time.
    neuronpop = nest.Create("iaf_psc_delta", Bits+2, params=ndict)

    W_arr, del_arr, W_arr_2, del_arr_2 = create_connections_TTFS2Binary(Bits)
    connect_layers(neuronpop, W_arr, del_arr)
    connect_layers(neuronpop, W_arr_2, del_arr_2)

    signal_generator = nest.Create("spike_generator",
            params={"spike_times": [1.]})
    
    #Signal generator to Activation neuron

    nest.Connect(signal_generator, neuronpop[-1], conn_spec='one_to_one', syn_spec={'weight': 100})

    signal_generator2 = nest.Create("spike_generator",
            params={"spike_times": spike_times})
    
    #Signal generator to Timing neuron

    nest.Connect(signal_generator2, neuronpop[0], conn_spec='one_to_one', syn_spec={'weight': 100})

    #print(nest.GetConnections(source=None, target=None, synapse_model=None, synapse_label=None))


    rec = nest.Create("spike_recorder")
    nest.Connect(neuronpop[-1], rec)


    recorders = []
    for i in range(0,Bits+2):
        rec = nest.Create("spike_recorder")
        recorders.append(rec)
        nest.Connect(neuronpop[i], rec)

    multimeter = nest.Create("multimeter")
    multimeter.set(record_from=["V_m"])



    nest.Connect(multimeter, neuronpop[index])


    nest.Simulate(200.)
    bin_arr = []
    for i in range(0,Bits+2):
        if 1<=i<=Bits: #Selects for Bit neurons
            bin_arr.append((len(recorders[i].get('events')['times'])-1)**2) #Invert binary list


    return spike_times[0]-1 ,get_decimal(bin_arr) #Returns GT time and decimal representation of the binary output of the circuit


def get_decimal(lst):
    sums = 0
    if len(lst)>0:
        for i,bin in enumerate(lst):
            sums += bin*2**(len(lst)-i-1)
    return sums



def test_script():
    Bits = 3 #If Bits are set too high, the circuit will fail due too low simulation time!
    Success = True
    for i in range(2**Bits):
        spike_times = [1+i]
        GT, result = create_TTFS2Binary_Circuit(Bits, spike_times)
        if GT != result:
            print("Test Failure!")
            print(f'Spike_times was {GT} output was {result}')
            Success = False
        nest.ResetKernel() #Removes previous nodes
    if Success:
        print('Test completed succesfully, 0 failures')


test_script()

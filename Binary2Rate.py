import nest
nest.set_verbosity("M_WARNING")
nest.ResetKernel()
import numpy
import random

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



def create_connections_Binary2Rate(Bits):

    N=3*Bits+1 #Number of neurons

    W_arr = numpy.array([[0 for j in range(N)] for i in range(N)], dtype=numpy.float32) #Weight array. Array type must be float due non-integer weights!
    del_arr = numpy.array([[1 for j in range(N)] for i in range(N)]) #Delay array

    for i in range(Bits):
        #Bit connections
        W_arr[Bits+i,i] = 15
        del_arr[Bits+i,i] = 2**i

        #Resonator Connections
        W_arr[Bits+i,Bits+i] = 15
        W_arr[2*Bits+i,Bits+i] = 15.001/2**(Bits-i-1) #Due to voltage leak
        W_arr[Bits+i,2*Bits+i] = -15

        del_arr[Bits+i,Bits+i] = 2**(i+1)

        #Resonator to Output
        W_arr[N-1,Bits+i] = 15

    return W_arr, del_arr
    




def create_Binary2Rate_Circuit(Bits, binary_nums):



    ndict = {"tau_m": 1e9, 't_ref': 0} #Voltage leak is minimized by setting tau_m high
    neuronpop = nest.Create("iaf_psc_delta", 3*Bits+1, params=ndict)

    W_arr, del_arr = create_connections_Binary2Rate(Bits)
    connect_layers(neuronpop, W_arr, del_arr)

    signal_generator = nest.Create("spike_generator",
            params={"spike_times": [1.0]})
    
    #Signal  generator to Bits
    for i in range(Bits):
            nest.Connect(signal_generator, neuronpop[i], conn_spec='one_to_one', syn_spec={'weight': 100*binary_nums[i]})


    #print(nest.GetConnections(source=None, target=None, synapse_model=None, synapse_label=None))


    rec = nest.Create("spike_recorder")
    nest.Connect(neuronpop[-1], rec)


    recorders = []
    for i in range(3*Bits+1):
        rec = nest.Create("spike_recorder")
        recorders.append(rec)
        nest.Connect(neuronpop[i], rec)


    nest.Simulate(200.)
    print_out=False
    if print_out:
        #print(rec.get('events')['times']-excess_delay) #The encoded time
        for i in range(3*Bits+1):
            print(recorders[i].get('events'),len(recorders[i].get('events')['times']))

    return rec.get('events')['times']

def test_script():

        #Generate a list of random binary numbers
        binary_list = [random.randint(0, 1) for _ in range(3)]
        binary_list = [1,0,0]
        Bits = len(binary_list)
        circuit_output = create_Binary2Rate_Circuit(Bits,binary_list)
        decimal_num=0
        for i in range(Bits):
            decimal_num += binary_list[i] * (2 ** (Bits - i - 1))
        print(decimal_num)
        print(f'Output is: {circuit_output}, which has length {len(circuit_output)}')
        nest.ResetKernel() #Removes previous nodes



def test_script2():
     #Generate random binary numbers and assess the circuits performance

     trials_per_bit = 6
     max_bits = 7 #If max bits is set too high, the circuitry will fail due too low simulation time, or voltage decay in neurons. Increase simulation time and tau_m to remedy this
     for Bits in range(2,max_bits+1):
            print(Bits)
            for trial in range(trials_per_bit):
                #Generate a list of random binary numbers
                binary_list = [random.randint(0, 1) for _ in range(Bits)]
                # Convert the binary list to decimal form
                decimal_num = 0
                for i in range(Bits):
                    decimal_num += binary_list[i] * (2 ** (Bits - i - 1))

                circuit_output = create_Binary2Rate_Circuit(Bits,binary_list)
                print(f'Output is: {len(circuit_output)}, the decimal_num is: {decimal_num}')
                nest.ResetKernel() #Removes previous nodes


test_script2()


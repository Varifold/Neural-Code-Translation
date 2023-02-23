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



def create_connections_Binary2TTFS(Bits):

    N=3*Bits+2 #Number of neurons

    W_arr = numpy.array([[0 for j in range(N)] for i in range(N)]) #Weight array
    del_arr = numpy.array([[1 for j in range(N)] for i in range(N)]) #Delay array

    for i in range(1,Bits):
        #Upper row
        W_arr[i+1,i]=15
        W_arr[2*Bits+1+i,i]=10

        #Lower row
        W_arr[1+i,2*Bits+i] = 15
        W_arr[2*Bits+i+1,2*Bits+i] = 10

        #Lower row delays
        del_arr[1+i,2*Bits+i] = 2**(Bits-i)+1
        del_arr[2*Bits+i+1,2*Bits+i] = 2**(Bits-i)+1
    
    #Connections to Output neuron
    W_arr[N-1,Bits] = 15
    W_arr[N-1,N-2] = 15
    del_arr[N-1,N-2] = 2

    #Connections from Activation neuron
    W_arr[1,0] = 15
    W_arr[2*Bits+1,0] = 10

    #Bit connections
    for i in range(1,Bits+1):
        W_arr[i,Bits+i] = -10
        W_arr[2*Bits+i,Bits+i] = 10

    return W_arr, del_arr
    




def create_Binary2TTFS_Circuit(Bits, binary_nums):

    excess_delay = Bits+3 

    ndict = {"tau_m": 100.0}
    neuronpop = nest.Create("iaf_psc_delta", 3*Bits+2, params=ndict)

    W_arr, del_arr = create_connections_Binary2TTFS(Bits)
    connect_layers(neuronpop, W_arr, del_arr)


    signal_generator = nest.Create("spike_generator",
            params={"spike_times": [1.0]})


    
    #Signal  generator to Bits
    for i in range(Bits):
            nest.Connect(signal_generator, neuronpop[Bits+i+1], conn_spec='one_to_one', syn_spec={'weight': 100*binary_nums[i]})

    #Signal generator to activation
    nest.Connect(signal_generator, neuronpop[0], conn_spec='one_to_one', syn_spec={'weight': 100})

    rec = nest.Create("spike_recorder")
    nest.Connect(neuronpop[-1], rec)
    
    nest.Simulate(200.)

    return rec.get('events')['times']-excess_delay


def test_circuit():
     #Generate random binary numbers and assess the circuits performance

     trials_per_bit = 3
     max_bits = 6 #If max bits is set too high, the circuitry will fail due too low simulation time, or voltage decay in neurons. Increase simulation time and tau_m to remedy this
     for Bits in range(2,max_bits+1):
            for trial in range(trials_per_bit):
                #Generate a list of random binary numbers
                binary_list = [random.randint(0, 1) for _ in range(Bits)]
                # Convert the binary list to decimal form
                decimal_num = 0
                for i in range(Bits):
                    decimal_num += binary_list[i] * (2 ** (Bits - i - 1))

                circuit_output = create_Binary2TTFS_Circuit(Bits,binary_list)
                print(f'Output is: {circuit_output}, the decimal_num is: {decimal_num}')
                nest.ResetKernel() #Removes previous nodes

test_circuit()

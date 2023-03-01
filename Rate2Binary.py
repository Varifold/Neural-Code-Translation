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



def create_connections_Rate2Binary(Bits, obs_time):

    N=Bits+2 #Number of neurons
    #We assume that the maximum input rate is one spike per one tick. 
    W_arr = numpy.array([[0 for j in range(N)] for i in range(N)], dtype=numpy.float32) #Weight array. Array type must be float due non-integer weights!
    del_arr = numpy.array([[1 for j in range(N)] for i in range(N)]) #Delay array


    #Rate neuron connections
    W_arr[1,0] = 15
    for i in range(Bits):
        W_arr[2+i,0] = 1

    #Activation neuron connections
    W_arr[0,1] = -105*Bits #After the observation time ends, the circuit stays active for #Bits Ticks, this negative pulse halts further rate input during that time
    del_arr[0,1] = obs_time 
    for i in range(Bits):
        W_arr[2+i,1] = 15.001+2**Bits-2**(Bits-1-i) #Due to voltage decay we need to add a small extra term
        del_arr[2+i,1] = obs_time+i

    #Bit connections
    for bit_index in range(Bits-1):
        for lower_bit in range(bit_index+1,Bits):
            W_arr[lower_bit+2,bit_index+2] = -2**(Bits-bit_index-1)
         
    return W_arr, del_arr
    


def create_Rate2Binary_Circuit(Bits, obs_time, spike_times):



    ndict = {"tau_m": 1e9, 't_ref': [0,obs_time+1]+[0]*Bits, 'V_th': [-55,-55]+[-55+2**Bits]*Bits} 
    neuronpop = nest.Create("iaf_psc_delta", Bits+2, params=ndict)

    W_arr, del_arr = create_connections_Rate2Binary(Bits,obs_time)
    connect_layers(neuronpop, W_arr, del_arr)

    signal_generator = nest.Create("spike_generator",
            params={"spike_times": spike_times})
    
    #Signal generator to Rate neuron

    nest.Connect(signal_generator, neuronpop[0], conn_spec='one_to_one', syn_spec={'weight': 100})


    #print(nest.GetConnections(source=None, target=None, synapse_model=None, synapse_label=None))


    rec = nest.Create("spike_recorder")
    nest.Connect(neuronpop[-1], rec)


    recorders = []
    for i in range(Bits+2):
        rec = nest.Create("spike_recorder")
        recorders.append(rec)
        nest.Connect(neuronpop[i], rec)


    nest.Simulate(200.)
    print_out=True
    if print_out:
        #print(rec.get('events')['times']-excess_delay) #The encoded time
        bit_arr = []
        for i in range(2,Bits+2):
            #print(recorders[i].get('events'),len(recorders[i].get('events')['times']))
            bit_arr.append(len(recorders[i].get('events')['times']))
        #print(f'Input digit was: {len(spike_times)}, output binary array is {bit_arr}, which equals {get_decimal(bit_arr)} in decimal')

    return len(spike_times), get_decimal(bit_arr)


def get_decimal(lst):
    sums = 0
    if len(lst)>0:
        for i,bin in enumerate(lst):
            sums += bin*2**(len(lst)-i-1)
    return sums


def test_script():
        #Generates a list of random spike times within obs_time and inputs them to Rate2Binary network
        fail_count = 0 
        tests_per_bit = 20 
        max_bits = 8 #If set higher, simulation time and tau_m must also be increased
        for Bits in range(2,max_bits):
            for _ in range(tests_per_bit):
                nest.ResetKernel() #Removes previous nodes
                obs_time = 2**Bits
                spike_times = [1]+[i*random.randint(0,1) for i in range(2,obs_time)]
                spike_times = [i for i in spike_times if i!=0]
                input, pred = create_Rate2Binary_Circuit(Bits, obs_time, spike_times)
                if input != pred:
                    print(f'Failure, input digit was {input}, predicted {pred}')
                    fail_count += 1
        
        print(f'Simulation ended. There were {fail_count} failures')
        nest.ResetKernel() #Removes previous nodes


test_script()


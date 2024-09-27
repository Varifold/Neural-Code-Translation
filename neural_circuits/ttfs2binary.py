import nest
nest.set_verbosity("M_WARNING")
nest.ResetKernel()
import numpy as np
import math
from shared.functions import connect_layers, get_decimal



def create_connections_TTFS2Binary_Parrot(Bits):
    N = 3*Bits
    W_arr = np.zeros((N, N), dtype=np.float32)
    del_arr = np.ones((N, N))

    W_arr_2 = np.zeros((N, N), dtype=np.float32)
    del_arr_2 = np.ones((N, N))

    # Timing neuron connections
    W_arr[2, 0] = 10
    W_arr[2*Bits, 0] = 10


    # Upper row connections
    for i in range(0, Bits - 2):
        W_arr[3 + i, 2 + i] = 10
        del_arr[3 + i, 2 + i] = 2**(Bits - i - 1) + 1

        W_arr[2*Bits + i + 1, 2 + i] = 10
        del_arr[2*Bits + i + 1, 2 + i] = 2**(Bits - i - 1) + 1

    W_arr[N-1, Bits] = 10
    del_arr[N-1, Bits] = 3

    # Bottom row connections
    for i in range(Bits - 2):
        W_arr[2*Bits + i + 1, 2*Bits + i] = 10
        W_arr[3 + i, 2*Bits + i] = 10

    W_arr[N - 1, N - 2] = 10

    #Activation neuron connections
    
    #To top row
    W_arr[2, 1] = 10
    
    #To bot row
    W_arr[2*Bits, 1] = 10
    del_arr[2*Bits, 1] = 2**(Bits-1)+1

    W_arr_2[2*Bits, 1] = -10
    del_arr_2[2*Bits, 1] = 2**(Bits-1)

    #To parrot neuron
    W_arr[Bits+1,1] = 100
    del_arr[Bits+1,1] = 2**(Bits - 1) + 1
    

    #Parrot neuron connections
    for i in range(0, Bits - 2):
        #To top row
        W_arr[3+i,Bits+1+i] = 10

        #Top reset
        W_arr_2[2+i,Bits+1+i] = -10
        del_arr_2[2+i,Bits+1+i] = 2**(Bits - 1 - i) + 1

        #To next parrot neuron
        W_arr[Bits+2+i,Bits+1+i] = 100
        del_arr[Bits+2+i,Bits+1+i] = 2**(Bits - 2 - i) + 1

        #To bot row
        W_arr[2*Bits +1 + i, Bits+1+i] = 10
        del_arr[2*Bits +1 + i, Bits+1+i] = 2**(Bits - 2 - i) + 1

        W_arr_2[2*Bits +1 + i, Bits+1+i] = -10
        del_arr_2[2*Bits +1 + i, Bits+1+i] = 2**(Bits - 2 - i)

        #Bot reset
        W_arr_2[2*Bits + i, Bits+1+i] = -10
        del_arr_2[2*Bits + i, Bits+1+i] = 2**(Bits - 1 - i) + 1

    #To final neuron
    #This final parrot neuron can be used to determine when the circuit is finished. The circuit will always be done 2 timesteps after the final parrot neuron spikes.
    W_arr[N-1, 2*Bits-1] = 10
    del_arr[N-1, 2*Bits-1] = 2

    W_arr_2[N-1, 2*Bits-1] = -10
    del_arr_2[N-1, 2*Bits-1] = 1

    return W_arr, del_arr, W_arr_2, del_arr_2



def create_TTFS2Binary_Circuit_Parrot(Bits, spike_times):

    neuron_num = 3*Bits
    simulate_time = 2**(Bits)+Bits+1
    timing_neuron_tau = [-(2**(Bits-i-1)-1+0.05)/math.log(0.5) for i in range(Bits)]#-t/log(0.5) is equal to tau_m which is such that the voltage halves in t time.
    ndict = {'V_min':-70, "tau_m": 2*[10]+timing_neuron_tau[:-1]+[10]*(Bits-1)+timing_neuron_tau[:-1]+timing_neuron_tau[-1:]} #This ordering is chosen for ease of defining the connections
    neuronpop = nest.Create("iaf_psc_delta", neuron_num, params=ndict)

    W_arr, del_arr, W_arr_2, del_arr_2 = create_connections_TTFS2Binary_Parrot(Bits)
    connect_layers(neuronpop, W_arr, del_arr)
    connect_layers(neuronpop, W_arr_2, del_arr_2)

    signal_generator = nest.Create("spike_generator",
            params={"spike_times": [1.]})
    
    #Signal generator to Activation neuron

    nest.Connect(signal_generator, neuronpop[1], conn_spec='one_to_one', syn_spec={'weight': 100})

    signal_generator2 = nest.Create("spike_generator",
            params={"spike_times": spike_times})
    
    #Signal generator to Timing neuron

    nest.Connect(signal_generator2, neuronpop[0], conn_spec='one_to_one', syn_spec={'weight': 100})

    #print(nest.GetConnections(source=None, target=None, synapse_model=None, synapse_label=None))


    rec = nest.Create("spike_recorder")
    nest.Connect(neuronpop[-1], rec)



    recorders = []
    for i in range(0,neuron_num):
        rec = nest.Create("spike_recorder")
        recorders.append(rec)
        nest.Connect(neuronpop[i], rec)

    multimeters = []
    for i in range(neuron_num):
        multimeter = nest.Create("multimeter")
        multimeter.set(record_from=["V_m"])
        nest.Connect(multimeter, neuronpop[i])
        multimeters.append(multimeter)


    nest.Simulate(simulate_time)

    bin_arr = []
    for i in range(0,neuron_num):
        if 2*Bits<=i: #Selects for bot row neurons
            bin_arr.append(len(recorders[i].get('events')['times'])) #Invert binary list
 
    bin_arr.reverse()#This bin_arr starts with the MsB. We reverse it so that it is in the common representation.
    return spike_times[0]-1 ,get_decimal(bin_arr) #Returns GT time and decimal representation of the binary output of the circuit


def test_script(max_bits=7):
    failure = 0
    for Bits in range(2,max_bits+1):
        for i in range(2**Bits):
            spike_times = [1+i]
            GT, circuit_output = create_TTFS2Binary_Circuit_Parrot(Bits, spike_times)
            if GT != circuit_output:
                print("Test Failure!")
                print(f'Spike_times was {GT} output was {circuit_output}')
                failure += 1
            nest.ResetKernel() #Removes previous nodes
    print(f"TTFS2Binary circuit tests completed with {failure} failures")


if __name__ == "__main__":

    test_script()


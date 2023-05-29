import nest
nest.set_verbosity("M_WARNING")
nest.ResetKernel()
import numpy
import random
import math
from shared.functions import connect_layers, get_decimal


def create_connections_Rate2Binary(Bits, obs_time):

    N=2*Bits+2 #Number of neurons
    #We assume that the maximum input rate is one spike per one tick. 
    W_arr = numpy.array([[0 for j in range(N)] for i in range(N)], dtype=numpy.float32) #Weight array. Array type must be float due non-integer weights!
    del_arr = numpy.array([[1 for j in range(N)] for i in range(N)]) #Delay array

    W_arr_2 = numpy.array([[0 for j in range(N)] for i in range(N)], dtype=numpy.float32) #Weight array. Array type must be float due non-integer weights!
    del_arr_2 = numpy.array([[1 for j in range(N)] for i in range(N)]) #Delay array

    #Rate neuron connection
    W_arr[2,0] = 10

    #Activation neuron

    #Rate Neuron
    W_arr_2[0,1] = -15*Bits #During the runtime after obs_time a maximum of #Bits spikes can arrive
    del_arr_2[0,1] = obs_time

    W_arr[0,1] = 15*(Bits+1) #This is guaranteed to cause a spike and so reset the state of the rate neuron
    del_arr[0,1] = obs_time + Bits

    #First counter bit
    W_arr_2[2,1] = -10 #We cancel the incoming spike from the rate neuron due to resetting
    del_arr_2[2,1] = obs_time + Bits + 1

    for i in range(Bits):
        #Readout
        W_arr[2+i,1] = 10
        del_arr[2+i,1] = obs_time + Bits
        #Reset_connection
        W_arr_2[2+i,1] = -20
        #Counter bit neuron
        W_arr[Bits+2+i,1] = 10
        del_arr[Bits+2+i,1] = obs_time + Bits + 1

    #Counter bit connections
    for i in range(Bits-1):
        W_arr[3+i,2+i] = 10
    
    #Counter bit to Bit neuron
    for i in range(Bits):
        W_arr[Bits+2+i,2+i] = 10
         
    return W_arr, del_arr, W_arr_2, del_arr_2
    


def create_Rate2Binary_Circuit(Bits, obs_time, spike_times):

    neuron_num = 2*Bits + 2
    simulation_time = obs_time+Bits+3
    ndict = {"tau_m": [1e9]*(2+Bits)+[-0.5/math.log(0.5)]*Bits, 'V_th': -55, 'V_min':-70, 't_ref': 0} #-t/log(0.5) is equal to tau_m which is such that the voltage halves in t time.
    neuronpop = nest.Create("iaf_psc_delta", neuron_num, params=ndict)

    W_arr, del_arr, W_arr_2, del_arr_2 = create_connections_Rate2Binary(Bits,obs_time)
    connect_layers(neuronpop, W_arr, del_arr)
    connect_layers(neuronpop, W_arr_2, del_arr_2)

    signal_generator_1 = nest.Create("spike_generator",
            params={"spike_times": spike_times})
    
    signal_generator_2 = nest.Create("spike_generator",
        params={"spike_times": [1]})
    
    #Signal generator to Rate neuron

    nest.Connect(signal_generator_1, neuronpop[0], conn_spec='one_to_one', syn_spec={'weight': 15})

    #Signal generator to Activation neuron

    nest.Connect(signal_generator_2, neuronpop[1], conn_spec='one_to_one', syn_spec={'weight': 15})


    #print(nest.GetConnections(source=None, target=None, synapse_model=None, synapse_label=None))



    recorders = []
    for i in range(neuron_num):
        rec = nest.Create("spike_recorder")
        recorders.append(rec)
        nest.Connect(neuronpop[i], rec)


    nest.Simulate(simulation_time)

    bit_arr = []
    for i in range(Bits+2,neuron_num): #Selects for the readout neurons
        bit_arr.append(len(recorders[i].get('events')['times']))


    return len(spike_times), get_decimal(bit_arr)




def test_script2(max_bits = 8,trials_per_bit = 50):
        #Generates a list of random spike times within obs_time and inputs them to Rate2Binary network
        failure = 0 
        for Bits in range(2,max_bits+1):
            for _ in range(trials_per_bit):
                obs_time = 2**Bits
                spike_times = [i*random.randint(0,1) for i in range(2,obs_time)]
                spike_times = [i for i in spike_times if i!=0]
                GT, circuit_output = create_Rate2Binary_Circuit(Bits, obs_time, spike_times)
                if GT != circuit_output:
                    print(f'Failure, input digit was {GT}, predicted {circuit_output}')
                    failure += 1
                nest.ResetKernel() #Removes previous nodes
        print(f"All tests completed with {failure} failures")

def test_script(max_bits = 7, trials_per_bit = 50):
    # Generates a list of random spike times within obs_time and inputs them to Rate2Binary network
    failure = 0 
    for Bits in range(2,max_bits+1):
        for _ in range(trials_per_bit):
            obs_time = 2**Bits
            num_spikes = random.randint(0, max(0, obs_time - 2)) # Random number of spikes
            spike_times = sorted(random.sample(range(2, obs_time), num_spikes))  # Uniformly distributed spike times
            GT, circuit_output = create_Rate2Binary_Circuit(Bits, obs_time, spike_times)
            if GT != circuit_output:
                print(f'Failure, input digit was {GT}, predicted {circuit_output}')
                failure += 1
            nest.ResetKernel() #Removes previous nodes
    print(f"Rate2Binary circuit tests completed with {failure} failures")
if __name__ == "__main__":

    test_script()

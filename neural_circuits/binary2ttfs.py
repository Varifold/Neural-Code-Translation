import nest
nest.set_verbosity("M_WARNING")
nest.ResetKernel()
import numpy
from shared.functions import connect_layers, decimal_to_binary_bits



def create_connections_Binary2TTFS(Bits):

    N=3*Bits+2 #Number of neurons

    W_arr = numpy.array([[0 for j in range(N)] for i in range(N)]) #Weight array
    del_arr = numpy.array([[1 for j in range(N)] for i in range(N)]) #Delay array

    for i in range(1,Bits):
        #Upper row
        W_arr[i+1,i] = 15
        W_arr[2*Bits+1+i,i] = 10
        W_arr[2*Bits+i,i] = -15

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
    del_arr[1,0] = 1
    del_arr[2*Bits+1,0] = 1

    #Bit connections
    for i in range(1,Bits+1):
        #To top
        W_arr[i,Bits+i] = -15
        #To bot
        W_arr[2*Bits+i,Bits+i] = 10

    return W_arr, del_arr
    




def create_Binary2TTFS_Circuit(Bits, binary_nums):

    excess_delay = Bits+3
    simulation_time = 2**Bits+Bits+2

    ndict = {"tau_m": 1000.0, 'V_min':[-90]+[-90]*2*Bits+[-70]*Bits+[-90]}
    neuronpop = nest.Create("iaf_psc_delta", 3*Bits+2, params=ndict)

    W_arr, del_arr = create_connections_Binary2TTFS(Bits)
    connect_layers(neuronpop, W_arr, del_arr)
    #connect_layers(neuronpop, W_arr_2, del_arr_2)


    signal_generator = nest.Create("spike_generator",
            params={"spike_times": [1.0]})


    
    #Signal  generator to Bits
    for i in range(Bits):
            nest.Connect(signal_generator, neuronpop[Bits+i+1], conn_spec='one_to_one', syn_spec={'weight': 100*binary_nums[i]})

    #Signal generator to activation
    nest.Connect(signal_generator, neuronpop[0], conn_spec='one_to_one', syn_spec={'weight': 100})

    #print(nest.GetConnections(source=None, target=None, synapse_model=None, synapse_label=None))

    rec = nest.Create("spike_recorder")
    nest.Connect(neuronpop[-1], rec)

    nest.Simulate(simulation_time)


    return rec.get('events')['times']-excess_delay


def test_script(max_bits = 7):
    #Generate random binary numbers and assess the circuits performance
    #If max bits is set too high, the circuitry will fail due too low simulation time, or voltage decay in neurons. Increase simulation time and tau_m to remedy this
    failure = 0
    for Bits in range(2,max_bits+1):
        for num in range(2**Bits):
            #Generate a list of random binary numbers
            binary_list = decimal_to_binary_bits(Bits, num) #Produces a binary representation of the number num. E.g if Bits=3, num = 2 output is [0,1,0]
            circuit_output = create_Binary2TTFS_Circuit(Bits,binary_list)
            if num != circuit_output or len(circuit_output)==0:
                print(f' num was: {num}, predicted: {circuit_output}', Bits)
                failure+=1
            nest.ResetKernel() #Removes previous nodes
    print(f"Binary2TTFS circuit tests completed with {failure} failures")


if __name__ == "__main__":

    test_script()

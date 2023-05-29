import nest
nest.set_verbosity("M_WARNING")
nest.ResetKernel()
import numpy
from shared.functions import connect_layers, decimal_to_binary_bits, get_decimal


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
        W_arr[2*Bits+i,Bits+i] = 15.001/2**(Bits-i-1)#Due to voltage decay we need to add a small positive term. This is implementation dependent and may cause issues if #Bits is very high
        W_arr[Bits+i,2*Bits+i] = -15

        del_arr[Bits+i,Bits+i] = 2**(i+1)

        #Resonator to Output
        W_arr[N-1,Bits+i] = 15

    return W_arr, del_arr
    




def create_Binary2Rate_Circuit(Bits, binary_nums):

    simulation_time = 2**Bits+2

    ndict = {"tau_m": 1e9, 't_ref': 0}
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


    nest.Simulate(simulation_time )
    print_out=False
    if print_out:
        #print(rec.get('events')['times']-excess_delay) #The encoded time
        for i in range(3*Bits+1):
            print(recorders[i].get('events'),len(recorders[i].get('events')['times']))

    return rec.get('events')['times']



def test_script(max_bits = 7):
    failure = 0
    for Bits in range(2,max_bits+1):
        for num in range(2**Bits):
            #Generate a list of random binary numbers
            binary_list = decimal_to_binary_bits(Bits, num) #Produces a binary representation of the number num. E.g if Bits=3, num = 2 output is [0,1,0]
            # Convert the binary list to decimal form
            GT = get_decimal(list(reversed(binary_list)))
            circuit_output = create_Binary2Rate_Circuit(Bits,binary_list)
            if len(circuit_output)!=GT:
                    failure+=1
                    print(f'Test failed! Output is: {len(circuit_output)}, the GT is: {GT}')
    
            nest.ResetKernel() #Removes previous nodes
    print(f"Binary2Rate circuit tests completed with {failure} failures")

if __name__ == "__main__":

    test_script()


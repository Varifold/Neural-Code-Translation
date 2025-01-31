import nest
nest.set_verbosity("M_WARNING")
nest.ResetKernel()

import numpy
import random
import math
import matplotlib.pyplot as plt  # For the raster plot

from shared.functions import connect_layers, get_decimal


def create_connections_Rate2Binary(Bits, obs_time):
    """
    Creates and returns two sets of weight and delay arrays (W_arr, del_arr, W_arr_2, del_arr_2).
    These define the synaptic connections and delays within the Rate2Binary circuit.
    """
    N = 2*Bits + 2  # Number of neurons

    # Initialize arrays
    W_arr = numpy.zeros((N, N), dtype=numpy.float32)
    del_arr = numpy.ones((N, N), dtype=numpy.int32)

    W_arr_2 = numpy.zeros((N, N), dtype=numpy.float32)
    del_arr_2 = numpy.ones((N, N), dtype=numpy.int32)

    # Rate neuron connection
    W_arr[2, 0] = 10

    # Activation neuron / Rate neuron interplay
    W_arr_2[0, 1] = -15*Bits
    del_arr_2[0, 1] = obs_time

    W_arr[0, 1] = 15*(Bits+1)
    del_arr[0, 1] = obs_time + Bits

    # Cancel incoming spike from the rate neuron (due to resetting)
    W_arr_2[2, 1] = -10
    del_arr_2[2, 1] = obs_time + Bits + 1

    # First counter bit
    for i in range(Bits):
        # Readout
        W_arr[2 + i, 1] = 10
        del_arr[2 + i, 1] = obs_time + Bits

        # Reset connection
        W_arr_2[2 + i, 1] = -20

        # Counter bit neuron
        W_arr[2 + Bits + i, 1] = 10
        del_arr[2 + Bits + i, 1] = obs_time + Bits + 1

    # Counter bit connections
    for i in range(Bits - 1):
        W_arr[3 + i, 2 + i] = 10

    # Counter bit to Bit neuron
    for i in range(Bits):
        W_arr[Bits + 2 + i, 2 + i] = 10

    return W_arr, del_arr, W_arr_2, del_arr_2


def create_Rate2Binary_Circuit(Bits, obs_time, spike_times, raster_plot=False):
    """
    Builds the Rate2Binary circuit for a given bit-width (Bits) and observation time (obs_time),
    injecting a spike train (spike_times) into the rate neuron.

    Returns:
        tuple(int, int, int):
            (number_of_input_spikes, decimal_value_of_output_bits, total_transmitted_spikes)
    """
    neuron_num = 2*Bits + 2
    # We run slightly past obs_time+Bits
    simulation_time = obs_time + Bits + 3

    # Custom tau_m for the first (2+Bits) neurons vs. the last Bits neurons
    ndict = {
        "tau_m": [1e9]*(2+Bits) + [-0.5/math.log(0.5)]*Bits, 
        "V_th": -55,
        "V_min": -70,
        "t_ref": 0
    }
    neuronpop = nest.Create("iaf_psc_delta", neuron_num, params=ndict)

    # Connect the circuit
    W_arr, del_arr, W_arr_2, del_arr_2 = create_connections_Rate2Binary(Bits, obs_time)
    connect_layers(neuronpop, W_arr, del_arr)
    connect_layers(neuronpop, W_arr_2, del_arr_2)

    # --- Count out-degree of each neuron (how many outgoing synapses) ---
    #    Because connect_layers only connects if weight != 0, we can
    #    just count nonzero entries in W_arr and W_arr_2 for each row i.
    out_degrees = [0]*neuron_num
    for i in range(neuron_num):
        count_1 = sum(1 for w in W_arr[:,i] if w != 0)
        count_2 = sum(1 for w in W_arr_2[:,i] if w != 0)
        out_degrees[i] = count_1 + count_2

    # Spike generators for input (rate neuron) and activation
    signal_generator_1 = nest.Create("spike_generator", params={"spike_times": spike_times})
    signal_generator_2 = nest.Create("spike_generator", params={"spike_times": [1]})

    # Connect the input generator to the rate neuron
    nest.Connect(signal_generator_1, neuronpop[0], 
                 conn_spec='one_to_one', syn_spec={'weight': 15})
    # Connect the activation generator to the activation neuron
    nest.Connect(signal_generator_2, neuronpop[1], 
                 conn_spec='one_to_one', syn_spec={'weight': 15})

    # Create spike recorders for all neurons
    recorders = []
    for i in range(neuron_num):
        rec = nest.Create("spike_recorder")
        recorders.append(rec)
        nest.Connect(neuronpop[i], rec)

    # Run the simulation
    nest.Simulate(simulation_time)

    # Optional raster plot
    if raster_plot:
        plt.figure(figsize=(8, 6))
        for i, rec in enumerate(recorders):
            times = rec.get('events')['times']
            plt.plot(times, i * numpy.ones_like(times), 'k.', markersize=3)
        plt.title(f"Spiking Raster (Rate2Binary) — Bits={Bits}")
        plt.xlabel("Time (ms)")
        plt.ylabel("Neuron Index")
        plt.ylim([-1, neuron_num + 1])
        plt.show()

    # --- Compute final bit-neuron output ---
    bit_arr = []
    for i in range(Bits + 2, neuron_num):
        spike_count = len(recorders[i].get('events')['times'])
        bit_arr.append(spike_count)
    decimal_output = get_decimal(bit_arr)

    # --- Compute total transmitted spikes ---
    #     Sum( (spike_count of neuron i) * (out_degrees[i]) ) for i in the circuit
    transmitted_spikes = 0
    total_spikes = 0
    for i in range(1,neuron_num): #Skip the rate neuron to get excess spikes!
        spike_count = len(recorders[i].get('events')['times'])
        total_spikes += spike_count
        transmitted_spikes += spike_count * out_degrees[i]

    return len(spike_times), decimal_output, transmitted_spikes, total_spikes


def test_script2(max_bits=8, trials_per_bit=50):
    """
    Generates random spike times within obs_time and tests the Rate2Binary network.
    This version constructs spike_times by stepping from i=2 to obs_time with random 0/1.
    """
    failure = 0 
    for Bits in range(2, max_bits+1):
        for _ in range(trials_per_bit):
            obs_time = 2**Bits
            spike_times = [i*random.randint(0,1) for i in range(2, obs_time)]
            spike_times = [i for i in spike_times if i!=0]
            GT, circuit_output = create_Rate2Binary_Circuit(Bits, obs_time, spike_times)
            if GT != circuit_output:
                print(f'Failure, input digit was {GT}, predicted {circuit_output}')
                failure += 1
            nest.ResetKernel()  # Clear nodes before next run
    print(f"All tests completed with {failure} failures")


def test_script(max_bits=7, trials_per_bit=50):
    """
    Generates a random number of spikes (num_spikes) within obs_time and tests the Rate2Binary network.
    spike_times are chosen uniformly at random from the range [2, obs_time).
    """
    failure = 0
    for Bits in range(2, max_bits + 1):
        for _ in range(trials_per_bit):
            obs_time = 2**Bits
            num_spikes = random.randint(0, max(0, obs_time - 2))  # Random number of spikes
            spike_times = sorted(random.sample(range(2, obs_time), num_spikes))
            GT, circuit_output, _, _ = create_Rate2Binary_Circuit(Bits, obs_time, spike_times)
            if GT != circuit_output:
                print(f'Failure, input digit was {GT}, predicted {circuit_output}')
                failure += 1
            nest.ResetKernel()  # Clear nodes before next run
    print(f"Rate2Binary circuit tests completed with {failure} failures")


def average_spikes_per_bits(min_bits=2, max_bits=8, K=5, seed=None):
    """
    For each Bits in [min_bits..max_bits], runs create_Rate2Binary_Circuit(K times).
    Computes and returns:
      - Average decimal output
      - Average transmitted spikes
      - Maximum transmitted spikes (for any one trial) for each Bits.
    """
    if seed is not None:
        random.seed(seed)

    results = {}
    for Bits in range(min_bits, max_bits + 1):
        obs_time = 2**Bits
        sum_decimal = 0
        sum_transmitted = 0
        sum_total = 0
        spike_time_lists = []

        for _ in range(K):
            # Generate random spike times
            num_spikes = random.randint(0, obs_time)
            spike_times = sorted(random.sample(range(1, obs_time + 1), num_spikes))
            spike_time_lists.append(spike_times)

            # Run the circuit
            _, decimal_output, transmitted_spikes, total_spikes = create_Rate2Binary_Circuit(
                Bits, obs_time, spike_times
            )

            # Accumulate totals
            sum_total += total_spikes
            sum_decimal += decimal_output
            sum_transmitted += transmitted_spikes

            # Reset kernel before the next run
            nest.ResetKernel()

        # Averages
        avg_decimal = sum_decimal / K
        avg_transmitted = sum_transmitted / K
        avg_total = sum_total/ K
        
        #Run maximal case:
        spike_times = list(range(1,obs_time+1))
        _, decimal_output, max_transmitted_spikes, max_total_spikes = create_Rate2Binary_Circuit(
                Bits, obs_time, spike_times)
        nest.ResetKernel()

        results[Bits] = {
            'obs_time': obs_time,
            'average_decimal_output': avg_decimal,
            'average_total_spikes': avg_total,
            'average_transmitted_spikes': avg_transmitted,
            'max_transmitted_spikes': max_transmitted_spikes,
            'max_total_spikes': max_total_spikes,
            'spike_time_lists': spike_time_lists
        }

    return results


if __name__ == "__main__":
    # Example usage
    run_example = True
    
    
    experiment_results = average_spikes_per_bits(min_bits=2, max_bits=8, K=10, seed=42)
    for bits, info in experiment_results.items():
        print(f"\n--- Bits = {bits} ---")
        print(f"obs_time = {info['obs_time']}")
        print(f"Average Total Spikes = {info['average_total_spikes']:.2f}")
        print(f"Average Transmitted Spikes = {info['average_transmitted_spikes']:.2f}")
        print(f"Max Transmitted Spikes = {info['max_transmitted_spikes']}")
        print(f"Max total Spikes = {info['max_total_spikes']}")
    
    if run_example:
        Bits = 8
        obs_time = 2**Bits
        # Example spike times
        spike_times = [i for i in range(1,int(obs_time-1),2)]
        input_count, output_value, _, _ = create_Rate2Binary_Circuit(
        Bits, obs_time, spike_times, raster_plot=True
        )
        print(f"Number of input spikes = {input_count}, Output (decimal) = {output_value}")

    # Or run the automated test
    # test_script()


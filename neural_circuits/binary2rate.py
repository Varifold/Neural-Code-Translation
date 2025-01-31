import nest
nest.set_verbosity("M_WARNING")
nest.ResetKernel()

import numpy
import matplotlib.pyplot as plt

from shared.functions import connect_layers, decimal_to_binary_bits, get_decimal


def create_connections_Binary2Rate(Bits):
    """
    Creates the weight and delay arrays for the Binary2Rate circuit.
    """
    N = 3*Bits+1  # Number of neurons

    # Weight array (use float to accommodate non-integer weights)
    W_arr = numpy.zeros((N, N), dtype=numpy.float32)
    # Delay array
    del_arr = numpy.ones((N, N), dtype=numpy.int32)

    for i in range(Bits):
        # Bit connections
        W_arr[Bits + i, i] = 15
        del_arr[Bits + i, i] = 2**i

        # Resonator connections
        W_arr[Bits + i, Bits + i] = 15
        W_arr[2*Bits + i, Bits + i] = 15.001 / 2**(Bits - i - 1)  
        # Above small offset in weight is to compensate for voltage decay
        W_arr[Bits + i, 2*Bits + i] = -15

        del_arr[Bits + i, Bits + i] = 2**(i + 1)

        # Resonator to Output
        W_arr[N - 1, Bits + i] = 15

    return W_arr, del_arr


def create_Binary2Rate_Circuit(Bits, binary_nums, raster_plot=False):
    """
    Builds and simulates the Binary2Rate circuit for the given bit-width (Bits)
    and binary input (binary_nums). Optionally produces a raster plot of the
    activity if raster_plot=True.
    """
    simulation_time = 2**Bits + 2

    # Use a large tau_m, so effectively "memory" lasts over the entire simulation
    ndict = {"tau_m": 1e9, 't_ref': 0}
    neuronpop = nest.Create("iaf_psc_delta", 3*Bits + 1, params=ndict)

    # Build connections
    W_arr, del_arr = create_connections_Binary2Rate(Bits)
    connect_layers(neuronpop, W_arr, del_arr)

    # Create spike generator that fires at time = 1.0 ms
    signal_generator = nest.Create("spike_generator",
                                   params={"spike_times": [1.0]})

    # Connect the spike generator to each "bit" neuron
    for i in range(Bits):
        # Multiply the weight by the binary value: 0 or 1
        nest.Connect(signal_generator, neuronpop[i],
                     syn_spec={'weight': 100*binary_nums[i]})

    # Spike recorder for the output neuron
    main_recorder = nest.Create("spike_recorder")
    nest.Connect(neuronpop[-1], main_recorder)

    # Individual spike recorders for all neurons (for raster plotting)
    recorders = []
    for i in range(3*Bits + 1):
        rec = nest.Create("spike_recorder")
        recorders.append(rec)
        nest.Connect(neuronpop[i], rec)

    # Run the simulation
    nest.Simulate(simulation_time)

    # If requested, produce a raster plot of spiking activity for all neurons
    if raster_plot:
        plt.figure(figsize=(8, 6))
        for i, rec in enumerate(recorders):
            times = rec.get('events')['times']
            # Plot each neuron's spikes at row = i
            plt.plot(times, i * numpy.ones_like(times), 'k.', markersize=3)
        plt.title(f"Spiking Raster for Binary2Rate Circuit ({Bits} bits)")
        plt.xlabel("Time (ms)")
        plt.ylabel("Neuron Index")
        plt.ylim([-1, 3*Bits+1])  # Pad to make sure raster dots are visible
        plt.show()

    # Return the spike times from the output neuron
    return main_recorder.get('events')['times']
    

def create_Binary2Rate_Circuit_spike_measures(Bits, binary_nums):
    """
    Creates the Binary2Rate circuit for a given bit-width (Bits) and an
    input bit pattern (binary_nums). Returns:
      - total_spikes (sum of spikes across all neurons)
      - total_transmitted_spikes (sum of spikes * out_degree for each neuron)
    Optionally displays a raster plot if raster_plot=True.
    """
    # Build W_arr with your existing function
    W_arr, del_arr = create_connections_Binary2Rate(Bits)
    N = 3*Bits + 1

    # --- Count out-degree of each neuron ---
    #    Because connect_layers connects i->j if W_arr[j, i] != 0,
    #    out_degree[i] = number of j's where W_arr[j, i] != 0.
    out_degrees = [0]*N
    for i in range(N):
        out_degrees[i] = sum(1 for w in W_arr[:,i] if w != 0)

    # Create the neurons
    simulation_time = 2**Bits + 2
    ndict = {"tau_m": 1e9, 't_ref': 0}
    neuronpop = nest.Create("iaf_psc_delta", N, params=ndict)

    # Connect them using your helper
    connect_layers(neuronpop, W_arr, del_arr)

    # Spike generator to trigger bit neurons at t=1ms
    signal_generator = nest.Create("spike_generator",
                                   params={"spike_times": [1.0]})
    for i in range(Bits):
        nest.Connect(signal_generator, neuronpop[i],
                     syn_spec={'weight': 100 * binary_nums[i]})

    # Spike recorders for ALL neurons
    recorders = []
    for i in range(N):
        rec = nest.Create("spike_recorder")
        recorders.append(rec)
        nest.Connect(neuronpop[i], rec)

    # Simulate
    nest.Simulate(simulation_time)

    # --- Compute total spikes and total transmitted spikes ---
    spike_counts = [0]*N
    for i in range(N-1): #Output neuron spikes removed from computation to get excess spikes
        spike_counts[i] = len(recorders[i].get('events')['times'])

    total_spikes = sum(spike_counts)
    total_transmitted_spikes = sum(spike_counts[i]*out_degrees[i] 
                                   for i in range(N))

    return total_spikes, total_transmitted_spikes

    
def sweep_spike_counts_Binary2Rate_transmitted(min_bits=2, max_bits=5):
    """
    For each Bits in [min_bits..max_bits], sweeps all binary inputs [0 .. 2**Bits - 1].
    Measures:
      - total_spikes (per input)
      - total_transmitted_spikes (per input)
    Then computes average and maximum across all inputs and returns a dictionary:
      {
        Bits : {
          "obs_time" : 2**Bits,
          "avg_total_spikes" : float,
          "max_total_spikes" : int,
          "avg_transmitted_spikes" : float,
          "max_transmitted_spikes" : int
        },
        ...
      }
    """
    results = {}
    for Bits in range(min_bits, max_bits + 1):
        obs_time = 2**Bits

        # We'll track sums and maximums
        sum_total_spikes = 0
        sum_transmitted_spikes = 0
        max_total_spikes = 0
        max_transmitted_spikes = 0

        # Number of possible inputs is obs_time (2**Bits)
        for num in range(obs_time):
            # Convert integer -> bit list
            binary_list = decimal_to_binary_bits(Bits, num)

            # Measure spikes
            total_spikes, total_transmitted_spikes = \
                create_Binary2Rate_Circuit_spike_measures(Bits, binary_list)

            # Update sums
            sum_total_spikes += total_spikes
            sum_transmitted_spikes += total_transmitted_spikes

            # Update maxima
            if total_spikes > max_total_spikes:
                max_total_spikes = total_spikes
            if total_transmitted_spikes > max_transmitted_spikes:
                max_transmitted_spikes = total_transmitted_spikes

            # Reset NEST kernel for the next trial
            nest.ResetKernel()

        # Averages
        n_inputs = obs_time
        avg_total_spikes = sum_total_spikes / n_inputs
        avg_trans_spikes = sum_transmitted_spikes / n_inputs

        results[Bits] = {
            "obs_time": obs_time,
            "avg_total_spikes": avg_total_spikes,
            "max_total_spikes": max_total_spikes,
            "avg_transmitted_spikes": avg_trans_spikes,
            "max_transmitted_spikes": max_transmitted_spikes,
        }
    return results


def test_script(max_bits=7):
    """
    Tests the Binary2Rate circuit from 2 bits up to max_bits.
    Ensures that the number of spikes in the output neuron matches
    the decimal value of the input.
    """
    failure = 0
    for Bits in range(2, max_bits + 1):
        for num in range(2**Bits):
            binary_list = decimal_to_binary_bits(Bits, num)
            GT = get_decimal(list(reversed(binary_list)))  # The ground truth decimal

            circuit_output = create_Binary2Rate_Circuit(Bits,
                                                        binary_list,
                                                        raster_plot=False)

            # Compare the number of output spikes to the ground truth
            if len(circuit_output) != GT:
                failure += 1
                print(f'Test failed! Bits={Bits}, Num={num}, '
                      f'Output spikes={len(circuit_output)}, GT={GT}')

            # Reset NEST to clear all nodes/synapses before the next run
            nest.ResetKernel()

    print(f"Binary2Rate circuit tests completed with {failure} failures.")


if __name__ == "__main__":
    # Collect the spike counts for Bits in [2..4].
    results = sweep_spike_counts_Binary2Rate_transmitted(min_bits=2, max_bits=8)

    for bits, info in results.items():
        print(f"\n--- Bits = {bits} ---")
        print(f"obs_time = {info['obs_time']}")
        print(f"Average Total Spikes:      {info['avg_total_spikes']:.1f}")
        print(f"Max Total Spikes:         {info['max_total_spikes']}")
        print(f"Average Transmitted Spikes: {info['avg_transmitted_spikes']:.1f}")
        print(f"Max Transmitted Spikes:     {info['max_transmitted_spikes']}")
        
    show_example = True
    if show_example:
        # Example usage:
        Bits = 8
        # Let’s pick a sample number to convert
        sample_num = 63
        # Convert the sample_num to binary
        binary_input = decimal_to_binary_bits(Bits, sample_num)

        print(f"Testing Binary2Rate with Bits={Bits}, binary_input={binary_input}")
        # Create the circuit, run simulation, and produce a raster plot
        output_spike_times = create_Binary2Rate_Circuit(Bits,
                                                    binary_input,
                                                    raster_plot=True)
        print(f"Output neuron spiked {len(output_spike_times)} times.")

    # Optionally, you can run the full test suite:
    # test_script()


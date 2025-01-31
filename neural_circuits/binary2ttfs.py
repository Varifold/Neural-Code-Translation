import nest
nest.set_verbosity("M_WARNING")
nest.ResetKernel()

import numpy
import matplotlib.pyplot as plt  # For plotting the raster
from shared.functions import connect_layers, decimal_to_binary_bits


def create_connections_Binary2TTFS(Bits):
    """
    Builds the weight and delay matrices for the Binary2TTFS circuit.
    Returns:
        W_arr (ndarray): Weight matrix
        del_arr (ndarray): Delay matrix
    """
    N = 3*Bits + 2  # Number of neurons

    W_arr = numpy.zeros((N, N), dtype=numpy.float32)
    del_arr = numpy.ones((N, N), dtype=numpy.int32)

    for i in range(1, Bits):
        # Upper row
        W_arr[i + 1, i] = 15
        W_arr[2*Bits + 1 + i, i] = 10
        W_arr[2*Bits + i, i] = -15

        # Lower row
        W_arr[1 + i, 2*Bits + i] = 15
        W_arr[2*Bits + i + 1, 2*Bits + i] = 10

        # Delays for lower row
        del_arr[1 + i, 2*Bits + i] = 2**(Bits - i) + 1
        del_arr[2*Bits + i + 1, 2*Bits + i] = 2**(Bits - i) + 1

    # Connections to output neuron
    W_arr[N - 1, Bits] = 15
    W_arr[N - 1, N - 2] = 15
    del_arr[N - 1, N - 2] = 2

    # Connections from activation neuron
    W_arr[1, 0] = 15
    W_arr[2*Bits + 1, 0] = 10
    del_arr[1, 0] = 1
    del_arr[2*Bits + 1, 0] = 1

    # Bit connections
    for i in range(1, Bits + 1):
        # To top
        W_arr[i, Bits + i] = -15
        # To bot
        W_arr[2*Bits + i, Bits + i] = 10

    return W_arr, del_arr


def create_Binary2TTFS_Circuit(Bits, binary_nums, raster_plot=False):
    """
    Creates and simulates the Binary2TTFS circuit for a given bit-width (Bits) and binary input list (binary_nums).
    
    Args:
        Bits (int): Number of bits.
        binary_nums (list): List of length Bits containing binary digits (0 or 1).
        raster_plot (bool): If True, displays a raster plot of spiking for all neurons.

    Returns:
        numpy.ndarray: The spike times of the output neuron, minus an 'excess_delay'.
    """
    # We'll subtract 'excess_delay' from output spikes so that 
    # the timing encodes the original decimal number. 
    excess_delay = Bits + 3

    simulation_time = 2**Bits + Bits + 2

    # Create neurons with custom membrane potentials, etc.
    ndict = {
        "tau_m": 1000.0,
        "V_min": [-90] + [-90]*(2*Bits) + [-70]*Bits + [-90]
    }
    neuronpop = nest.Create("iaf_psc_delta", 3*Bits + 2, params=ndict)

    # Build weight & delay matrices, connect
    W_arr, del_arr = create_connections_Binary2TTFS(Bits)
    connect_layers(neuronpop, W_arr, del_arr)

    # Spike generator to feed activation and bits
    signal_generator = nest.Create("spike_generator",
                                   params={"spike_times": [1.0]})

    # Connect spike generator to each bit neuron
    for i in range(Bits):
        nest.Connect(signal_generator,
                     neuronpop[Bits + i + 1],
                     conn_spec='one_to_one',
                     syn_spec={'weight': 100*binary_nums[i]})

    # Connect spike generator to activation neuron
    nest.Connect(signal_generator, neuronpop[0],
                 conn_spec='one_to_one',
                 syn_spec={'weight': 100})

    # Spike recorder for the output neuron (last index)
    main_recorder = nest.Create("spike_recorder")
    nest.Connect(neuronpop[-1], main_recorder)

    # --- Optional: record all neurons for raster ---
    recorders = []
    for i in range(3*Bits + 2):
        rec = nest.Create("spike_recorder")
        recorders.append(rec)
        nest.Connect(neuronpop[i], rec)

    # Run simulation
    nest.Simulate(simulation_time)

    # Generate raster plot if requested
    if raster_plot:
        plt.figure(figsize=(8, 6))
        for i, rec in enumerate(recorders):
            # For consistent timing, subtract the 'excess_delay' from spikes
            times = rec.get('events')['times'] - excess_delay
            plt.plot(times, i*numpy.ones_like(times), 'k.', markersize=3)
        plt.title(f"Binary2TTFS Circuit - Bits={Bits}")
        plt.xlabel("Time (ms, shifted)")
        plt.ylabel("Neuron Index")
        plt.ylim([-1, 3*Bits + 3])
        plt.axvline(x=0, color='r', linestyle='--', label='Excess Delay = 0')
        plt.legend()
        plt.show()

    # Return the SHIFTED spike times of the output neuron
    return main_recorder.get('events')['times'] - excess_delay

import nest
import math

def create_Binary2TTFS_Circuit_spike_measures(Bits, binary_nums):
    """
    Similar to create_Binary2TTFS_Circuit, but returns two additional metrics:
      (1) total_spikes across all neurons
      (2) total_transmitted_spikes across all neurons

    We do NOT shift the spike times by 'excess_delay' here, because we only need
    the count of spikes, not the precise timing.

    Returns:
      total_spikes (int)
      total_transmitted_spikes (int)
    """
    # 1) Build W_arr, del_arr
    W_arr, del_arr = create_connections_Binary2TTFS(Bits)
    N = 3*Bits + 2  # total number of neurons

    # 2) Count out-degree of each neuron
    #    The connection is from i->j if W_arr[i, j] != 0
    out_degrees = [0]*N
    for i in range(N):
        out_degrees[i] = sum(1 for w in W_arr[:, i] if w != 0)

    # 3) Create neurons
    simulation_time = 2**Bits + Bits + 2
    ndict = {
        "tau_m": 1000.0,
        "V_min": [-90] + [-90]*(2*Bits) + [-70]*Bits + [-90]
    }
    neuronpop = nest.Create("iaf_psc_delta", N, params=ndict)

    # 4) Connect them
    connect_layers(neuronpop, W_arr, del_arr)

    # 5) Spike generator to feed activation + bits
    signal_generator = nest.Create("spike_generator", params={"spike_times": [1.0]})

    # Connect spike generator to each bit neuron
    for i in range(Bits):
        nest.Connect(signal_generator,
                     neuronpop[Bits + i + 1],
                     syn_spec={'weight': 100 * binary_nums[i]})
    # Connect the activation neuron
    nest.Connect(signal_generator, neuronpop[0], syn_spec={'weight': 100})

    # 6) Spike recorders for ALL neurons
    recorders = []
    for i in range(N):
        rec = nest.Create("spike_recorder")
        recorders.append(rec)
        nest.Connect(neuronpop[i], rec)

    # 7) Run the simulation
    nest.Simulate(simulation_time)


    # 8) Compute total_spikes and total_transmitted_spikes
    spike_counts = [len(rec.get('events')['times']) for rec in recorders]
    total_spikes = sum(spike_counts)
    total_transmitted_spikes = sum(spike_counts[i] * out_degrees[i]
                                   for i in range(N))

    return total_spikes, total_transmitted_spikes

def sweep_spike_counts_Binary2TTFS_transmitted(min_bits=2, max_bits=5):
    """
    For each Bits in [min_bits..max_bits], sweep all possible integer inputs (0..2^Bits -1),
    build/run the Binary2TTFS circuit, and measure:
      - total_spikes
      - total_transmitted_spikes
    Then compute average and maximum across all inputs for that Bits.

    Returns a dictionary of the form:
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

        sum_total_spikes = 0
        sum_transmitted_spikes = 0
        max_total_spikes = 0
        max_transmitted_spikes = 0

        # For each possible input
        for num in range(obs_time):
            # Convert decimal --> bit list
            binary_list = decimal_to_binary_bits(Bits, num)

            # Run circuit, measure
            total_spikes, total_trans_spikes = create_Binary2TTFS_Circuit_spike_measures(
                Bits, binary_list
            )

            sum_total_spikes += total_spikes
            sum_transmitted_spikes += total_trans_spikes

            if total_spikes > max_total_spikes:
                max_total_spikes = total_spikes
            if total_trans_spikes > max_transmitted_spikes:
                max_transmitted_spikes = total_trans_spikes

            # Reset NEST for next input
            nest.ResetKernel()

        # Compute averages
        n_inputs = obs_time
        avg_total_spikes = sum_total_spikes / n_inputs
        avg_trans_spikes = sum_transmitted_spikes / n_inputs

        results[Bits] = {
            "obs_time": obs_time,
            "avg_total_spikes": avg_total_spikes,
            "max_total_spikes": max_total_spikes,
            "avg_transmitted_spikes": avg_trans_spikes,
            "max_transmitted_spikes": max_transmitted_spikes
        }

    return results


def test_script(max_bits=7):
    """
    Tests the Binary2TTFS circuit on random binary inputs for bit-widths from 2..max_bits.
    Compares the decimal integer 'num' to the circuit output timing (which should match 'num').
    """
    failure = 0
    for Bits in range(2, max_bits+1):
        for num in range(2**Bits):
            # Convert decimal num to a binary list of length Bits
            binary_list = decimal_to_binary_bits(Bits, num)

            # Run the circuit and get the (shifted) output spike times
            circuit_output_times = create_Binary2TTFS_Circuit(Bits, binary_list)

            # The circuit is expected to produce exactly one spike (or none if there's a problem),
            # with time == 'num' (shifted).
            if (len(circuit_output_times) != 1) or (abs(circuit_output_times[0] - num) > 1e-9):
                print(f'Failure: Bits={Bits}, num={num}, output_times={circuit_output_times}')
                failure += 1

            nest.ResetKernel()  # Clear nodes before next iteration

    print(f"Binary2TTFS circuit tests completed with {failure} failures")


if __name__ == "__main__":

    results = sweep_spike_counts_Binary2TTFS_transmitted(min_bits=2, max_bits=8)

    for bits, info in results.items():
        print(f"\n--- Bits = {bits} ---")
        print(f"obs_time = {info['obs_time']}")
        print(f"Average Total Spikes: {info['avg_total_spikes']:.1f}")
        print(f"Max Total Spikes:     {info['max_total_spikes']}")
        print(f"Average Transmitted Spikes: {info['avg_transmitted_spikes']:.1f}")
        print(f"Max Transmitted Spikes:     {info['max_transmitted_spikes']}")
    run_example = True
    if run_example:
        # Example usage:
            # Let’s pick 4 bits and an example decimal number
        Bits = 4
        example_num = 4
        # Convert that to binary
        binary_input = decimal_to_binary_bits(Bits, example_num)
            
        print(f"Testing Binary2TTFS with Bits={Bits}, binary_input={binary_input}")
        # Generate a raster plot
        output_times = create_Binary2TTFS_Circuit(Bits, binary_input, raster_plot=True)
        print(f"Output neuron spiked at (shifted) time: {output_times}")

    # Optionally run the full test
    # test_script()

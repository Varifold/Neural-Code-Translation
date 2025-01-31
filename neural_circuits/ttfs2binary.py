import nest
nest.set_verbosity("M_WARNING")
nest.ResetKernel()

import numpy as np
import math
import matplotlib.pyplot as plt  # For the raster plot

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

    # Activation neuron connections

    # To top row
    W_arr[2, 1] = 10

    # To bot row
    W_arr[2*Bits, 1] = 10
    del_arr[2*Bits, 1] = 2**(Bits-1) + 1

    W_arr_2[2*Bits, 1] = -10
    del_arr_2[2*Bits, 1] = 2**(Bits-1)

    # To parrot neuron
    W_arr[Bits+1, 1] = 100
    del_arr[Bits+1, 1] = 2**(Bits - 1) + 1

    # Parrot neuron connections
    for i in range(0, Bits - 2):
        # To top row
        W_arr[3 + i, Bits + 1 + i] = 10

        # Top reset
        W_arr_2[2 + i, Bits + 1 + i] = -10
        del_arr_2[2 + i, Bits + 1 + i] = 2**(Bits - 1 - i) + 1

        # To next parrot neuron
        W_arr[Bits + 2 + i, Bits + 1 + i] = 100
        del_arr[Bits + 2 + i, Bits + 1 + i] = 2**(Bits - 2 - i) + 1

        # To bot row
        W_arr[2*Bits + 1 + i, Bits + 1 + i] = 10
        del_arr[2*Bits + 1 + i, Bits + 1 + i] = 2**(Bits - 2 - i) + 1

        W_arr_2[2*Bits + 1 + i, Bits + 1 + i] = -10
        del_arr_2[2*Bits + 1 + i, Bits + 1 + i] = 2**(Bits - 2 - i)

        # Bot reset
        W_arr_2[2*Bits + i, Bits + 1 + i] = -10
        del_arr_2[2*Bits + i, Bits + 1 + i] = 2**(Bits - 1 - i) + 1

    # To final neuron
    W_arr[N-1, 2*Bits-1] = 10
    del_arr[N-1, 2*Bits-1] = 2

    W_arr_2[N-1, 2*Bits-1] = -10
    del_arr_2[N-1, 2*Bits-1] = 1

    return W_arr, del_arr, W_arr_2, del_arr_2


def create_TTFS2Binary_Circuit_Parrot(Bits, spike_times, raster_plot=False):
    """
    Builds and simulates the TTFS2Binary circuit with parrot neurons for the given bit-width (Bits).
    The input 'spike_times' are sent to the timing neuron.

    Args:
        Bits (int): Number of bits for the circuit.
        spike_times (list): Times (ms) at which the input spikes occur.
        raster_plot (bool): If True, display a raster plot of all neuron spikes.

    Returns:
        (int, int): A tuple (GT, circuit_output) where GT is spike_times[0] - 1,
                    and circuit_output is the decimal interpretation of the bottom-row neurons.
    """
    neuron_num = 3 * Bits
    simulate_time = 2**Bits + Bits + 1

    # Tau for timing neurons. 
    # -t/log(0.5) => voltage halves in t ms. 
    timing_neuron_tau = [-(2**(Bits - i - 1) - 1 + 0.05) / math.log(0.5) for i in range(Bits)]

    # The neuron ordering in 'ndict' must match the connection logic
    # We'll use a custom tau_m pattern for various subsets.
    ndict = {
        'V_min': -70,
        "tau_m": (
            2*[10] +
            timing_neuron_tau[:-1] +
            [10]*(Bits - 1) +
            timing_neuron_tau[:-1] +
            timing_neuron_tau[-1:]
        )
    }
    neuronpop = nest.Create("iaf_psc_delta", neuron_num, params=ndict)

    # Create and apply connections
    W_arr, del_arr, W_arr_2, del_arr_2 = create_connections_TTFS2Binary_Parrot(Bits)
    connect_layers(neuronpop, W_arr, del_arr)
    connect_layers(neuronpop, W_arr_2, del_arr_2)

    # Spike generator for the activation neuron
    signal_generator = nest.Create("spike_generator", params={"spike_times": [1.]})
    nest.Connect(signal_generator, neuronpop[1], conn_spec='one_to_one', syn_spec={'weight': 100})

    # Spike generator for the timing neuron
    signal_generator2 = nest.Create("spike_generator", params={"spike_times": spike_times})
    nest.Connect(signal_generator2, neuronpop[0], conn_spec='one_to_one', syn_spec={'weight': 100})

    # Recorder for final neuron
    rec_final = nest.Create("spike_recorder")
    nest.Connect(neuronpop[-1], rec_final)

    # Recorders for all neurons
    recorders = []
    for i in range(neuron_num):
        rec = nest.Create("spike_recorder")
        recorders.append(rec)
        nest.Connect(neuronpop[i], rec)

    multimeters = []
    for i in range(neuron_num):
        mm = nest.Create("multimeter", params={"record_from": ["V_m"]})
        multimeters.append(mm)
        nest.Connect(mm, neuronpop[i])

    # Run the simulation
    nest.Simulate(simulate_time)

    # Generate raster plot if requested
    if raster_plot:
        plt.figure(figsize=(8, 6))
        for i, rec in enumerate(recorders):
            spike_t = rec.get('events')['times']
            plt.plot(spike_t, i * np.ones_like(spike_t), 'k.', markersize=3)
        plt.title(f"Spiking Raster (TTFS2Binary Parrot) — Bits={Bits}")
        plt.xlabel("Time (ms)")
        plt.ylabel("Neuron index")
        plt.ylim([-1, neuron_num + 1])
        plt.show()

    # Build binary output from the bottom-row neurons (the last Bits in the 3*Bits ordering).
    bin_arr = []
    for i in range(neuron_num):
        if 2*Bits <= i:  # bottom row neuron
            bin_arr.append(len(recorders[i].get('events')['times']))

    # The bottom row, from left to right, is typically MSB -> LSB,
    # so we reverse to get the standard binary "MSB left, LSB right" indexing:
    bin_arr.reverse()
    
    # GT is spike_times[0] - 1
    GT = spike_times[0] - 1
    circuit_output = get_decimal(bin_arr)

    return GT, circuit_output

def create_TTFS2Binary_Circuit_Parrot_spike_measures(Bits, spike_times):
    """
    Builds and simulates the TTFS2Binary circuit with parrot neurons for the given bit-width (Bits).
    The input 'spike_times' are sent to the timing neuron.
    
    Returns:
        GT (int) :  spike_times[0] - 1  (ground truth, from your prior logic)
        circuit_output (int) : decimal interpretation of the bottom-row neurons
        total_spikes (int) : sum of spikes across all neurons
        total_transmitted_spikes (int) : sum of (spikes_i * out_degree_i) for all neurons
    """
    # --- 1. Build adjacency matrices ---
    W_arr, del_arr, W_arr_2, del_arr_2 = create_connections_TTFS2Binary_Parrot(Bits)
    neuron_num = 3 * Bits

    # --- 2. Count out-degree of each neuron ---
    out_degrees = [0]*neuron_num
    for i in range(neuron_num):
        # out_degree = # of j where W_arr[i,j] != 0 or W_arr_2[i,j] != 0
        count_w1 = sum(1 for w in W_arr[:, i] if w != 0)
        count_w2 = sum(1 for w in W_arr_2[:, i] if w != 0)
        out_degrees[i] = count_w1 + count_w2

    # --- 3. Create neurons ---
    simulate_time = 2**Bits + Bits + 1

    # Tau for timing neurons:
    timing_neuron_tau = [-(2**(Bits - i - 1) - 1 + 0.05) / math.log(0.5) for i in range(Bits)]
    ndict = {
        'V_min': -70,
        "tau_m": (
            2*[10] +
            timing_neuron_tau[:-1] +
            [10]*(Bits - 1) +
            timing_neuron_tau[:-1] +
            timing_neuron_tau[-1:]
        )
    }
    neuronpop = nest.Create("iaf_psc_delta", neuron_num, params=ndict)

    # --- 4. Connect them ---
    connect_layers(neuronpop, W_arr, del_arr)
    connect_layers(neuronpop, W_arr_2, del_arr_2)

    # --- 5. Spike generators for activation + timing neuron ---
    signal_generator = nest.Create("spike_generator", params={"spike_times": [1.]})
    nest.Connect(signal_generator, neuronpop[1], syn_spec={'weight': 100})

    signal_generator2 = nest.Create("spike_generator", params={"spike_times": spike_times})
    nest.Connect(signal_generator2, neuronpop[0], syn_spec={'weight': 100})

    # --- 6. Spike recorders for ALL neurons ---
    recorders = []
    for i in range(neuron_num):
        rec = nest.Create("spike_recorder")
        recorders.append(rec)
        nest.Connect(neuronpop[i], rec)

    # Optionally track final neuron spikes, but we’ll read them from `recorders` anyway
    # rec_final = nest.Create("spike_recorder")
    # nest.Connect(neuronpop[-1], rec_final)

    # --- 7. Simulate ---
    nest.Simulate(simulate_time)


    # --- 8. Compute bottom-row output, total_spikes, total_transmitted_spikes ---
    spike_counts = [len(rec.get('events')['times']) for rec in recorders]
    total_spikes = sum(spike_counts)
    total_transmitted_spikes = sum(spike_counts[i] * out_degrees[i]
                                   for i in range(neuron_num))

    # The bottom row: indices from [2*Bits .. 3*Bits-1]
    # Reverse them to interpret MSB..LSB in the typical left->right order:
    bottom_row_counts = [spike_counts[i] for i in range(2*Bits, 3*Bits)][::-1]
    circuit_output = get_decimal(bottom_row_counts)


    GT = spike_times[0] - 1 if len(spike_times) > 0 else -1

    return GT, circuit_output, total_spikes, total_transmitted_spikes

def sweep_spike_counts_TTFS2Binary_Parrot_transmitted(min_bits=2, max_bits=5):
    """
    For each Bits in [min_bits..max_bits], tries all possible input spike times = 1..2^Bits.
    For each input (i.e. i from 0..2^Bits-1 => spike_times=[1 + i]):
      - runs create_TTFS2Binary_Circuit_Parrot_spike_measures
      - measures total_spikes, total_transmitted_spikes
    Returns a dictionary:
        {
          Bits: {
            "obs_time": 2**Bits,
            "avg_total_spikes": float,
            "max_total_spikes": int,
            "avg_transmitted_spikes": float,
            "max_transmitted_spikes": int
          },
          ...
        }
    """

    results = {}
    for Bits in range(min_bits, max_bits+1):
        obs_time = 2**Bits

        sum_total_spikes = 0
        sum_transmitted_spikes = 0
        max_total_spikes = 0
        max_transmitted_spikes = 0

        # Iterate over all possible inputs
        for i in range(obs_time):
            spike_times = [1 + i]

            _, _, total_spikes, transmitted_spikes = create_TTFS2Binary_Circuit_Parrot_spike_measures(
                Bits, spike_times
            )
            # Add spike_generator contribution (T and A neurons)
            sum_total_spikes += total_spikes +2
            sum_transmitted_spikes += transmitted_spikes +6

            if total_spikes > max_total_spikes:
                max_total_spikes = total_spikes
            if transmitted_spikes > max_transmitted_spikes:
                max_transmitted_spikes = transmitted_spikes

            # Reset kernel for the next run
            nest.ResetKernel()

        # Compute average
        n_inputs = obs_time
        avg_total = (sum_total_spikes) / n_inputs
        avg_trans = (sum_transmitted_spikes) / n_inputs

        # Add spike_generator contribution (T and A neurons)
        results[Bits] = {
            "obs_time": obs_time,
            "avg_total_spikes": avg_total,
            "max_total_spikes": max_total_spikes+2,
            "avg_transmitted_spikes": avg_trans,
            "max_transmitted_spikes": max_transmitted_spikes+6
        }

    return results


def test_script(max_bits=7):
    failure = 0
    for Bits in range(2, max_bits + 1):
        for i in range(2**Bits):
            spike_times = [1 + i]
            GT, circuit_output = create_TTFS2Binary_Circuit_Parrot(Bits, spike_times)
            if GT != circuit_output:
                print("Test Failure!")
                print(f'Spike_times was {GT}, output was {circuit_output}')
                failure += 1
            nest.ResetKernel()  # Clears all nodes
    print(f"TTFS2Binary circuit tests completed with {failure} failures")


if __name__ == "__main__":
    # Example: sweep Bits=2..4
    results = sweep_spike_counts_TTFS2Binary_Parrot_transmitted(min_bits=2, max_bits=8)

    for bits, info in results.items():
        print(f"\n--- Bits = {bits} ---")
        print(f"obs_time = {info['obs_time']}")
        print(f"Average total spikes = {info['avg_total_spikes']:.2f}")
        print(f"Max total spikes = {info['max_total_spikes']}")
        print(f"Average transmitted spikes = {info['avg_transmitted_spikes']:.2f}")
        print(f"Max transmitted spikes = {info['max_transmitted_spikes']}")
    run_example = True
    if run_example:    
    # Example usage:
        bits = 8
        input_spike = 63
        # If the input spike is at time=4, the ground truth is 4 - 1 = 3
        # We'll run with raster_plot=True to see the spiking
        GT, output = create_TTFS2Binary_Circuit_Parrot(
            Bits=bits,
            spike_times=[input_spike],
            raster_plot=True
        )
        print(f"GT = {GT}, circuit output = {output}")

    # Or run the test script:
    # test_script()


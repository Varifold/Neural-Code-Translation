import nest
nest.set_verbosity("M_WARNING")
import numpy

def connect_layers(self_layer, weight_matrix, delay_matrix):
    #This is based on code from NEST documentation https://nest-simulator.readthedocs.io/en/v3.3/guides/connection_management.html
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

        if list(post)!=[]: #Skip empty
            nest.Connect(pre_array, post, conn_spec='one_to_one', syn_spec={'weight': weights, 'delay': delay})

def get_decimal(lst):
    sums = 0
    if len(lst)>0:
        for i,bin in enumerate(lst):
            sums += bin*2**(i)
    return sums


def decimal_to_binary_bits(Bits, decimal):
    binary_representation = [int(b) for b in bin(decimal)[2:]]  # Convert to binary and make a list of bits
    while len(binary_representation) < Bits:  # If not enough bits, prepend with zeros
        binary_representation.insert(0, 0)
    return binary_representation

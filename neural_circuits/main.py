import ttfs2binary, rate2binary, binary2rate, binary2ttfs

if __name__ == "__main__":

    #Runs all tests for all circuits. Check neural_circuits for details
    #Setting Max_Bits too high can cause issues due to constant tau_m or very large size of the circuit
    ttfs2binary.test_script(max_bits=5)
    rate2binary.test_script(max_bits=5)
    binary2rate.test_script(max_bits=5)
    binary2ttfs.test_script(max_bits=5)

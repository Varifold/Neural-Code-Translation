"""
This script runs tests for all circuits. Check the neural_circuits module for details.
Setting Max_Bits too high can cause issues due to constant tau_m or very large size of the circuit.
"""

import ttfs2binary, rate2binary, binary2rate, binary2ttfs

def run_tests(max_bits):
    """
    Runs test scripts for various circuits with the provided max_bits value.
    
    Args:
        max_bits (int): The maximum number of bits for the test scripts. 
    """
    print("Running Tests...")
    try:
        ttfs2binary.test_script(max_bits)
    except Exception as e:
        print(f"Error in ttfs2binary test: {e}")
        
    try:
        rate2binary.test_script(max_bits)
    except Exception as e:
        print(f"Error in rate2binary test: {e}")
    
    try:
        binary2rate.test_script(max_bits)
    except Exception as e:
        print(f"Error in binary2rate test: {e}")
    
    try:
        binary2ttfs.test_script(max_bits)
    except Exception as e:
        print(f"Error in binary2ttfs test: {e}")

if __name__ == "__main__":
    MAX_BITS = 8  # Set your max_bits value here
    run_tests(MAX_BITS)

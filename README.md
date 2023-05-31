# Neural-Code-Translation

Welcome to the repository for our research on neural circuits that facilitate translation between various spiking neural encodings. Our work leverages the Leaky Integrate-and-Fire (LIF) neuron model to enable translation between rate, time-to-first-spike, and binary codes.


Repository Contents

This repository contains the following circuits:

    Binary to Time-To-First-Spike (TTFS)
    Time-To-First-Spike (TTFS) to Binary
    Binary to Rate
    Rate to Binary

Each of these circuits is located in its respective Python script. A main script (main.py) is provided that executes test cases for each of these circuits.
Requirements

    NEST Simulator version 3.4

Usage

To execute the test cases for each circuit, simply run the main.py script from the root directory of the project:

bash

python main.py

This will run a series of tests for each circuit using a maximum of 5 bits for the binary tests.

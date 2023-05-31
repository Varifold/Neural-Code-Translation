# Neural-Code-Translation

Welcome to the repository for our research on neural circuits that facilitate translation between various spiking neural and non-neural encodings. Our work leverages the Leaky Integrate-and-Fire (LIF) neuron model to enable seamless translation between rate, time-to-first-spike, and binary codes.
Abstract

In our study, we introduce neural circuits that enable translation between various spiking neural and non-neural encodings, such as rate, time-to-first-spike, and binary codes using the Leaky Integrate-and-Fire (LIF) neuron model. The development of such circuits is of great interest in the field of neuromorphic computing due to the distinct advantages of different neural encodings.

As an application of these circuits, we introduce a high-bandwidth neural transmitter that encodes and transmits binary data through a single axon and then decodes the data at the target site. The techniques employed in our handcrafted neural circuits offer valuable insights into the capabilities of LIF neurons, illustrating the potential and versatility of these small-scale circuit designs in the broader domains of computational neuroscience and artificial intelligence.
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

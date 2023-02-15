# Structured code after this resource: https://github.com/desh2608/dnn-hmm-asr/blob/master/submission.py
# And this resource: https://github.com/raminnakhli/HMM-DNN-Speech-Recognition

# implementing a HMM from scratch: https://towardsdatascience.com/hidden-markov-model-implemented-from-scratch-72865bda430e

import mlp
import numpy as np

class HMM():
    def __init__(self, num_states):
        self.pi = np.zeros(num_states)
        self.pi[0] = 1
        # next state?
        self.num_states = num_states
        # chose 16000 for 16kHz sample rate. That is 1 second of audio and 44 for number of phonemes in english language.
        self.states = [mlp.MLP(16000, 44) for i in range(num_states)]
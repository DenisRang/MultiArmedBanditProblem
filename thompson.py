import numpy as np
import matplotlib.pyplot as plt
import math


# Goal is to maximize payout from 2 unfair coins"
N = 2  # number machines
means = np.array([0.7, 0.5])  # prob of a win, each coins
probs = np.zeros(N)  # sampling prob win, each machine
S = np.zeros(N, dtype=np.int)  # number successes each machine
F = np.zeros(N, dtype=np.int)  # number failures each machine
rnd = np.random.RandomState(7)  # for machine payouts and Beta

for trial in range(10):
    print("\ntrial " + str(trial))
    for i in range(N):
        probs[i] = rnd.beta(S[i] + 1, F[i] + 1)

    print("sampling probs =  ", end="")
    for i in range(N):
        print("%0.4f  " % probs[i], end="")
        print("")
        A_t = np.argmax(probs)
        print("chose coin " + str(A_t), end="")

        p = rnd.random_sample()  # [0.0, 1.0)
        if p < means[A_t]:
            print(" - win")
            S[A_t] += 1
        else:
            print(" - lose")
            F[A_t] += 1

print("\nfinal Success vector: ", end="")
print(S)
print("final Failure vector: ", end="")
print(F)
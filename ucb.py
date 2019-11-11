import numpy as np
import matplotlib.pyplot as plt
import math


def ucb(k=10, timesteps=1000, runs=500):
    q_star = np.random.normal(0, 1, (runs, k))  # Generated values for the k-armed bandits

    S = np.zeros((runs, k))  # Knowledge of sum reward
    N = np.ones((runs, k))  # Number of previously used values (to calculate Q correctly)
    R = []  # Array of average reward per step

    for timestep in range(timesteps):
        R_t = []  # Array of rewards on the current step
        for problem in range(runs):
            maximum_ucb = 0
            A_t = -1
            for a_t in range(k):
                if (N[problem, a_t] > 0):
                    r_a = q_star[problem, a_t]  # Current reward of an arm
                    n_a = N[problem, a_t] + 1
                    av_a = (S[problem, a_t] + r_a) / n_a
                    ucb_a = av_a + math.sqrt(2 * math.log(problem + 1) / n_a)
                else:
                    ucb_a = 1e400  # Set ucb some small constant number because we can't estimate until an arm will be chosen
                if ucb_a > maximum_ucb:
                    maximum_ucb = ucb_a
                    A_t = a_t
            R_current = q_star[problem, A_t]  # Current reward
            R_t.append(R_current)
            N[problem, A_t] += 1
            S[problem, A_t] += R_current

        R.append(np.mean(R_t))

    return R

R = ucb()
fig, ax = plt.subplots()
fig.set_figwidth(15)
fig.set_figheight(10)
ax.plot(R, label="ucb")
plt.xlabel("steps")
plt.ylabel("average reward")
plt.title("Average reward per step")
plt.legend()
plt.show()
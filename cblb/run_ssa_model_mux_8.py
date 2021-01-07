import numpy as np
import matplotlib.pyplot as plt

from parameters import *
from models import *


def simulate_stochastic_mux_8(params, Y0, Omega, T_end, dt=1):
    state = np.array(Y0)

    Y_total = np.zeros([1 + T_end // dt, len(state)])
    T = np.zeros(1 + T_end // dt)
    t = 0

    Y_total[0, :] = state
    T[0] = t

    N = MUX_8_1_generate_stoichiometry()

    i = 1
    last_time = t

    while t < T_end:

        # choose two random numbers
        r = np.random.uniform(size=2)
        r1 = r[0]
        r2 = r[1]

        a = MUX_8_1_model_stochastic(state, params, Omega)

        asum = np.cumsum(a)
        a0 = np.sum(a)
        # get tau
        tau = (1.0 / a0) * np.log(1.0 / r1)

        # print(t)
        # select reaction
        reaction_number = np.argwhere(asum > r2 * a0)[0, 0]  # get first element

        # update concentrations
        state = state + N[:, reaction_number]

        # update time
        t = t + tau

        if (t - last_time >= dt) or (t >= T_end):
            last_time = t
            Y_total[i, :] = state
            T[i] = t
            i += 1

    return T[:i], Y_total[:i, :]


rho_x = 0
params = [delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, rho_x, gamma_x, theta_x, r_X]
# reaction space volume for the whole cell population
# N_cells should be set to 1
Omega = 10

# I0, I1, I2, I3, I4, I5, I6, I7
I = np.array([0, 0, 0, 0, 0, 1, 0, 0]) * 100
# S0, S1, S2
S = np.array([0, 1, 0])

Y0 = np.zeros(88)
Y0[47:87] = 1  # number of cells

Y0[:8] = I
Y0[8:11] = S

T, Y = simulate_stochastic_mux_8(params, Y0, Omega, 100)

out = Y[:, -1]
# plt.plot(T,out)
# plt.show()

I0_a, I0_b = Y[:, 2], Y[:, 3]
I1_a, I1_b = Y[:, 8], Y[:, 9]
I2_a, I2_b = Y[:, 14], Y[:, 15]
I3_a, I3_b = Y[:, 20], Y[:, 21]
I4_a, I4_b = Y[:, 26], Y[:, 27]
I5_a, I5_b = Y[:, 32], Y[:, 33]
I6_a, I6_b = Y[:, 38], Y[:, 39]
I7_a, I7_b = Y[:, 44], Y[:, 45]

out = Y[:, -1]

# plot

I0_out = Y[:, 11]
I1_out = Y[:, 12]
I2_out = Y[:, 13]
I3_out = Y[:, 14]
I4_out = Y[:, 15]
I5_out = Y[:, 16]
I6_out = Y[:, 17]
I7_out = Y[:, 18]

out = Y[:, -1]

# plot
# plot
fig, axes = plt.subplots(nrows=3, ncols=4)
ax1, ax2, ax3, ax4 = axes[0]
ax5, ax6, ax7, ax8 = axes[1]

ax1.plot(T, I0_out)
ax1.legend(["I0_out"])
ax2.plot(T, I1_out)
ax2.legend(["I1_out"])
ax3.plot(T, I2_out)
ax3.legend(["I2_out"])
ax4.plot(T, I3_out)
ax4.legend(["3_out"])
ax5.plot(T, I4_out)
ax5.legend(["I4_out"])
ax6.plot(T, I5_out)
ax6.legend(["I5_out"])
ax7.plot(T, I6_out)
ax7.legend(["I6_out"])
ax8.plot(T, I7_out)
ax8.legend(["I7_out"])


ax9 = plt.subplot(313)
ax9.plot(T, out)
ax9.set_title('out')

plt.suptitle(f"S = [{S[0]},{S[1]},{S[0]}]")
plt.show()

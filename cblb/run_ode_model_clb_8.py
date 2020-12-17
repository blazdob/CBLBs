from scipy.integrate import ode
import matplotlib.pyplot as plt
import platform
from models import *
from parameters import *

"""
[[(S1, I1)], []]
"""

states = [
    ([0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]),
    ([0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0]),
    ([1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0]),
    ([1, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0]),
    ([0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0]),
    ([0, 1, 0], [0, 0, 1, 0, 0, 0, 0, 0]),
    ([1, 1, 0], [0, 0, 1, 0, 0, 0, 0, 0]),
    ([1, 1, 0], [0, 0, 0, 1, 0, 0, 0, 0]),
    ([0, 0, 1], [0, 0, 0, 1, 0, 0, 0, 0]),
    ([0, 0, 1], [0, 0, 0, 0, 1, 0, 0, 0]),
    ([1, 0, 1], [0, 0, 0, 0, 1, 0, 0, 0]),
    ([1, 0, 1], [0, 0, 0, 0, 0, 1, 0, 0]),
    ([0, 1, 1], [0, 0, 0, 0, 0, 1, 0, 0]),
    ([0, 1, 1], [0, 0, 0, 0, 0, 0, 1, 0]),
    ([1, 1, 1], [0, 0, 0, 0, 0, 0, 1, 0]),
    ([1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 1])
]

# simulation parameters (for a single state)
t_end = 500
N = t_end

rho_x = 0
rho_y = 0

"""
rho_I0_a, rho_I0_b, rho_I1_a, rho_I1_b, rho_I2_a, rho_I2_b, rho_I3_a, rho_I3_b = 0, 5, 5, 0, 5, 0, 5, 0

params = (delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, delta_y, rho_x, rho_y, gamma_x, theta_x, r_X, r_Y, 
         rho_I0_a, rho_I0_b, rho_I1_a, rho_I1_b, rho_I2_a, rho_I2_b, rho_I3_a, rho_I3_b)
"""

Y0 = np.zeros(128)

# number of cells: toggle switches
N_I0 = np.array([1, 1])
N_I1 = np.array([1, 1])
N_I2 = np.array([1, 1])
N_I3 = np.array([1, 1])
N_I4 = np.array([1, 1])
N_I5 = np.array([1, 1])
N_I6 = np.array([1, 1])
N_I7 = np.array([1, 1])

Y0[4:6] = N_I0
Y0[10:12] = N_I1
Y0[16:18] = N_I2
Y0[22:24] = N_I3
Y0[28:30] = N_I4
Y0[34:36] = N_I5
Y0[40:42] = N_I6
Y0[46:48] = N_I7

# number of cells: mux_8
# Y0[47-8+48:87-8+48] = 1 # number of cells
Y0[87:127] = 1  # number of cells

"""
simulations
"""

for iteration, state in enumerate(states):

    S = state[0]
    I = state[1]
    I0, I1, I2, I3, I4, I5, I6, I7 = I

    if iteration > 0 and states[iteration - 1][1] == I:
        # rho_I0_a, rho_I0_b, rho_I1_a, rho_I1_b, rho_I2_a, rho_I2_b, rho_I3_a, rho_I3_b = (1-I0) * 5, I0*5, (1-I1)*5, I1*5, (1-I2)*5, I2*5, (1-I3)*5, I3*5
        rho_I0_a, rho_I0_b, rho_I1_a, rho_I1_b, rho_I2_a, rho_I2_b, rho_I3_a, rho_I3_b, \
        rho_I4_a, rho_I4_b, rho_I5_a, rho_I5_b, rho_I6_a, rho_I6_b, rho_I7_a, rho_I7_b = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    else:
        rho_I0_a, rho_I0_b, rho_I1_a, rho_I1_b, rho_I2_a, rho_I2_b, rho_I3_a, rho_I3_b, \
        rho_I4_a, rho_I4_b, rho_I5_a, rho_I5_b, rho_I6_a, rho_I6_b, rho_I7_a, rho_I7_b = (1 - I0) * 5, I0 * 5, \
                                                                                         (1 - I1) * 5, I1 * 5, (
                                                                                                     1 - I2) * 5, I2 * 5, (
                                                                                                     1 - I3) * 5, \
                                                                                         I3 * 5, (1 - I4) * 5, I4 * 5, (
                                                                                                     1 - I5) * 5, I5 * 5, (
                                                                                                     1 - I6) * 5, I6 * 5, (
                                                                                                     1 - I7) * 5, I7 * 5

    rho_x, rho_y = 0, 0
    params = (
        delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, delta_y, rho_x, rho_y, gamma_x, theta_x, r_X,
        r_Y,
        rho_I0_a, rho_I0_b, rho_I1_a, rho_I1_b, rho_I2_a, rho_I2_b, rho_I3_a, rho_I3_b,
        rho_I4_a, rho_I4_b, rho_I5_a, rho_I5_b, rho_I6_a, rho_I6_b, rho_I7_a, rho_I7_b)

    if iteration:
        Y0 = Y_last[-1, :]

    Y0[48:51] = S

    # initialization

    T = np.linspace(0, t_end, N)

    t1 = t_end
    dt = t_end / N
    T = np.arange(0, t1 + dt, dt)
    Y = np.zeros([1 + N, 128])
    Y[0, :] = Y0

    # simulation
    r = ode(CLB_8_model_ODE).set_integrator('zvode', method='bdf')
    r.set_initial_value(Y0, T[0]).set_f_params(params)

    i = 1
    while r.successful() and r.t < t1:

        Y[i, :] = r.integrate(r.t + dt)
        i += 1

        # hold the state after half of the simulation time!
        if r.t > t1 / 2:
            params = (
                delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, delta_y, rho_x, rho_y, gamma_x,
                theta_x,
                r_X, r_Y,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            r.set_f_params(params)

    Y_last = Y
    if not iteration:
        Y_full = Y
        T_full = T
    else:
        Y_full = np.append(Y_full, Y, axis=0)
        T_full = np.append(T_full, T + iteration * t_end, axis=0)

Y = Y_full
T = T_full

S0, S1, S2 = Y[:, 48], Y[:, 49], Y[:, 50]

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
fig, axes = plt.subplots(nrows=4, ncols=4)
ax1, ax2, ax3, ax4 = axes[0]
ax5, ax6, ax7, ax8 = axes[1]

ax1.plot(T, I0_a, color="#800000ff", alpha=0.75)
ax1.plot(T, I0_b, color="#999999ff", alpha=0.75)
ax1.legend(["$I_0$", "$\\overline{I_0}$"])
# ax1.set_title('$I_0$ toggle')
ax1.set_xlabel("Time [min]")
ax1.set_ylabel("Concentrations [nM]")


ax2.plot(T, I1_a, color="#00ff00ff", alpha=0.75)
ax2.plot(T, I1_b, color="#666666ff")  # , alpha=0.75)
ax2.legend(["$I_1$", "$\\overline{I_1}$"])
# ax2.set_title('$I_1$ toggle')
ax2.set_xlabel("Time [min]")
ax2.set_ylabel("Concentrations [nM]")


ax3.plot(T, I2_a, color="#0000ffff", alpha=0.75)
ax3.plot(T, I2_b, color="#ecececfe")  # , alpha=0.75)
ax3.legend(["$I_2$", "$\\overline{I_2}$"])
# ax3.set_title('$I_2$ toggle')
ax3.set_xlabel("Time [min]")
ax3.set_ylabel("Concentrations [nM]")


ax4.plot(T, I3_a, color="#800080ff", alpha=0.75)
ax4.plot(T, I3_b, color="#999999fc")  # , alpha=0.75)
ax4.legend(["$I_3$", "$\\overline{I_3}$"])
# ax4.set_title('$I_3$ toggle')
ax4.set_xlabel("Time [min]")
ax4.set_ylabel("Concentrations [nM]")



ax5.plot(T, I4_a, color="#800000ff", alpha=0.75)
ax5.plot(T, I4_b, color="#999999ff", alpha=0.75)
ax5.legend(["$I_4$", "$\\overline{I_4}$"])
# ax1.set_title('$I_0$ toggle')
ax5.set_xlabel("Time [min]")
ax5.set_ylabel("Concentrations [nM]")

ax6.plot(T, I5_a, color="#00ff00ff", alpha=0.75)
ax6.plot(T, I5_b, color="#666666ff")  # , alpha=0.75)
ax6.legend(["$I_5$", "$\\overline{I_5}$"])
# ax2.set_title('$I_1$ toggle')
ax6.set_xlabel("Time [min]")
ax6.set_ylabel("Concentrations [nM]")

ax7.plot(T, I6_a, color="#0000ffff", alpha=0.75)
ax7.plot(T, I6_b, color="#ecececfe")  # , alpha=0.75)
ax7.legend(["$I_6$", "$\\overline{I_6}$"])
# ax3.set_title('$I_2$ toggle')
ax7.set_xlabel("Time [min]")
ax7.set_ylabel("Concentrations [nM]")

ax8.plot(T, I7_a, color="#800080ff", alpha=0.75)
ax8.plot(T, I7_b, color="#999999fc")  # , alpha=0.75)
ax8.legend(["$I_7$", "$\\overline{I_7}$"])
# ax4.set_title('$I_3$ toggle')
ax8.set_xlabel("Time [min]")
ax8.set_ylabel("Concentrations [nM]")

ax9 = plt.subplot(413)
ax9.plot(T, S0, color="#ff6600ff", alpha=0.75)
ax9.plot(T, S1, color="#ffff00ff")  # , alpha=0.75)
ax9.plot(T, S2, color="#ff0000ff")  # , alpha=0.75)
ax9.legend(["$S_0$", "$S_1$", "$S_2$"])
# ax5.set_title('Select inputs')
ax9.set_xlabel("Time [min]")
ax9.set_ylabel("Concentrations [nM]")

ax10 = plt.subplot(414)
ax10.plot(T, out, color="#8080805a", alpha=0.75)
# ax6.set_title('out')
ax10.legend(['out'])
ax10.set_xlabel("Time [min]")
ax10.set_ylabel("Concentrations [nM]")

# plt.suptitle("$out = \\overline{S}_1 \\overline{S}_0 I_0 \\vee \\overline{S}_1 S_0 I_1 \\vee S_1 \\overline{S}_0 I_2 \\vee S_1 S_0 I_3$")
plt.gcf().set_size_inches(15, 10)

if platform.system() == "Linux":
    plt.savefig("figs/CBLB_8_ode.pdf", bbox_inches='tight')
else:
    plt.savefig("figs\\CBLB_8_ode.pdf", bbox_inches='tight')
plt.show()

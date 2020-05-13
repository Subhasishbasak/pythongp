import matplotlib.pyplot as plt

plt.figure()

plt.vlines(2, -0.2, 0.2, colors = 'g', label = 'x_n')
plt.vlines(4, -0.2, 0.2, label = 'min_{i < n} (x_i)')
plt.xlim((-3, 6))
plt.ylim((-5, 5))

plt.plot([-3, 6], [0, 0])
plt.plot([2, 4], [0, 0], 'r', label = 'improvement')

plt.legend()
plt.show()
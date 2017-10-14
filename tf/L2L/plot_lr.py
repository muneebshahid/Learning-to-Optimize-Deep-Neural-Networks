import matplotlib.pyplot as plt
import numpy as np

decay_steps = 10000
learning_rate = 0.001
end_learning_rate = 0.00001
power = 5

learrning_rates = []
for global_step in range(decay_steps):
    global_step = np.minimum(global_step, decay_steps)
    decayed_learning_rate = (learning_rate - end_learning_rate) * np.power((1.0 - float(global_step) / decay_steps), power) + end_learning_rate
    learrning_rates.append(decayed_learning_rate)

learrning_rates = np.array(learrning_rates)
plt.plot(range(decay_steps), learrning_rates)
plt.show()
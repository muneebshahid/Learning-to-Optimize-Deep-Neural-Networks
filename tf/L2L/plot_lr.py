import matplotlib.pyplot as plt
import numpy as np

decay_steps = 50000
learning_rate = 0.0003
end_learning_rate = 0.000005
power = 4

learrning_rates = []
for global_step in range(decay_steps):
    global_step = np.minimum(global_step, decay_steps)
    decayed_learning_rate = (learning_rate - end_learning_rate) * np.power((1.0 - float(global_step) / decay_steps), power) + end_learning_rate
    learrning_rates.append(decayed_learning_rate)

learrning_rates = np.array(learrning_rates)
plt.plot(range(decay_steps), learrning_rates)
plt.show()

# -decay_steps = 50000
# -learning_rate = 0.0003
# -end_learning_rate = 0.000005
# +decay_steps = 500*390
# +learning_rate = 0.05
# +end_learning_rate = 0.0
#  power = 4
 
#  learrning_rates = []
#  for global_step in range(decay_steps):
# -    global_step = np.minimum(global_step, decay_steps)
# -    decayed_learning_rate = (learning_rate - end_learning_rate) * np.power((1.0 - float(global_step) / decay_steps), power) + end_learning_rate
# +    global_step = float(np.minimum(global_step, decay_steps))
# +    # decayed_learning_rate = (learning_rate - end_learning_rate) * np.power((1.0 - float(global_step) / decay_steps), power) + end_learning_rate
# +    decayed_learning_rate = end_learning_rate + 0.5 * (learning_rate - end_learning_rate) * (1 + np.cos(global_step / decay_steps) * np.pi)

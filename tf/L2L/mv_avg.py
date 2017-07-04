def avg(avg, iterates, beta):
    for iterate in iterates:
        avg = beta * avg + (1 - beta) * iterate
    return avg
prev_avg = 100
iterates = [9, 8, 7, 6]
avg(prev_avg, iterates, .9)

import numpy as np
import matplotlib.pyplot as plt
import csv
import random
import time

dest_x = []
dest_y = []
dest_num = 0    # number of destinations
with open("data5.csv", 'r') as f:
    reader = csv.reader(f)
    fieldnames = next(reader)
    csv_reader = csv.DictReader(f, fieldnames=fieldnames)
    for row in csv_reader:
        d = {}
        for k, v in row.items():
            d[k] = float(v)
        dest_x.append(d['x'])
        dest_y.append(d['y'])
        dest_num += 1
# parameters
ant_num = 100
alpha = 2
beta = 1
rho = 0.95
Q = 100
dest_x = np.array(dest_x)
dest_y = np.array(dest_y)

dis = np.zeros((dest_num, dest_num))  # distance between two destinations
for i in range(dest_num):
    for j in range(dest_num):
        dis[i][j] = ((dest_x[i] - dest_x[j]) ** 2 +
                     (dest_y[i] - dest_y[j]) ** 2) ** 0.5
phe = np.ones((dest_num, dest_num))  # Pheromones chart
# traveled chart, -1 means not traveled yet
traveled = np.zeros((ant_num, dest_num))-1

iteration = 50
best_route = np.zeros((iteration, dest_num))
best_dis = np.zeros((iteration))
avg_dis = np.zeros((iteration))
t0 = time.time()
for n in range(iteration):
    # put every ant in a random position to start
    start_pos = np.zeros((ant_num))
    for i in range(ant_num):
        start_pos[i] = random.randint(0, dest_num - 1)
        traveled[i][0] = start_pos[i]
    # every ant should travel through all the destinations
    for j in range(1, dest_num):
        for i in range(ant_num):
            prob = np.zeros((dest_num - j))
            not_traveled = np.zeros((dest_num - j))
            not_traveled_idx = 0
            for k in range(dest_num):
                if k not in traveled[i]:
                    not_traveled[not_traveled_idx] = k
                    not_traveled_idx += 1
            # record the ant's recent position, which is that last one in the traveled[i]
            pos_now = int(traveled[i][j-1])
            # the posibility for the ant to travel from the recent destination to another
            for k in range(dest_num - j):
                tmp_next = int(not_traveled[k])
                prob[k] = (phe[pos_now][tmp_next]**alpha) * \
                    ((1/dis[pos_now][tmp_next])**beta)
            prob_sum = np.sum(prob)
            prob = prob / prob_sum
            # using random number with the probility to get the next destination
            prob_cumsum = np.cumsum(prob)
            tmp_rand = random.randint(0, 9999) / 10000
            next_dest_idx = 0
            while prob_cumsum[next_dest_idx] < tmp_rand:
                next_dest_idx += 1
            next_dest = not_traveled[next_dest_idx]
            traveled[i][j] = next_dest
    # every ant has finished traveling
    # determine the total traveling distance
    total_dis = np.zeros((ant_num))
    for i in range(ant_num):
        for j in range(dest_num-1):
            total_dis[i] += dis[int(traveled[i][j])][int(traveled[i][j + 1])]
    # print(traveled)
    # print(total_dis)
    # find the best ant whose traveling distance is the shortest
    best_ant = np.where(total_dis == np.min(total_dis))
    # best_ant'datatype is an array, get the first item number
    best_ant = best_ant[0][0].item()
    best_route[n] = traveled[best_ant]
    best_dis[n] = total_dis[best_ant]
    avg_dis[n] = np.mean(total_dis)
    # ants leave pheromones on traveled route
    phe_delta = np.zeros((dest_num, dest_num))
    for i in range(ant_num):
        for j in range(dest_num-1):
            phe_delta[int(traveled[i][j])][int(
                traveled[i][j + 1])] += Q/total_dis[i]
    phe = rho * phe + phe_delta
    # update the traveld matrix
    traveled = np.zeros((ant_num, dest_num)) - 1
# print(best_route)
# print(best_dis)
# print(avg_dis)
t1 = time.time()
print('Run time: {}s'.format(t1-t0))
# reorganize the last best_route for plot
plot_route_x = np.zeros((dest_num - 1, 2))
plot_route_y = np.zeros((dest_num - 1, 2))

best_iter = np.where(best_dis == np.min(best_dis))
# print(best_iter)
# find the best iteration whose best_dis is the shortest
best_iter = best_iter[0][0].item()

for i in range(dest_num - 1):
    #plot_route_x[i][0] = dest_x[int(best_route[iteration - 1][i])]
    #plot_route_x[i][1] = dest_x[int(best_route[iteration - 1][i + 1])]
    #plot_route_y[i][0] = dest_y[int(best_route[iteration - 1][i])]
    #plot_route_y[i][1] = dest_y[int(best_route[iteration - 1][i + 1])]
    plot_route_x[i][0] = dest_x[int(best_route[best_iter][i])]
    plot_route_x[i][1] = dest_x[int(best_route[best_iter][i + 1])]
    plot_route_y[i][0] = dest_y[int(best_route[best_iter][i])]
    plot_route_y[i][1] = dest_y[int(best_route[best_iter][i + 1])]


x = np.arange(iteration)
plt.figure()
plt.subplot(1, 3, 1)
plt.plot(x, best_dis)
plt.title('best distance')
plt.xlabel('iteration')
plt.ylabel('distance')
plt.subplot(1, 3, 2)
plt.plot(x, avg_dis)
plt.title('average distance')
plt.xlabel('iteration')
plt.ylabel('distance')
plt.subplot(1, 3, 3)
plt.plot(plot_route_x, plot_route_y, 'tan')
plt.scatter(plot_route_x, plot_route_y, color='orange')
plt.title('final best route')
plt.show()

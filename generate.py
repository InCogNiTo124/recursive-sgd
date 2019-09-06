import numpy as np
import matplotlib.pyplot as plt
def assign(x, y):
    if x < 0.5:
        if y < 0.5:
            return int((y - 0.5) ** 2 + x ** 2 >= 0.25)
        else:
            return 0
    else:
        if y < 0.5:
            return 1
        else:
            return int((y-0.5) ** 2 + (x-1)**2 <= 0.25)
assign = np.vectorize(assign)

x = np.arange(0, 1, 1/100)
y = np.arange(0, 1, 1/100)
a = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
b = assign(a[:, 0], a[:, 1])
plt.scatter(a[:, 0], a[:, 1], c=b, s=1)
plt.show()
idx = np.random.choice(len(a), 1024, False)
selected_a_0 = list(a[idx, 0])
selected_a_1 = list(a[idx, 1])
selected_b = list(b[idx])
print(sum(selected_b)/1024)
if int(input("Is this acceptable?")):
    csv = "\n".join(["{:.2}, {:.2}, {}".format(x, y, v) for x, y, v in zip(selected_a_0, selected_a_1, selected_b)])
    with open("dataset.csv", "w") as file:
        file.write(csv)
print("Done.")


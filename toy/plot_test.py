import matplotlib.pyplot as plt
import numpy as np

# Generate a list of 2000 numbers
x_axis = np.arange(0, 2000, 100)

# Generate a list of y-axis values
y_axis = [i * 2 for i in x_axis]

plt.figure()
# Plot the graph
plt.plot(x_axis, y_axis)

# Set the x-axis interval to 100
plt.xticks(range(0, 2000, 100))

# Add labels and title
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Line Graph with Interval of 100")

# Show the graph
plt.draw()
plt.show()
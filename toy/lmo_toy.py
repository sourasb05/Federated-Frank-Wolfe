import numpy as np

def argmin(A, b, c):
    m, n = A.shape

    # Initialize the solution vector
    x = np.zeros(n)
    print("m :",m)
    print("n :",n)
    print("x :",x)
    #i=0
    while True:
        #i+=1
        #print(i)
        # Check if the current solution satisfies all constraints
        if np.all(A @ x >= b):
            return x

        # Determine the column to enter the basis
        j_star = np.argmin(c)
        print("j_star :",j_star)
        # Calculate the step size
        print(" A @ x", A @ x)
        print("A :",A)
        print("A[:, j_star]",A[:, j_star])
        step_size = np.min((b - A @ x) / A[:, j_star])
        print("step_size :",step_size)
        # Update the solution
        x += step_size * np.squeeze(A[:, j_star])
        print("x :",x)
        input("press")




A = np.array([[1, 2], [3, 4]])
b = np.array([3, 5])
c = np.array([-3, 1])

solution = argmin(A, b, c)
print("Optimal solution:", solution)
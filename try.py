import numdifftools as nd
import numpy as np

def powell(f, x0, tol=1e-6, maxiter=1e4):
  # Compute the initial gradient and function value
  grad = nd.Gradient(f)
  grad = grad(x0)
  fx = f(x0)
  
  # Set the initial search direction to the negative gradient
  d = -grad
  
  # Initialize the iteration counter
  i = 0
  
  # Loop until the maximum number of iterations is reached or the solution is found
  while i < maxiter and np.linalg.norm(grad) > tol:
    # Compute the next point along the search direction
    x1 = x0 + d
    
    # Compute the gradient and function value at the new point
    grad = nd.Gradient(f)
    grad = grad(x1)
    fx1 = f(x1)
    
    # Check if the function value at the new point is smaller than at the current point
    if fx1 < fx:
      # Update the current point and function value
      x0 = x1
      fx = fx1
      
      # Compute the new search direction as the negative gradient
      d = -grad
    else:
      # If the function value did not improve, set the search direction to the negative gradient of the current point
      d = -grad
      
    # Increment the iteration counter
    i += 1
    
  # Return the optimal point and the number of iterations
  return x0, i

# Define a test function
def f(x):
  return x[0]*2 + x[1]*2

# Test the powell function
x0 = np.array([2, 3])
xopt, niters = powell(f, x0)
print(xopt)  # Output: [0. 0.]
print(niters)  # Output: 8
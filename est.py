import csv
import numpy as np
import matplotlib.pyplot as plt

# 1) LOAD DATA

liste_x = []
liste_y = []

with open('data.csv') as f:
	reader = csv.reader(f, delimiter=',')
	next(reader, None)

	for row in reader:
		km = float(row[0])
		price = float(row[1])
		liste_x.append(km)
		liste_y.append(price)

arr_x = np.array(liste_x).reshape(-1, 1)
arr_y = np.array(liste_y).reshape(-1, 1)

print("Shape X:", arr_x.shape)
print("Shape Y:", arr_y.shape)

# 2) NORMALIZATION OF X

mean_x = np.mean(arr_x)
std_x = np.std(arr_x)

arr_x_norm = (arr_x - mean_x) / std_x

# Construct design matrix
X = np.hstack((arr_x_norm, np.ones((arr_x_norm.shape[0], 1)))).astype(float)

# 3) INITIALIZE THETA

rng = np.random.default_rng()
theta = rng.standard_normal((X.shape[1], 1))

# 4) MODEL / COST / GRADIENT

def model(X, theta):
	"""Return predictions."""
	return X.dot(theta)

def cost_function(X, y, theta):
	m = len(y)
	preds = model(X, theta)
	return (1/(2*m)) * np.sum((preds - y)**2)

def grad(X, y, theta):
	m = len(y)
	return (1/m) * X.T.dot(model(X, theta) - y)

def gradient_descent(X, y, theta, learning_rate, iterations):
	for i in range(iterations):
		theta = theta - learning_rate * grad(X, y, theta)
	return theta

def precision_coeff(y, pred):
	"""Compute R² score."""
	u = ((y - pred)**2).sum()
	v = ((y - y.mean())**2).sum()
	return 1 - u/v

# 5) TRAIN MODEL

theta_final = gradient_descent(X, arr_y, theta, learning_rate=0.01, iterations=10000)

predictions = model(X, theta_final)

print("Coefficient de précision (R²) :", precision_coeff(arr_y, predictions))

# 6) PLOT RESULTS

plt.scatter(arr_x, arr_y, label="Data points")
plt.plot(arr_x, predictions, color='red', label="Linear regression")

plt.xlabel("Kilometers")
plt.ylabel("Price")
plt.title("Linear Regression on Car Dataset")
plt.legend()
plt.grid(True)
plt.show()

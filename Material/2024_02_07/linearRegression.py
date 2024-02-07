import numpy as np

class LinearRegression:
    def __init__(self):
        self.b0 = 0
        self.b1 = 0

    def fit (self, X, y):
        x_mean = np.mean (X)
        y_mean = np.mean (y)
        numerator, denominator = 0, 0
        
        for _ in range (len (X)):
            numerator += (X[_]-x_mean)*(y[_]-y_mean)
            denominator += (X[_]-x_mean)**2
        
        self.b1 = numerator / denominator
        self.b0 = y_mean - (self.b1 * x_mean)

        return self.b0, self.b1
    
    def predict (self, T_x):
        y_hat = self.b0 + (self.b1 * T_x)
        return y_hat

if __name__ == '__main__':

    X = np.array ([180, 155, 167, 175], ndmin=2)
    X = X.reshape (4, 1)
    y = np.array ([80, 65, 70, 74])

    model = LinearRegression ()
    b0, b1 = model.fit (X, y)
    print (f'The value of b0 : {b0}')
    print (f'The value of b1 : {b1}')

    y_pred = model.predict ([[158]])
    print (f'The output is {y_pred}')
# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 01/09/2026



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# Data Load
data=pd.read_csv('/content/Sample - Superstore.csv', encoding='latin1')
N=1000
plt.rcParams['figure.figsize'] = [12, 6] #plt.rcParams is a dictionary-like object in Mat
X=data['Sales']
plt.plot(X)
plt.title('Original Data')
plt.show()
plt.subplot(2, 1, 1)
plot_acf(X, lags=len(X)/4, ax=plt.gca())
plt.title('Original Data ACF')
plt.subplot(2, 1, 2)
plot_pacf(X, lags=len(X)/4, ax=plt.gca())
plt.title('Original Data PACF')
plt.tight_layout()
plt.show()
arma11_model = ARIMA(X, order=(1, 0, 1)).fit()
phi1_arma11 = arma11_model.params['ar.L1']
theta1_arma11 = arma11_model.params['ma.L1']
# Simulate ARMA(1,1) Process
ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)
plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 500])
plt.show()
# Plot ACF and PACF for ARMA(1,1)
plot_acf(ARMA_1)
plt.show()
plot_pacf(ARMA_1)
plt.show()
# Fitting the ARMA(1,1) model and deriving parameters
arma22_model = ARIMA(X, order=(2, 0, 2)).fit()
phi1_arma22 = arma22_model.params['ar.L1']
phi2_arma22 = arma22_model.params['ar.L2']
theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']
```

# OUTPUT:
SIMULATED ARMA(1,1) PROCESS:
<img width="1037" height="552" alt="image" src="https://github.com/user-attachments/assets/bfebe8b1-e26e-41b3-9ad4-f5397f1e3924" />



Partial Autocorrelation
<img width="1228" height="672" alt="image" src="https://github.com/user-attachments/assets/0a0efe57-82ec-45a9-a18a-7278e912b1aa" />


Autocorrelation

<img width="1217" height="666" alt="image" src="https://github.com/user-attachments/assets/9f6d2bc4-5fb7-4dfd-972c-057847694e96" />


SIMULATED ARMA(2,2) PROCESS:
<img width="1227" height="676" alt="image" src="https://github.com/user-attachments/assets/11c98bef-bf3a-4a53-a315-39540e38fe87" />


Partial Autocorrelation
<img width="1255" height="652" alt="image" src="https://github.com/user-attachments/assets/6cc0b275-b1d4-47c6-b97c-f099c797ef8a" />



Autocorrelation
<img width="1247" height="661" alt="image" src="https://github.com/user-attachments/assets/f8d48515-ff8b-4dff-a3e6-1cce00573b2c" />


# RESULT:
Thus, a python program is created to fir ARMA Model successfully.

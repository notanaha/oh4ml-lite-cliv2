import os, argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import joblib
import mlflow

from pandas import Grouper
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, help="model_name", default="arima-model")
parser.add_argument('--input', type=str, help="input")
parser.add_argument('--output', type=str, help="output", default="./outputs")
args = parser.parse_args()

print("Argument 1: %s" % args.model_name)
print("Argument 2: %s" % args.input)
print("Argument 3: %s" % args.output)

df = pd.read_csv(args.input)

df.set_index('TransactionDate',inplace=True)
df.columns = ['PaidAmount']
series = pd.Series(df['PaidAmount'])

def mean_and_variance(X):
    split = int(len(X) / 2)
    X1, X2 = X[0:split], X[split:]
    mean1, mean2 = X1.mean(), X2.mean()
    var1, var2 = X1.var(), X2.var()
    print('mean1=%f, mean2=%f' % (mean1, mean2))
    print('variance1=%f, variance2=%f' % (var1, var2))

mean_and_variance(series.values)

def fuller_test(X):
    result = adfuller(X)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
    	print('\t%s: %.3f' % (key, value))

fuller_test(series.values)

plot_acf(series)

plot_pacf(series)

X = series.values
size = int(len(X) * 0.9)
train, test = X[0:size], X[size:len(X)]

model = ARIMA(train, order=(2,0,2))
model_fit = model.fit()
print(model_fit.summary())

# plot residual errors
residuals = pd.DataFrame(model_fit.resid)
residuals.plot(title="Residuals Error Plot")
plt.savefig(os.path.join(args.output, "res.png"))
#residuals.plot(kind='kde')
#plt.show()
#print(residuals.describe())

predictions=model_fit.forecast(steps=test.size)

mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(test,predictions)
print('Test RMSE: %.3f' % rmse)
print('Test R2: %.3f' % r2)

# plot
plt.plot(test)
plt.plot(predictions, color='red')
plt.title("Test Data Vs. Predictions")
plt.savefig(os.path.join(args.output, "pred.png"))

mlflow.log_metric('RMSE', rmse)
mlflow.log_metric('R2', r2)

model_file_path = 'models/' + 'arima_model.pkl'
filename = os.path.join(args.output, model_file_path)
print(filename)

os.makedirs(os.path.join(args.output, 'models'), exist_ok=True)
joblib.dump(value=model_fit, filename=filename)


metric = {}
metric['model_name'] = args.model_name
metric['RMSE'] = rmse
metric['R2'] = r2
print(metric)

with open(os.path.join(args.output, 'metric.json'), "w") as outfile:
    json.dump(metric, outfile)

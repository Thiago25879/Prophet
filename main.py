import pandas as pd
from prophet import Prophet
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

y = pd.read_csv('Data/air_visit_data.csv.zip')
y = y.pivot(index='visit_date', columns='air_store_id')['visitors']
y = y.fillna(0)
y = pd.DataFrame(y.sum(axis=1))
print(y)

y = y.reset_index(drop=False)
y.columns = ['ds', 'y']

# Test set consists of the last 28 days
train = y.iloc[:450, :]
test = y.iloc[450:, :]

m = Prophet()
m.fit(train)

future = m.make_future_dataframe(periods=len(test))
forecast = m.predict(future)

print(r2_score(list(test['y']), list(forecast.loc[450:,'yhat'] )))

plt.plot(list(test['y']))
plt.plot(list(forecast.loc[450:,'yhat'] ))
plt.show()

fig1 = m.plot(forecast)
plt.show()

fig2 = m.plot_components(forecast)
plt.show()
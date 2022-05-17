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

holidays = pd.read_csv('Data/date_info.csv.zip')
holidays = holidays[holidays['holiday_flg'] == 1]
holidays = holidays[['calendar_date', 'holiday_flg']]
holidays = holidays.drop(['holiday_flg'], axis=1)
holidays['holiday'] = 'holiday'
holidays.columns = ['ds', 'holiday']

X_reservations = pd.read_csv('Data/air_reserve.csv.zip')
X_reservations['visit_date'] = pd.to_datetime(X_reservations['visit_datetime']).dt.date
X_reservations = pd.DataFrame(X_reservations.groupby('visit_date')
['reserve_visitors'].sum())
X_reservations = X_reservations.reset_index(drop = False)
train = train.copy()
train['ds'] = pd.to_datetime(train['ds']).dt.date
train = train.merge(X_reservations, left_on = 'ds', right_on = 'visit_date', how = 'left')[['ds', 'y', 'reserve_visitors']].fillna(0)

m = Prophet(holidays=holidays)
m.add_regressor('reserve_visitors')
m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
m.fit(train)

future = m.make_future_dataframe(periods=len(test))
future['ds'] = pd.to_datetime(future['ds']).dt.date
future = future.merge(X_reservations, left_on='ds', right_on='visit_date', how='left')[['ds', 'reserve_visitors']].fillna(0)
forecast = m.predict(future)

print(r2_score(list(test['y']), list(forecast.loc[450:,'yhat'] )))

plt.plot(list(test['y']))
plt.plot(list(forecast.loc[450:,'yhat'] ))
plt.show()

fig1 = m.plot(forecast)
plt.show()

fig2 = m.plot_components(forecast)
plt.show()
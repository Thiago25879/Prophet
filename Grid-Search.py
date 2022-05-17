import pandas as pd
from prophet import Prophet
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

y = pd.read_csv('Data/air_visit_data.csv.zip')
y = y.pivot(index='visit_date', columns='air_store_id')['visitors']
y = y.fillna(0)
y = pd.DataFrame(y.sum(axis=1))
# print(y)

y = y.reset_index(drop=False)
y.columns = ['ds', 'y']

# Test set consists of the last 28 days
train4 = y.iloc[:450, :]
test = y.iloc[450:, :]

holidays = pd.read_csv('Data/date_info.csv.zip')
holidays = holidays[holidays['holiday_flg'] == 1]
holidays = holidays[['calendar_date', 'holiday_flg']]
holidays = holidays.drop(['holiday_flg'], axis=1)
holidays['holiday'] = 'holiday'
holidays.columns = ['ds', 'holiday']

X_reservations = pd.read_csv('Data/air_reserve.csv.zip')
X_reservations['visit_date'] = pd.to_datetime(X_reservations['visit_datetime']).dt.date
X_reservations = pd.DataFrame(X_reservations.groupby('visit_date')['reserve_visitors'].sum())
X_reservations = X_reservations.reset_index(drop=False)
train4 = train4.copy()
train4['ds'] = pd.to_datetime(train4['ds']).dt.date
train4 = train4.merge(X_reservations, left_on='ds', right_on='visit_date', how='left')[['ds', 'y', 'reserve_visitors']].fillna(0)


def model_test(holidays, weekly_seasonality,
               yearly_seasonality, add_monthly, add_reserve, changepoint_prior_scale,
               holidays_prior_scale, month_fourier):
    m4 = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        holidays=holidays,
        changepoint_prior_scale=changepoint_prior_scale,
        holidays_prior_scale=holidays_prior_scale)
    if add_monthly:
        m4.add_seasonality(
            name='monthly',
            period=30.5,
            fourier_order=month_fourier)
    if add_reserve:
        m4.add_regressor('reserve_visitors')
    m4.fit(train4)
    future4 = m4.make_future_dataframe(periods=len(test))
    future4['ds'] = pd.to_datetime(future4['ds']).dt.date
    if add_reserve:
        future4 = future4.merge(X_reservations, left_on='ds', right_on='visit_date', how='left')[['ds', 'reserve_visitors']].fillna(0)
        forecast4 = m4.predict(future4)
        return r2_score(list(test['y']), list(forecast4.loc[450:, 'yhat']))


# Setting the grid
holidays_opt = [holidays, None]
weekly_seas = [5, 10, 30, 50]
yearly_seas = [5, 10, 30, 50]
add_monthly = [True, False]
add_reserve = [True, False]
changepoint_prior_scale = [0.1, 0.3, 0.5]
holidays_prior_scale = [0.1, 0.3, 0.5]
month_fourier = [5, 10, 30, 50]
# Looping through the grid
grid_results = []
for h in holidays_opt:
    for w in weekly_seas:
        for ys in yearly_seas:
            for m in add_monthly:
                for r in add_reserve:
                    for c in changepoint_prior_scale:
                        for hp in holidays_prior_scale:
                            for mf in month_fourier:
                                r2 = model_test(h, w, ys, m, r, c, hp, mf)
                                print([w, ys, m, r, c, hp, mf, r2])
                                grid_results.append([h, w, ys, m, r, c, hp, mf, r2])
# adding it all to a dataframe and extract the best model
benchmark = pd.DataFrame(grid_results)
benchmark = benchmark.sort_values(8, ascending=False)
h, w, ys, m, r, c, hp, mf, r2 = list(benchmark.iloc[0, :])
# Fit the Prophet with those best hyperparameters
m4 = Prophet(
    yearly_seasonality=ys,
    weekly_seasonality=w,
    holidays=h,
    changepoint_prior_scale=c,
    holidays_prior_scale=hp)
if m:
    m4.add_seasonality(
        name='monthly',
        period=30.5,
        fourier_order=mf)
if r:
    m4.add_regressor('reserve_visitors')
m4.fit(train4)
future4 = m4.make_future_dataframe(periods=len(test))
future4['ds'] = pd.to_datetime(future4['ds']).dt.date
if r:
    future4 = future4.merge(
        X_reservations,
        left_on='ds',
        right_on='visit_date',
        how='left')
    future4 = future4[['ds', 'reserve_visitors']]
    future4 = future4.fillna(0)

forecast4 = m4.predict(future4)
print(r2_score(list(test['y']), list(forecast4.loc[450:, 'yhat'])))
plt.plot(list(test['y']))
plt.plot(list(forecast4.loc[450:, 'yhat']))
plt.show()
fig1 = m.plot(forecast4)
plt.show()
fig2 = m.plot_components(forecast4)
plt.show()
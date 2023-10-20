import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

data = pd.read_csv('./курс.csv').assign(Дата=lambda x: pd.to_datetime(x['Дата'], format='%m/%d/%Y')) \
    .set_index('Дата').sort_index()

# Вычисление разницы между текущим и предыдущим значением курса
data['Изменение доллара'] = data['Курс доллара в рублях'] - data['Курс доллара в рублях'].shift(1)
data['Изменение евро'] = data['Курс евро в рублях'] - data['Курс евро в рублях'].shift(1)

# Вычисление скользящей средней для курса доллара и евро
window_size = 30  # Размер окна скользящей средней
data['MA доллара'] = data['Курс доллара в рублях'].rolling(window=window_size).mean()
data['MA евро'] = data['Курс евро в рублях'].rolling(window=window_size).mean()

# Вычисление скользящего коэффициента вариации для курса доллара и евро
data['STD доллара'] = data['Курс доллара в рублях'].rolling(window=window_size).std()
data['STD евро'] = data['Курс евро в рублях'].rolling(window=window_size).std()
data['CV доллара'] = data['STD доллара'] / data['Курс доллара в рублях'].rolling(window=window_size).mean() * 100
data['CV евро'] = data['STD евро'] / data['Курс евро в рублях'].rolling(window=window_size).mean() * 100

# Коэффициенты вариации для доллара и евро
cv_dollar, cv_euro = data['CV доллара'].iloc[-1], data['CV евро'].iloc[-1]

# Рассчет среднего курса доллара и евро
mean_dollar, mean_euro = data[['Курс доллара в рублях', 'Курс евро в рублях']].mean()

# Максимальный и минимальный Курс доллара в рублях и евро
max_dollar, min_dollar = data['Курс доллара в рублях'].max(), data['Курс доллара в рублях'].min()
max_euro, min_euro = data['Курс евро в рублях'].max(), data['Курс евро в рублях'].min()

# автокрреляция
autocorr_dollar, autocorr_euro = data['Курс доллара в рублях'].autocorr(), data['Курс евро в рублях'].autocorr()

# Вычисление коэффициента корреляции между курсом доллара и евро
correlation = data['Курс доллара в рублях'].corr(data['Курс евро в рублях'])

# прогнозирование
forecast_dollar = sm.tsa.ARIMA(data['Курс доллара в рублях'], order=(1, 1, 1)).fit().forecast(steps=10)
forecast_euro = sm.tsa.ARIMA(data['Курс евро в рублях'], order=(1, 1, 1)).fit().forecast(steps=10)

# Тест на стационарность курса
adf_test_dollar = sm.tsa.stattools.adfuller(data['Курс доллара в рублях'])
adf_test_euro = sm.tsa.stattools.adfuller(data['Курс евро в рублях'])

print(f'Коэффициент корреляции между курсом доллара и евро: {round(correlation, 4)}\n'
      f'Средний курс доллара: {round(mean_dollar, 4)}\n'
      f'Средний курс евро: {round(mean_euro, 4)}\n'
      f'Максимальный курс доллара: {max_dollar}\n'
      f'Минимальный курс доллара: {min_dollar}\n'
      f'Максимальный курс евро: {max_euro}\n'
      f'Минимальный курс евро: {min_euro}\n'
      f'Автокорреляция курса доллара: {round(autocorr_dollar, 4)}\n'
      f'Автокорреляция курса евро: {round(autocorr_euro, 4)}\n'
      f'Коэффициент вариации для доллара: {cv_dollar * 100:.4f}%\n'
      f'Коэффициент вариации для евро: {cv_euro * 100:.4f}%\n'
      f'Прогноз курса доллара на следующие 10 дней: \n{forecast_dollar}\n'
      f'Прогноз курса евро на следующие 10 дней: \n{forecast_euro}\n'
      f'ADF статистика для курса доллара: {round(adf_test_dollar[0], 4)}\n'
      f'p-value для курса доллара: {round(adf_test_dollar[1], 4)}\n'
      f'ADF статистика для курса евро: {round(adf_test_euro[0], 4)}\n'
      f'p-value для курса евро: {round(adf_test_euro[1], 4)}')

data_columns = ['Курс доллара в рублях', 'Курс евро в рублях', 'Изменение доллара', 'Изменение евро']
titles = ['Изменение курса доллара в рублях', 'Изменение курса евро в рублях', 'Изменение курса доллара (разница)', 'Изменение курса евро (разница)']
plots = [
    ('Курс доллара в рублях', 'MA доллара', 'Курс доллара', 'Скользящая средняя (доллар)', 'Курс доллара в рублях', 'Изменение курса доллара в рублях и скользящая средняя'),
    ('Курс евро в рублях', 'MA евро', 'Курс евро', 'Скользящая средняя (евро)', 'Курс евро в рублях', 'Изменение курса евро в рублях и скользящая средняя'),
    ('CV доллара', None, None, None, 'Скользящий коэффициент вариации (доллар)', 'Скользящий коэффициент вариации для курса доллара'),
    ('CV евро', None, None, None, 'Скользящий коэффициент вариации (евро)', 'Скользящий коэффициент вариации для курса евро')
]

for column, title in zip(data_columns, titles):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data[column])
    plt.xlabel('Дата')
    plt.ylabel(column)
    plt.title(title)
    plt.grid(True)
    plt.show()

for plot in plots:
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data[plot[0]], label=plot[2])
    if plot[1]:
        plt.plot(data.index, data[plot[1]], label=plot[3])
    plt.xlabel('Дата')
    plt.ylabel(plot[4])
    plt.title(plot[5])
    plt.legend()
    plt.grid(True)
    plt.show()

decompose_and_plot = lambda data: (sm.tsa.seasonal_decompose(data, model='additive').plot(), plt.show())
acf_pacf_and_plot = lambda data, title: (plot_acf(data), plot_pacf(data), plt.title(title), plt.show())

decompose_and_plot(data['Курс доллара в рублях'])
decompose_and_plot(data['Курс евро в рублях'])
acf_pacf_and_plot(data['Курс доллара в рублях'], 'Частичная автокорреляция курса доллара')
acf_pacf_and_plot(data['Курс евро в рублях'], 'Частичная автокорреляция курса евро')

# тренды
def plot_trends(currency):
    window_sizes = [30, 60, 90, 120]
    fig, axes = plt.subplots(len(window_sizes), 1, figsize=(8, len(window_sizes) * 2))

    for i, window_size in enumerate(window_sizes):
        subset = data[currency].iloc[-window_size:]
        trend = sm.tsa.seasonal_decompose(subset, model='additive').trend
        axes[i].plot(trend)
        axes[i].set_title(f'Тренд ({window_size} дней)')
        axes[i].set_ylim(trend.min(), trend.max())

    plt.xlabel('Дата')
    plt.tight_layout()
    plt.show()

plot_trends('Курс доллара в рублях')
plot_trends('Курс евро в рублях')

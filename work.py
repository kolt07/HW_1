import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Завантаження файлу
file_path = "Oschadbank (USD).xls"
df = pd.read_excel(file_path, engine="xlrd")

# Конвертація дати
df["Дата"] = pd.to_datetime(df["Дата"], dayfirst=True)
df = df.sort_values("Дата")

# Використовуємо курс продажу як основний показник
rates = df[["Дата", "Продаж"]]

# 1. Генерація випадкової величини (хі-квадрат розподіл)
n = len(rates)
random_values = np.random.chisquare(10, n)  # хі квадрат закон розподілу ВВ з вибіркою єб'ємом iter та параметрами: dsig
random_values = random_values - np.mean(random_values)  # Центрування

# 2. Побудова квадратичного тренду
x = (df["Дата"] - df["Дата"].min()).dt.days # Нормалізуємо данні для побудови

coeffs = np.polyfit(x, rates["Продаж"], 2)
trend = np.polyval(coeffs, x)
trend_equation = f"y = {coeffs[0]:.5f}x^2 + {coeffs[1]:.5f}x + {coeffs[2]:.5f}"

# 3. Адитивна модель (тренд + стохастична складова)
synthetic_data = trend + random_values

# 4. Статистичні характеристики
variance = np.var(random_values)
std_dev = np.std(random_values)
mS = np.median(random_values)
mean_value = np.mean(random_values)

print('-------- статистичні характеристики ХІ КВАДРАТ закону розподілу ВВ ---------')
print(f"математичне сподівання (медіана): {mS}")
print(f"Математичне очікування (середня): {mean_value}")
print(f"Дисперсія: {variance}")
print(f"Середньоквадратичне відхилення: {std_dev}")
print(f"Рівняння квадратичного тренду: {trend_equation}")
print('----------------------------------------------------------------------------')


# 5. Візуалізація
df_plot = rates.copy()
df_plot["Тренд"] = trend
df_plot["Синтетичні дані"] = synthetic_data

plt.figure(figsize=(12, 6))
plt.plot(df_plot["Дата"], df_plot["Продаж"], label="Реальні дані")
plt.plot(df_plot["Дата"], df_plot["Тренд"], label="Квадратичний тренд")
plt.plot(df_plot["Дата"], df_plot["Синтетичні дані"], label="Адитивна модель", linestyle="--")
plt.legend()
plt.xlabel("Дата")
plt.ylabel("Курс продажу")
plt.title("Моделювання змін курсу валют")
plt.grid()
plt.show()

# Гістограма випадкової складової
plt.figure(figsize=(8, 4))
plt.hist(random_values, bins=20, density=True, alpha=0.6, color='g')
plt.title("Гістограма випадкової складової (Хі-квадрат розподіл)")
plt.xlabel("Значення")
plt.ylabel("Щільність")
plt.grid()
plt.show()


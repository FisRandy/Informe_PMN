import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

# Definir función de ajuste log-log
def ajuste_minimos_cuadrados(x, y):
    log_x = np.log(x)
    log_y = np.log(y)
    n = len(x)

    sum_x = np.sum(log_x)
    sum_y = np.sum(log_y)
    sum_xy = np.sum(log_x * log_y)
    sum_x2 = np.sum(log_x ** 2)

    b = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
    a_log = (sum_y - b * sum_x) / n
    a = np.exp(a_log)  # Deslogaritmamos a

    return a, b

# Datos del tiempo
tiempo_medido = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.94, 0.9, 0.9, 1.04, 1.12, 0.94, 0.97, 0.94, 1.06, 0.93],
    [1.2, 1.09, 1.31, 1.25, 1.16, 1.22, 1.38, 1.31, 1.25, 1.25],
    [1.72, 1.59, 1.81, 1.62, 1.6, 1.65, 1.72, 1.66, 1.62, 1.65],
    [1.82, 1.97, 1.9, 1.82, 1.97, 2.03, 2, 1.95, 1.85, 1.96],
    [2, 2.28, 2.06, 2.13, 2.28, 2.06, 2.31, 2.06, 2, 2.13],
    [2.31, 2.37, 2.34, 2.38, 2.41, 2.34, 2.38, 2.31, 2.34, 2.41],
    [2.47, 2.46, 2.5, 2.38, 2.56, 2.53, 2.47, 2.53, 2.53, 2.44],
    [2.5, 2.56, 2.32, 2.62, 2.6, 2.56, 2.63, 2.66, 2.62, 2.56],
    [2.84, 2.84, 2.97, 2.93, 2.94, 2.88, 2.97, 2.88, 3, 3]
]

posicion = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])  # cm
tiempo_promedio = np.mean(tiempo_medido, axis=1)
std_tiempo = np.std(tiempo_medido, axis=1, ddof=1)

# Evitamos el cero (no puede hacerse log(0))
posicion_no_cero = posicion[1:]
tiempo_no_cero = tiempo_promedio[1:]

# Ajuste por mínimos cuadrados en escala log-log
a_tm, b_tm = ajuste_minimos_cuadrados(tiempo_no_cero, posicion_no_cero)

# Crear la gráfica de puntos
plt.figure(figsize=(8, 5))
plt.scatter(tiempo_promedio, posicion, color='red', label="Datos experimentales", s=20)

# Crear valores de tiempo desde muy cerca de 0
t_vals = np.linspace(0.001, max(tiempo_no_cero), 400)
x_fit = a_tm * t_vals ** b_tm

# Graficar curva ajustada
plt.plot(t_vals, x_fit, color='blue', linestyle='dashed', label=f"Ajuste: x(t) = {a_tm:.2f}·t^{b_tm:.2f}")

# Etiquetas y título
plt.xlabel("Tiempo (segundos)")
plt.ylabel("Posición (cm)")
plt.title("Ajuste por mínimos cuadrados: MRUV (potencial)")
plt.grid(True)
plt.legend()
plt.xlim(0, 3.5)
plt.ylim(0, 100)

plt.savefig('mi_grafica.png')
plt.show()



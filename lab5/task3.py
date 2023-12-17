import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl

# Завантаження вхідних даних
text = np.loadtxt('data_perceptron.txt')

# Поділ точок даних та міток
data = text[:, : 2]
labels = text[:, 2]. reshape((text. shape[0],1))

plt. figure()
plt.scatter(data[:, 0], data[:, 1])
plt. xlabel('Размерность 1 ')
plt.ylabel('Размерность 2')
plt. title('Входные данные')

# Визначення максимального та мінімального значень для кожного виміру
dim1_min,dim1_max,dim2_min,dim2_max = 0,1,0,1

# Кількість нейронів у вихідному шарі
num_output = labels.shape[1]

# Визначення перцептрону з двома вхідними нейронами (оскільки Вхідні дані - двовимірні)
dim1 = [dim1_min,dim1_max]
dim2 = [dim2_min,dim2_max]
perceptron = nl.net.newp([dim1,dim2],num_output)

# Тренування перцептрону з використанням наших даних
error_progress = perceptron.train(data, labels, epochs=100, show=20, lr=0.03)

# Побудова графіка процесу навчання
plt.figure()
plt.plot(error_progress)
plt.xlabel('Количество эпох')
plt.ylabel('Ошибка обучения')
plt.title('Изменение ошибки обучения')
plt.grid()
plt.show()
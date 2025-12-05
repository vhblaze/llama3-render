import numpy as np, random

#esse cara faz as medições entre os pesos e bias de um perceptron simples possui 2 neuronios na camada de entrada e 1 na de saída, utilizando as bibliotecas numpy

X = np.array([[0,0], [0,1], [1,0], [1,1]]) # Representa os dados de entrada (exemplo: valores binários)
y = np.array([0, 0, 0, 1]) #Saída esperada — o que a rede deve aprender

w = np.random.rand(2) #Cada peso é um número que indica a “importância” de cada entrada
b = np.random.rand(1) #Ajusta o deslocamento da função, permitindo que o modelo generalize
lr = 0.1 # Decide se o neurônio “dispara” (1) ou não (0)

def step(z):
    return 1 if z >= 0.5 else 0

for epoch in range(100): #Cada repetição é uma tentativa de reduzir o erro e ajustar o modelo
    for xi, target in zip(X, y):
        z = np.dot(w, xi) + b #Soma ponderada → o cálculo central do neurônio
        output = step(z)
        error = target - output
        w += lr * error * xi # Ajuste dos pesos com base no erro — isso é aprendizado supervisionado
        b += lr * error  #Ajuste das Bias com base no erro

print("Pesos finais:", w, "Bias:", b)

#abaixo versão sem numpy feito na mão no seco, precisei usar o gpt para ajudar a criar

# Entradas (X) e saídas desejadas (y) para o treinamento do perceptron
X = [[0,0], [0,1], [1,0], [1,1]]
y = [0, 0, 0, 1]

w = [random.random(), random.random()]
b = random.random()

# Taxa de aprendizado
lr = 0.1

# Função de ativação degrau
def step(z):
    return 1 if z >= 0.5 else 0

# Função para calcular o produto escalar (dot product)
def dot(a, b):
    s = 0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s

# Treinamento
for epoch in range(100):
    for i in range(len(X)):
        xi = X[i]
        target = y[i]

        # Soma ponderada: z = w1*x1 + w2*x2 + b
        z = dot(w, xi) + b

        # Saída após ativação
        output = step(z)

        # Erro
        error = target - output

        # Atualização dos pesos e bias
        for j in range(len(w)):
            w[j] += lr * error * xi[j]
        b += lr * error

print("Pesos finais:", w)
print("Bias final:", b)
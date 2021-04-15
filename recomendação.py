from rbm import RBM

import numpy as np

rbm = RBM(num_visible=6, num_hidden=2)

base = np.array([[1, 1, 1, 0, 0, 0], 
                 [1, 0, 1, 0, 0, 0],
                [1, 1, 0, 1, 0, 1]])


rbm.train(base, max_epochs=5000)

rbm.weights

import pandas as pd
pd.DataFrame(rbm.weights)

df = pd.DataFrame(base)

usuario1 = np.array([[1, 1, 0, 1, 0, 0]])
rbm.run_visible(usuario1)

user2 = np.array([[0, 0, 0, 1, 1, 0]])
rbm.run_visible(user2)

camada = np.array([[0, 1]])
recomen = rbm.run_hidden(camada)

print(recomen)

filmes = ["A bruxa", "Invocação do Mal", "O chamado", "Se beber não case", "Gente Grande", "American Pie"]

for i in range(len(user2[0])):
    #print(usuario1[0, i])
    if user2[0, i] == 0 and recomen[0, i] == 1:
        print(filmes[i])
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import BernoulliRBM
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

base = datasets.load_digits()
import pandas as pd
previsores = np.asarray(base.data, 'float32')
classe = base.target


norm = MinMaxScaler(feature_range=(0, 1))

previsores = norm.fit_transform(previsores)

pd.DataFrame(previsores).head(5)

x_train, x_test, y_train, y_test = train_test_split(previsores, classe, test_size=0.2, random_state=0)

rbm = BernoulliRBM(random_state=0)
rbm.n_iter = 25
rbm.n_components = 50
naive_rbm = GaussianNB()
classificador_rbm = Pipeline(steps=[('rbm', rbm), ('naive', naive_rbm)])
classificador_rbm.fit(x_train, y_train)

plt.figure(figsize=(20, 20))
for i, comp in enumerate(rbm.components_):
    plt.subplot(10, 10, i+1)
    plt.imshow(comp.reshape((8, 8)), cmap=plt.cm.gray_r)
    plt.xticks(())
    plt.yticks(())
plt.show()
previsores_rbm = classificador_rbm.predict(x_test)
from sklearn.metrics import accuracy_score
print("{:.2f}%" .format(accuracy_score(previsores_rbm, y_test)*100))
naive_simples = GaussianNB()
naive_simples.fit(x_train, y_train)
previsoes = naive_simples.predict(x_test)
print("{:.2f}%" .format(accuracy_score(previsoes, y_test)*100))

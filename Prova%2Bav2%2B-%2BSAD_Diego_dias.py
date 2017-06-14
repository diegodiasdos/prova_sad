
# coding: utf-8

# DISCIPLINA: Sistemas de Apoio à Decisão 	PROFESSOR(A): Alex Salgado 
#  
# PERÍODO: 8o. 				TURNO: manhã 		AVALIAÇÃO: 
#  
# ALUNO(A): __Diego Dias Dos Santos Maartins _______________________________________________________
#  
# GRAU: 							VISTO DO PROFESSOR:
#  
# Preencha sua resposta no próprio arquivo do Jupyter e depois me envie no link abaixo:
# resposta: https://goo.gl/forms/aOml4FAMaTmB4u482
# 

# ## Questão 1 - valor (0,5)
# 1.1 - Importar os modulos python para machine learn e carregar o arquivo cargas.xlsx usando o método read_excel do pandas

# In[62]:

get_ipython().magic('matplotlib notebook')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
carga = pd.read_excel('cargas.xlsx')
print(carga)


# 1.2 - Exibir os primeiros registros desta tabela

# In[64]:

carga.head()


# ## Questão 2 - valor (0,5)
# Utilizando as terminologias de Machine Learning(observações e features):  
# 2.1 - Quantas observações têm nessa base de dados?

# In[66]:

carga.count()


# 2.2 - Quantas "features" têm nessa base de dados?

# In[67]:


"5 features"


# 2.2 - Quantas "features" têm nessa base de dados

# In[68]:


"5 features"


# ## Questão 3 - valor (1,0)
# 3.1 - Usando o algoritmo de KNN (com 5 vizinhos, k=5), qual a previsão? Adivinhe qual é carga com peso 327g, largura 14 cm, altura 8 cm, ou seja, com as seguintes features (peso = 327, largura=14, altura=8).
# 

# In[69]:


X = carga[['peso(gramas)','largura','altura']]
y = carga[['nome_carga']]

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X,y.values.ravel())
knn.predict([[327,14,8]])




# ## Questão 4 - valor (2,0)
# Usando o método de avaliação de acurácia (Treinar e testar SEPARADAMENTE-Train/test split), qual dos 3 métodos abaixo é mais eficiente?  
# 4.1 - Acurácia usando o algoritmo de KNN (com 1 vizinho, k=1)
# 

# In[70]:


from sklearn.cross_validation import train_test_split
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size=0.4, random_state=4)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))


# 4.2 - Acurácia usando o algoritmo de KNN (com 5 vizinho, k=5)

# In[71]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))


# 4.3 - Acurácia usando o algoritmo de LogisticRegression

# In[72]:

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))


# In[ ]:




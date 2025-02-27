from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix

#ATRIBUTOS
x = [
    [0,0],
    [0,1],
    [1,0],
    [1,1]
]

#CLASSES
y_true = [0,1,1,0]

#TREINANDO O MODELO DE ML
model = LinearSVC()

model.fit(x, y_true)

#FRAZENDO PREDIÇÕES
y_pred = model.predict(x)


print(y_pred)

accuracy = accuracy_score(y_true, y_pred)
print("ACURACIA DO TESTE DO MODELO: {}".format(accuracy * 100))

confusion = confusion_matrix(y_true
, y_pred)
print("MATRIZ DE CONFUSAO DOS TESTES:")
print(confusion)
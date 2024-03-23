import numpy as np
import time
import pandas as pd
from sklearn.svm import SVC # Jada SVM inplementatutako pythoneko pakete bat
import pickle # Modeloak gordetzeko



# -------------------------------------------------------------------------------
# -----------------------------DATU BASEA INPORTATU------------------------------
# -------------------------------------------------------------------------------
entrenamendu_datuak = pd.read_csv("mnist_train.csv")
testeatzeko_datuak = pd.read_csv("mnist_test.csv")
# print(entrenamendu_datuak.head())
X_entrenamendu = entrenamendu_datuak.iloc[:,1:]
Y_entrenamendu = entrenamendu_datuak["label"]

X_test = testeatzeko_datuak.iloc[:,1:]
Y_test = testeatzeko_datuak["label"]




# -------------------------------------------------------------------------------
# -------------------------------MODELOA ENTRENATU-------------------------------
# -------------------------------------------------------------------------------

modeloa = SVC(C = 4)

hasierako_denbora = time.time()
modeloa.fit(X_entrenamendu.values,Y_entrenamendu)
amaierako_denbora = time.time()

print(f"Entrenatzeko beharrezko denbora: {amaierako_denbora-hasierako_denbora}")
# print(f"Entrenamendu errorea: {modeloa.score(X_entrenamendu,Y_entrenamendu)}")
# print(f"Testak erabiliz lortutako errorea: {modeloa.score(X_test,Y_test)}")

pickle.dump(modeloa,open("SkLearn_SVC_model_C_4.pkl","wb"))

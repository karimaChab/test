
  #### l'lgorithme principal est décomposé en 4 étepes 

# configuration de base et initialisation des poids et biais 
# Mappez les étiquettes de classe de {0, 1} à {-1, 1}
#effectuez une descents de gradient pour n itérations, ce qui implique le calcul des  gradients et la mise à jour des poids et biais en conséquence
# faire de la predécition finale

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SVM:
    # configuration de base et initialisation des poids et biais 
    def __init__(self, learning_rate=1e-3, lambda_param=1e-2, nbr_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = nbr_iters
        self.w = None
        self.b = None

    def _init_weights_bias(self, X):
        n_features = X.shape[1]
        self.w = np.zeros(n_features)
        self.b = 0
  #  Mappez les étiquettes de classe de {0, 1} à {-1, 1}
    def _get_cls_map(self, y):
        return np.where(y <= 0, -1, 1)
  # vérifier si la contrainte de l'hyperplande séparation est satisfaite
    def _satisfy_constraint(self, x, idx):
        linear_model = np.dot(x, self.w) + self.b 
        return self.cls_map[idx] * linear_model >= 1
    
    def _get_gradients(self, constrain, x, idx):
          # si le point de données se trouve du bon coté 
        if constrain:
            dw = self.lambda_param * self.w
            db = 0
            return dw, db
          # si le point de données est du mauvais coté 
        dw = self.lambda_param * self.w - np.dot(self.cls_map[idx], x)
        db = - self.cls_map[idx]
        return dw, db
    
      
       # mise à jour les paramétres en conséqunence
    def _update_weights_bias(self, dw, db):
        self.w -= self.lr * dw
        self.b -= self.lr * db
    
    def fit(self, X, y):
        #init le poids et le biais 
        self._init_weights_bias(X)
         # mapper la classe binaire sur {-1,1 }
        self.cls_map = self._get_cls_map(y)

        for _ in range(self.n_iters):
            for idx, x in enumerate(X):
              # vérifier si le point de données satisfait la contrainte
                constrain = self._satisfy_constraint(x, idx)
              # claculer le gradients en conséquence
                dw, db = self._get_gradients(constrain, x, idx)
             # mettre à jour les poids et les biais
                self._update_weights_bias(dw, db)
    
    def predict(self, X):
        estimate = np.dot(X, self.w) + self.b
          # calculer le signe des étiquettes de classe 
        prediction = np.sign(estimate)
        # mapper la classe de {-1 , 1} aux valeurs d'origine {0,1 } avant de renvoyer l'étiqutte 
        return np.where(prediction == -1, 0, 1)


    



# Déclaration de fonctions
# ########################################################
     
def convert_data(df) ->list:
    r = []
    for index, line in df.iterrows():
        cont = [
            float(line ['age']),
            float(line ['sex']),
            float(line ['cp']),
            float(line ['trestbps']),
            float(line ['chol']),
            float(line ['fbs']),
            float(line ['restecg']),
            float(line ['thalach']),
            float(line ['exang']),
            float(line ['oldpeak']),
            float(line ['slope']),
            float(line ['ca']),
            float(line ['thal']),
            float(line ['target']),
            
        ]
        r.append(cont)
        
    return r


def accuracy(predictions, trues) -> float:
    total = 0
    for i in range(0, len(predictions)):
        if predictions[i] == trues[i]:
            total += 1
    return total / len(predictions)



    
# Fonction Main
# ########################################################

       
def main(filename: str = 'dataset.csv'):
     # Chargement des données
      df = pd.read_csv(filename)
      print("L'entête du dataframe: \n\n", df.head())


     # Convertir les données
      dataset = convert_data(df)
      dataset = np.array(dataset)
      
      # Mélanger les données
      np.random.shuffle(dataset)
      
     # Extraction des données
      x = dataset[:,0:13]
      y = dataset[:,13]

     # Normalisation des données
      mean = x.mean(axis = 0)
      std  = x.std(axis = 0)
      x = (x - mean) / std
      
      
     # division des données
     
      div_index = int(0.70 * len(x))
      X_train = x[0:div_index]
      y_train = y[0:div_index]
      X_test = x[0:div_index]
      y_test = y[0:div_index]

      clf =SVM(nbr_iters=1000)
      clf.fit(X_train, y_train)
      predictions = clf.predict(X_test)
      acc = accuracy(predictions, y_test)
      print('Accuracy=', acc)


main()





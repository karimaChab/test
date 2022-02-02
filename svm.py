
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

kjnjdbolabclobefcbpabfcbaz
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

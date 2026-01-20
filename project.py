# python -m pip install --upgrade pip
# pip install tensorflow-2.10.0-cp39-cp39-win_amd64.whl ** tem que baixar o tensorflow-2.10.0-cp39-cp39-win_amd64.whl na web
# pip install numpy==1.24.4
# pip install scikit-learn matplotlib seaborn

#2. PREPARAR DADOS
import tensorflow as tf  
from tensorflow.keras.datasets import mnist  
from tensorflow.keras.utils import to_categorical 

# 3. CRIAR MODELO
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Dense, Dropout  

# 5. AVALIAÇÃO
import numpy as np 
from sklearn.metrics import confusion_matrix, classification_report 
import matplotlib.pyplot as plt 

# 6. VISUALIZAÇÃO
import matplotlib.pyplot as plt
import seaborn as sns 

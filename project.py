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
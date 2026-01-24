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

#2. PREPARAR DADOS

(dados_treino, resultados_treino), (dados_teste, resultados_teste) = mnist.load_data()  

dados_treino = dados_treino.astype('float32') / 255  
dados_teste = dados_teste.astype('float32') / 255  

dados_treino = dados_treino.reshape(-1, 28*28)  
dados_teste = dados_teste.reshape(-1, 28*28)  

resultados_treino = to_categorical(resultados_treino, 10)  
resultados_teste = to_categorical(resultados_teste, 10)  

print("Dados preparados!") 

# 3. CRIAR MODELO

modelo = Sequential([
    tf.keras.Input(shape=(784,)),
    Dense(512, activation='relu'),
    Dropout(0.2),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

# Compilar modelo
modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("Modelo ANN construído e compilado!")

# 4. TREINAR MODELO

treinando = modelo.fit(dados_treino, resultados_treino, 
                    epochs=20, 
                    batch_size=128, 
                    validation_split=0.2) 

modelo.save("modelo_mnist.h5")

# 5. AVALIAÇÃO

# Avaliar no conjunto de teste: nota geral
teste_loss, teste_acc = modelo.evaluate(dados_teste, resultados_teste)
print(f"Acurácia no teste: {teste_acc*100:.2f}%")
resposta_do_modelo_dist = modelo.predict(dados_teste) 
resposta_do_modelo_1 = np.argmax(resposta_do_modelo_dist, axis=1)  
respostas_certas = np.argmax(resultados_teste, axis=1)  

# Matriz de confusão em formato numérico
matriz_confusao = confusion_matrix(respostas_certas, resposta_do_modelo_1)
print("Matriz de Confusão:")
print(matriz_confusao)

# Relatório de classificação
print("Relatório de Classificação:")
print(classification_report(respostas_certas, resposta_do_modelo_1))

# 5. VISUALIZAÇÃO

# Matriz de confusão colorida
plt.figure(figsize=(8,6))
sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()

# Curva de perda (loss)
plt.plot(treinando.history['loss'], label='Treino')           
plt.plot(treinando.history['val_loss'], label='Validação')    
plt.title('Curva de Perda')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend() 
plt.show()   

# Curva de acurácia (accuracy)
plt.plot(treinando.history['accuracy'], label='Treino')        
plt.plot(treinando.history['val_accuracy'], label='Validação') 
plt.title('Curva de Acurácia')
plt.xlabel('Época')
plt.ylabel('Accuracy')
plt.legend() 
plt.show()   

# casos aleatórios
fig, axes = plt.subplots(2, 5, figsize=(12,5)) 
for i, ax in enumerate(axes.flat):              
    ax.imshow(dados_teste[i].reshape(28,28), cmap='gray')  
    ax.set_title(f"Prev: {resposta_do_modelo_1[i]}\nReal: {respostas_certas[i]}")  
    ax.axis('off')  
plt.show()  

# Encontrar índices onde o modelo errou
indices_errados = np.where(resposta_do_modelo_1 != respostas_certas)[0]
print(f"Total de erros do modelo: {len(indices_errados)}")
np.random.seed(42) 
indices_escolhidos = np.random.choice(indices_errados, 8, replace=False)

# casos errados
fig, axes = plt.subplots(2, 4, figsize=(12,6))

for ax, idx in zip(axes.flat, indices_escolhidos):
    ax.imshow(dados_teste[idx].reshape(28,28), cmap='gray')
    ax.set_title(
        f"Previsto: {resposta_do_modelo_1[idx]}\nReal: {respostas_certas[idx]}"
    )
    ax.axis('off')

plt.suptitle("Exemplos de Classificações Erradas do Modelo", fontsize=14)
plt.tight_layout()
plt.show()



import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw, ImageTk, ImageEnhance, ImageOps, ImageFilter
import numpy as np
import tensorflow as tf

# CARREGAR MODELO 
try:
    modelo = tf.keras.models.load_model("modelo_mnist.h5")
    print("✓ Modelo carregado com sucesso!")
except Exception as e:
    print(f"✗ Erro ao carregar modelo: {e}")
    modelo = None

# ==================== PRÉ-PROCESSAMENTO MELHORADO ====================
def preprocess_imagem(img):
    """
    Pré-processamento OTIMIZADO para reconhecimento robusto
    """
    if img.mode != 'L':
        img = img.convert('L')
    
    # 1. DETECTAR E INVERTER SE NECESSÁRIO
    img_array_test = np.array(img, dtype='float32') / 255.0
    media_pixels = np.mean(img_array_test)
    
    if media_pixels > 0.5:
        print("[DEBUG] Fundo CLARO detectado - invertendo")
        img = ImageOps.invert(img)
    else:
        print("[DEBUG] ✓ Fundo ESCURO - mantendo original")
    
    # 2. MELHORAR CONTRASTE
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.5)  # Aumentado de 2.0 para 2.5
    
    # 3. REMOVER RUÍDO com filtro gaussiano suave
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    # 4. CENTRALIZAR O DÍGITO (critical for MNIST)
    img_array = np.array(img, dtype='float32')
    
    # Encontrar bounding box do dígito
    threshold = np.max(img_array) * 0.3
    coords = np.argwhere(img_array > threshold)
    
    if len(coords) > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # Adicionar margem de 10%
        altura = y_max - y_min
        largura = x_max - x_min
        margem_y = int(altura * 0.1)
        margem_x = int(largura * 0.1)
        
        y_min = max(0, y_min - margem_y)
        y_max = min(img_array.shape[0], y_max + margem_y)
        x_min = max(0, x_min - margem_x)
        x_max = min(img_array.shape[1], x_max + margem_x)
        
        # Recortar
        img_cropped = img.crop((x_min, y_min, x_max, y_max))
        
        # Redimensionar mantendo proporção
        img_cropped.thumbnail((20, 20), Image.Resampling.LANCZOS)
        
        # Colocar em canvas 28x28 centralizado
        img_final = Image.new('L', (28, 28), color=0)
        offset_x = (28 - img_cropped.width) // 2
        offset_y = (28 - img_cropped.height) // 2
        img_final.paste(img_cropped, (offset_x, offset_y))
        
        print(f"[DEBUG] ✓ Dígito centralizado: crop=({x_min},{y_min},{x_max},{y_max})")
    else:
        # Fallback: apenas redimensionar
        img_final = img.resize((28, 28), Image.Resampling.LANCZOS)
        print("[DEBUG] ⚠ Nenhum dígito detectado - usando redimensionamento simples")
    
    # 5. NORMALIZAR E BINARIZAR
    img_array = np.array(img_final, dtype='float32') / 255.0
    img_array = np.where(img_array > 0.4, 1.0, 0.0)  # Threshold reduzido de 0.5 para 0.4
    
    print(f"[DEBUG] Shape: {img_array.shape}, Pixels ativos: {(img_array > 0.5).sum()}")
    
    # 6. RESHAPE PARA O MODELO
    img_array = img_array.reshape(1, 784)
    
    return img_array

# ==================== FUNÇÕES DA INTERFACE ====================
def reconhecer():
    """Reconhece o dígito desenhado ou carregado"""
    global imagem
    
    print("\n" + "="*50)
    print("RECONHECENDO DÍGITO...")
    print("="*50)
    
    if modelo is None:
        messagebox.showerror("Erro", "Modelo não carregado!")
        return
    
    try:
        img_preparada = preprocess_imagem(imagem)
        predicao = modelo.predict(img_preparada, verbose=0)
        
        classe = np.argmax(predicao[0])
        confianca = predicao[0][classe] * 100
        
        print(f"✓ RESULTADO: Dígito {classe} ({confianca:.1f}%)")
        
        # Top 3 previsões
        top3_idx = np.argsort(predicao[0])[::-1][:3]
        
        # Atualizar resultado com animação visual
        resultado_numero.config(text=str(classe))
        resultado_confianca.config(text=f"{confianca:.1f}% confiança")
        
        # Mostrar top 3
        top3_text = ""
        for i, idx in enumerate(top3_idx):
            top3_text += f"{' ' if i==0 else ' ' if i==1 else ' '} {idx}: {predicao[0][idx]*100:.1f}%\n"
        
        resultado_top3.config(text=top3_text)
        
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"✗ ERRO: {e}")
        import traceback
        traceback.print_exc()
        messagebox.showerror("Erro", f"Erro na predição:\n{str(e)}")

def carregar_imagem():
    """Carrega uma imagem de arquivo"""
    arquivo = filedialog.askopenfilename(
        title="Selecione uma imagem", 
        filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp *.gif"), ("Todos", "*.*")]
    )
    if arquivo:
        try:
            img_carregada = Image.open(arquivo).convert('L')
            img_carregada.thumbnail((320, 320), Image.Resampling.LANCZOS)
            
            nova_img = Image.new('L', (320, 320), color=0)
            offset_x = (320 - img_carregada.width) // 2
            offset_y = (320 - img_carregada.height) // 2
            nova_img.paste(img_carregada, (offset_x, offset_y))
            
            global imagem, draw
            imagem = nova_img
            draw = ImageDraw.Draw(imagem)
            
            canvas.delete("all")
            img_tk = ImageTk.PhotoImage(imagem)
            canvas.create_image(160, 160, image=img_tk)
            canvas.image = img_tk
            
            resultado_numero.config(text="?")
            resultado_confianca.config(text="Clique em Reconhecer")
            resultado_top3.config(text="")
            
            print("✓ Imagem carregada")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Não foi possível carregar:\n{str(e)}")

def desenhar(event):
    """Desenha no canvas com pincel branco"""
    x, y = event.x, event.y
    r = 15
    
    canvas.create_oval(x-r, y-r, x+r, y+r, fill="#FFFFFF", outline="")
    draw.ellipse([x-r, y-r, x+r, y+r], fill=255)

def limpar():
    """Limpa o canvas"""
    global imagem, draw
    canvas.delete("all")
    imagem = Image.new("L", (320, 320), color=0)
    draw = ImageDraw.Draw(imagem)
    
    resultado_numero.config(text="?")
    resultado_confianca.config(text="Desenhe um número")
    resultado_top3.config(text="")
    
    print("🧹 Canvas limpo")

# ==================== INTERFACE TKINTER ====================
janela = tk.Tk()
janela.title("Reconhecimento de Dígitos MNIST")
janela.geometry("1000x650")
janela.configure(bg="#0f0f23")
janela.resizable(False, False)

# Estilos customizados
COR_BG = "#0f0f23"
COR_CARD = "#1a1a2e"
COR_ACCENT = "#16213e"
COR_DESTAQUE = "#0f4c75"
COR_TEXTO = "#e4e4e4"
COR_TEXTO_SEC = "#94a1b2"
COR_GRADIENT_1 = "#667eea"
COR_GRADIENT_2 = "#764ba2"

# Container principal
container = tk.Frame(janela, bg=COR_BG)
container.pack(fill="both", expand=True, padx=20, pady=20)

#  HEADER 
header = tk.Frame(container, bg=COR_BG)
header.pack(fill="x", pady=(0, 15))

titulo = tk.Label(header, 
                 text="RECONHECIMENTO DE DÍGITOS", 
                 font=("Helvetica Neue", 24, "bold"), 
                 fg="#667eea", 
                 bg=COR_BG)
titulo.pack()

subtitulo = tk.Label(header, 
                    text="Inteligência Artificial treinada com 60.000 imagens do dataset MNIST", 
                    font=("Helvetica Neue", 9), 
                    fg=COR_TEXTO_SEC, 
                    bg=COR_BG)
subtitulo.pack(pady=(3, 0))

separador = tk.Frame(header, bg="#667eea", height=2)
separador.pack(pady=(10, 0), fill="x")

#CONTEÚDO PRINCIPAL 
content_frame = tk.Frame(container, bg=COR_BG)
content_frame.pack(fill="both", expand=True)

# COLUNA ESQUERDA - Canvas de Desenho
left_panel = tk.Frame(content_frame, bg=COR_CARD, relief="flat")
left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))

# Título do painel
canvas_titulo = tk.Label(left_panel, 
                        text="ÁREA DE DESENHO", 
                        font=("Helvetica Neue", 12, "bold"), 
                        fg=COR_TEXTO, 
                        bg=COR_CARD)
canvas_titulo.pack(pady=(15, 10))

# Botão carregar
btn_carregar = tk.Button(left_panel, 
                        text="Carregar Imagem", 
                        font=("Helvetica Neue", 10, "bold"), 
                        bg=COR_DESTAQUE, 
                        fg="#ffffff",
                        activebackground="#0a3a5e",
                        activeforeground="#ffffff",
                        relief="flat",
                        cursor="hand2", 
                        command=carregar_imagem,
                        padx=15,
                        pady=6)
btn_carregar.pack(pady=(0, 10))

# Canvas com borda estilizada
canvas_container = tk.Frame(left_panel, bg="#667eea", padx=3, pady=3)
canvas_container.pack(pady=8)

canvas = tk.Canvas(canvas_container, 
                  width=320, 
                  height=320, 
                  bg="#000000", 
                  highlightthickness=0,
                  cursor="pencil")
canvas.pack()

# Inicializar imagem
imagem = Image.new("L", (320, 320), color=0)
draw = ImageDraw.Draw(imagem)
canvas.bind("<B1-Motion>", desenhar)

# Dica
dica_label = tk.Label(left_panel, 
                     text="Dica: Desenhe o número GRANDE e CENTRALIZADO", 
                     font=("Helvetica Neue", 9, "italic"), 
                     fg=COR_TEXTO_SEC, 
                     bg=COR_CARD)
dica_label.pack(pady=(8, 0))

# Botões de ação
botoes_frame = tk.Frame(left_panel, bg=COR_CARD)
botoes_frame.pack(pady=15)

btn_reconhecer = tk.Button(botoes_frame, 
                          text="RECONHECER", 
                          font=("Arial", 12, "bold"), 
                          bg="#667eea", 
                          fg="#ffffff",
                          activebackground="#5568d3",
                          activeforeground="#ffffff",
                          relief="raised",
                          bd=0,
                          cursor="hand2", 
                          command=reconhecer, 
                          state="normal",
                          width=13,
                          height=2)
btn_reconhecer.pack(side="left", padx=8)

btn_limpar = tk.Button(botoes_frame, 
                      text="LIMPAR", 
                      font=("Arial", 12, "bold"), 
                      bg="#e74c3c", 
                      fg="#ffffff",
                      activebackground="#c0392b",
                      activeforeground="#ffffff",
                      relief="raised",
                      bd=0,
                      cursor="hand2", 
                      command=limpar,
                      width=13,
                      height=2)
btn_limpar.pack(side="left", padx=8)

# ===== COLUNA DIREITA - Resultados =====
right_panel = tk.Frame(content_frame, bg=COR_CARD, relief="flat")
right_panel.pack(side="right", fill="both", expand=True, padx=(10, 0))

# Título
resultado_titulo = tk.Label(right_panel, 
                            text="RESULTADO DA PREDIÇÃO", 
                            font=("Helvetica Neue", 12, "bold"), 
                            fg=COR_TEXTO, 
                            bg=COR_CARD)
resultado_titulo.pack(pady=(15, 20))

# Card do resultado principal
resultado_card = tk.Frame(right_panel, bg=COR_ACCENT, relief="flat")
resultado_card.pack(pady=15, padx=30, fill="x")

resultado_label_texto = tk.Label(resultado_card, 
                                 text="DÍGITO IDENTIFICADO", 
                                 font=("Helvetica Neue", 10), 
                                 fg=COR_TEXTO_SEC, 
                                 bg=COR_ACCENT)
resultado_label_texto.pack(pady=(15, 5))

resultado_numero = tk.Label(resultado_card, 
                           text="?", 
                           font=("Helvetica Neue", 70, "bold"), 
                           fg="#667eea", 
                           bg=COR_ACCENT)
resultado_numero.pack(pady=8)

resultado_confianca = tk.Label(resultado_card, 
                              text="Desenhe um número", 
                              font=("Helvetica Neue", 11), 
                              fg=COR_TEXTO_SEC, 
                              bg=COR_ACCENT)
resultado_confianca.pack(pady=(0, 15))

# Separador
sep2 = tk.Frame(right_panel, bg="#667eea", height=2)
sep2.pack(pady=15, fill="x", padx=30)

# Top 3 previsões
top3_titulo = tk.Label(right_panel, 
                      text="TOP 3 PREVISÕES", 
                      font=("Helvetica Neue", 11, "bold"), 
                      fg=COR_TEXTO, 
                      bg=COR_CARD)
top3_titulo.pack(pady=(10, 10))

resultado_top3 = tk.Label(right_panel, 
                         text="", 
                         font=("Helvetica Neue", 10), 
                         fg=COR_TEXTO_SEC, 
                         bg=COR_CARD,
                         justify="left")
resultado_top3.pack(pady=8)

# Footer com informação
footer = tk.Frame(right_panel, bg=COR_CARD)
footer.pack(side="bottom", pady=15)

info_label = tk.Label(footer, 
                     text="Rede Neural MLP\n512→256→10 neurônios\nAcurácia: ~98%", 
                     font=("Helvetica Neue", 8), 
                     fg=COR_TEXTO_SEC, 
                     bg=COR_CARD,
                     justify="center")
info_label.pack()

# Executar
print("\n" + "="*50)
print("Interface MNIST PREMIUM iniciada!")
janela.mainloop()
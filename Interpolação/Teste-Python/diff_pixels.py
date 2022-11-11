# Dadas duas imagens de mesmo tamanho, verifica se existem pixels divergentes entre elas
# Obs. Programa não otimizado!

from PIL import Image

# Caminho para as imagens consideradas
image1 = "image1.jpg"
image2 = "image2.jpg"

# Criação dos objetos para manipular aS imageNS
image_obj1 = Image.open(image1)
image_obj2 = Image.open(image2)

# Recuperar informações referente a dimensão das imagens (considera-se que são iguais para ambas)
in_width = image_obj1.size[0]
in_height = image_obj1.size[1]

# Lista par armazenar os pixels que são diferentes entre as imagens
diff_pixels = []

# Percorrendo (por índice) as linhas das imagens
for i in range(in_width):
    # Percorrendo (por índice) as colunas da imagem
    for j in range(in_height):

        # Verifica se os pixels em uma mesma posição (i,j) são diferentes nas imagens
        if image_obj1.getpixel((i,j)) != image_obj2.getpixel((i,j)):
            # Se sim, armazena a posição de divergência na lista
            diff_pixels.append((i,j))

print(f"Quantidade de pixels divergentes: {len(diff_pixels)}")

# Se existem pixels divergentes, solicitar ao usuário se a lista deve ser exibida
if len(diff_pixels) != 0 and bool(int(input("Exibir diff_pixels? 0 - Não / 1 - Sim: "))):
    # Se sim, exibe a lista com as posições dos pixels que são diferentes entre as imagens
    print(diff_pixels)

# Fecha os objetos que manipulam as imagens
image_obj1.close()
image_obj2.close()
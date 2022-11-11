from PIL import Image
from math import floor, ceil

# Método para interpolação linear
def linear_interpolation(x, x0, y0, x1, y1):
    u = (x-x0)/(x1-x0)
    return (1-u)*y0 + u*y1

# Método para interpolação bilinear
def bilinear_interpolation(x, y, x0, y0, x1, y1, f00, f01, f10, f11):
    r1 = linear_interpolation(x, x0, f00, x1, f10)
    r2 = linear_interpolation(x, x0, f01, x1, f11)

    return linear_interpolation(y, y0, r1, y1, r2)


# Caminho para as imagens de entrada e saída
# Obs. A imagem de saída não é salva, sendo apenas exibida ao final do programa!
in_image = "image.jpg"
out_image = "output_teste.jpg"

# Objeto para manipular a imagem de entrada
in_image_obj = Image.open(in_image)

# Obtendo dimensões da imagem para calculos futuros
in_width = in_image_obj.size[0]
in_height = in_image_obj.size[1]

# Fator pelo qual a imagem será redimensionada (será aplicado o mesmo para largura e altura)
# Obs. Tente modificar esse valor (inclusive, como exemplo para diminuir a imagem cita-se o valor 0.75)
scale = 2

# Tupla contendo as dimensões da imagem de saída, calculada com base nas dimensões da imagem inicial
# e do valor de escala. Note que devemos utilizar a transformações para o tipo inteiro - caso contrário,
# casos como scale=0.75 produzem problemas!
out_size = (int(in_width * scale), int(in_height * scale)) 

# Objeto para manipulação da imagem de saída. Informa-se o modo ("L" -> gray scale) e o tamanho
out_image_obj = Image.new(mode="L", size=out_size)

# Percorrendo as linhas da imagem de saída
for i in range(out_size[0]):
    # Percorrendo as colunas da imagem de saída
    for j in range(out_size[1]):

        # Verifica se o pixel (i,j) corresponde a um ponto já existente na imagem de entrada
        if (i%scale == 0 and j%scale == 0):
            # Se sim, basta copiar o valor daquele ponto
            value = in_image_obj.getpixel((i/scale, j/scale))

        # Se não, significa que teremos que interpolar o valor para aquele pixel na imagem de saída
        else:

            # Calculo dos 4 pixels de referência da imagem de entrada.
            
            # Considera os pixels mais à direita em caso do arredondamento de x0 = x1 
            #x0 = floor(i/scale)
            #y0 = floor(j/scale)
            #x1 = max(x0+1, ceil(i/scale))
            #y1 = max(y0+1, ceil(j/scale))

            # Considera os pixels mais à direita em caso do arredondamento de x0 = x1 
            x1 = ceil(i/scale)
            y1 = ceil(j/scale)
            x0 = min(x1-1, floor(i/scale))
            y0 = min(y1-1, floor(j/scale))

            # Verifica se (i,j) possui algum ponto de referêcia que vai além da imagem.
            # Por exemplo, em uma imagem 4x4 que foi redimensionada para 8x8, ao tentar
            # calcular os pixels de referência para a última coluna e linha, você obtêm
            # x1 e y1 que não existem - estão além da imagem. 
            # Como esses valores não existem, não é possível realizar uma interpolação do
            # tipo bilinear - não temos pontos para usar! - sendo assim, utiliza-se a
            # interpolação Nearest Neighbor, onde copia-se o valor do pixel vizinho mais
            # próximo.
            if (x1 > in_width-1 or y1 > in_height-1):
                value = in_image_obj.getpixel((floor(i/scale), floor(j/scale)))
                out_image_obj.putpixel((i,j), int(value))

                # O código pode ser melhor trabalhado/estruturado para evitar usar
                # "continue" e ficar quebrando o fluxo de leitura (do programador).
                continue
                
            # Cálculo do valor dos pixels de referência para interpolação
            f00 = in_image_obj.getpixel((x0,y0))
            f01 = in_image_obj.getpixel((x0,y1))
            f10 = in_image_obj.getpixel((x1,y0))
            f11 = in_image_obj.getpixel((x1,y1))

            # Realização da interpolação bilinear
            value = bilinear_interpolation(i/scale, j/scale, x0, y0, x1, y1, f00, f01, f10, f11)

        # Escrita do valor do pixel de saída obtido na imagem (em sua respectiva posição)
        out_image_obj.putpixel((i,j), int(value))

# Apenas para exibir a imagem. Caso deseje, você pode descomentar o código em sequência
# para salvar a imagem também.
out_image_obj.save(out_image)
#out_image_obj.show()

# Fechar os objetos que manipulam as imagens de entrada e saída
in_image_obj.close()
out_image_obj.close()
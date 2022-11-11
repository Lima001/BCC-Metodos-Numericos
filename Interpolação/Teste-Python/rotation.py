# Utiliazando interpolação, realiza a rotação de uma imagem

from PIL import Image
from math import floor, ceil, pi, sin, cos

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
# Obs. A imagem de saída não é salva por padrão, sendo apenas exibida ao final do programa!
in_image = "image.jpg"
out_image = "output.jpg"

# Objeto para manipular a imagem de entrada
in_image_obj = Image.open(in_image)

# Obtendo dimensões da imagem para calculos futuros
in_width =  in_image_obj.size[0]
in_height = in_image_obj.size[1]

# Definindo coordenadas do centro da imagem de entrada
in_center = (in_width//2, in_height//2)

# Informações das dimensões da imagem de saída. No programa em questão, a imagem não será escalada, 
# mas sim o plano de fundo. Isso ocorre, pois caso as dimensões forem as mesmas da imagem de entrada,
# ao efetuar certas rotações a imagem de saída apresentará "cortes"

# Caso deseje observar a imagem rotacionada com cortes, remova o escalar utilizado para multiplicar
# as dimensões da imagem de entrada
out_width  = 2*in_width
out_height = 2*in_height

# Definindo coordenadas do centro da imagem de saída
out_center = (out_width//2, out_height//2)

# Tupla contendo as dimensões da imagem de saída
out_size = (out_width, out_height)

# Objeto para manipulação da imagem de saída. Informa-se o modo ("L" -> gray scale) e o tamanho
out_image_obj = Image.new(mode="L", size=out_size)

# Angulo em radianos que imagem deve ser rotacionada
angle = -40*pi/180

# Visando facilitar o processo de rotação, ao invés de percorrer a imagem de entrada
# e calcular o valor dos pixels na imagem de saída, faz se o processo inverso. Dessa forma
# é possível resolver mais facilmente problemas de mapeamento. Para uma discussão um pouco
# mais detalhada sobre, recomenda-se acessar o link: 
# https://gamedev.stackexchange.com/questions/120023/image-interpolaton-for-rotation

# Percorrendo (por índice) as linhas da imagem de saída
for i in range(out_width):
    # Percorrendo (por índice) as colunas da imagem de saída
    for j in range(out_height):

        # Calculando a posição do pixel inversamente rotacionado (com base na origem definida no centro da imagem)
        # na imagem de entrada
        i_ = (i-out_center[0])*cos(angle) - (j-out_center[1])*sin(angle) + in_center[0]
        j_ = (i-out_center[0])*sin(angle) + (j-out_center[1])*cos(angle) + in_center[1]

        # Se o valor obtido está além das dimensões da imagem de entrada, um dos dois casos
        # abaixo ocorreu:
        # (I) O pixel inversamente rotacionado corresponde a um pixel de fundo (sem cor) - ocorre
        #     quando a imagem de saída possui dimensões maiores do que a imagem de entrada
        # (II) O pixel iversamente rotacionado cprresponde a um elemento da imagem de entrada,
        #      porém ele ultrapassa os limites da imagem - nesse caso, obtem-se o corte da imagem.
        #      Ocorre quando as dimensões da imagem de saída são iguais, ou menores a imagem de entrada
        if (int(i_) >= in_width or int(i_) < 0 or int(j_) >= in_height or int(j_) < 0):
            # Se isso acontecer, continue o processo de iteração - por padrão ao criar o objeto
            # para a imagem de saída, as posições são "de fundo/sem cor".
            continue

        # Em sequência tem-se o código que realiza a interpolação do valor do pixel rotacionado.
        # Nesse caso, as posições anteriores correspodem a um ponto na imagem de entrada. Todavia,
        # esse ponto não é discreto, logo diferentes possibilidades podem ser feitas para obter
        # uma correspondência com um pixel real na imagem.  Por padrão, optou-se por realziar a 
        # interpolação bilinear desse valor.
        #
        # Todavia, caso queira, descomente o código abaixo para realizar interpolação por vizinho 
        # mais próximo.
        #
        # out_image_obj.putpixel((i,j), in_image_obj.getpixel((int(i_),int(j_))))
        # continue # Apenas para facilitar e não executar o código da interpolação bilinear
        
        # Calculando as coordenadas dos quatro pontos de referência na imagem de entrada
        # para realizar a interpolação bilinear do pixel na coordenada rotacionada obtida anteriormente
        x1 = ceil(i_)
        y1 = ceil(j_)
        x0 = min(x1-1, floor(i_))
        y0 = min(y1-1, floor(j_))

        # Verifica se (i,j) possui algum ponto de referêcia que vai além da imagem.
        # Como esses valores não existem, não é possível realizar uma interpolação do
        # tipo linear - não temos dois pontos para usar! - sendo assim, utiliza-se a
        # interpolação Nearest Neighbor, onde copia-se o valor do pixel vizinho mais
        # próximo.
        if (x1 > in_width-1 or y1 > in_height-1):
            value = in_image_obj.getpixel((int(i_), int(j_)))
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
        value = bilinear_interpolation(i_, j_, x0, y0, x1, y1, f00, f01, f10, f11)

        # Escrita do valor do pixel de saída obtido na imagem (em sua respectiva posição)
        out_image_obj.putpixel((i,j), int(value))

# Apenas para exibir a imagem. Caso deseje, você pode descomentar o código em sequência
# para salvar a imagem também.
#out_image_obj.save(out_image)
out_image_obj.show()

# Fechar os objetos que manipulam as imagens de entrada e saída
in_image_obj.close()
out_image_obj.close()
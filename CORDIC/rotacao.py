from traceback import print_tb
import pygame
from math import cos, sin, pi

def fcos(x):
    return 1 - 1/2 * x*x + 1/24 * x**4 - 1/720 * x**6

def fsin(x):
    return x - 1/6 * x**3 + 1/120 * x**5

def rodar(x, y, angulo):
    return (x*fcos(angulo) - y*fsin(angulo), x*fsin(angulo) + y*fcos(angulo))

def rodar_final(x, y, angulo):
    c, s = cos_sin(angulo) 
    return (x*c - y*s, x*s + y*c)

def cos_sin(angulo_alvo):
    angulo = 0

    x0 = 1
    y0 = 0

    if angulo_alvo > 2 * 3.14:
        #print(angulo_alvo)
        angulo_alvo -= (angulo_alvo // (2 * 3.14)) * (2* 3.14)
        #print(angulo_alvo)

    while abs(angulo - angulo_alvo) > 3.14/4:
        angulo += 3.14/4
        x0, y0 = rodar(x0, y0, 3.14/4)

    for i in range(0,20):
        if angulo == angulo_alvo:
            break

        d = -1 if angulo_alvo - angulo < 0 else +1
        angulo += d*3.14/(4*2**i)
        x0, y0 = rodar(x0, y0, d*3.14/(4*2**i))
    
    #return (cos(angulo_alvo), sin(angulo_alvo))
    return (x0,y0)


#cores
branco = (255,255,255)
verde = (0,255,0)
vermelho = (255,0,0)
preto = (0,0,0)

#tela
largura = 800
altura = 600
velocidade = 30

#variaveis do jogo
executar = True
pia = 3.14
angulo = 0
crescimento = pi/120

#Circunferencia
pos_x = largura//2
pos_y = altura//2
raio = 100

#Linha
pos_xf = raio
pos_yf = 0

pygame.init()

tela = pygame.display.set_mode((largura,altura))

relogio = pygame.time.Clock()

fonte = pygame.font.SysFont(None, 20)

while executar:

    tela.fill(preto)

    for evento in pygame.event.get():
        if evento.type == pygame.QUIT:
            executar = False

    pos_xf, pos_yf = rodar_final(pos_xf, pos_yf, angulo)
    
    pygame.draw.circle(tela, branco, (pos_x,pos_y), raio, 1)
    pygame.draw.line(tela, vermelho, (pos_x,pos_y), (int(pos_x+pos_xf), int(pos_y-pos_yf)))

    angulo += crescimento
    pos_xf = raio
    pos_yf = 0

    relogio.tick(velocidade)
    pygame.display.update()

pygame.quit()

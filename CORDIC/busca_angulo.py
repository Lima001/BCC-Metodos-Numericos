angulo_alvo = 5*3.14
angulo = 0

if angulo_alvo > 2 * 3.14:
    angulo_alvo -= angulo_alvo // (2 * 3.14)

while abs(angulo - angulo_alvo) > 3.14/4:
    angulo += 3.14/4

for i in range(0,20):
    if angulo == angulo_alvo:
        break

    d = -1 if angulo_alvo - angulo < 0 else +1
    angulo += d*3.14/(4*2**i)

print(angulo)
print(angulo_alvo) 
# Implementação teste do filtro gaussiano

from PIL import Image
from math import pow, pi, e

def print_kernel(kernel):
    size = len(kernel)
    for i in range(size):
        for j in range(size):
            print(kernel[i][j], end=" ")
        print()

def generate_gaussian_kernel(size, sigma):
    
    if size%2 == 0:
        return

    radius = (size-1)//2

    kernel = [[0 for i in range(size)] for j in range(size)]

    a = 1/(2*pi*pow(sigma,2))
    b = 2*pow(sigma,2)

    for i in range(-radius,radius+1):
        for j in range(-radius,radius+1):
            kernel[i+radius][j+radius] = a*pow(e,(-(i*i+j*j)/b))

    s = sum([sum(kernel[i]) for i in range(size)])
    
    for i in range(size):
        for j in range(size):
            kernel[i][j] /= s 
    
    return kernel

def blur_image(in_image, out_image, size, sigma, save=0):
    kernel = generate_gaussian_kernel(size,sigma)
    radius = (size-1)//2

    in_image_obj = Image.open(in_image)
    width = in_image_obj.size[0]
    height = in_image_obj.size[1]

    out_image_obj = Image.new(mode="L", size=(width,height))

    for i in range(width):
        for j in range(height):
            
            value = 0
            
            for ki in range(-radius,radius+1):
                for kj in range(-radius,radius+1):
            
                    if i+ki < 0 or i+ki >= width or j+kj < 0 or j+kj >= height:
                        continue

                    value += in_image_obj.getpixel((i+ki,j+kj))*kernel[ki+radius][kj+radius]

            out_image_obj.putpixel((i,j), int(value))

    if save:
        out_image_obj.save(out_image)
    else:
        out_image_obj.show()

    in_image_obj.close()
    out_image_obj.close()


if __name__ == "__main__":
    blur_image("image.jpg", "out.jpg", 7, 64)
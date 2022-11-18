from gaussian_blur import *
from resize import *
from math import sqrt

def generate_escale_space(in_image, out_dir, n_octaves, n_blur_level, sigma):

    # Armazena o nome das imagens geradas
    scale_space_names = [[] for i in range(n_octaves)]
    
    # Produção das imagens bases
    scale = 1
    for i in range(n_octaves):
        out_image = f"{out_dir}/output_scale_{i}_{0}.jpg"
        scale_space_names[i].append(out_image)    
        resize(in_image,out_image,scale,True)
        scale *= 0.5

    # Aplicando blur nos octaves
    for i in range(n_octaves):
        blur = sigma
        for j in range(1,n_blur_level):
            out_image = f"{out_dir}/output_scale_{i}_{j}.jpg"

            base_image = scale_space_names[i][j-1]
            blur_image(base_image, out_image, 3, blur, 1)

            scale_space_names[i].append(out_image)
            blur *= sigma

    return scale_space_names

if __name__ == "__main__":
    generate_escale_space("image.jpg", "out", 4, 5, sqrt(2))
from scale_space import *

def sub_images(img1, img2, output, save=1):
    img1_obj = Image.open(img1)
    img2_obj = Image.open(img2)
    
    width = img1_obj.size[0]
    height = img1_obj.size[1]
    
    output_obj = Image.new(mode="L", size=(width, height))

    for i in range(width):
        for j in range(height):
            value = img1_obj.getpixel((i,j)) - img2_obj.getpixel((i,j))
            output_obj.putpixel((i,j), value)

    if save:
        output_obj.save(output)
    else:
        output_obj.show()

    img1_obj.close()
    img2_obj.close()
    output_obj.close()

def difference_of_gaussians(in_image, out_dir, n_octaves, n_blur_level, sigma):
    scale_space = generate_escale_space(in_image, out_dir, n_octaves, n_blur_level, sigma)
    
    dog = [[] for i in range(n_octaves)]
    
    for i in range(n_octaves):
        for j in range(1,n_blur_level):
            output = f"{out_dir}/output_dog_{i}_{j-1}.jpg"
            sub_images(scale_space[i][j-1], scale_space[i][j], output)

    return dog

if __name__ == "__main__":
    difference_of_gaussians("image.jpg", "out", 4, 5, sqrt(2))
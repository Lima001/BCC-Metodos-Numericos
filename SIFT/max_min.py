from dog import *

def locate_approx_max_min(in_image, out_dir, n_octaves, n_blur_level, sigma):
    dog = difference_of_gaussians(in_image, out_dir, n_octaves, n_blur_level, sigma)
    
    approx_max_min = [[] for i in range(n_octaves)]

    for i in range(n_octaves):
        for j in range(1,len(dog[0])-1):
            output = f"{out_dir}/output_approx_max_min_{i}_{j-1}.jpg"
            img1_obj = Image.open(dog[i][j-1])
            img2_obj = Image.open(dog[i][j])
            img3_obj = Image.open(dog[i][j+1])
            
            width = img1_obj.size[0]
            height = img1_obj.size[1]
            
            output_obj = Image.new(mode="L", size=(width, height))

            for i in range(width):
                for j in range(height):
                    x = img2_obj.getpixel((i,j))
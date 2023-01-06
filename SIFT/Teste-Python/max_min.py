from dog import *

def calc_approx_max_min(img1, img2, img3, output, save=1):
    img1_obj = Image.open(img1)
    img2_obj = Image.open(img2)
    img3_obj = Image.open(img3)
    
    width = img1_obj.size[0]
    height = img1_obj.size[1]
    
    output_obj = Image.new(mode="L", size=(width, height))

    for i in range(width):
        for j in range(height):
            
            x = img2_obj.getpixel((i,j))
            
            stop = 0
            for ik in range(i-1, i+2):
                
                if stop:
                    break

                for jk in range(j-1, j+2):
                    
                    if ik < 0 or ik > width or jk < 0 or jk > height:
                        continue
                    
                    v1 = img1_obj.getpixel((i,j))
                    v2 = img2_obj.getpixel((i,j))
                    v3 = img3_obj.getpixel((i,j))

                    if (max(x,v1,v2,v3) != x and min(x,v1,v2,v3) != x):
                        stop = 1
                        break

            if not stop:
                output_obj.putpixel((i,j), x)

    img1_obj.close()
    img2_obj.close()
    img3_obj.close()

    if save:
        output_obj.save(output)
    else:
        output_obj.show()

    output_obj.close()

def locate_approx_max_min(in_image, out_dir, n_octaves, n_blur_level, sigma):
    dog = difference_of_gaussians(in_image, out_dir, n_octaves, n_blur_level, sigma)
    
    approx_max_min = [[] for i in range(n_octaves)]

    for i in range(n_octaves):
        for j in range(1,len(dog[0])-1):
            
            output = f"{out_dir}/output_approx_max_min_{i}_{j-1}.jpg"
            calc_approx_max_min(dog[i][j-1], dog[i][j], dog[i][j+1], output)
            approx_max_min.append(output)

    return approx_max_min

if __name__ == "__main__":
    locate_approx_max_min("image.jpg", "out", 4, 5, sqrt(2))
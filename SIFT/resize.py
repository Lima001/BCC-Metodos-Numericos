from PIL import Image
from math import floor, ceil

def linear_interpolation(x, x0, y0, x1, y1):
    u = (x-x0)/(x1-x0)
    return (1-u)*y0 + u*y1


def bilinear_interpolation(x, y, x0, y0, x1, y1, f00, f01, f10, f11):
    r1 = linear_interpolation(x, x0, f00, x1, f10)
    r2 = linear_interpolation(x, x0, f01, x1, f11)

    return linear_interpolation(y, y0, r1, y1, r2)


def resize(in_image, out_image, scale, save=0):

    in_image_obj = Image.open(in_image)
    in_width = in_image_obj.size[0]
    in_height = in_image_obj.size[1]

    out_size = (int(in_width * scale), int(in_height * scale)) 
    out_image_obj = Image.new(mode="L", size=out_size)

    for i in range(out_size[0]):

        for j in range(out_size[1]):

            if (i%scale == 0 and j%scale == 0):
                value = in_image_obj.getpixel((i/scale, j/scale))

            else:

                x1 = ceil(i/scale)
                y1 = ceil(j/scale)
                x0 = min(x1-1, floor(i/scale))
                y0 = min(y1-1, floor(j/scale))

                if (x1 > in_width-1 or y1 > in_height-1):
                    value = in_image_obj.getpixel((floor(i/scale), floor(j/scale)))
                    out_image_obj.putpixel((i,j), int(value))
                    continue
                    
                f00 = in_image_obj.getpixel((x0,y0))
                f01 = in_image_obj.getpixel((x0,y1))
                f10 = in_image_obj.getpixel((x1,y0))
                f11 = in_image_obj.getpixel((x1,y1))

                value = bilinear_interpolation(i/scale, j/scale, x0, y0, x1, y1, f00, f01, f10, f11)

            out_image_obj.putpixel((i,j), int(value))

    if save:
        out_image_obj.save(out_image)
    else:
        out_image_obj.show()

    in_image_obj.close()
    out_image_obj.close()

if __name__ == "__main__":
    resize("image.jpg","output.jpg",0.5)
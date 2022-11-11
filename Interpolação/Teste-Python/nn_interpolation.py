from PIL import Image
from math import floor

in_image = "image1.jpg"
out_image = "output.jpg"

in_image_obj = Image.open(in_image)

in_width = in_image_obj.size[0]
in_height = in_image_obj.size[1]
scale = 6
out_size = (in_width * scale, in_height * scale) 

out_image_obj = Image.new(mode="L", size=out_size)

for i in range(out_size[0]):
    for j in range(out_size[1]):
        value = in_image_obj.getpixel((floor(i/scale), floor(j/scale)))
        out_image_obj.putpixel((i,j), value)

out_image_obj.show()

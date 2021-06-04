from pixilizer.image import Pixilizer
from pixilizer.utils import *

path = "234.jpg"

image = read_source_image(path)

Px = Pixilizer(pixel_relative_size=1)

rez, _ = Px.forward(image, True, 4, 10)

pil_rez = np_to_pil(rez, 1.2)
pil_rez.show()
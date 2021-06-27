from pixilizer.utils import pixilize_single_image, pixilize_single_image_with_quantization


class Pixilizer(object):
    def __init__(self, pixel_relative_size=6) -> None:
        super().__init__()
        assert (pixel_relative_size > 0 and pixel_relative_size <= 100)
        self.pixel_relative_size = pixel_relative_size

    def forward(self, image, pix=False, num_cetroids=8, color_offset=3):
        if pix:
            self.pixelised_image, quant = pixilize_single_image_with_quantization(image, self.pixel_relative_size,
                                                                                  num_cetroids, color_offset)
            return self.pixelised_image, quant
        else:
            self.pixelised_image = pixilize_single_image(image, self.pixel_relative_size)

            return self.pixelised_image

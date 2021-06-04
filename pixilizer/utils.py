import cv2
import imageio
from PIL import Image, ImageEnhance
import numpy as np
from sklearn.cluster import KMeans


def pixilize_single_image(image, pixel_relative_size):

    height, width = image.shape[:2]

    pixel_absolute_size = pixel_relative_size / 100 * height

    number_full_pixels_height = height // int(pixel_absolute_size)
    number_full_pixels_width = width // int(pixel_absolute_size)

    temp = cv2.resize(image,
                      (number_full_pixels_width, number_full_pixels_height),
                      interpolation=cv2.INTER_LINEAR)

    output = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)

    return output


def pixilize_single_image_with_quantization(image,
                                            pixel_relative_size,
                                            clusters=8,
                                            colorOffset=6):

    height, width = image.shape[:2]

    pixel_absolute_size = pixel_relative_size / 100 * height

    number_full_pixels_height = height // int(pixel_absolute_size)
    number_full_pixels_width = width // int(pixel_absolute_size)

    temp = cv2.resize(image,
                      (number_full_pixels_width, number_full_pixels_height),
                      interpolation=cv2.INTER_LINEAR)

    copy_temp = temp.copy()
    copy_temp = copy_temp.reshape((copy_temp.shape[0] * copy_temp.shape[1], 3))
    # Perform clustering
    clt = KMeans(n_clusters=clusters + (colorOffset))
    clt.fit(copy_temp)
    # Create centroids hist
    centroids = clt.cluster_centers_
    pinformed = np.array([[copy_temp[i], clt.labels_[i], i]
                          for i in range(copy_temp.shape[0])])
    quantzd = temp.copy()

    for i in pinformed:
        quantzd[i[2] // number_full_pixels_width][
            i[2] % number_full_pixels_width] = centroids[i[1]]

    output = cv2.resize(quantzd, (width, height),
                        interpolation=cv2.INTER_NEAREST)

    return output, quantzd


def read_source_image(image_path):
    source_image = imageio.imread(image_path)
    if len(source_image.shape) < 3:
        source_image = cv2.cvtColor(source_image, cv2.COLOR_GRAY2RGB)
    source_image = source_image[..., :3]
    return source_image


def np_to_pil(image, enhance=1.0):
    im = Image.fromarray(image)
    converter = ImageEnhance.Color(im)
    img2 = converter.enhance(enhance)
    return img2


def show_np_img(image, enhance=1.0):
    im = Image.fromarray(image)
    converter = ImageEnhance.Color(im)
    img2 = converter.enhance(enhance)
    img2.show("result")


def blend_images(image_one, image_two, mask):
    result = image_one.copy()
    # foreground = (image_two.copy()) * np.stack([mask, mask, mask], axis=2)
    # background = result * np.stack([~mask, ~mask, ~mask], axis=2)
    # blend = cv2.addWeighted(background, 0.5, foreground, 0.5, 0.0)
    result[np.where(mask != 0)] = image_two[np.where(mask != 0)]

    return result

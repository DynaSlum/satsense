import numpy as np
import rasterio

# import below has side effects
from satsense.features import Lacunarity  # noqa: F401
from satsense.features import Texton  # noqa: F401
from satsense.image import Image


def write_target(target_image, image_name, crs, transform):
    with rasterio.open(
            image_name,
            'w',
            driver='GTiff',
            height=target_image.shape[1],
            width=target_image.shape[2],
            count=target_image.shape[0],
            dtype=str(target_image.dtype),
            crs=crs,
            transform=transform,
            nodata=target_image.fill_value) as target:
        target.write(target_image.filled())


def generate_grayscale(img, prefix):
    grayscale = img['grayscale'][np.newaxis, :, :]

    write_target(grayscale, prefix + 'grayscale.tif', img.crs, img.transform)


def generate_canny_edge(img, prefix):
    canny = img['canny_edge'][np.newaxis, :, :].astype(np.uint8)
    canny.set_fill_value(255)
    write_target(canny, prefix + 'canny_edge.tif', img.crs, img.transform)


def generate_gray_ubyte(img, prefix):
    gray_ubyte = img['gray_ubyte'][np.newaxis, :, :]
    gray_ubyte[gray_ubyte == 255] = 254

    gray_ubyte.set_fill_value(255)

    write_target(gray_ubyte, prefix + 'gray_ubyte.tif', img.crs, img.transform)


if __name__ == "__main__":
    image = Image('../source/section_2_sentinel.tif', 'quickbird')
    image.precompute_normalization()

    generate_grayscale(image, '../baseimage/section_2_sentinel_')
    generate_canny_edge(image, '../baseimage/section_2_sentinel_')
    generate_gray_ubyte(image, '../baseimage/section_2_sentinel_')

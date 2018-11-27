import rasterio

from satsense.features.lacunarity import get_canny_edge_image


def test_canny_edge(image):
    with rasterio.open('test/data/baseimage/section_2_sentinel_canny_edge.tif'
                       ) as dataset:
        target = dataset.read(1, masked=True)

        result = get_canny_edge_image(image)

        same = target == result

        assert same.all()

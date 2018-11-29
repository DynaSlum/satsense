import py.path
import rasterio

from satsense import FeatureVector, Image
from satsense.extract import extract_feature
from satsense.features import HistogramOfGradients
from satsense.generators import FullGenerator


def test_netcdf_save(image, tmpdir):
    netcdf_save(image, tmpdir)


def netcdf_save(image, tmpdir):
    feature = HistogramOfGradients([(25, 25), (50, 50)])
    generator = FullGenerator(image, (25, 25))
    vector = extract_feature(feature, generator)
    fv = FeatureVector(
        feature, vector, crs=generator.crs, transform=generator.transform)

    paths = fv.save(str(tmpdir) + '/')

    with rasterio.open(paths[0]) as dataset:
        assert dataset.shape == vector.shape[0:2]


if __name__ == "__main__":
    image = Image('test/data/source/section_2_sentinel.tif', 'quickbird')
    image.precompute_normalization()

    netcdf_save(image, py.path.local('.'))

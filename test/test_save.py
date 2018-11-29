from satsense import Image, FeatureVector
from satsense.generators import FullGenerator
from satsense.features import HistogramOfGradients
from satsense.extract import extract_feature


def test_netcdf_save(image):
    feature = HistogramOfGradients([(50, 50)])
    generator = FullGenerator(image, (25, 25))
    vector = extract_feature(feature, generator)
    fv = FeatureVector(feature, vector, generator.step_size)

    fv.crs = image.crs
    fv.transform = image.transform

    fv.save()


if __name__ == "__main__":
    img = Image('test/data/source/section_2_sentinel.tif', 'quickbird')
    img.precompute_normalization()

    test_netcdf_save(img)

from satsense.image import Image


def test_roundtrip_normalization(tmpdir):
    img = Image('test/data/source/section_2_sentinel.tif', 'quickbird')
    img2 = Image('test/data/source/section_2_sentinel.tif', 'quickbird')

    img.precompute_normalization()

    img.save_normalization_limits(filename_prefix=str(tmpdir))

    img2.load_normalization_limits(filename_prefix=str(tmpdir))

    assert img.normalization_parameters == img2.normalization_parameters

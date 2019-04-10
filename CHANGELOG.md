# Changelog

## version 0.9

- Added a large amount of documentation
  - Available at https://satsense.readthedocs.io/en/latest/
  - Includes:
    - Installation instructions
    - Example notebook for feature extraction
    - API Documentation and docstrings

- Bug fixes:
  - Histogram of Gradients
    - fixed the absolute sine difference calculation
  - Fixed the padding around the data when splitting the generator.
  - Fixed the generation of windows

- Development:
  - Added automated versioning
  - Increased code maintainability

## version 0.8
- Initial release
- Features included:
  - Histogram of Gradients
  - Pantex
  - NDVI
    - also available:
    - RgNDVI (Red-green based)
    - RbNDVI (Red-blue based)
    - NDSI (Snow Cover Index)
    - NDWI (Water Cover Index)
    - WVSI (Soil Cover Index)
  - Lacunarity
  - SIFT
  - Texton
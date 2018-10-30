import hypothesis.strategies as st
from hypothesis.extra.numpy import (floating_dtypes, integer_dtypes,
                                    unsigned_integer_dtypes)
from rasterio import check_dtype

n_jobs = st.integers(min_value=1, max_value=10)

rasterio_dtypes = st.one_of(
    integer_dtypes(),
    unsigned_integer_dtypes(),
    floating_dtypes(),
).filter(check_dtype)

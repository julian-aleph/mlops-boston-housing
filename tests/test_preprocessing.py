import numpy as np
import pandas as pd
from preprocessing import QuantileClipper


def test_quantile_clipper_dataframe_clips_without_mutating_input() -> None:
    data = pd.DataFrame({"a": [1.0, 2.0, 100.0], "b": [5.0, 6.0, 7.0]})
    original = data.copy(deep=True)

    clipper = QuantileClipper(lower_quantile=0.25, upper_quantile=0.75)
    transformed = clipper.fit(data).transform(data)

    pd.testing.assert_frame_equal(data, original)
    assert isinstance(transformed, pd.DataFrame)
    assert transformed["a"].min() >= clipper.lower_bounds_["a"]
    assert transformed["a"].max() <= clipper.upper_bounds_["a"]


def test_quantile_clipper_numpy_array() -> None:
    data = np.array([[1.0, 5.0], [2.0, 6.0], [100.0, 7.0]])

    clipper = QuantileClipper(lower_quantile=0.25, upper_quantile=0.75)
    transformed = clipper.fit(data).transform(data)

    assert isinstance(transformed, np.ndarray)
    assert transformed.shape == data.shape
    assert transformed[:, 0].min() >= clipper.lower_bounds_[0]
    assert transformed[:, 0].max() <= clipper.upper_bounds_[0]

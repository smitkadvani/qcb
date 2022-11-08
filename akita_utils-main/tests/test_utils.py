import pandas as pd
import numpy as np
from akita_utils import filter_by_chrmlen, ut_dense
from io import StringIO


def test_ut_dense():

    # toy output representing upper-triangular output for two targets
    ut_vecs = np.vstack(([4, 5, 6], [-2, -3, -4])).T

    # (3 entries) x (2 targets) input, two empty diagonals --> 4x4x2 output
    assert np.shape(ut_dense(ut_vecs, 2)) == (4, 4, 2)

    # (3 entries) x (2 targets) input, one empty diagonals --> 3x3x2 output
    dense_mats = ut_dense(ut_vecs, 1)
    assert np.shape(dense_mats) == (3, 3, 2)

    # outputs are symmetric dense matrices with the 3 original entries
    # and zeros at the diagonal
    target_0 = np.array([[0, 4, 5], [4, 0, 6], [5, 6, 0]])
    target_1 = np.array([[0, -2, -3], [-2, 0, -4], [-3, -4, 0]])

    assert (dense_mats[:, :, 0] == target_0).all()
    assert (dense_mats[:, :, 1] == target_1).all()


def test_split_df_equally():
    
    df = pd.DataFrame(np.linspace(0, 99, 100), columns=['col1'])
    fifth_chunk = akita_utils.split_df_equally(df, 20, 5)
    assert (fifth_chunk["col1"].to_numpy() == np.linspace(25, 29, 5)).all() == True

    
def test_filter_by_chrmlen():

    df1 = pd.DataFrame(
        [["chrX", 3, 8], ["chr1", 4, 5], ["chrX", 1, 5]],
        columns=["chrom", "start", "end"],
    )

    # get the same result with chrmsizes provided as dict or via StringIO

    # one interval is dropped for exceeding chrX len of 7
    assert filter_by_chrmlen(df1, {"chr1": 10, "chrX": 7}, 0).shape == (2, 3)

    # both chrX intervals are dropped if the buffer_bp are increased
    assert filter_by_chrmlen(df1, {"chr1": 10, "chrX": 7}, 3).shape == (1, 3)

    # no intervals remain if all of chr1 is excluded as well
    assert filter_by_chrmlen(df1, {"chr1": 10, "chrX": 7}, 5).shape == (0, 3)

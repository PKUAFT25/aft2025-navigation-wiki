## Note:
## 复现 MASTER ：若需脱离 qlib 框架，只需新开空的./qlib/data, 将以下文件放在项目的 ./qlib/data/dataset.py；可能需要配上 ./qlib/data/__init__.py

import pandas as pd
import numpy as np
import bisect
from typing import Union, List, Tuple

def get_level_index(df: pd.DataFrame, level: Union[str, int]) -> int:
    """

    get the level index of `df` given `level`

    Parameters
    ----------
    df : pd.DataFrame
        data
    level : Union[str, int]
        index level

    Returns
    -------
    int:
        The level index in the multiple index
    """
    if isinstance(level, str):
        try:
            return df.index.names.index(level)
        except (AttributeError, ValueError):
            # NOTE: If level index is not given in the data, the default level index will be ('datetime', 'instrument')
            return ("datetime", "instrument").index(level)
    elif isinstance(level, int):
        return level
    else:
        raise NotImplementedError(f"This type of input is not supported")


def np_ffill(arr: np.array):
    """
    forward fill a 1D numpy array

    Parameters
    ----------
    arr : np.array
        Input numpy 1D array
    """
    mask = np.isnan(arr.astype(float))  # np.isnan only works on np.float
    # get fill index
    idx = np.where(~mask, np.arange(mask.shape[0]), 0)
    np.maximum.accumulate(idx, out=idx)
    return arr[idx]

class TSDataSampler:
    """
    (T)ime-(S)eries DataSampler
    This is the result of TSDatasetH

    It works like `torch.data.utils.Dataset`, it provides a very convenient interface for constructing time-series
    dataset based on tabular data.
    - On time step dimension, the smaller index indicates the historical data and the larger index indicates the future
      data.

    If user have further requirements for processing data, user could process them based on `TSDataSampler` or create
    more powerful subclasses.

    Known Issues:
    - For performance issues, this Sampler will convert dataframe into arrays for better performance. This could result
      in a different data type


    Indices design:
        TSDataSampler has a index mechanism to help users query time-series data efficiently.

        The definition of related variables:
            data_arr: np.ndarray
                The original data. it will contains all the original data.
                The querying are often for time-series of a specific stock.
                By leveraging this data charactoristics to speed up querying, the multi-index of data_arr is rearranged in (instrument, datetime) order

            data_index: pd.MultiIndex with index order <instrument, datetime>
                it has the same shape with `idx_map`. Each elements of them are expected to be aligned.

            idx_map: np.ndarray
                It is the indexable data. It originates from data_arr, and then filtered by 1) `start` and `end`  2) `flt_data`
                    The extra data in data_arr is useful in following cases
                    1) creating meaningful time series data before `start` instead of padding them with zeros
                    2) some data are excluded by `flt_data` (e.g. no <X, y> sample pair for that index). but they are still used in time-series in X

                Finnally, it will look like.

                array([[  0,   0],
                       [  1,   0],
                       [  2,   0],
                       ...,
                       [241, 348],
                       [242, 348],
                       [243, 348]], dtype=int32)

                It list all indexable data(some data only used in historical time series data may not be indexabla), the values are the corresponding row and col in idx_df
            idx_df: pd.DataFrame
                It aims to map the <datetime, instrument> key to the original position in data_arr

                For example, it may look like (NOTE: the index for a instrument time-series is continoues in memory)

                    instrument SH600000 SH600008 SH600009 SH600010 SH600011 SH600015  ...
                    datetime
                    2017-01-03        0      242      473      717      NaN      974  ...
                    2017-01-04        1      243      474      718      NaN      975  ...
                    2017-01-05        2      244      475      719      NaN      976  ...
                    2017-01-06        3      245      476      720      NaN      977  ...

            With these two indices(idx_map, idx_df) and original data(data_arr), we can make the following queries fast (implemented in __getitem__)
            (1) Get the i-th indexable sample(time-series):   (indexable sample index) -> [idx_map] -> (row col) -> [idx_df] -> (index in data_arr)
            (2) Get the specific sample by <datetime, instrument>:  (<datetime, instrument>, i.e. <row, col>) -> [idx_df] -> (index in data_arr)
            (3) Get the index of a time-series data:   (get the <row, col>, refer to (1), (2)) -> [idx_df] -> (all indices in data_arr for time-series)
    """

    # Please refer to the docstring of TSDataSampler for the definition of following attributes
    data_arr: np.ndarray
    data_index: pd.MultiIndex
    idx_map: np.ndarray
    idx_df: pd.DataFrame

    def __init__(
        self,
        data: pd.DataFrame,
        start,
        end,
        step_len: int,
        fillna_type: str = "none",
        dtype=None,
        flt_data=None,
    ):
        """
        Build a dataset which looks like torch.data.utils.Dataset.

        Parameters
        ----------
        data : pd.DataFrame
            The raw tabular data whose index order is <"datetime", "instrument">
        start :
            The indexable start time
        end :
            The indexable end time
        step_len : int
            The length of the time-series step
        fillna_type : int
            How will qlib handle the sample if there is on sample in a specific date.
            none:
                fill with np.nan
            ffill:
                ffill with previous sample
            ffill+bfill:
                ffill with previous samples first and fill with later samples second
        flt_data : pd.Series
            a column of data(True or False) to filter data. Its index order is <"datetime", "instrument">
            None:
                kepp all data

        """
        self.start = start
        self.end = end
        self.step_len = step_len
        self.fillna_type = fillna_type
        assert get_level_index(data, "datetime") == 0
        self.data = data.swaplevel().sort_index().copy()
        data.drop(
            data.columns, axis=1, inplace=True
        )  # data is useless since it's passed to a transposed one, hard code to free the memory of this dataframe to avoid three big dataframe in the memory(including: data, self.data, self.data_arr)

        kwargs = {"object": self.data}
        if dtype is not None:
            kwargs["dtype"] = dtype

        self.data_arr = np.array(**kwargs)  # Get index from numpy.array will much faster than DataFrame.values!
        # NOTE:
        # - append last line with full NaN for better performance in `__getitem__`
        # - Keep the same dtype will result in a better performance
        self.data_arr = np.append(
            self.data_arr,
            np.full((1, self.data_arr.shape[1]), np.nan, dtype=self.data_arr.dtype),
            axis=0,
        )
        self.nan_idx = len(self.data_arr) - 1  # The last line is all NaN; setting it to -1 can cause bug #1716

        # the data type will be changed
        # The index of usable data is between start_idx and end_idx
        self.idx_df, self.idx_map = self.build_index(self.data)
        self.data_index = deepcopy(self.data.index)

        if flt_data is not None:
            if isinstance(flt_data, pd.DataFrame):
                assert len(flt_data.columns) == 1
                flt_data = flt_data.iloc[:, 0]
            # NOTE: bool(np.nan) is True !!!!!!!!
            # make sure reindex comes first. Otherwise extra NaN may appear.
            flt_data = flt_data.swaplevel()
            flt_data = flt_data.reindex(self.data_index).fillna(False).astype(bool)
            self.flt_data = flt_data.values
            self.idx_map = self.flt_idx_map(self.flt_data, self.idx_map)
            self.data_index = self.data_index[np.where(self.flt_data)[0]]
        self.idx_map = self.idx_map2arr(self.idx_map)
        self.idx_map, self.data_index = self.slice_idx_map_and_data_index(
            self.idx_map, self.idx_df, self.data_index, start, end
        )

        self.idx_arr = np.array(self.idx_df.values, dtype=np.float64)  # for better performance
        del self.data  # save memory

    @staticmethod
    def slice_idx_map_and_data_index(
        idx_map,
        idx_df,
        data_index,
        start,
        end,
    ):
        assert (
            len(idx_map) == data_index.shape[0]
        )  # make sure idx_map and data_index is same so index of idx_map can be used on data_index

        start_row_idx, end_row_idx = idx_df.index.slice_locs(start=time_to_slc_point(start), end=time_to_slc_point(end))

        time_flter_idx = (idx_map[:, 0] < end_row_idx) & (idx_map[:, 0] >= start_row_idx)
        return idx_map[time_flter_idx], data_index[time_flter_idx]

    @staticmethod
    def idx_map2arr(idx_map):
        # pytorch data sampler will have better memory control without large dict or list
        # - https://github.com/pytorch/pytorch/issues/13243
        # - https://github.com/airctic/icevision/issues/613
        # So we convert the dict into int array.
        # The arr_map is expected to behave the same as idx_map

        dtype = np.int32
        # set a index out of bound to indicate the none existing
        no_existing_idx = (np.iinfo(dtype).max, np.iinfo(dtype).max)

        max_idx = max(idx_map.keys())
        arr_map = []
        for i in range(max_idx + 1):
            arr_map.append(idx_map.get(i, no_existing_idx))
        arr_map = np.array(arr_map, dtype=dtype)
        return arr_map

    @staticmethod
    def flt_idx_map(flt_data, idx_map):
        idx = 0
        new_idx_map = {}
        for i, exist in enumerate(flt_data):
            if exist:
                new_idx_map[idx] = idx_map[i]
                idx += 1
        return new_idx_map

    def get_index(self):
        """
        Get the pandas index of the data, it will be useful in following scenarios
        - Special sampler will be used (e.g. user want to sample day by day)
        """
        return self.data_index.swaplevel()  # to align the order of multiple index of original data received by __init__

    def config(self, **kwargs):
        # Config the attributes
        for k, v in kwargs.items():
            setattr(self, k, v)

    @staticmethod
    def build_index(data: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """
        The relation of the data

        Parameters
        ----------
        data : pd.DataFrame
            A DataFrame with index in order <instrument, datetime>

                                      RSQR5     RESI5     WVMA5    LABEL0
            instrument datetime
            SH600000   2017-01-03  0.016389  0.461632 -1.154788 -0.048056
                       2017-01-04  0.884545 -0.110597 -1.059332 -0.030139
                       2017-01-05  0.507540 -0.535493 -1.099665 -0.644983
                       2017-01-06 -1.267771 -0.669685 -1.636733  0.295366
                       2017-01-09  0.339346  0.074317 -0.984989  0.765540

        Returns
        -------
        Tuple[pd.DataFrame, dict]:
            1) the first element:  reshape the original index into a <datetime(row), instrument(column)> 2D dataframe
                instrument SH600000 SH600008 SH600009 SH600010 SH600011 SH600015  ...
                datetime
                2017-01-03        0      242      473      717      NaN      974  ...
                2017-01-04        1      243      474      718      NaN      975  ...
                2017-01-05        2      244      475      719      NaN      976  ...
                2017-01-06        3      245      476      720      NaN      977  ...
            2) the second element:  {<original index>: <row, col>}
        """
        # object incase of pandas converting int to float
        idx_df = pd.Series(range(data.shape[0]), index=data.index, dtype=object)
        idx_df = lazy_sort_index(idx_df.unstack())
        # NOTE: the correctness of `__getitem__` depends on columns sorted here
        idx_df = lazy_sort_index(idx_df, axis=1).T

        idx_map = {}
        for i, (_, row) in enumerate(idx_df.iterrows()):
            for j, real_idx in enumerate(row):
                if not np.isnan(real_idx):
                    idx_map[real_idx] = (i, j)
        return idx_df, idx_map

    @property
    def empty(self):
        return len(self) == 0

    def _get_indices(self, row: int, col: int) -> np.array:
        """
        get series indices of self.data_arr from the row, col indices of self.idx_df

        Parameters
        ----------
        row : int
            the row in self.idx_df
        col : int
            the col in self.idx_df

        Returns
        -------
        np.array:
            The indices of data of the data
        """
        indices = self.idx_arr[max(row - self.step_len + 1, 0) : row + 1, col]

        if len(indices) < self.step_len:
            indices = np.concatenate([np.full((self.step_len - len(indices),), np.nan), indices])

        if self.fillna_type == "ffill":
            indices = np_ffill(indices)
        elif self.fillna_type == "ffill+bfill":
            indices = np_ffill(np_ffill(indices)[::-1])[::-1]
        else:
            assert self.fillna_type == "none"
        return indices

    def _get_row_col(self, idx) -> Tuple[int]:
        """
        get the col index and row index of a given sample index in self.idx_df

        Parameters
        ----------
        idx :
            the input of  `__getitem__`

        Returns
        -------
        Tuple[int]:
            the row and col index
        """
        # The the right row number `i` and col number `j` in idx_df
        if isinstance(idx, (int, np.integer)):
            real_idx = idx
            if 0 <= real_idx < len(self.idx_map):
                i, j = self.idx_map[real_idx]  # TODO: The performance of this line is not good
            else:
                raise KeyError(f"{real_idx} is out of [0, {len(self.idx_map)})")
        elif isinstance(idx, tuple):
            # <TSDataSampler object>["datetime", "instruments"]
            date, inst = idx
            date = pd.Timestamp(date)
            i = bisect.bisect_right(self.idx_df.index, date) - 1
            # NOTE: This relies on the idx_df columns sorted in `__init__`
            j = bisect.bisect_left(self.idx_df.columns, inst)
        else:
            raise NotImplementedError(f"This type of input is not supported")
        return i, j

    def __getitem__(self, idx: Union[int, Tuple[object, str], List[int]]):
        """
        # We have two method to get the time-series of a sample
        tsds is a instance of TSDataSampler

        # 1) sample by int index directly
        tsds[len(tsds) - 1]

        # 2) sample by <datetime,instrument> index
        tsds['2016-12-31', "SZ300315"]

        # The return value will be similar to the data retrieved by following code
        df.loc(axis=0)['2015-01-01':'2016-12-31', "SZ300315"].iloc[-30:]

        Parameters
        ----------
        idx : Union[int, Tuple[object, str]]
        """
        # Multi-index type
        mtit = (list, np.ndarray)
        if isinstance(idx, mtit):
            indices = [self._get_indices(*self._get_row_col(i)) for i in idx]
            indices = np.concatenate(indices)
        else:
            indices = self._get_indices(*self._get_row_col(idx))

        # 1) for better performance, use the last nan line for padding the lost date
        # 2) In case of precision problems. We use np.float64. # TODO: I'm not sure if whether np.float64 will result in
        # precision problems. It will not cause any problems in my tests at least
        indices = np.nan_to_num(indices.astype(np.float64), nan=self.nan_idx).astype(int)

        if (np.diff(indices) == 1).all():  # slicing instead of indexing for speeding up.
            data = self.data_arr[indices[0] : indices[-1] + 1]
        else:
            data = self.data_arr[indices]
        if isinstance(idx, mtit):
            # if we get multiple indexes, addition dimension should be added.
            # <sample_idx, step_idx, feature_idx>
            data = data.reshape(-1, self.step_len, *data.shape[1:])
        return data

    def __len__(self):
        return len(self.idx_map)


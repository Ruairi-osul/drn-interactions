from typing import List, Tuple
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import cross_val_score


class OffsetDecoder:
    def __init__(self, estimator, cv, scoring):
        self.estimator = estimator
        self.cv = cv
        self.scoring = scoring

    @staticmethod
    def _preprocess(frame_sets):
        frames = []
        for i, (pos, neg) in enumerate(frame_sets):
            positive_i = pd.concat({f"Positive_{i}": pos}, names=["group"])
            negative_i = pd.concat({f"Negative_{i}": neg}, names=["group"])
            frames.append(positive_i)
            frames.append(negative_i)
        df = pd.concat(frames)
        return df

    def fit_models(self, frame_sets: List[Tuple[pd.DataFrame, pd.DataFrame]]):
        df = self._preprocess(frame_sets)
        num_sets = len(frame_sets)
        offsets = df.index.unique(2).values

        results = {}
        for offset in offsets:
            X, y = self._get_X_y_at_offset(df, offset=offset, num_sets=num_sets)
            # print(offset)
            scores = cross_val_score(
                clone(self.estimator), X, y, cv=self.cv, scoring=self.scoring
            )
            results[offset] = scores
        return self.tidy_scores(results)

    def _get_X_y_at_offset(self, df, offset, num_sets):
        arrays = []
        for i in range(num_sets):
            positive_i = df.loc[(f"Positive_{i}", slice(None), offset), :].values
            negative_i = df.loc[(f"Negative_{i}", slice(None), offset), :].values
            arrays.append(positive_i)
            arrays.append(negative_i)
        X = np.concatenate(arrays)
        y = np.concatenate([np.repeat(i, len(array)) for i, array in enumerate(arrays)])
        return X, y

    @staticmethod
    def tidy_scores(results):
        return (
            pd.DataFrame(results)
            .reset_index()
            .rename(columns={"index": "fold"})
            .melt(var_name="offset", value_name="f1score", id_vars="fold")
        )

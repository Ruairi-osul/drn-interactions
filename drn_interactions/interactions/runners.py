from typing import List
import pandas as pd
from copy import deepcopy


class InteractionsRunner:
    def __init__(self, loader, preprocessor, pcup, corr, pcorr):
        self.loader = loader
        self.preprocessor = preprocessor
        self.pcup = pcup
        self.corr = corr
        self.pcorr = pcorr
        self._df_piv = None

    @property
    def df_piv(self):
        if self._df_piv is None:
            df_piv = self.loader()
            df_piv = self.preprocessor(df_piv)
            self._df_piv = df_piv
        return self._df_piv

    def run_pcorr(self):
        return self.pcorr(self.df_piv)

    def run_corr(self):
        return self.corr(self.df_piv)

    def run_pcup(self):
        return self.pcup(self.df_piv)

    def _run_bootstrap(self, f, n_boot: int = 500):
        frames: List[pd.DataFrame] = []
        current_loader = deepcopy(self.loader)
        self.loader.shuffle = True
        for i in range(n_boot):
            self._df_piv = None
            frames.append(f().assign(rep=i))

        self.loader = current_loader
        return pd.concat(frames).reset_index(drop=True)

    def _run_multi(self, sessions, f):
        frames = []
        for session in sessions:
            self._df_piv = None
            self.loader.set_session(session)
            frames.append(f().assign(session_name=session))
        return pd.concat(frames).reset_index(drop=True)

    def run_pcorr_multi(self, sessions):
        return self._run_multi(sessions, self.run_pcorr)

    def run_corr_multi(self, sessions):
        return self._run_multi(sessions, self.run_corr)

    def run_pcup_multi(self, sessions):
        return self._run_multi(sessions, self.run_pcup)

    def corr_bootstrap(self, n_boot: int = 500):
        return self._run_bootstrap(self.run_corr, n_boot)

    def pcorr_bootstrap(self, n_boot: int = 500):
        return self._run_bootstrap(self.run_pcorr, n_boot)

    def pcup_bootstrap(self, n_boot: int = 500):
        return self._run_bootstrap(self.run_pcup, n_boot)

    def corr_bootstrap_multi(self, sessions, n_boot: int = 500):
        return self._run_multi(sessions, lambda: self.corr_bootstrap(n_boot))

    def pcorr_bootstrap_multi(self, sessions, n_boot: int = 500):
        return self._run_multi(sessions, lambda: self.pcorr_bootstrap(n_boot))

    def pcup_bootstrap_multi(self, sessions, n_boot: int = 500):
        return self._run_multi(sessions, lambda: self.pcup_bootstrap(n_boot))

    def __call__(self, df_piv):
        ...

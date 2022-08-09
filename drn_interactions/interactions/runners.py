from typing import List
import pandas as pd
from copy import deepcopy


class InteractionsRunner:
    def __init__(self, loader, preprocessor, pcup=None, corr=None, pcorr=None):
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
        return self.pcorr.fit(self.df_piv).get_edge_df()

    def run_corr(self):
        return self.corr.fit(self.df_piv).get_edge_df()

    def run_pcup(self):
        return self.pcup.fit(self.df_piv).get_couplings()

    def _run_bootstrap(self, f, n_boot: int = 500):
        frames: List[pd.DataFrame] = []
        current_loader = deepcopy(self.loader)
        self.loader.shuffle = True
        for i in range(n_boot):
            self._df_piv = None
            frames.append(f().assign(rep=i))

        self.loader = current_loader
        return pd.concat(frames).reset_index(drop=True)

    def _run_multi(self, sessions, f, on_error="ignore"):
        frames = []
        for session in sessions:
            self._df_piv = None
            self.loader.set_session(session)
            try:
                frames.append(f().assign(session_name=session))
            except ValueError as e:
                if on_error == "ignore":
                    continue
                else:
                    raise e
        return pd.concat(frames).reset_index(drop=True)

    def run_pcorr_multi(self, sessions, on_error="ignore"):
        return self._run_multi(sessions, self.run_pcorr, on_error=on_error)

    def run_corr_multi(self, sessions, on_error="ignore"):
        return self._run_multi(sessions, self.run_corr, on_error=on_error)

    def run_pcup_multi(self, sessions, on_error="ignore"):
        return self._run_multi(sessions, self.run_pcup, on_error=on_error)

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

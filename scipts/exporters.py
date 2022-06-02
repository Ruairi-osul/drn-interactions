import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
from drn_interactions.export import (
    load_neurons,
    load_spiketimes,
    load_waveforms,
)
from drn_interactions.load import get_group_names
from dotenv import load_dotenv
from ephys_queries import (
    select_recording_sessions,
    select_discrete_data,
    select_spike_times,
    select_stft,
)
from spiketimes.df.statistics import mean_firing_rate_by, cv2_isi_by
from waveform_utils.waveforms import (
    waveform_peaks_by_neuron,
    waveform_width_by_neuron,
    peak_asymmetry_by_neuron,
)
import math
from itertools import combinations


class Exporter:
    def __init__(self, engine, metadata):
        self.engine = engine
        self.metadata = metadata
        self._raw_data = None
        self._processed_data = None

    def _get_raw_data(self):
        ...

    def _process_data(self, raw_data):
        return raw_data

    @property
    def raw_data(self):
        if self._raw_data is None:
            self._raw_data = self._get_raw_data()
        return self._raw_data

    @property
    def processed_data(self):
        if self._processed_data is None:
            self._processed_data = self._process_data(self.raw_data)
        return self._processed_data


class NeuronsExporter(Exporter):
    def _get_raw_data(self):
        return load_neurons(self.engine, self.metadata).rename(
            columns={"id": "neuron_id"}
        )

    def _process_data(self, raw_data):
        df_spikes = load_spiketimes(self.engine, self.metadata, block_name="pre")
        waveforms = load_waveforms(self.engine, self.metadata)
        peaks = waveform_peaks_by_neuron(
            waveforms,
            neuron_col="neuron_id",
            index_col="waveform_index",
            value_col="waveform_value",
        ).dropna()
        width = waveform_width_by_neuron(peaks, peak_names=["initiation", "ahp"])
        peak_asym = peak_asymmetry_by_neuron(peaks, peak_names=["initiation", "ahp"])
        mfr = mean_firing_rate_by(
            df_spikes, spiketimes_col="spiketimes", spiketrain_col="neuron_id"
        )
        cv_isi = cv2_isi_by(
            df_spikes, spiketimes_col="spiketimes", spiketrain_col="neuron_id"
        )
        return width.merge(peak_asym).merge(mfr).merge(cv_isi).merge(raw_data)


class WaveformsExporter(Exporter):
    def _get_raw_data(self):
        return load_waveforms(self.engine, self.metadata)

    def _process_data(self, raw_data):
        df_spikes = load_spiketimes(self.engine, self.metadata, block_name="pre")
        waveforms = load_waveforms(self.engine, self.metadata)
        peaks = waveform_peaks_by_neuron(
            waveforms,
            neuron_col="neuron_id",
            index_col="waveform_index",
            value_col="waveform_value",
        ).dropna()
        width = waveform_width_by_neuron(peaks, peak_names=["initiation", "ahp"])
        peak_asym = peak_asymmetry_by_neuron(peaks, peak_names=["initiation", "ahp"])
        mfr = mean_firing_rate_by(
            df_spikes, spiketimes_col="spiketimes", spiketrain_col="neuron_id"
        )
        cv_isi = cv2_isi_by(
            df_spikes, spiketimes_col="spiketimes", spiketrain_col="neuron_id"
        )
        return width.merge(peak_asym).merge(mfr).merge(cv_isi).merge(raw_data)


class DistanceExporter(Exporter):
    @staticmethod
    def _distance_between_chans(ch1, ch2):
        """
        Calculate distances between two channels on a cambridge neurotech 32 channel P series probe.

        Electrode spec:
            2 shanks 250um appart
            Each shank has 16 channels in two columns of 8, spaced 22.5um appart
            Contacts are placed 25um above eachother
        """
        # Shank
        shank_1 = 1 if ch1 <= 15 else 2
        shank_2 = 1 if ch2 <= 15 else 2
        width = 250 if shank_1 != shank_2 else 0

        # Column
        col_1 = 1 if ch1 % 2 == 0 else 2
        col_2 = 1 if ch2 % 2 == 0 else 2
        width = 22.5 if (col_1 != col_2) and (width == 0) else width

        #
        ch1t = ch1 - 16 if ch1 > 15 else ch1
        ch2t = ch2 - 16 if ch2 > 15 else ch2
        height = abs(ch1t - ch2t) * 25

        return math.hypot(height, width) if width else height

    def _get_raw_data(self):
        return load_neurons(self.engine, self.metadata)

    def _process_data(self, raw_data):
        dfs = []
        for session_name in raw_data.session_name.unique():
            neurons = raw_data.loc[lambda x: x.session_name == session_name]
            combs = combinations(neurons.id, r=2)
            c1s, c2s = [], []
            for comb in combs:
                c1s.append(comb[0])
                c2s.append(comb[1])
            by_comb = pd.DataFrame({"neuron1": c1s, "neuron2": c2s})
            by_comb = (
                by_comb.merge(
                    neurons[["id", "channel"]], left_on="neuron1", right_on="id"
                )
                .drop("id", axis=1)
                .rename(columns={"channel": "channel_n1"})
            )
            by_comb = (
                by_comb.merge(
                    neurons[["id", "channel"]], left_on="neuron2", right_on="id"
                )
                .drop("id", axis=1)
                .rename(columns={"channel": "channel_n2"})
            )
            by_comb["distance"] = by_comb.apply(
                lambda x: self._distance_between_chans(x.channel_n1, x.channel_n2),
                axis=1,
            )
            by_comb["session_name"] = session_name
            dfs.append(by_comb)
        return pd.concat(dfs)


class RecordingsExporter(Exporter):
    def _get_raw_data(self):
        return select_recording_sessions(
            self.engine, self.metadata, group_names=get_group_names()
        )

    def _process_data(self, raw_data):
        return raw_data.rename(columns={"id": "session_id"})


class BlockExporter(Exporter):
    def __init__(self, engine, metadata, block=None):
        super().__init__(engine=engine, metadata=metadata)
        self.block = block
        self.t_before = 0 if self.block == "pre" else 600
        self.align_to_block = False if self.block == "pre" else True


class EventsExporter(BlockExporter):
    FS = 30000

    def _get_raw_data(self):
        return select_discrete_data(
            self.engine,
            self.metadata,
            group_names=get_group_names(),
            block_name=self.block,
            align_to_block=self.align_to_block,
        )

    def _process_data(self, raw_data):
        return raw_data.assign(
            event_s=lambda x: x["timepoint_sample"].divide(EventsExporter.FS)
        )


class SpikesExporter(BlockExporter):
    FS = 30000

    def _get_raw_data(self):
        return select_spike_times(
            self.engine,
            self.metadata,
            block_name=self.block,
            group_names=get_group_names(),
            t_before=self.t_before,
            align_to_block=self.align_to_block,
        )

    def _process_data(self, raw_data):
        return raw_data.assign(
            spiketimes=lambda x: x["spike_time_samples"].divide(SpikesExporter.FS)
        )


class EEGExporter(BlockExporter):
    signal = "eeg_occ"

    def _get_raw_data(self):
        try:
            df = select_stft(
                self.engine,
                self.metadata,
                group_names=get_group_names(),
                signal_names=[EEGExporter.signal],
                align_to_block=self.align_to_block,
                t_before=self.t_before,
            )
        except IndexError:
            df = pd.DataFrame()
        return df

    def get_band_ts(self):
        return self._mean_band_psd_ts(self._assign_band(self.processed_data))

    @staticmethod
    def _assign_band(df_fft, freq_col="frequency"):
        """
        Filter freqs between 0 - 8 Hz and assign each freq to
        delta or theta band
        """
        return df_fft.loc[lambda x: (x[freq_col] < 8) & (x[freq_col] > 0)].assign(
            band=lambda x: x[freq_col].apply(lambda y: "delta" if y < 4 else "theta")
        )

    @staticmethod
    def _mean_band_psd_ts(
        df_fft,
        sig_col="signal_name",
        sesh_col="session_name",
        time_col="timepoint_s",
        group_col="group_name",
        band_col="band",
        psd_col="fft_value",
    ):
        """
        Create df with one row per timepoint per band and the mean
        power in the band at that timepoint.
        """
        return (
            df_fft.groupby([sesh_col, sig_col, time_col, group_col, band_col])
            .apply(lambda x: np.mean(x[psd_col]))
            .reset_index()
            .rename(columns={0: "psd"})
            .pivot_table(index=[sesh_col, time_col], columns=band_col, values="psd")
            .reset_index()
            .assign(
                delta_smooth=lambda x: gaussian_filter1d(x.delta, 5),
                theta_smooth=lambda x: gaussian_filter1d(x.theta, 5),
                delta_to_theta=lambda x: x.delta / x.theta,
                delta_to_theta_smooth=lambda x: x.delta_smooth / x.theta_smooth,
            )
            .rename(columns={time_col: "time"})
            .assign(time=lambda x: x["time"] - 2)
        )


class LFPExporter(BlockExporter):
    signal = "lfp_lr"

    def _get_raw_data(self):
        try:
            df = select_stft(
                self.engine,
                self.metadata,
                group_names=get_group_names(),
                signal_names=[LFPExporter.signal],
                align_to_block=self.align_to_block,
                t_before=self.t_before,
            )
        except IndexError:
            df = pd.DataFrame()
        return df

    def get_band_ts(self):
        return self._mean_band_psd_ts(self._assign_band(self.processed_data))

    @staticmethod
    def _assign_band(df_fft, freq_col="frequency"):
        """
        Filter freqs between 0 - 8 Hz and assign each freq to
        delta or theta band
        """
        return df_fft.loc[lambda x: (x[freq_col] < 8) & (x[freq_col] > 0)].assign(
            band=lambda x: x[freq_col].apply(lambda y: "delta" if y < 4 else "theta")
        )

    @staticmethod
    def _mean_band_psd_ts(
        df_fft,
        sig_col="signal_name",
        sesh_col="session_name",
        time_col="timepoint_s",
        group_col="group_name",
        band_col="band",
        psd_col="fft_value",
    ):
        """
        Create df with one row per timepoint per band and the mean
        power in the band at that timepoint.
        """
        return (
            df_fft.groupby([sesh_col, sig_col, time_col, group_col, band_col])
            .apply(lambda x: np.mean(x[psd_col]))
            .reset_index()
            .rename(columns={0: "psd"})
            .pivot_table(index=[sesh_col, time_col], columns=band_col, values="psd")
            .reset_index()
            .assign(
                delta_smooth=lambda x: gaussian_filter1d(x.delta, 5),
                theta_smooth=lambda x: gaussian_filter1d(x.theta, 5),
                delta_to_theta=lambda x: x.delta / x.theta,
                delta_to_theta_smooth=lambda x: x.delta_smooth / x.theta_smooth,
            )
            .rename(columns={time_col: "time"})
            .assign(time=lambda x: x["time"] - 2)
        )

# TODO add exporter for raw LFP and EEG
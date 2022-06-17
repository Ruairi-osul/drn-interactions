from .spikes import SpikesHandler, SpikesHandlerMulti


def shuffle_spikes(sh: SpikesHandler) -> SpikesHandler:
    return SpikesHandler(
        block=sh.block,
        bin_width=sh.bin_width,
        session_names=sh.session_names,
        t_start=sh.t_start,
        t_stop=sh.t_stop,
        shuffle=True,
    )


def shuffle_spikes_multi(shm: SpikesHandlerMulti) -> SpikesHandlerMulti:
    return SpikesHandlerMulti(
        block=shm.block,
        bin_width=shm.bin_width,
        session_names=shm.session_names,
        t_start=shm.t_start,
        t_stop=shm.t_stop,
        shuffle=True,
    )

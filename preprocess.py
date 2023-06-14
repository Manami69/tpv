import matplotlib.pyplot as plt
from mne import Epochs, pick_types, events_from_annotations
import mne
from utils import load_filter_dataset


def plot_datas(raw):
    """plot data as asked in the subject
    there is more ways to plot and study data with the mme library"""
    event_id = dict(left=1, right=2)
    events, _ = events_from_annotations(raw, event_id=dict(T1=1, T2=2))
    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False)

    # see signals for each channel
    # [fr]signal pour chaque electrode placee sur le cerveau
    raw.plot(scalings="auto")
    plt.show()

    # power of the signal by frequency and by channel
    # [fr] la repartition de la  puissance du signal en
    # fonction des bandes de frequence
    raw.compute_psd().plot(picks=picks, exclude="bads")
    plt.show()
    # TODO: [Bonus] filtered signal vue with wavelet or fourier transform
    picks_shadow = pick_types(
        raw.info, meg="grad", eeg=True, eog=False, stim=False, exclude="bads"
    )
    tmin, tmax = -20.0, 40.0  # Y-axis
    baseline = (None, 0)
    epochs = Epochs(
        raw,
        events,
        event_id,
        tmin,
        tmax,
        picks=picks_shadow,
        baseline=baseline,
        reject=dict(),
        preload=True,
    )
    epochs.compute_psd(fmin=0.0, fmax=80.0).plot(
        average=True, picks=picks_shadow, exclude="bads"
    )
    plt.show()
    # events through time
    fig = mne.viz.plot_events(
        events, sfreq=raw.info["sfreq"], first_samp=raw.first_samp, event_id=event_id
    )
    fig.subplots_adjust(right=0.7)  # to see the legend


def main():
    raw = load_filter_dataset()
    plot_datas(raw)


if __name__ == "__main__":
    try:
        main()
    except Exception as msg:
        print(f"Error: {msg}")

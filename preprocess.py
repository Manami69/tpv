import matplotlib.pyplot as plt
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.channels import make_standard_montage
from mne import Epochs, pick_types, events_from_annotations


def load_filter_dataset(subject=1, runs=[3, 7, 11], Hz_min=7.0, Hz_max=30):
    # mne library loader of datas
    raw_fnames = eegbci.load_data(subject, runs, path="./")
    raws = [read_raw_edf(f, preload=True) for f in raw_fnames]
    raw = concatenate_raws(raws)
    eegbci.standardize(raw)
    # set color for each channel
    montage = make_standard_montage("standard_1005")
    raw.set_montage(montage)
    raw.filter(7.0, 30.0, fir_design="firwin")
    return raw


def plot_datas(raw):
    events, event_id = events_from_annotations(raw, event_id=dict(T1=0, T2=1))
    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False)

    # see signals for each channel
    # fr signal pour chaque electrode placee sur le cerveau
    raw.plot(scalings="auto")
    plt.show()
    # fr la repartition de la  puissance du signal en
    # fonction des bandes de frequence
    raw.compute_psd().plot(picks=picks, exclude="bads")
    plt.show()

    # la meme une fois que le signal a ete filtre
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


def main():
    raw = load_filter_dataset()
    plot_datas(raw)


if __name__ == "__main__":
    main()

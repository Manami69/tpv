from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne import Epochs, pick_types
from mne.channels import make_standard_montage
import mne


def load_filter_dataset(subject=1, runs=[3, 7, 11], freq_min=7.0, freq_max=30):
    """Load  and filter dataset for one subject for defined runs

default runs are ones associated with `task 1` : \
(opening and closing left or right fist)"""
    # mne library loader of datas
    raw_fnames = eegbci.load_data(subject, runs, path="./", verbose="ERROR")
    raws = [read_raw_edf(f, preload=True) for f in raw_fnames]
    raw = concatenate_raws(raws)
    eegbci.standardize(raw)
    # set color for each channel
    montage = make_standard_montage("standard_1005")
    raw.set_montage(montage)
    raw.filter(freq_min, freq_max, fir_design="firwin", skip_by_annotation="edge")
    return raw


def get_filtered_events(raw, tmin=-1.0, tmax=2.0, freq_min=7.0, freq_max=30):
    """get events from dataset

See T1 and T2's motions for defined task

By default we'll use `task 1` motions :
 - `T1` => closing then opening left fist
 - `T2` => closing then opening right fist"""
    raw.filter(freq_min, freq_max, fir_design="firwin", skip_by_annotation="edge")
    events, event_id = mne.events_from_annotations(raw,
                                                   event_id=dict(T1=1, T2=2))
    picks = pick_types(raw.info, meg=False, eeg=True,
                       stim=False, eog=False, exclude="bads")
    # Read epochs (train will be done only between 1 and 2s)
    # Testing will be done with a running classifier
    epochs = Epochs(
        raw,
        events,
        event_id,
        tmin,
        tmax,
        proj=True,
        picks=picks,
        baseline=None,
        preload=True,
    )
    epochs_train = epochs.copy().crop(tmin=tmin, tmax=tmax)
    epochs_data_train = epochs_train.get_data()
    return (epochs_data_train, epochs.events[:, -1] - 1)

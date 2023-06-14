from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.channels import make_standard_montage
import mne


def load_filter_dataset(subject=1, runs=[3, 7, 11], freq_min=7.0, freq_max=30):
    """Load  and filter dataset for one subject for defined runs

default runs are ones associated with `task 1` : \
(opening and closing left or right fist)"""
    # mne library loader of datas
    raw_fnames = eegbci.load_data(subject, runs, path="./")
    raws = [read_raw_edf(f, preload=True) for f in raw_fnames]
    raw = concatenate_raws(raws)
    eegbci.standardize(raw)
    # set color for each channel
    montage = make_standard_montage("standard_1005")
    raw.set_montage(montage)
    raw.filter(freq_min, freq_max, fir_design="firwin")
    return raw


def get_events(raw, event_id=dict(left=1, right=2)):
    """get events from dataset

See T1 and T2's motions for defined task

By default we'll use `task 1` motions :
 - `T1` => closing then opening left fist
 - `T2` => closing then opening right fist"""
    events, _ = mne.events_from_annotations(raw, event_id=dict(T1=1, T2=2))
    return (events, event_id)

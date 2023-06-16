#%%
import mne
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.channels import make_standard_montage


#%%
# mne library loader of datas
raw_fnames = eegbci.load_data(1, [3, 7, 11], path="./")
raws = [read_raw_edf(f, preload=True) for f in raw_fnames]
raw = concatenate_raws(raws)
eegbci.standardize(raw)
# set color for each channel
montage = make_standard_montage("standard_1005")
raw.set_montage(montage)
raw.filter(7.0, 30., fir_design="firwin")


# %%
event_id=dict(left=1, right=2)
mne.events_from_annotations(raw, event_id=dict(T1=1, T2=2))
# %%
test = raw._data

# %%

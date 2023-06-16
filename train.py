from utils import load_filter_dataset, get_events
from mne import Epochs, pick_types, events_from_annotations

## STEPS OF 
# dimensionalilty reduction algorithm (homemade)
# classification algorithm (with sklearn)
# Playback reading on the file to simulate a data stream
# You have to use the pipeline object
# from sklearn (use baseEstimator and transformer-Mixin
# classes of sklearn)


def main():
    #%%
    datas = load_filter_dataset()
    events, event_id = get_events(datas)
    picks = pick_types(datas.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude="bads")
    tmin, tmax = -20.0, 40.0  # Y-axis
    #%%
    epochs = Epochs(
        datas,
        events,
        event_id,
        tmin,
        tmax,
        proj=True,
        picks=picks,
        baseline=None,
        preload=True,
    )
    epochs_train = epochs.copy().crop(tmin, tmax)
    labels = epochs.events[:, -1] - 2
    print(events)



if __name__ == "__main__":
    try:
        main()
    except Exception as msg:
        print(f"Error: {msg}")

# %%

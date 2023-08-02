#%%
import mne
from mne import Epochs, pick_types, events_from_annotations
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.channels import make_standard_montage
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score
import numpy as np


#%%
# mne library loader of datas
tmin, tmax = -1.0, 4.0
raw_fnames = eegbci.load_data(1, [3, 7, 11], path="./")
raws = [read_raw_edf(f, preload=True) for f in raw_fnames]
raw = concatenate_raws(raws)
eegbci.standardize(raw)
# set color for each channel
montage = make_standard_montage("standard_1005")
raw.set_montage(montage)
# Apply band-pass filter
raw.filter(7.0, 30., fir_design="firwin")


# %%
event_id = dict(T1=1, T2=2)
events, _ = events_from_annotations(raw, event_id=dict(T1=1, T2=2))
picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                   exclude="bads")
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
epochs_train = epochs.copy().crop(tmin=1.0, tmax=2.0)
Y = epochs.events[:, -1] - 1

# %%
epochs_data = epochs.get_data()
epochs_data_train = epochs_train.get_data()

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(epochs_data_train, Y, test_size=0.2, random_state=42)
# Créer la pipeline avec CSP comme prétraitement et LDA comme modèle de classification
pipeline = Pipeline([
    ('CSP', CSP(n_components=4, reg=None, log=True, norm_trace=False)),  # Prétraitement CSP
    ('LDA', LinearDiscriminantAnalysis())  # Modèle de classification LDA
])

# Entraîner le modèle avec la validation croisée
scores = cross_val_score(pipeline, X_train, y_train, cv=5)  # 5-fold cross-validation

# Afficher les scores de validation croisée
print("Scores de validation croisée:", scores)
print("Score moyen:", np.mean(scores))

# Entraîner le modèle final sur l'ensemble d'entraînement complet
pipeline.fit_transform(X_train, y_train)

# Faire des prédictions sur l'ensemble de test
predictions = pipeline.predict(X_test)

for index, value in enumerate(predictions):
    print(f"predicted [{value}] - real [{y_test[index]}]")
# Évaluer les performances du modèle sur l'ensemble de test
accuracy = np.mean(predictions == y_test)
print("Précision sur l'ensemble de test:", accuracy)
print(pipeline.score(X_train, y_train))
print(pipeline.score(X_test, y_test))
# %%

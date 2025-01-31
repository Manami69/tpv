# TOTAL PERSPECTIVE VORTEX

## Python 3.11 with mne lib
### install conda
[install conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html)
### install mne
```bash
conda create --strict-channel-priority --channel=conda-forge --name=mne mne-base
```
### launch env
```bash
# To activate this environment,use
conda activate mne
# To deactivate an active environment, use
conda deactivate
```

## dataset
 - [dataset link](https://physionet.org/content/eegmmidb/1.0.0/)
 - [load this dataset with mne](https://mne.tools/stable/overview/datasets_index.html#eegbci-motor-imagery)

## infos

## Links 
 - [useful video for common spatial pattern](https://www.youtube.com/watch?v=EAQcu6DLAS0)
 - [scikit-learn Baseestimator and TransformerMixin](https://sklearn-template.readthedocs.io/en/latest/user_guide.html)
 - [same](https://medium.com/mlearning-ai/workflow-to-build-sklearn-pipelines-54abffddccb1)
 - [csp mne source code](https://github.com/mne-tools/mne-python/blob/main/mne/decoding/csp.py)
 - [cross validation explanation](https://datascientest.com/cross-validation)
 - [brain explosion](https://www.scielo.org.mx/scielo.php?pid=S0035-001X2022000400012&script=sci_arttext)
 - [choose classifier](https://neuro.inf.unibe.ch/AlgorithmsNeuroscience/Tutorial_files/ApplyingMachineLearningMethods_1.html)

 # CSP
voir these page 74 et suivantes
- on separe les epochs en classes (une tache = une classe)
- on passe la matrice 3D (epoch, channels, time) en matrice 2D (channel, epoch * time) 
- on compute toutes les matrices de covariance pour chaque epoch et on fait une moyennede ces matrices par classe
- on va chercher les filtres spaciaux qui maximisent une classe et minimise l'autre. Onva chercher les n valeurs propres les plus grandes.
on applique le filtre CSP sur nos donnees. Pour pouvoir utiliser l'algorithme declassification, on va garder la puissance moyenne (average power) du resultat pouravoir une matrice 2D.
 ## Sources
 - ["Réduction de dimension en apprentissage supervisé : applications à l’étude de l’activité cérébrale", Laurent Vezard, 2013](https://www.theses.fr/2013BOR15005)
 - [maths -> variance, covariance, vecteur et valeur propres](https://courses.cs.washington.edu/courses/csep546/16sp/slides/PCA_csep546.pdf)
 - [changement d'axe de matrices avec schema](https://stackoverflow.com/questions/32034237/how-does-numpys-transpose-method-permute-the-axes-of-an-array)


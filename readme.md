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
>[total-perspective-vortex] Hello, I have trouble understanding what "You have to use the pipeline object from sklearn (use baseEstimator and transformerMixin classes of sklearn)" means from V.1.3 Implementation. If anyone could explain it to me I would greatly appreciate it.

>>Hello, use these two functions to create your pipeline
https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html

## Links 
 - [useful video for common spatial pattern](https://www.youtube.com/watch?v=EAQcu6DLAS0)
 - [scikit-learn Baseestimator and TransformerMixin](https://sklearn-template.readthedocs.io/en/latest/user_guide.html)
 - [cross validation explanation](https://datascientest.com/cross-validation)
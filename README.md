# PyQSOFitSpec

## A code to fit the spectrum of quasar


## What's new
This repo is adapted from [PyQSOFit](https://github.com/legolason/PyQSOFit) and [PyQSOFit_SBL](https://github.com/JackHon55/PyQSOFit_SBL). The main modification includes fitting the spectral line with a different line fitting model. The spectrum is bootstrapped from the flux and error and refitted at each iteration. The mean and standard deviation are calculated to yield the output line properties. For other updates, see `CHANGELOG.md`.

The code is used in Yong et al. (2022, submitted) to perform spectral line analysis on orientation selected SDSS samples. <!-- TODO add link -->


## Description from PyQSOFit
We provide a brief guide of the Python QSO fitting code (PyQSOFit) to measure spectral properties of SDSS quasars. The code is currently transferred from Yue's IDL to Python. The package includes the main routine, Fe II templates, an input line-fitting parameter list, host galaxy templates, and dust reddening map to extract spectral measurements from the raw fits. Monte Carlo estimation of the measurement uncertainties of the fitting results can be conducted with the same fitting code.

The code takes an input spectrum (observed-frame wavelength, flux density and error arrays) and the redshift as input parameters, performs the fitting in the rest frame, and outputs the best-fit parameters and quality-checking plots to the paths specified by the user.

The code uses an input line-fitting parameter list to specify the fitting range and parameter constraints of the individual emission line components. An example of such a file is provided in the example.ipynb. Within the code, the user can switch on/off components to fit to the pseudo-continuum. For example, for some objects the UV/optical Fe II emission cannot be well constrained and the user may want to exclude this component in the continuum fit. The code is highly flexible and can be modified to meet the specific needs of the user.


## Usage
- To run:
  ```shell
  $ python main.py <PATH_FITSFILE>
  ```
  - `<PATH_FITSFILE>`: Optional path to fits file or directory. Assume fits file in current directory if none provided.
- Set configurations in `main.py`. Some selected configs:
  - `nboot_fit`: Number of bootstrap to fit individual lines
  - `linemodel`: str of line model define in `fit_linemodel.py`
- Define other line fitting model in `fit_linemodel.py`
- Output results: `pyqsofitspec_out.csv`
  - The fitted measurements are saved per file, and hence can be resumed if there is any error. Make a copy of the output result file and rerun.
- Warning: Ensure that number of files and `nboot_fit` are reasonable, else will encounter OSError due to too many open files.


## Example
Some spectra are in [example](https://github.com/yongsukyee/PyQSOFitSpec/tree/main/example) folder. In the following example, the broad lines are fitted with single skewed normal. To reproduce:
```shell
$ python main.py example
```


## Related materials
- Paper: Yong et al. (2022, submitted) <!-- TODO add link -->
- Repos where this code is adapted from:
  - [PyQSOFit](https://github.com/legolason/PyQSOFit)
  - [PyQSOFit_SBL](https://github.com/JackHon55/PyQSOFit_SBL)
- Other demos (accessed on 18 June 2022):
  - [archive/example_PyQSOFit](https://github.com/yongsukyee/PyQSOFitSpec/tree/main/archive/example_PyQSOFit): from [PyQSOFit example](https://github.com/legolason/PyQSOFit/tree/master/example)
  - [archive/example_PyQSOFit_SBL](https://github.com/yongsukyee/PyQSOFitSpec/tree/main/archive/example_PyQSOFit_SBL): from [PyQSOFit_SBL example_sdss](https://github.com/JackHon55/PyQSOFit_SBL/tree/master/example_sdss)


## TODO
- [ ] In PyQSOFitSpec.py:
  - [ ] output line_prop as dict for easy access
  - [ ] change color scheme for decomposition_host, BC
- [ ] In main.py:
  - [ ] could improve speed by running in parallel
  - [ ] tidy up and separate configs if nboot_fit
  - [ ] change output of run_qsofit so that don't need to manually include/exclude fitted lines
  - [x] add command line arguments for fits file path or directory


## Cite the original authors of the PyQSOFit code
```bibtex
@MISC{2018ascl.soft09008G,
       author = {{Guo}, Hengxiao and {Shen}, Yue and {Wang}, Shu},
        title = "{PyQSOFit: Python code to fit the spectrum of quasars}",
     keywords = {Software},
 howpublished = {Astrophysics Source Code Library, record ascl:1809.008},
         year = 2018,
        month = sep,
          eid = {ascl:1809.008},
        pages = {ascl:1809.008},
archivePrefix = {ascl},
       eprint = {1809.008},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2018ascl.soft09008G},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```


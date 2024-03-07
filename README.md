# MUWCLASS_CSCv2
 
## Classifying Unidentified X-ray Sources in the Chandra Source Catalog Using a Multi-wavelength Machine Learning Approach

### Hui Yang<sup>1</sup>, Jeremy Hare<sup>2</sup>, Oleg Kargaltsev<sup>1</sup>, Steven Chen<sup>1</sup>, Igor Volkov<sup>1</sup>,  Blagoy Rangelov<sup>3</sup>, Yichao Lin<sup>1</sup>,
<sup>1</sup>The George Washington University <sup>2</sup>NASA GSFC <sup>3</sup>Texas State University

## CHECK our [MUWCLASS paper](https://ui.adsabs.harvard.edu/abs/2022ApJ...941..104Y/abstract)
## Related papers: [NGC 3532](https://ui.adsabs.harvard.edu/abs/2023ApJ...948...59C/abstract), [13 FGL-LAT source](https://ui.adsabs.harvard.edu/abs/2024ApJ...961...26R/abstract), [Visualization tool](https://ui.adsabs.harvard.edu/abs/2021RNAAS...5..102Y/abstract), [4XMM-DR13 TD](https://ui.adsabs.harvard.edu/abs/2024arXiv240215684L/abstract)

### contact huiyang@gwu.edu if you have any questions

--- 

This github repo provides the MUltiWavelength Machine Learning CLASSification Pipeline (MUWCLASS) and the classification results on the Chandra Source Catalog v2 (CSCv2).

The main components of this github repo are

demos/
- There are notebooks of demonstrations of classifying CSCv2 sources using MUWCLASS with CSCv2 and multiwavelength data

files/{CSC_TD_11042023_MW_allcolumns.csv, tbabs.data}
- Some other CSV files including the raw training dataset with more properties (CSC_TD_MW_remove.csv), the photoelectric absorption cross-section file (tbabs.data).

--- 

## Softwares Installation Instruction (using conda)

### simple Python 3.9 environment (without CIAO), this version turn offs the plotting functions of X-ray images

* run the follow code to create a new conda environment muwclass; if you already have Python 3.9, you can use your own conda environment with additional Python packages installed from below

* conda create -n muwclass python=3.9

* run 'bash install-packages.sh' under ciao environment to install all required packages 

#### other required packages 

* run 'bash install-packages.sh' under ciao-4.14-muwclass environment to install all required packages 

* then, make sure to enable widgetsnbextension and ipyaladin, run 
* jupyter nbextension enable --py widgetsnbextension
* jupyter nbextension enable --py --sys-prefix ipyaladin
- on your terminal 

* You might also need to manually register the existing ds9 with the xapns name server by selecting the ds9 File->XPA->Connect menu option so your ds9 will be fully accessible to pyds9.

* You need to install gcc to properly install some of the packages

* If you have problems importing some packages, try to pip install the packages that fail to import manually under the conda environment

* Currently there is an issue with ipyaladin package (Failed to load model class 'ModelAladin' from module 'ipyaladin')
 





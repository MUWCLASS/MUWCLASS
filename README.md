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

* clone the [MUWCLASS package](https://github.com/MUWCLASS/MUWCLASS) to your local desktop

* run the follow code to create a new conda environment muwclass; if you already have Python 3.9, you can use your own conda environment with additional Python packages installed from below

* conda create -n muwclass python=3.9

* run 'bash install-packages.sh' under muwclass environment to install all required packages 

* clone (NOT pip install) the [NWAY package](https://github.com/JohannesBuchner/nway) to your local desktop and change the nway_dir variable in nway_match.py line 25 to the directory where you clone the nway package

 





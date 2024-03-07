#!/bin/bash

# --------------------------------------------------------------#
# Script to install packages for MUWCLASS working environment   #
#---------------------------------------------------------------#
#Autor: Hui Yang		                              	#
#Date: 01/20/2022			                        #
#                                                               #
# INSTRUCTIONS: When you run this script, make sure you         #
# ------------------------------------------------------------- #


# ----------------- Python 3.x ------------------------------------
conda install notebook
conda install scikit-learn-intelex
conda install astropy
conda install conda-forge::pyvo
conda install numpy
conda install pandas
conda install -c conda-forge astroquery
conda install -c astropy astroML
pip install gdpyc
conda install -c conda-forge extinction
conda install -c conda-forge imbalanced-learn
conda install bokeh
conda install colorcet
conda install healpy
conda install -c conda-forge tqdm

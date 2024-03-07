#!/usr/bin/env python
# coding: utf-8

# version 1.0

# from sklearnex import patch_sklearn
# patch_sklearn()

from sklearn import tree
import colorcet as cc
from bokeh.transform import dodge, linear_cmap
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, FuncTickFormatter
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from gdpyc import GasMap, DustMap
from astropy.coordinates import SkyCoord, Angle
from astropy import units as u
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy.special as sc
import extinction
from imblearn.over_sampling import SMOTE, KMeansSMOTE, ADASYN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import json
from random import randint
import matplotlib.pyplot as plt
from collections import Counter
# import lightgbm as lgb
import glob


from scipy.signal import find_peaks

import pylab
from scipy.optimize import least_squares
from astroquery.vizier import Vizier
from astroquery.gaia import Gaia
Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"  # Reselect Data Release 3, default
Gaia.ROW_LIMIT = -1 # Ensure the default row limit.


import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

import shap

from physical_oversampling import physical_oversample_csv, test_reddening_grid_csv
from nway_match import CSCviewsearch

# from cuml import RandomForestClassifier as cuRF

MW_names = {'gaia':  ['Gmag', 'BPmag', 'RPmag'],
            '2mass': ['Jmag', 'Hmag', 'Kmag'],
            'wise':  ['W1mag', 'W2mag', 'W3mag'],
            'glimpse': ['3.6mag', '4.5mag', '5.8mag', '8.0mag'],
            'hugs': ['F275W', 'F336W', 'F438W', 'F606W', 'F814W'],
            }

gaia_features = ['Gmag', 'BPmag', 'RPmag']
gaia_limits = [21.5,   21.5,   21.]
gaia_zeros = [2.5e-9, 4.08e-9, 1.27e-9]  # [3228.75, 3552.01, 2554.95]
gaia_eff_waves = [5822.39, 5035.75, 7619.96]
gaia_width_waves = [4052.97, 2157.50, 2924.44]
twomass_features = ['Jmag', 'Hmag', 'Kmag']
twomass_limits = [18.5,   18.0,  17.0]
twomass_zeros = [3.13e-10, 1.13e-10, 4.28e-11]  # [1594.,  1024., 666.7]
twomass_eff_waves = [12350., 16620., 21590.]
twomass_width_waves = [1624.32, 2509.40, 2618.87]
wise_features = ['W1mag', 'W2mag', 'W3mag']
wise_limits = [18.5,   17.5, 14.5]
wise_zeros = [8.18e-12, 2.42e-12, 6.52e-14]  # [309.54, 171.787, 31.674]
wise_eff_waves = [33526, 46028, 115608]
wise_width_waves = [6626.42, 10422.66, 55055.71]  # [34000., 46000., 120000.]

# WFC3/UVIS F275W, F336W, F438W; ACS/WFC F606W, F814W
hugs_features = ['F275W', 'F336W', 'F438W', 'F606W', 'F814W']
hugs_limits = [28]*5
hugs_zeros = [3.79e-9, 3.31e-9, 6.77e-9, 2.91e-9, 1.13e-9]
hugs_eff_waves = [2720.03, 3358.95, 4323.35, 5809.26, 7973.39]
hugs_width_waves = [423.76, 512.40, 589.21, 1771.88, 1888.66]


MW_features = gaia_features + twomass_features + wise_features
MW_limits = gaia_limits + twomass_limits + wise_limits  # limiting magnitudes
# zero points to convert magnitude to flux in wavelength space
MW_zeros = gaia_zeros + twomass_zeros + wise_zeros
MW_width_waves = gaia_width_waves + twomass_width_waves + \
    wise_width_waves    # effective wavelength widths
MW_eff_waves = gaia_eff_waves + twomass_eff_waves + \
    wise_eff_waves    # effective wavelength

CSC_flux_features = ['Fcsc_s', 'Fcsc_m', 'Fcsc_h', 'Fcsc_b']
CSC_HR_features = ['HR_hm', 'HR_ms', 'HR_hms']
CSC_var_features = ['var_inter_prob', 'var_intra_prob']

CSC_features = CSC_flux_features + CSC_HR_features + CSC_var_features
# CSC_features = list(set(CSC_features) - set(['Fcsc_b']))

# colors = ['G-BP','G-RP','G-J','G-H','G-K','BP-RP','RP-J','J-H','J-K','H-K','W1-W2','W1-W3','W2-W3']
# colors_v08262022 = ['G-BP', 'G-RP', 'G-J', 'G-H', 'G-K', 'BP-RP','J-H', 'J-K', 'H-K', 'W1-W2', 'W1-W3', 'W2-W3']
# colors_before05012023 =['G-BP','G-RP','J-H','J-K','H-K','H-W2','K-W1','K-W2','W1-W2','W1-W3', 'W2-W3']
# colors_physical_v1 = ['G-BP', 'G-RP', 'J-H', 'J-K', 'H-K','H-W1','H-W2','K-W1','K-W2', 'W1-W2', 'W1-W3', 'W2-W3']#
colors = ['G-BP', 'G-RP', 'G-J', 'G-H', 'G-K', 'BP-RP',
          'J-H', 'J-K', 'H-K', 'W1-W2', 'W1-W3', 'W2-W3']

colors_all = ['G-BP', 'G-RP', 'G-J', 'G-H', 'G-K', 'G-W1', 'G-W2', 'G-W3', 'BP-RP', 'BP-J', 'BP-H', 'BP-K', 'BP-W1', 'BP-W2', 'BP-W3', 'RP-J', 'RP-H',
              'RP-K', 'RP-W1', 'RP-W2', 'RP-W3', 'J-H', 'J-K', 'J-W1', 'J-W2', 'J-W3', 'H-K', 'H-W1', 'H-W2', 'H-W3', 'K-W1', 'K-W2', 'K-W3', 'W1-W2', 'W1-W3', 'W2-W3']

# CSC_all_features = CSC_features + MW_features + colors
CSC_all_features = CSC_features + gaia_features + \
    twomass_features + ['W1mag', 'W2mag'] + colors
# CSC_all_features = CSC_features + list(set(MW_features) - set(['W3mag'])) + colors
CSC_all_features_test = CSC_features + MW_features + colors_all

XMM_all_features = CSC_flux_features + CSC_HR_features + MW_features + colors

Flux_features = CSC_flux_features + MW_features

# dictionary for features based on distance, first item is the distance feature, the rest are luminosity features
dist_features_dict = {'nodist': [],
                      #   'rgeo': ['rgeo'],
                      #   'rpgeo':  ['rpgeo'],
                      #   'plx': ['Plx_dist'],
                      'rgeo_lum': ['rgeo', 'Lcsc_b', 'Gmag_lum', 'Jmag_lum'],
                      'rpgeo_lum': ['rpgeo', 'Lcsc_b', 'Gmag_lum', 'Jmag_lum'],
                      'plx_lum': ['Plx_dist', 'Lcsc_b', 'Gmag_lum', 'Jmag_lum'],
                      'gc_dist': ['HELIO_DISTANCE'] + [flux.replace('F', 'L') for flux in CSC_flux_features] + [mag+'_lum' for mag in hugs_features],
                      }

CSC_avgflux_prefix = 'flux_aper90_avg_'

exnum = -9999999.  # some extra-large negtive number to replace NULL

class_labels = {'AGN': 'AGN', 'NS': 'NS', 'CV': 'CV', 'HMXB': 'HMXB',
                'LMXB': 'LMXB', 'HM-STAR': 'HM-STAR', 'LM-STAR': 'LM-STAR', 'YSO': 'YSO'}
# class_labels = {'AGN':'AGN','NS':'NS','NS_BIN':'NS_BIN','CV':'CV','HMXB':'HMXB','LMXB':'LMXB','HM-STAR':'HM-STAR','LM-STAR':'LM-STAR','YSO':'YSO'}
n_classes = 8  # 9

class_colors = ['blue', 'orange', 'red', 'c',
                'g', 'purple', 'magenta', 'olive', 'Aqua']

MW_cols = {'xray': ['name', 'ra', 'dec', 'PU', 'significance', 'flux_aper90_avg_s', 'e_flux_aper90_avg_s', 'flux_aper90_avg_m', 'e_flux_aper90_avg_m', 'flux_aper90_avg_h', 'e_flux_aper90_avg_h',
                    'flux_aper90_avg_b', 'e_flux_aper90_avg_b', 'kp_prob_b_max', 'var_inter_prob'],
           'gaia': ['DR3Name_gaia', 'RA_pmcor_gaia', 'DEC_pmcor_gaia', 'Gmag_gaia', 'e_Gmag_gaia', 'BPmag_gaia', 'e_BPmag_gaia', 'RPmag_gaia', 'e_RPmag_gaia', 'GAIA_Plx', 'GAIA_e_Plx', 'rgeo_gaiadist', 'b_rgeo_gaiadist', 'B_rgeo_gaiadist', 'rpgeo_gaiadist', 'b_rpgeo_gaiadist', 'B_rpgeo_gaiadist'],
           '2mass': ['_2MASS_2mass', 'Jmag_2mass', 'e_Jmag_2mass', 'Hmag_2mass', 'e_Hmag_2mass', 'Kmag_2mass', 'e_Kmag_2mass'],
           'catwise': ['Name_catwise', 'W1mag_catwise', 'e_W1mag_catwise', 'W2mag_catwise', 'e_W2mag_catwise'],
           'unwise': ['objID_unwise', 'W1mag_unwise', 'e_W1mag_unwise', 'W2mag_unwise', 'e_W2mag_unwise'],
           'allwise': ['AllWISE_allwise', 'W1mag_allwise', 'e_W1mag_allwise', 'W2mag_allwise', 'e_W2mag_allwise', 'W3mag_allwise', 'e_W3mag_allwise', 'W4mag_allwise', 'e_W4mag_allwise'],
           'vphas': ['VPHASDR2_vphas', 'Gmag_vphas', 'RPmag_vphas', 'BPmag_vphas', 'e_Gmag_vphas', 'e_RPmag_vphas', 'e_BPmag_vphas'],
           '2mass_gaia': ['_2MASS_2mass_gaia', 'Jmag_2mass_gaia', 'e_Jmag_2mass_gaia', 'Hmag_2mass_gaia', 'e_Hmag_2mass_gaia', 'Kmag_2mass_gaia', 'e_Kmag_2mass_gaia'],
           'allwise_gaia': ['AllWISE_allwise_gaia', 'W1mag_allwise_gaia', 'e_W1mag_allwise_gaia', 'W2mag_allwise_gaia', 'e_W2mag_allwise_gaia', 'W3mag_allwise_gaia', 'e_W3mag_allwise_gaia', 'W4mag_allwise_gaia', 'e_W4mag_allwise_gaia']
           }

########################### Default Scaler  ####################################
#   default = StandardScaler to remove the mean and scale to unit variance
standscaler = StandardScaler()
ML_scaler = standscaler  # the scaler selected

scaler_switch = False  # for ML_model = RFmodel


def stats(df, flx='flux_aper90_sym_', end='.1', drop=False):
    print("Run stats......")
    df = df.fillna(exnum)
    s0 = np.where((df[flx+'h'+end] != 0) & (df[flx+'h'+end] != exnum) & (df[flx+'m'+end] != 0)
                  & (df[flx+'m'+end] != exnum) & (df[flx+'s'+end] != 0) & (df[flx+'s'+end] != exnum))[0]
    s1 = np.where((df[flx+'h'+end] != exnum) & (df['e_'+flx+'h'+end] != exnum) & (df[flx+'m'+end] != exnum)
                  & (df['e_'+flx+'m'+end] != exnum) & (df[flx+'s'+end] != exnum) & (df['e_'+flx+'s'+end] != exnum))[0]
    s2 = np.where(((df[flx+'h'+end] != exnum) & (df['e_'+flx+'h'+end] != exnum)) & ((df[flx+'m'+end] != exnum)
                  & (df['e_'+flx+'m'+end] != exnum)) & ((df[flx+'s'+end] == exnum) | (df['e_'+flx+'s'+end] == exnum)))[0]
    s3 = np.where(((df[flx+'h'+end] == exnum) | (df['e_'+flx+'h'+end] == exnum)) & ((df[flx+'m'+end] != exnum)
                  & (df['e_'+flx+'m'+end] != exnum)) & ((df[flx+'s'+end] != exnum) & (df['e_'+flx+'s'+end] != exnum)))[0]
    s4 = np.where(((df[flx+'h'+end] != exnum) & (df['e_'+flx+'h'+end] != exnum)) & ((df[flx+'m'+end] == exnum)
                  | (df['e_'+flx+'m'+end] == exnum)) & ((df[flx+'s'+end] != exnum) & (df['e_'+flx+'s'+end] != exnum)))[0]
    s5 = np.where(((df[flx+'h'+end] == exnum) | (df['e_'+flx+'h'+end] == exnum)) & ((df[flx+'m'+end] != exnum)
                  & (df['e_'+flx+'m'+end] != exnum)) & ((df[flx+'s'+end] == exnum) | (df['e_'+flx+'s'+end] == exnum)))[0]
    s6 = np.where(((df[flx+'h'+end] != exnum) & (df['e_'+flx+'h'+end] != exnum)) & ((df[flx+'m'+end] == exnum)
                  | (df['e_'+flx+'m'+end] == exnum)) & ((df[flx+'s'+end] == exnum) | (df['e_'+flx+'s'+end] == exnum)))[0]
    s7 = np.where(((df[flx+'h'+end] == exnum) | (df['e_'+flx+'h'+end] == exnum)) & ((df[flx+'m'+end] == exnum)
                  | (df['e_'+flx+'m'+end] == exnum)) & ((df[flx+'s'+end] != exnum) & (df['e_'+flx+'s'+end] != exnum)))[0]
    s8 = np.where(((df[flx+'h'+end] == exnum) | (df['e_'+flx+'h'+end] == exnum)) & ((df[flx+'m'+end] == exnum)
                  | (df['e_'+flx+'m'+end] == exnum)) & ((df[flx+'s'+end] == exnum) | (df['e_'+flx+'s'+end] == exnum)))[0]
    s9 = np.where((df[flx+'h'+end] == exnum) | (df['e_'+flx+'h'+end] == exnum) | (df[flx+'m'+end] == exnum)
                  | (df['e_'+flx+'m'+end] == exnum) | (df[flx+'s'+end] == exnum) | (df['e_'+flx+'s'+end] == exnum))[0]
    df = df.replace(exnum, np.nan)

    tot = len(df)
    df_rows = [('Y', 'Y', 'Y', len(s1), int(len(s1)/tot*1000)/10.),
               ('Y', 'Y', 'N', len(s2), int(len(s2)/tot*1000)/10.),
               ('N', 'Y', 'Y', len(s3), int(len(s3)/tot*1000)/10.),
               ('Y', 'N', 'Y', len(s4), int(len(s4)/tot*1000)/10.),
               ('N', 'Y', 'N', len(s5), int(len(s5)/tot*1000)/10.),
               ('Y', 'N', 'N', len(s6), int(len(s6)/tot*1000)/10.),
               ('N', 'N', 'Y', len(s7), int(len(s7)/tot*1000)/10.),
               ('N', 'N', 'N', len(s8), int(len(s8)/tot*1000)/10.),
               ('~Y', 'Y', 'Y', len(s9), int(len(s9)/tot*1000)/10.)]

    tt = Table(rows=df_rows, names=('H', 'M', 'S', '#', '%'))
    print(tt)
    print('-----------------')
    print('total:     ', tot)

    print("Only ", len(s1), " detections have valid fluxes at all bands.")

    if drop == True:
        # print("Only ", len(s1), " detections have valid fluxes at all bands.")
        df.loc[s9, 'per_remove_code'] = df.loc[s9, 'per_remove_code']+64
        print('After dropping', str(len(s9)), 'detections with NaNs,',
              len(df[df['per_remove_code'] == 0]), 'remain.')
        return df

    elif drop == False:
        return df


def cal_bflux(df, flx='flux_aper90_', end='.1'):
    # manually calculate the mean and the variance of the broad band flux to compare against CSC values

    df[flx+'b_manual'+end] = df[flx+'s'+end]+df[flx+'m'+end]+df[flx+'h'+end]
    df['e_'+flx+'b_manual'+end] = np.sqrt(df['e_'+flx+'s'+end]
                                          ** 2+df['e_'+flx+'m'+end]**2+df['e_'+flx+'h'+end]**2)

    return df


def powlaw2symmetric(df, cols=['flux_powlaw', 'powlaw_gamma', 'powlaw_nh', 'powlaw_ampl'], end='.1'):
    # calculate the left & right uncertainties, the mean, the variance of the Fechner distribution for band fluxes
    # print("Run powlaw2symmetric......")

    for col in cols:
        df['e_'+col+'_hilim'+end] = df[col+'_hilim'+end] - df[col+end]
        df['e_'+col+'_lolim'+end] = df[col+end] - df[col+'_lolim'+end]
        df[col+'_mean'+end] = df[col+end] + \
            np.sqrt(2/np.pi) * (df['e_'+col+'_hilim' +
                                   end] - df['e_'+col+'_lolim'+end])
        df['e_'+col+'_mean'+end] = np.sqrt((1. - 2./np.pi) * (df['e_'+col+'_hilim'+end] -
                                           df['e_'+col+'_lolim'+end])**2 + df['e_'+col+'_hilim'+end]*df['e_'+col+'_lolim'+end])
        df = df.drop(['e_'+col+'_hilim'+end, 'e_'+col+'_lolim'+end], axis=1)

    return df


def add_newdata(data, data_dir):
    '''
    Manually add fluxes for some sources in the TD based on Rafael's calculations
    '''
    print("Run add_newdata......")

    # Adding new data
    data['obs_reg'] = data['obsid']*10000+data['region_id']

    print('Before adding new data:')
    stats(data)
    # stats(data[data['per_remove_code']==0])

    bands = ['s', 'm', 'h']
    files = [f'{data_dir}/newdata/output_gwu_snull.txt_May_29_2020_15_32_39_clean.csv',
             f'{data_dir}/newdata/output_gwu_mnull.txt_Jun_01_2020_09_36_59_clean.csv', f'{data_dir}/newdata/gwu_hnull_output.txt_May_01_2020_13_16_39_clean.csv']

    for band, fil in zip(bands, files):

        data_new = pd.read_csv(fil)

        data_new['obs'] = data_new['#current_obsid'].str[:5].astype(int)
        data_new['obs_reg'] = data_new['obs']*10000+data_new['reg']

        data_new = data_new.rename(columns={'mode': 'flux_aper90_'+band+'.1',
                                   'lolim': 'flux_aper90_lolim_'+band+'.1', 'hilim': 'flux_aper90_hilim_'+band+'.1'})
        data_new = flux2symmetric(
            data_new, flx='flux_aper90_', bands=[band], end='.1')
        data_new = data_new.set_index('obs_reg')

        data = data.set_index('obs_reg')
        # df = data.copy()

        data.update(data_new)

        # print(np.count_nonzero(data!=df)/3)

        data.reset_index(inplace=True)

        print('After adding new ', str(len(data_new)), band, 'band data:')
        data = stats(data)
        # stats(data[data['per_remove_code']==0])

    return data


def apply_flags_filter(data, instrument=True, sig=False, theta_flag=True, dup=True, sat_flag=True, pileup_warning=True, streak_flag=True, pu_signa_fil=False, verb=False):
    # print("Run apply_flags_filter......")

    data = data.fillna(exnum)

    if verb:
        stats(data[data['per_remove_code'] == 0])

    if instrument:
        s = np.where(data['instrument'] == ' HRC')[0]
        data.loc[s, 'per_remove_code'] = data.loc[s, 'per_remove_code']+1
        # print('After dropping', str(len(s)),'detections with HRC instrument,', len(data[data['per_remove_code']==0]),'remain.')

    if theta_flag:
        s = np.where(data['theta'] > 10)[0]
        data.loc[s, 'per_remove_code'] = data.loc[s, 'per_remove_code']+2
        # print('After dropping', str(len(s)),'detections with theta larger than 10\',', len(data[data['per_remove_code']==0]),'remain.')
        if verb:
            stats(data[data['per_remove_code'] == 0])

    if sat_flag:
        s = np.where(data['sat_src_flag.1'] == True)[0]
        # print(str(sorted(Counter(data['Class'].iloc[s]).items())))
        data.loc[s, 'per_remove_code'] = data.loc[s, 'per_remove_code']+4
        # print("After dropping", len(s), " detections with sat_src_flag = TRUE,",  len(data[data['per_remove_code']==0]),'remain.')
        if verb:
            stats(data[data['per_remove_code'] == 0])

    if pileup_warning:
        s = np.where((data['pileup_warning'] > 0.3) &
                     (data['pileup_warning'] != exnum))[0]
        data.loc[s, 'per_remove_code'] = data.loc[s, 'per_remove_code']+8
        # print("After dropping", len(s), " detections with pile_warning>0.3,", len(data[data['per_remove_code']==0]),'remain.')
        if verb:
            stats(data[data['per_remove_code'] == 0])

    if streak_flag:
        s = np.where(data['streak_src_flag.1'] == True)[0]
        data.loc[s, 'per_remove_code'] = data.loc[s, 'per_remove_code']+16
        # print("After dropping", len(s), " detections with streak_src_flag = TRUE,",  len(data[data['per_remove_code']==0]),'remain.')
        if verb:
            stats(data[data['per_remove_code'] == 0])

    if dup:
        # print(data.groupby(['obsid','region_id','obi']).filter(lambda g: len(g['name'].unique()) > 1) )
        s = np.where(data.set_index(['obsid', 'region_id', 'obi']).index.isin(data.groupby(['obsid', 'region_id', 'obi']).filter(
            lambda g:  len(g['name'].unique()) > 1).set_index(['obsid', 'region_id', 'obi']).index))[0]
        # data.iloc[s].to_csv('dup.csv',index=False)
        data.loc[s, 'per_remove_code'] = data.loc[s, 'per_remove_code']+32
        # print("After dropping", len(s), " detections assigned to different sources,",  len(data[data['per_remove_code']==0]),'remain.')
        if verb:
            stats(data[data['per_remove_code'] == 0])

    if pu_signa_fil:
        # s = np.where( (data['flux_significance_b']==exnum) | (data['flux_significance_b']==0)  | (np.isinf(data['PU'])) | (data['PU']==exnum))[0]
        s = np.where((np.isinf(data['PU'])) | (data['PU'] == exnum))[0]
        data.loc[s, 'per_remove_code'] = data.loc[s, 'per_remove_code']+64
        # print("After dropping", len(s), " detections having nan sig or inf PU,",  len(data[data['per_remove_code']==0]),'remain.')
        if verb:
            stats(data[data['per_remove_code'] == 0])

    if sig:
        s = np.where(data['flux_significance_b'] < sig)[0]
        data.loc[s, 'per_remove_code'] = data.loc[s, 'per_remove_code']+1
        # print('After dropping', str(len(s)),'detections with flux_significance_b less than', sig, len(data[data['per_remove_code']==0]),'remain.')
        if verb:
            stats(data[data['per_remove_code'] == 0])

    data = data.replace(exnum, np.nan)

    return data


def cal_sig(df, df_ave, sig):
    # print("Run cal_sig......")

    df_ave[sig+'_max'] = np.nan

    for src in df.usrid.unique():
        idx = np.where((df.usrid == src) & (df.per_remove_code == 0))[0]
        idx2 = np.where(df_ave.usrid == src)[0]

        if len(idx) == 0:
            continue
        elif len(idx) == 1:
            sig_max = df.loc[idx, sig].values
        else:
            sig_max = np.nanmax(df.loc[idx, sig])

        df_ave.loc[idx2, sig+'_max'] = sig_max

    return df, df_ave


def cal_cnt(df, df_ave, cnt, cnt_hi, cnt_lo):
    df_ave[cnt+'_max'] = np.nan

    for src in df.usrid.unique():
        idx = np.where((df.usrid == src) & (df.per_remove_code == 0))[0]
        idx2 = np.where(df_ave.usrid == src)[0]

        if len(idx) == 0:
            continue
        elif len(idx) == 1:
            cnt_max = df.loc[idx, cnt].values
            cnt_max_hi = df.loc[idx, cnt_hi].values
            cnt_max_lo = df.loc[idx, cnt_lo].values
        else:
            max_ind = np.nanargmax(df.loc[idx, cnt])
            cnt_max = df.loc[max_ind, cnt]
            cnt_max_hi = df.loc[max_idx, cnt_hi]
            cnt_max_lo = df.loc[max_idx, cnt_lo]
            # print(cnt_max, np.nanmax(df.loc[idx,cnt]))

        df_ave.loc[idx2, cnt+'_max'] = cnt_max
        df_ave.loc[idx2, cnt+'_max_hi'] = cnt_max_hi
        df_ave.loc[idx2, cnt+'_max_lo'] = cnt_max_lo

    return df, df_ave


def cal_aveflux(df, df_ave, bands, flux_name, per_flux_name, fil=False, add2df=False):
    # print("Run cal_aveflux......")

    for band in bands:
        col = flux_name+band
        p_col = per_flux_name+band+'.1'
        df_ave[col] = np.nan
        df_ave['e_'+col] = np.nan
        if add2df:
            df[col] = np.nan
            df['e_'+col] = np.nan

        for uid in df.usrid.unique():

            if fil == True:
                idx = np.where((~df[p_col].isna()) & (~df['e_'+p_col].isna()) & (df.per_remove_code == 0) & (
                    df.usrid == uid) & (df.theta <= 10) & (df['sat_src_flag.1'] != True) & (df.pileup_warning <= 0.3))[0]
            elif fil == 'strict':
                idx = np.where((~df[p_col].isna()) & (~df['e_'+p_col].isna()) & (df.per_remove_code == 0) & (df.usird == uid) & (df.theta <= 10) & (df['sat_src_flag.1']
                               == False) & (df.conf_code <= 7) & (df.pileup_warning <= 0.1) & (df.edge_code <= 1) & (df.extent_code <= 0) & (df['streak_src_flag.1'] == False))[0]
            else:
                idx = np.where((~df[p_col].isna()) & (
                    ~df['e_'+p_col].isna()) & (df.per_remove_code == 0) & (df.usrid == uid))[0]

            idx2 = np.where(df_ave.usrid == uid)[0]

            if len(idx) == 0:
                # df_ave.loc[idx2, 'remove_code'] = 1
                continue

            elif len(idx) == 1:
                ave = df.loc[idx, p_col].values
                err = df.loc[idx, 'e_'+p_col].values
                # df_ave.loc[idx2, col]      = ave
                # df_ave.loc[idx2, 'e_'+col] = err

            else:
                ave = np.average(
                    df.loc[idx, p_col].values, weights=1./(df.loc[idx, 'e_'+p_col].values)**2)
                err = np.sqrt(1./sum(1./(df.loc[idx, 'e_'+p_col].values)**2))

            df_ave.loc[idx2, col] = ave
            df_ave.loc[idx2, 'e_'+col] = err
            if add2df:
                df.loc[idx, col] = ave
                df.loc[idx, 'e_'+col] = err

    return df, df_ave


def cal_theta_counts(df, df_ave, theta, net_count, err_count):
    # print("Run cal_theta_counts......")

    for col in [theta+'_mean', theta+'_median', 'e_'+theta, net_count, err_count]:
        df_ave[col] = np.nan

    for src in df.usrid.unique():
        idx = np.where((df.usrid == src))[0]  # & (df.per_remove_code==0) )[0]
        idx2 = np.where(df_ave.usrid == src)[0]

        theta_mean = np.nanmean(df.loc[idx, theta].values)
        theta_median = np.nanmedian(df.loc[idx, theta].values)
        theta_std = np.nanstd(df.loc[idx, theta].values)
        counts = np.nansum(df.loc[idx, net_count].values)
        e_counts = np.sqrt(
            np.nansum([e**2 for e in df.loc[idx, err_count].values]))

        df_ave.loc[idx2, theta+'_mean'] = theta_mean
        df_ave.loc[idx2, theta+'_median'] = theta_median
        df_ave.loc[idx2, 'e_'+theta] = theta_std
        df_ave.loc[idx2, net_count] = counts
        df_ave.loc[idx2, err_count] = e_counts
        df_ave.loc[idx2, theta+'_median'] = theta_median

    return df, df_ave


def nan_flux(df_ave, flux_name, flux_flag='flux_flag', end=''):
    # print("Run nan_flux......")

    if end == '':
        df_ave[flux_flag] = 0

        for band, code, flux_hilim in zip(['s', 'm', 'h'], [1, 2, 4], [1e-17, 1e-17, 1e-17]):
            col = flux_name+band+end
            idx = np.where((df_ave[col].isna()) | (df_ave['e_'+col].isna()))[0]

            df_ave.loc[idx, col] = np.sqrt(2/np.pi) * flux_hilim
            df_ave.loc[idx, 'e_'+col] = np.sqrt((1. - 2./np.pi)) * flux_hilim

            df_ave.loc[idx, flux_flag] = df_ave.loc[idx, flux_flag] + code

    elif end == '.1':
        df_ave[flux_flag+'.1'] = 0
        for band, code, flux_hilim in zip(['s', 'm', 'h'], [1, 2, 4], [1e-17, 1e-17, 1e-17]):
            col = flux_name+band+end
            idx = np.where((df_ave[col].isna()) | (df_ave['e_'+col].isna()))[0]

            df_ave.loc[idx, col] = np.sqrt(2/np.pi) * flux_hilim
            df_ave.loc[idx, 'e_'+col] = np.sqrt((1. - 2./np.pi)) * flux_hilim

            df_ave.loc[idx, flux_flag +
                       '.1'] = df_ave.loc[idx, flux_flag+'.1'] + code
        df_ave.loc[df_ave[flux_flag+'.1'] == 7,
                   'per_remove_code'] = df_ave.loc[df_ave[flux_flag+'.1'] == 7, 'per_remove_code']+128

    return df_ave


def cal_var(df, df_ave, b_ave, b_per):
    # print("Run cal_var......")

    new_cols = ['chisqr', 'dof', 'kp_prob_b_max',
                'var_inter_prob', 'significance_max']
    for col in new_cols:
        df_ave[col] = np.nan

    # b_per     = 'flux_aper90_sym_b.1'

    for uid in sorted(df.usrid.unique()):
        idx = np.where((~df[b_per].isna()) & (
            ~df['e_'+b_per].isna()) & (df.per_remove_code == 0) & (df.usrid == uid))[0]

        dof = len(idx)-1.

        idx2 = np.where(df_ave.usrid == uid)[0]
        df_ave.loc[idx2, 'dof'] = dof

        if dof > -1:
            df_ave.loc[idx2, 'kp_prob_b_max'] = np.nanmax(
                df.loc[idx, 'kp_prob_b'].values)
            df_ave.loc[idx2, 'significance_max'] = np.nanmax(
                df.loc[idx, 'flux_significance_b'].values)

        if (dof == 0) or (dof == -1):
            continue

        chisqr = np.sum((df.loc[idx, b_per].values-df.loc[idx,
                        b_ave].values)**2/(df.loc[idx, 'e_'+b_per].values)**2)
        df_ave.loc[idx2, 'chisqr'] = chisqr

    df_ave['var_inter_prob'] = df_ave.apply(
        lambda row: sc.chdtr(row.dof, row.chisqr), axis=1)
    df_ave = df_ave.round({'var_inter_prob': 3})

    return df, df_ave


def cal_ave_pos(df, df_ave, ra_col, dec_col, pu_col, add2df=False):

    for uid in df.usrid.unique():

        idx = np.where((df.usrid == uid))[0]  # & (df.per_remove_code==0) )[0]
        idx2 = np.where(df_ave.usrid == uid)[0]

        if len(idx) == 1:
            ra_ave = df.loc[idx, ra_col].values
            dec_ave = df.loc[idx, dec_col].values
            pu_ave = df.loc[idx, pu_col].values
            # df_ave.loc[idx2, col]      = ave
            # df_ave.loc[idx2, 'e_'+col] = err

        else:
            ra_ave = np.average(
                df.loc[idx, ra_col].values, weights=1./(df.loc[idx, pu_col].values)**2)
            dec_ave = np.average(
                df.loc[idx, dec_col].values, weights=1./(df.loc[idx, pu_col].values)**2)
            pu_ave = np.sqrt(1./sum(1./(df.loc[idx, pu_col].values)**2))

        df_ave.loc[idx2, 'ra_ave'] = ra_ave
        df_ave.loc[idx2, 'dec_ave'] = dec_ave
        df_ave.loc[idx2, 'PU_ave'] = pu_ave
        if add2df:
            df.loc[idx, 'ra_ave'] = ra_ave
            df.loc[idx, 'dec_ave'] = dec_ave
            df.loc[idx, 'PU_ave'] = pu_ave

    return df, df_ave


def cal_ave_v2(df, data_dir, dtype='TD', Chandratype='CSC', PU=False, cnt=False, plot=False, verb=False, convert_hms_to_deg=True, cal_ave_coordinate=False):
    '''
    description:
        calculate the averaged data from per-observation CSC data

    input: 
        df: the DataFrame of per-observation data
        dtype: 'TD' for training dataset and 'field' for field (testing) dataset
        plot: plot mode

    output: 
        df_ave: the DataFrame of averaged data from per-observation CSC data
        df: the per-observation data used to calculate the averaged data

    '''

    # print("Run cal_ave......")
    # print('There are', str(len(df)), 'per-obs data.')

    df = df.fillna(exnum)
    df = df.replace(r'^\s*$', exnum, regex=True)
    df = df.apply(pd.to_numeric, errors='ignore')
    df = df.replace({' TRUE': True, 'False': False, 'FALSE': False})
    df = df.replace(exnum, np.nan)

    # convert asymmetric fluxes to symmetric fluxes
    df = flux2symmetric(df, end='.1')

    if Chandratype == 'CSC' or Chandratype == 'CSC-CXO':
        df = flux2symmetric(df, end='')
        df = cal_bflux(df, flx='flux_aper90_sym_', end='')
        df = powlaw2symmetric(df, end='')

    if dtype == 'TD' and Chandratype == 'CSC':
        # Adding new data
        df = add_newdata(df, data_dir)

    # fxs = ['flux_aper90_'+band+'.1' for band in ['b','s','m','h']]
    # los = ['flux_aper90_lolim_'+band+'.1' for band in ['b','s','m','h']]
    # his = ['flux_aper90_hilim_'+band+'.1' for band in ['b','s','m','h']]
    # bs = ['flux_aper90_'+b+'b.1' for b in ['','lolim_','hilim_']]
    # ss = ['flux_aper90_'+b+'s.1' for b in ['','lolim_','hilim_']]
    # ms = ['flux_aper90_'+b+'m.1' for b in ['','lolim_','hilim_']]
    # hs = ['flux_aper90_'+b+'h.1' for b in ['','lolim_','hilim_']]
    # fxs_cols = ['name','var_inter_prob_b', *bs, *ss, *ms, *hs]
    # df[fxs_cols].to_csv('df_per_test.csv',index=False)
    # a mode of 0 and an upper limit of 1E-17 are used to replace the Null values of per-obs flux
    # df = nan_flux(df, flux_name='flux_aper90_sym_',flux_flag='per_flux_flag',end='.1')
    # print(df['per_flux_flag'].value_counts())

    # Apply with some filters on sat_src_flag and pile_warning at per-obs level
    if Chandratype == 'CSC' or Chandratype == 'CSC-CXO':
        # theta_flag=True,dup=True,sat_flag=True,pileup_warning=True,streak_flag=True
        df = apply_flags_filter(df, verb=verb)
    elif Chandratype == 'CXO':
        df = apply_flags_filter(df, instrument=False, sig=False, theta_flag=True, dup=False,
                                sat_flag=False, pileup_warning=False, streak_flag=False, pu_signa_fil=False, verb=verb)
    # '''

    # df.to_csv('TD_test.csv',index=False)

    if Chandratype == 'CSC':
        cols_copy = ['name', 'usrid', 'ra', 'dec', 'err_ellipse_r0', 'err_ellipse_r1', 'err_ellipse_ang', 'significance',
                     'extent_flag', 'pileup_flag', 'sat_src_flag', 'streak_src_flag', 'conf_flag',
                     'flux_aper90_sym_b', 'e_flux_aper90_sym_b', 'flux_aper90_sym_h', 'e_flux_aper90_sym_h',
                     'flux_aper90_sym_m', 'e_flux_aper90_sym_m', 'flux_aper90_sym_s', 'e_flux_aper90_sym_s',
                     'kp_intra_prob_b', 'ks_intra_prob_b', 'var_inter_prob_b',
                     'nh_gal', 'flux_powlaw_mean', 'e_flux_powlaw_mean', 'powlaw_gamma_mean', 'e_powlaw_gamma_mean',
                     'powlaw_nh_mean', 'e_powlaw_nh_mean', 'powlaw_ampl_mean', 'e_powlaw_ampl_mean', 'powlaw_stat']  # 'ra_pnt','dec_pnt',

    elif Chandratype == 'CXO':
        cols_copy = ['name', 'usrid', 'ra',
                     'dec', 'significance', 'net_counts']
    elif Chandratype == 'CSC-CXO':
        cols_copy = ['COMPONENT', 'name', 'usrid', 'ra', 'dec', 'err_ellipse_r0', 'err_ellipse_r1', 'err_ellipse_ang', 'significance',
                     'extent_flag', 'pileup_flag', 'sat_src_flag', 'streak_src_flag', 'conf_flag',
                     'flux_aper90_sym_b', 'e_flux_aper90_sym_b', 'flux_aper90_sym_h', 'e_flux_aper90_sym_h',
                     'flux_aper90_sym_m', 'e_flux_aper90_sym_m', 'flux_aper90_sym_s', 'e_flux_aper90_sym_s',
                     'kp_intra_prob_b', 'ks_intra_prob_b', 'var_inter_prob_b',
                     'nh_gal', 'flux_powlaw_mean', 'e_flux_powlaw_mean', 'powlaw_gamma_mean', 'e_powlaw_gamma_mean',
                     'powlaw_nh_mean', 'e_powlaw_nh_mean', 'powlaw_ampl_mean', 'e_powlaw_ampl_mean', 'powlaw_stat']

    df = df[df['per_remove_code'] == 0].reset_index(drop=True)
    # df.to_csv('TD_test.csv',index=False)
    if PU:
        df_ave = df[cols_copy+PU].copy()
    else:
        df_ave = df[cols_copy].copy()

    df_ave['prod_per_remove_code'] = 0
    for uid in df_ave.usrid.unique():
        idx = np.where(df.usrid == uid)[0]
        idx2 = np.where(df_ave.usrid == uid)[0]
        df_ave.loc[idx2, 'prod_per_remove_code'] = np.prod(
            df.loc[idx, 'per_remove_code'].values)

    df_ave = df_ave.drop_duplicates(subset=['usrid'], keep='first')
    df_ave = df_ave.reset_index(drop=True)
    df_ave['remove_code'] = 0
    df_ave.loc[df_ave.prod_per_remove_code > 0, 'remove_code'] = 1
    df_ave = df_ave.drop('prod_per_remove_code', axis=1)

    df, df_ave = cal_sig(df, df_ave, 'flux_significance_b')
    if Chandratype == 'CXO' or Chandratype == 'CSC-CXO':
        df, df_ave = cal_theta_counts(
            df, df_ave, 'theta', 'NET_COUNTS_broad', 'NET_ERR_broad')

    # Calculating average fluxes
    df, df_ave = cal_aveflux(df, df_ave, [
                             's', 'm', 'h'], 'flux_aper90_avg_', 'flux_aper90_sym_')  # fil =False)

########
    # still removing those with all 3 band null fluxes
    df = nan_flux(df, 'flux_aper90_sym_', end='.1')
    df = df[(df['per_remove_code'] == 0) | (
        df['flux_aper90_sym_b.1'] >= 0)].reset_index(drop=True)
    df['flux_aper90_sym_ori_b.1'] = df['flux_aper90_sym_b.1'].copy()
    df['e_flux_aper90_sym_ori_b.1'] = df['e_flux_aper90_sym_b.1'].copy()
    df = cal_bflux(df, flx='flux_aper90_sym_', end='.1')
    # print(df.loc[df['flux_aper90_sym_ori_b.1'].isnull(), 'flux_aper90_sym_ori_b.1'])
    df.loc[df['flux_aper90_sym_ori_b.1'].isnull(
    ), 'flux_aper90_sym_ori_b.1'] = df.loc[df['flux_aper90_sym_ori_b.1'].isnull(), 'flux_aper90_sym_b.1']
    df.loc[df['flux_aper90_sym_ori_b.1'].isnull(
    ), 'e_flux_aper90_sym_ori_b.1'] = df.loc[df['flux_aper90_sym_ori_b.1'].isnull(), 'e_flux_aper90_sym_b.1']

    # print(df.loc[df['flux_aper90_sym_ori_b.1'].isnull(), 'flux_aper90_sym_ori_b.1'])
    df, df_ave = cal_aveflux(
        df, df_ave, ['b'], 'flux_aper90_avg2_', 'flux_aper90_sym_ori_', add2df=True)

#######
    # Calculating inter-variability
    df, df_ave = cal_var(df, df_ave, 'flux_aper90_avg2_b',
                         'flux_aper90_sym_ori_b.1')
    df_ave = df_ave.drop(
        ['flux_aper90_avg2_b', 'e_flux_aper90_avg2_b'], axis=1)

    if (Chandratype == 'CSC' or Chandratype == 'CSC-CXO') and convert_hms_to_deg == True:
        # combine additional useful master flux
        # df_ave = combine_master(df_ave)
        df_ave['ra'] = Angle(df_ave['ra'], 'hourangle').degree
        df_ave['dec'] = Angle(df_ave['dec'], 'deg').degree

    df_ave = nan_flux(df_ave, 'flux_aper90_avg_')

    df_ave = cal_bflux(df_ave, 'flux_aper90_avg_', end='')

    if Chandratype == 'CXO' and cal_ave_coordinate == True:
        df, df_ave = cal_ave_pos(
            df, df_ave, ra_col='ra.1', dec_col='dec.1', pu_col='PU.1')

    if cnt:
        df, df_ave = cal_cnt(df, df_ave, 'src_cnts_aper90_b',
                             'src_cnts_aper90_hilim_b', 'src_cnts_aper90_lolim_b')
    # df_ave.to_csv('ave_test.csv',index=False)
    # '''
    return df_ave, df


def flux2symmetric(df, flx='flux_aper90_', bands=['b', 's', 'm', 'h'], end='.1', rename=False):
    # calculate the left & right uncertainties, the mean, the variance of the Fechner distribution for band fluxes
    # rename: if True, rename flux columns to be consistent with Fcsc_ format
    # print("Run flux2symmetric......")

    for band in bands:
        df['e_'+flx+'hilim_'+band+end] = df[flx +
                                            'hilim_'+band+end] - df[flx+''+band+end]
        df['e_'+flx+'lolim_'+band+end] = df[flx +
                                            ''+band+end] - df[flx+'lolim_'+band+end]
        df[flx+'sym_'+band+end] = df[flx+''+band+end] + np.sqrt(2/np.pi) * (
            df['e_'+flx+'hilim_'+band+end] - df['e_'+flx+'lolim_'+band+end])
        df['e_'+flx+'sym_'+band+end] = np.sqrt((1. - 2./np.pi) * (df['e_'+flx+'hilim_'+band+end] -
                                               df['e_'+flx+'lolim_'+band+end])**2 + df['e_'+flx+'hilim_'+band+end]*df['e_'+flx+'lolim_'+band+end])
        df = df.drop(['e_'+flx+'hilim_'+band+end, 'e_' +
                     flx+'lolim_'+band+end], axis=1)
        if rename:
            df = df.rename(columns={
                           flx+'sym_'+band+end: 'Fcsc_'+band, 'e_'+flx+'sym_'+band+end: 'e_Fcsc_'+band})

    return df


# MC sampling
def nonzero_sample(df, col, out_col, random_state=None, factor=np.sqrt(2.)):
    '''
    description: 
        sampling the col column of df with its Gaussian uncertainty e_col column while making sure the sampled value is larger than zero (cases for fluxes)

    input:
        df: the dataframe 
        col: the sampled column name (the uncertainty column is e_col by default)
        out_col: output column name
    '''
    if random_state is None:
        np.random.seed(randint(1, 999999999))
    else:
        np.random.seed(random_state)

    df['temp_'+col] = np.random.randn(df[col].size) * \
        df['e_'+col] * factor + df[col]
    s = df.loc[df['temp_'+col] <= 0].index

    while len(s) > 0:
        df.loc[s, 'temp_'+col] = np.random.randn(
            df.loc[s, col].size) * df.loc[s, 'e_'+col] * factor + df.loc[s, col]
        s = df.loc[df['temp_'+col] <= 0].index

    df[out_col] = df['temp_'+col]
    df = df.drop(columns='temp_'+col)

    return df


def asymmetric_errors(df, dist):
    # calculate the errors for distances based on 84% and 16% percentile values

    df['e_B_'+dist] = df['B_'+dist] - df[dist]
    df['e_b_'+dist] = df[dist] - df['b_'+dist]

    # assume mode is median for this
    df['mean_'+dist] = df[dist] + \
        np.sqrt(2/np.pi) * (df['e_B_'+dist] - df['e_b_'+dist])
    df['e_'+dist] = np.sqrt((1. - 2./np.pi) * (df['e_B_'+dist] -
                            df['e_b_'+dist])**2 + df['e_B_'+dist]*df['e_b_'+dist])

    return df


def sample_data(df, Xray='CSC', gc=False, distance_feature='nodist', Uncer_flag=False, random_state=None, rep_num=False, factor=np.sqrt(2.), verb=False):
    '''
    description: create sampled data from (Gaussian) distributions of measurements

    input:
        df: the dataframe 
        Xray: 'XMM' or 'CSC' X-ray data set 
        fIR: 'WISE' or 'Glimpse' far-Infrared data 
        Xray_level: 'ave' when averaged fluxes are sampled or 'obs' when per-observation fluxes are sampled
        distance: 
        verb: 
    '''

    if rep_num != False:
        df = pd.DataFrame(np.repeat(df.values, rep_num,
                          axis=0), columns=df.columns)

    if random_state is None:
        np.random.seed(randint(1, 999999999))
    else:
        np.random.seed(random_state)

    if verb:
        print('Run......sample_data')
        print('Sampling ' + Xray + ' X-ray data.')

    if Uncer_flag == True:
        if Xray == 'XMM':
            # simulate fluxes assuming gaussian distribution of flux for XMM energy bands

            bands = ['2', '3', '4', '5', '8']
            for band in bands:
                df = nonzero_sample(
                    df, 'Fxmm_'+band, 'Fxmm_'+band, random_state=random_state, factor=factor)

        if Xray == 'CSC':

            bands = ['s', 'm', 'h']

            for band in bands:

                df = nonzero_sample(
                    df, 'Fcsc_'+band, 'Fcsc_'+band, random_state=random_state, factor=factor)

            df['Fcsc_b'] = df['Fcsc_s'] + df['Fcsc_m'] + df['Fcsc_h']

        if verb:
            print('Sampling MW data.')
        if gc:
            MW_cats = ['hugs']
        else:
            MW_cats = ['gaia', '2mass', 'wise']
        for cat in MW_cats:
            for band in MW_names[cat]:
                df[band] = np.random.randn(
                    df[band].size) * df['e_'+band] * factor + df[band]

        if distance_feature not in ['nodist', 'gc_dist']:

            dist_feature = dist_features_dict[distance_feature][0]

            # set distances to zero for sources without DR3Name_gaia

            if 'DR3Name_gaia' in df.columns:
                df.loc[df['DR3Name_gaia'].isna(), [dist_feature, 'e_' +
                                                   dist_feature]] = np.nan
            elif 'Gaia' in df.columns:
                df.loc[df['Gaia'].isna(), [dist_feature, 'e_' +
                                           dist_feature]] = np.nan
            else:
                raise ValueError('No Gaia DR3 name column in the data frame.')

            # set distance of sources with negative parallaxes and parallaxes with large errors (fpu<2) to nan, already done in making of TD?
            # the cleaning of features should be done when creating the test data
            df.loc[df['GAIA_Plx'] < 0, [
                dist_feature, 'e_'+dist_feature]] = np.nan
            df.loc[df['GAIA_Plx']/df['GAIA_e_Plx'] < 2,
                   [dist_feature, 'e_'+dist_feature]] = np.nan

            df = asymmetric_errors(df, dist_feature)

            if dist_feature == 'Plx_dist':
                df = nonzero_sample(df, 'GAIA_Plx', 'GAIA_Plx',
                                    random_state=random_state, factor=factor)
                # GAIA_Plx in units of mas
                df['Plx_dist'] = 1000./df['GAIA_Plx']
            else:
                # set distance of sources with no parallax measurements to nan.
                df.loc[df['GAIA_Plx'].isna(), [dist_feature, 'e_' +
                                               dist_feature]] = np.nan
                df = nonzero_sample(
                    df, dist_feature, dist_feature, random_state=random_state, factor=factor)

    elif Uncer_flag == False:

        if Xray == 'XMM':
            bands = ['2', '3', '4', '5', '8']
            for band in bands:
                # can be implemented when producing the XMM TD
                df.loc[df['Fxmm_'+band] == 0, 'Fxmm_'+band] = 1e-22

        if Xray == 'CSC':
            df['Fcsc_b'] = df['Fcsc_s'] + df['Fcsc_m'] + df['Fcsc_h']

        if verb:
            print('Copying MW data where FIR is from ', fIR,
                  ' and distance feature ', distance_feature, '.')

        if distance_feature not in ['nodist', 'gc_dist']:

            dist_feature = dist_features_dict[distance_feature][0]

            df.loc[df['DR3Name_gaia'].isna(), [dist_feature, 'e_' +
                                               dist_feature]] = np.nan

            # set distance of sources with negative parallaxes and parallaxes with large errors (fpu<2) to nan, already done in making of TD?
            df.loc[df['GAIA_Plx'] < 0, [
                dist_feature, 'e_'+dist_feature]] = np.nan
            df.loc[df['GAIA_Plx']/df['GAIA_e_Plx'] < 2,
                   [dist_feature, 'e_'+dist_feature]] = np.nan

            df = asymmetric_errors(df, dist_feature)

            if dist_feature == 'Plx_dist':
                # set distance of sources with no parallax measurements to nan.
                # GAIA_Plx in units of mas
                df['Plx_dist'] = 1000./df['GAIA_Plx']
            else:
                # set distance of sources with no parallax measurements to nan.
                df.loc[df['GAIA_Plx'].isna(), [dist_feature, 'e_' +
                                               dist_feature]] = np.nan

    return df


def convert2csc(data, method='simple', Gamma=2., verb=False):
    # Convert XMM fluxes to CSC fluxes with method='simple' using simple scaling factors assuming Gamma=2
    # or method='LR' with linear regression using paramters from fitting the same sources from XMM and CSC TD

    CSC_fluxs, XMM_fluxs = ['Fcsc_s', 'Fcsc_m', 'Fcsc_h'], [
        'Fxmm_2',   'Fxmm_3',  'Fxmm_4',   'Fxmm_5']
    CSC_bands, XMM_bands = [[0.5, 1.2], [1.2, 2.],  [2., 7.]],   [
        [0.5, 1.], [1., 2.], [2., 4.5], [4.5, 12.]]

    simple_coefs = [[(np.log(1.2)-np.log(0.5))/(np.log(1.0)-np.log(0.5))],  # [0.5, 1.0] keV -> (0.5, 1.2) keV
                    # [1.0, 2.0] -> (1.2, 2.0) keV
                    [(np.log(2.0)-np.log(1.2))/(np.log(2.0)-np.log(1.0))],
                    [1.0, (np.log(7.0)-np.log(4.5))/(np.log(12.0)-np.log(4.5))]]  # [2.0, 4.5] [4.5, 12.0] -> [2.0, 7.0]
    if Gamma != 2.:
        simple_coefs = [[(1.2**(2.-Gamma)-0.5**(2.-Gamma))/(1.**(2.-Gamma)-0.5**(2.-Gamma))],
                        [(2.**(2.-Gamma) - 1.2**(2.-Gamma))/(2.**(2.-Gamma)-1.)],
                        [1., (7.**(2.-Gamma)-4.5**(2.-Gamma))/(12.**(2.-Gamma)-4.5**(2.-Gamma))]]

    LR_coefs = [[0.95141998],
                [0.54679595, 0.44769412],
                [1.00513222],
                [0.66209453, 0.31911049]]

    XMMfluxs = [[data['Fxmm_2']],
                [data['Fxmm_3']],
                [data['Fxmm_4'], data['Fxmm_5']]]
    CSCfluxs = ['Fcsc_s', 'Fcsc_m', 'Fcsc_h']

    if method == 'simple':
        if verb:
            print("Run convert2csc with simple method and Gamma = "+str(Gamma)+".")

        # Simple scaling assuming a flat spectrum (default Gamma=2)
        for col_n, Xflux, coef in zip(CSCfluxs, XMMfluxs, simple_coefs):
            # print("Converbting to", col_n, "with simple method.")
            data[col_n] = sum(
                [flux*c for (flux, c) in zip(Xflux, np.array(coef))])

    if method == 'LR':
        if verb:
            print("Run convert2csc with LR method......")

        # Linear Regression
        for col_n, Xflux, LR_coef, sim_coef in zip(cols_new, XMMfluxs, LR_coefs, simple_coefs):
            # print("Converbting to", col_n, "with LR method.")
            data[col_n] = np.prod(
                [flux**c for (flux, c) in zip(Xflux, np.array(LR_coef))], axis=0)
            if col_n == 'Fcsc_s_lr2':
                data.loc[(data['xmm_f2'] == 0) | (data['xmm_f3'] == 0), col_n] = \
                    sum([flux*c for (flux, c) in zip([data.loc[(data['xmm_f2'] == 0) | (data['xmm_f3'] == 0), 'xmm_f2'],
                                                      data.loc[(data['xmm_f2'] == 0) | (data['xmm_f3'] == 0), 'xmm_f3']], np.array(sim_coef))])

            if col_n == 'Fcsc_h_lr2':
                data.loc[(data['xmm_f4'] == 0) | (data['xmm_f5'] == 0), col_n] = \
                    sum([flux*c for (flux, c) in zip([data.loc[(data['xmm_f4'] == 0) | (data['xmm_f5'] == 0), 'xmm_f4'],
                                                      data.loc[(data['xmm_f4'] == 0) | (data['xmm_f5'] == 0), 'xmm_f5']], np.array(sim_coef))])

    data['Fcsc_b'] = data['Fcsc_s']+data['Fcsc_m']+data['Fcsc_h']

    return data


def get_red_par(ra, dec, dustmap='SFD', nhmap='LAB'):

    coords = SkyCoord(ra, dec, unit='deg')
    # 0.86 is the correction described in Schlafly et al. 2010 and Schlafly & Finkbeiner 2011
    ebv = DustMap.ebv(coords, dustmap=dustmap) * 0.86
    nH_from_AV = 2.21 * 3.1 * ebv
    nH = GasMap.nh(coords, nhmap=nhmap).value / \
        1.e21  # nH in unit of 1.e21 atoms /cm2

    return ebv, nH_from_AV


def red_factor(ene, nH, Gamma, tbabs_ene, tbabs_cross):

    if Gamma == 2:
        flux_unred_int = np.log(ene[1]) - np.log(ene[0])
    else:
        flux_unred_int = (ene[1]**(2.-Gamma)-ene[0]**(2.-Gamma))/(2.-Gamma)

    _ = np.array([_**(1 - Gamma) for _ in tbabs_ene])
    tbabs_flux_red = _ * np.exp(-nH * 1e-3 * tbabs_cross)

    finterp = InterpolatedUnivariateSpline(tbabs_ene, tbabs_flux_red, k=1)

    flux_red_int = finterp.integral(*ene)

    return flux_red_int / flux_unred_int


def apply_red2csc(data, nh, tbabs_ene, tbabs_cross, red_class=['AGN'], deredden=False, self_unred=False, Gamma=2):
    bands = ['Fcsc_s', 'Fcsc_m', 'Fcsc_h', 'Fcsc_b']
    enes = [[0.5, 1.2], [1.2, 2.0], [2.0, 7.0], [0.5, 7.0]]
    for ene, band in zip(enes, bands):
        red_fact = red_factor(ene, nh, Gamma, tbabs_ene, tbabs_cross)

        # remove total galactic absorption in direction of source
        if deredden:
            data[band] = data.apply(lambda row: row[band] / red_factor(ene, row['nH'], Gamma, tbabs_ene, tbabs_cross), axis=1)
        else:
            # this case is to remove intrinsic (local) absorption for AGN, then add the Galactic absorption
            if self_unred == True:
                data.loc[data['Class'].isin(red_class), band] = data.loc[data['Class'].isin(red_class)].apply(lambda row: row[band] * red_factor(ene, nh - row['nH'], Gamma, tbabs_ene, tbabs_cross), axis=1)

            if self_unred == False:
                data.loc[data['Class'].isin(red_class), band] = data.loc[data.Class == red_class, band]*red_fact
    return data


def apply_red2mw(data, ebv, red_class='AGN', deredden=False, self_unred=False, gc=False):
    # extinction.fitzpatrick99 https://extinction.readthedocs.io/en/latest/
    # wavelengths of B, R, I (in USNO-B1), J, H, K (in 2MASS), W1, W2, W3 (in WISE) bands in Angstroms
    # wavelengths of G, Gbp, Grp (in Gaia), J, H, K (in 2MASS), W1, W2, W3 (in WISE) bands in Angstroms

    if gc:
        waves = hugs_eff_waves
        bands = hugs_features
    else:
        waves = MW_eff_waves
        bands = MW_features

    if deredden:
        data[bands] = data.apply(lambda row: row[bands] - extinction.fitzpatrick99(np.array(waves), 3.1*row['ebv']), axis=1)
    else:
        # this case is to remove intrinsic (local) absorption for AGN, then add the Galactic absorption
        if self_unred == True:
            data.loc[data['Class'].isin(red_class), bands] = data.loc[data['Class'].isin(red_class)].apply(lambda row: row[bands] + extinction.fitzpatrick99(np.array(waves), 3.1*(ebv-row['ebv'])), axis=1)
        if self_unred == False:
            data.loc[data['Class'].isin(red_class), bands] = data.loc[data['Class'].isin(red_class)].apply(lambda row: row[bands] + extinction.fitzpatrick99(np.array(waves), 3.1*ebv), axis=1)

    return data


def create_colors(data, apply_limit=True, gc=False):

    if gc:
        bands = hugs_features
        limits = hugs_limits
    else:
        bands = MW_features
        limits = MW_limits

    if apply_limit:
        # data = mw2limit(data, verb=verb)
        for band, limit in zip(bands, limits):
            data.loc[data[band] >= limit, band] = np.nan

    bands2 = bands.copy()
    created_colors = []
    for col1 in bands:
        bands2.remove(col1)
        for col2 in bands2:
            if gc:
                color = col1 + "-" + col2
            else:
                color = col1[:-3] + "-" + col2[:-3]
            data[color] = data[col1] - data[col2]
            created_colors.append(color)

    return data, created_colors


def create_Xfeatures(data):
    '''
    create X-ray features including EP052Flux, EP127Flux, EP057Flux, in erg/cm^2/s
    and hardness ratios EPHR4, EPHR2
    '''

    data['Fcsc_sm'] = data['Fcsc_s'] + data['Fcsc_m']
    data['Fcsc_mh'] = data['Fcsc_m'] + data['Fcsc_h']

    '''
    data.loc[data.Fcsc_sm <=10**(-19.),'Fcsc_sm']= 10**(-19.)
    data.loc[data.Fcsc_mh  <=10**(-21.),'Fcsc_mh']= 10**(-21.)
    data.loc[data.Lcsc_b <=10**(-20.),'Lcsc_b']= 10**(-20.)
    '''

    data['HR_ms'] = (data['Fcsc_m'] - data['Fcsc_s']) / \
        (data['Fcsc_m'] + data['Fcsc_s'])
    data['HR_hm'] = (data['Fcsc_h'] - data['Fcsc_m']) / \
        (data['Fcsc_h'] + data['Fcsc_m'])
    data['HR_hms'] = (data['Fcsc_h'] - data['Fcsc_m'] -
                      data['Fcsc_s'])/(data['Fcsc_h'] + data['Fcsc_m'] + data['Fcsc_s'])

    return data


def mag2flux(data, gc=False):
    '''
    from magnitude to flux in erg/cm^s/s
    '''

    if gc:
        bands = hugs_features
        zeros = hugs_zeros
        width_waves = hugs_width_waves
    else:
        bands = MW_features
        zeros = MW_zeros
        width_waves = MW_width_waves

    for band, zero, wave in zip(bands, zeros, width_waves):
        data[band] = zero * 10**(-data[band]/2.5) * wave

    return data


def luminosity(data, distance_feature, distance_value=None, verb=False):
    '''
    define luminosities based on distance
    '''
    if verb:
        print("Adding Luminosities based on " + distance_feature)

    if distance_feature == 'rgeo_lum' or distance_feature == 'rpgeo_lum' or distance_feature == 'plx_lum':
        for col in ['Lcsc_b', 'Gmag', 'Jmag']:
            if col == 'Lcsc_b':
                data['Lcsc_b'] = ((data['Lcsc_b'].values)*u.erg/u.s/u.cm**2*4*np.pi*np.power(
                    data[dist_features_dict[distance_feature][0]].values*u.pc, 2)).to(u.erg/u.s).value
            else:
                data[col+'_lum'] = ((data[col].values)*u.erg/u.s/u.cm**2*4*np.pi*np.power(
                    data[dist_features_dict[distance_feature][0]].values*u.pc, 2)).to(u.erg/u.s).value

    # for GC data, use distance to GC to calculate luminosities
    # for AGN, use distance to GC of evaluation data to calculate luminosities
    if distance_feature == 'gc_dist':
        for col in CSC_flux_features:
            data[col.replace('F', 'L')] = ((data[col].values)*u.erg/u.s/u.cm**2*4*np.pi*np.power(
                data[dist_features_dict[distance_feature][0]].values*u.kpc, 2)).to(u.erg/u.s).value
            if 'Class' in data.columns:
                data.loc[data['Class'] == 'AGN', col.replace('F', 'L')] = (
                    (data.loc[data['Class'] == 'AGN', col].values)*u.erg/u.s/u.cm**2*4*np.pi*np.power(distance_value*u.kpc, 2)).to(u.erg/u.s).value
        for col in hugs_features:
            data[col + '_lum'] = ((data[col].values)*u.erg/u.s/u.cm**2*4*np.pi*np.power(
                data[dist_features_dict[distance_feature][0]].values*u.kpc, 2)).to(u.erg/u.s).value
            if 'Class' in data.columns:
                data.loc[data['Class'] == 'AGN', col + '_lum'] = (
                    (data.loc[data['Class'] == 'AGN', col].values)*u.erg/u.s/u.cm**2*4*np.pi*np.power(distance_value*u.kpc, 2)).to(u.erg/u.s).value

    # if distance == 'plx_lum':
        # data['Lcsc_b'] = (data['Lcsc_b']*4*np.pi*np.power(1/data[dist_features[0]], 2)).astype('float64')
        # data['Gmag_lum'] = (data['Gmag']*4*np.pi*np.power(1/data[dist_features[0]], 2)).astype('float64')
        # data['Jmag_lum'] = (data['Jmag']*4*np.pi*np.power(1/data[dist_features[0]], 2)).astype('float64')

    return data


def standardize_log(data, by='Lcsc_b', gc=False, distance_feature='nodist'):
    # standardizing data by dividing all flux features (except by feature-EP057Flux) by 'by'-broad band flux to mitigate the impact of their unknown distances
    # if using gc_dist, then just log flux features

    cols = dist_features_dict[distance_feature][1:
                                                ] if distance_feature != 'nodist' else []

    if gc:
        cols = CSC_flux_features + hugs_features + cols
    else:
        cols = Flux_features.copy() + cols

    if gc and distance_feature == 'gc_dist':
        for col in cols:
            data[col] = np.log10(data[col])
    else:
        cols.remove(by)
        for col in cols:
            data[col] = np.log10((data[col]/data[by]).astype(float))
        data[by] = np.log10(data[by].astype(float))

    return data


def postprocessing(data,
                   Xcat='CSC',
                   gc=False,
                   distance_feature='nodist',
                   distance_value=None,
                   add_cols=['Class', 'name'],
                   apply_limit=True,
                   mag2flux_switch=True,
                   stand_switch=True,
                   Xfeature_only=False,
                   color_select=True):
    '''
    description:
        postprocess the data to be fed into classifier

    input:
        data: the input DataFrame
        Xcat: 'CSC' or 'XMM' based
        distance: 
        add_cols: columns to be added besides the features used to be trained

    output: the DataFrame after post-processing

    '''

    # apply_limit = True # apply the magnitude limit cuts if apply_limit=True, otherwise not

    # Create colors from magnitudes and apply magnitude limit cut if apply_limit=True
    data, created_colors = create_colors(data, apply_limit=apply_limit, gc=gc)

    # Create X-ray features defined in X_features
    data = create_Xfeatures(data)

    if mag2flux_switch:
        # Convert MW magnitudes to flux in erg/s/cm^2
        data = mag2flux(data, gc=gc)

    if distance_feature != 'nodist':
        data = luminosity(data, distance_feature,
                          distance_value=distance_value)

    # all flux features are divided by broad band X-ray flux for standardization except for Fb
    standidize_by = 'Lcsc_b'
    if stand_switch:
        # Standardizing data by dividing flux features (except EP057Flux) by EP057Flux
        data = standardize_log(data, by=standidize_by,
                               gc=gc, distance_feature=distance_feature)

    # if Xcat == 'CSC':
    #     if Xfeature_only == False:
    #         if color_select == True:
    #             data = data[CSC_all_features +
    #                         dist_features_dict[distance]+add_cols]
    #         elif color_select == False:
    #             data = data[CSC_all_features_test +
    #                         dist_features_dict[distance]+add_cols]
    #     elif Xfeature_only == True:
    #         data = data[CSC_features+dist_features_dict[distance]+add_cols]
    # if Xcat == 'XMM':
    #     data = data[XMM_all_features+add_cols]

    if Xcat == 'CSC':
        # make sure to exclude distance itself as a feature
        features = CSC_features + \
            dist_features_dict[distance_feature][1:] + add_cols
        if stand_switch and distance_feature != 'gc_dist':
            features.remove(standidize_by)
        if Xfeature_only == False:
            if gc:
                features = features + hugs_features + created_colors
                # remove fluxes if luminosities are added
                features = [
                    feature for feature in features if feature not in CSC_flux_features + hugs_features]
            else:
                if color_select == True:
                    features = features + gaia_features + \
                        twomass_features + ['W1mag', 'W2mag'] + colors
                elif color_select == False:
                    features = features + MW_features + colors_all
        else:
            if gc:
                features = [feature for feature in features if feature not in [
                    'F275W_lum', 'F336W_lum', 'F438W_lum', 'F606W_lum', 'F814W_lum']]
    if Xcat == 'XMM':
        features = XMM_all_features + add_cols

    data = data[features]

    return data


def scaling(scaler, X_train, unscales, verb=False):
    # apply scaler on training set and other data
    # default = StandardScaler to remove the mean and scale to unit variance
    if verb:
        print("Run scaling......")
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    if verb:
        print("Train DS transformed shape: {}".format(X_train_scaled.shape))

    X_train_scaled_df = pd.DataFrame(
        X_train_scaled, index=X_train.index, columns=X_train.columns)

    scaleds = []
    for un_scale in unscales:
        scaled = scaler.transform(un_scale)
        scaled_df = pd.DataFrame(scaled, columns=X_train.columns)
        scaleds.append(scaled_df)

        if verb:
            print("Transformed shape: {}".format(un_scale.shape))

    return X_train_scaled_df, scaleds


def oversampling(method, X_train, y_train):
    # oversampling training dataset to mitigate for the imbalanced TD
    # default = SMOTE

    X_train.replace(np.nan, exnum, inplace=True)
    X_res, y_res = method.fit_resample(X_train, y_train)
    res = X_res.values

    X_train[X_train == exnum] = np.nan
    X_train_min = np.nanmin(X_train, axis=0)

    for i in np.arange(len(res[:, 0])):
        for j in np.arange(len(res[0, :])):
            if res[i, j] < X_train_min[j]:
                res[i, j] = np.nan

    X_res[:] = res
    return X_res, y_res


def loo_prepare(args):

    (i, df, red_switch, Xcat, gc, distance_feature, Uncer_flag, ran_feature, random_state_sample, random_state_smote, tbabs_ene, tbabs_cross,
     apply_limit, mag2flux_switch, stand_switch, oversample_switch, scaler_switch, color_select, ran_factor, physical_sample, df_reds, Xfeature_only) = args

    print(i)

    df_test = df[df.name == df.name[i]]
    df_train = df[df.name != df.name[i]]

    df_backup = df_train.copy()

    field_ra = df_test['ra'].values
    field_dec = df_test['dec'].values

    df_test = sample_data(df_test, Xcat, gc=gc, distance_feature=distance_feature, Uncer_flag=Uncer_flag,
                          random_state=random_state_sample, rep_num=False, factor=ran_factor)
    df_train = sample_data(df_train, Xcat, gc=gc, distance_feature=distance_feature, Uncer_flag=Uncer_flag,
                           random_state=random_state_sample, factor=ran_factor)

    if physical_sample == True:
        # df_test = test_reddening_grid_csv(df_test, df_reds, random_state=random_state_sample)
        df_train = physical_oversample_csv(
            df_train, df_reds, random_state=random_state_sample, ebv_pdf='gamma')

    if Xcat == 'XMM':
        df_test = convert2csc(df_test, method='simple', Gamma=2.)
        df_train = convert2csc(df_train, method='simple', Gamma=2.)

    '''
    df = sample_data(df,Xcat,distance,Uncer_flag,random_state_sample)

    df_test = df[df.name == df.name[i]]
    df_train = df[df.name != df.name[i]]
    '''

    if red_switch:

        # Extract reddening parameters from SFD dustmap & DL HI map
        ebv, nh = get_red_par(field_ra, field_dec)

        deredden = gc
        # Applying reddening to AGNs
        red_class = ['AGN']

        data_red2csc = apply_red2csc(
            df_train.copy(), nh, tbabs_ene, tbabs_cross, red_class=red_class, deredden=deredden, self_unred=False, Gamma=2)
        df_train = apply_red2mw(
            data_red2csc, ebv, red_class=red_class, deredden=deredden, self_unred=False, gc=gc)

    # calculate luminosity of AGNs assuming distance to test source
    # if test source has no distance, assume 4 kpc
    if gc:
        distance_value = df_test['HELIO_DISTANCE'][i]
        if np.isnan(distance_value):
            distance_value = 4
    else:
        distance_value = None

    df_train = postprocessing(df_train, Xcat, gc, distance_feature, distance_value=distance_value, add_cols=[
                              'Class', 'name'], apply_limit=apply_limit, mag2flux_switch=mag2flux_switch, stand_switch=stand_switch, Xfeature_only=Xfeature_only, color_select=color_select)
    df_test = postprocessing(df_test, Xcat, gc, distance_feature, distance_value=distance_value, add_cols=[
                             'Class', 'name'], apply_limit=apply_limit, mag2flux_switch=mag2flux_switch, stand_switch=stand_switch, Xfeature_only=Xfeature_only, color_select=color_select)

    X_train, y_train = df_train.drop(['Class', 'name'], axis=1), df_train.Class
    X_test, y_test, test_name = df_test.drop(
        ['Class', 'name'], axis=1), df_test.Class, df_test.name

    if scaler_switch == True:
        X_train, [X_test] = scaling(ML_scaler, X_train, [X_test])

    # save unsmoted TD for XCLASS
    # df_save = X_train.merge(df_backup, left_index=True, right_index=True, suffixes=(None, 'notprocessed'))
    # rename_cols = {'name':'name_csc', 'ra':'ra_csc', 'dec':'dec_csc', 'Fcsc_b':'flux_aper90_avg_b', 'Fcsc_s':'flux_aper90_avg_s', 'Fcsc_m':'flux_aper90_avg_m', 'Fcsc_h':'flux_aper90_avg_h', 'Lcsc_b':'lum_aper90_avg_b', 'Lcsc_b':'lum_aper90_avg_s', 'Lcsc_m':'lum_aper90_avg_m', 'Lcsc_h':'lum_aper90_avg_h', 'var_inter_prob':'var_inter_prob_b','var_intra_prob':'var_intra_prob_b'}
    # rename_cols.update({col+'_lum':col+'_HUGS_abs' for col in hugs_features})
    # rename_cols.update({col:col+'_HUGS' for col in ['F275W-F336W', 'F275W-F438W', 'F275W-F606W', 'F275W-F814W', 'F336W-F438W', 'F336W-F606W', 'F336W-F814W', 'F438W-F606W', 'F438W-F814W', 'F606W-F814W']})
    # df_save = df_save.rename(columns=rename_cols)
    # df_save = df_save.fillna(-10)
    # df_save.to_csv(f'./data/LOO/td_gc_csc_hugs_rescaled.csv',index=False)

    if oversample_switch == True:
        ML_oversampler = SMOTE(
            random_state=random_state_smote, k_neighbors=4, n_jobs=-1)
        # ML_oversampler = KMeansSMOTE(random_state=random_state_smote, k_neighbors=4, n_jobs=-1)
        # ML_oversampler = ADASYN(random_state=random_state_smote, n_neighbors=4, n_jobs=-1)

        X_train, y_train = oversampling(ML_oversampler, X_train, y_train)
    # print(Counter(y_train))
    X_train = X_train.fillna(-100)
    X_test = X_test.fillna(-100)
    # print(X_train)

    if ran_feature == 'normal':
        X_train['ran_fea'] = np.random.randn(X_train.shape[0])
        X_test['ran_fea'] = np.random.randn(X_test.shape[0])
    elif ran_feature == 'uniform':
        X_train['ran_fea'] = np.random.rand(X_train.shape[0])
        X_test['ran_fea'] = np.random.rand(X_test.shape[0])

    return [i, X_train, y_train, X_test, y_test, test_name]


def class_prepare_FGL(args):
    '''
    description:

    preprocessing both TD and unclassified X-ray sources from a particular region 
    (like a FGL field or a cluster so an uniformed extinction/absorption parameter (ebv) can be applied)

    input:

    TD: Training dataset DataFrame
    field: field dataframe
    red_switch: reddening switch, if set to False, no reddening will be applied to AGN in TD to accomodate the reddening bias of AGN
    ebv: E(B-V) extinction parameter, nH is calculated as 2.21 * 3.1 * ebv 
    Xcat: 'CSC' for Chandra Source Catalog, 'XMM' for XMM 
    distance: if set to True, the distance will be used to calculate luminosity as features 
    Uncer_flag: if set to False, Monte-Carlo Sampling will not be used to take into account the measurement uncertainties 
    random_state_sample: random_state for MC sampling 
    random_state_smote: random_state for SMOTE 
    tbabs_ene, tbabs_cross: absorption files of energy and cross-section
    apply_limit: apply the magnitude limit cuts if apply_limit=True, otherwise not
    mag2flux_switch: if set to True, convert MW magnitudes to flux in erg/s/cm^2
    standard_switch: if set to True, standardizing data by dividing flux features (except EP057Flux) by EP057Flux
    oversample_switch: if set to True, apply SMOTE oversampling, nothing will change if physical_sample is already set to True
    scaler_switch: if set to True, standard scaling will be applied to both TD and field data
    color_select: if set to True, feature selection will be applied to remove unimportant (color) features
    ran_factor: the factor multiplied to uncertainties, default is sqrt(2)
    physical_sample: if set to True, physical oversampling will be applied
    df_reds: reddening dataframe needed for physical oversampling
    Xfeature_only: if set to True, only X-ray features will be used for training
    missingvalues_replace: if set to True, missing values will be replaced to -100
    '''

    (TD, field, red_switch, ebv, Xcat, gc, distance_feature, Uncer_flag, random_state_sample, random_state_smote, tbabs_ene, tbabs_cross,
     apply_limit, mag2flux_switch, standard_switch, oversample_switch, scaler_switch, color_select, ran_factor, physical_sample, df_reds, Xfeature_only, missingvalues_replace) = args

    TD = sample_data(TD, Xray=Xcat, gc=gc, distance_feature=distance_feature, Uncer_flag=Uncer_flag,
                     random_state=random_state_sample, factor=ran_factor)
    field = sample_data(field, Xray=Xcat, gc=gc, distance_feature=distance_feature, Uncer_flag=Uncer_flag,
                        random_state=random_state_sample, factor=ran_factor)

    if physical_sample == True:
        # df_test = test_reddening_grid_csv(df_test, df_reds, random_state=random_state_sample)
        TD = physical_oversample_csv(
            TD, df_reds, random_state=random_state_sample, ebv_pdf='gamma')

    # ['name','ra','dec','Class','flux_aper90_avg_b','flux_aper90_avg_h','flux_aper90_avg_m','flux_aper90_avg_s','var_inter_prob','kp_prob_b_max','Signif.','Gmag','BPmag','RPmag','Jmag','Hmag','Kmag','W1mag_comb','W2mag_comb','W3mag']].to_csv(f'{dir_out}/{field_name}_XCLASS.csv')

    # for col in ['ra_cat', 'dec_cat','ref','significance_max','rgeo']:
        # TD[col] = np.nan
    # TD.rename(columns={'Fcsc_s':'F_s','Fcsc_m':'F_m','Fcsc_h':'F_h','Fcsc_b':'F_b','var_inter_prob':'P_inter','var_intra_prob':'P_intra','significance_max':'significance','Gmag':'G','BPmag':'BP','RPmag':'RP','Jmag':'J','Hmag':'H','Kmag':'K','W1mag':'W1','W2mag':'W2','W3mag':'W3'})[['name','ra','dec','Class','F_b','F_h','F_m','F_s','P_inter','P_intra','significance','G','BP','RP','J','H','K','W1','W2','W3','ref','rgeo']].to_csv('TD_physical_oversample.csv',index=False)

    if red_switch:

        # Extract reddening parameters from SFD dustmap & DL HI map
        # ebv, nh = get_red_par(field_ra, field_dec)

        nh = 2.21 * 3.1 * ebv
        red_class = ['AGN']

        # Applying reddening to AGNs
        # if GC, deredden all sources instead

        deredden = gc
        TD_red2csc = apply_red2csc(
            TD.copy(), nh, tbabs_ene, tbabs_cross, red_class=red_class, deredden=deredden, self_unred=False, Gamma=2)
        TD = apply_red2mw(TD_red2csc, ebv, red_class=red_class, deredden=deredden, self_unred=False, gc=gc)
        
        if deredden:
            field_red2csc = apply_red2csc(field.copy(), nh, tbabs_ene, tbabs_cross, red_class=red_class, deredden=deredden, self_unred=False, Gamma=2)
            field = apply_red2mw(field_red2csc, ebv, red_class=red_class, deredden=deredden, self_unred=False, gc=gc)

    if gc:
        distance_value = field['HELIO_DISTANCE'].reset_index(drop=True)[0]
    else:
        distance_value = None
    TD = postprocessing(TD, Xcat, gc=gc, distance_feature=distance_feature, distance_value=distance_value, add_cols=[
                        'Class', 'name'], apply_limit=apply_limit, mag2flux_switch=mag2flux_switch, stand_switch=standard_switch, Xfeature_only=Xfeature_only, color_select=color_select)
    field = postprocessing(field, Xcat, gc=gc, distance_feature=distance_feature, distance_value=distance_value, add_cols=[
                           'name'], apply_limit=apply_limit, mag2flux_switch=mag2flux_switch, stand_switch=standard_switch, Xfeature_only=Xfeature_only, color_select=color_select)

    field.to_csv(f'field_processed.csv', index=False)

    X_train, y_train, train_name = TD.drop(
        ['Class', 'name'], axis=1), TD.Class, TD.name
    X_test, test_name = field.drop('name', axis=1), field.name

    if scaler_switch == True:
        X_train, [X_test] = scaling(ML_scaler, X_train, [X_test])
    if oversample_switch == True:
        ML_oversampler = SMOTE(
            random_state=random_state_smote, k_neighbors=4)  # , n_jobs=-1)
        X_train, y_train = oversampling(ML_oversampler, X_train, y_train)

    if missingvalues_replace == True:
        X_train = X_train.fillna(-100)
        X_test = X_test.fillna(-100)

    '''
    if save_file != False:
        X_train_save = X_train.copy()
        X_train_save['name'] = TD['name']
        X_train_save['Class'] = TD['Class']
        X_train_save['Class_prob'] = 1.
        X_train_save.to_csv(f'{save_file}TD_checking.csv',index=False)
        #X_test_save = X_test.copy()
        #X_test_save['name'] = field['name']
        #X_test_save.to_csv(f'{save_file}test_checking.csv',index=False)
    '''
    return [X_train, y_train, X_test, test_name, train_name]


def class_prepare_eRASS1(args):
    '''
    description:

    preprocessing both TD and unclassified X-ray sources from a particular region 
    (like a FGL field or a cluster so an uniformed extinction/absorption parameter (ebv) can be applied)

    input:

    TD: Training dataset DataFrame
    field: field dataframe
    red_switch: reddening switch, if set to False, no reddening will be applied to AGN in TD to accomodate the reddening bias of AGN
    ebv: E(B-V) extinction parameter, nH is calculated as 2.21 * 3.1 * ebv 
    Xcat: 'CSC' for Chandra Source Catalog, 'XMM' for XMM 
    distance: if set to True, the distance will be used to calculate luminosity as features 
    Uncer_flag: if set to False, Monte-Carlo Sampling will not be used to take into account the measurement uncertainties 
    random_state_sample: random_state for MC sampling 
    random_state_smote: random_state for SMOTE 
    tbabs_ene, tbabs_cross: absorption files of energy and cross-section
    apply_limit: apply the magnitude limit cuts if apply_limit=True, otherwise not
    mag2flux_switch: if set to True, convert MW magnitudes to flux in erg/s/cm^2
    standard_switch: if set to True, standardizing data by dividing flux features (except EP057Flux) by EP057Flux
    oversample_switch: if set to True, apply SMOTE oversampling, nothing will change if physical_sample is already set to True
    scaler_switch: if set to True, standard scaling will be applied to both TD and field data
    color_select: if set to True, feature selection will be applied to remove unimportant (color) features
    ran_factor: the factor multiplied to uncertainties, default is sqrt(2)
    physical_sample: if set to True, physical oversampling will be applied
    df_reds: reddening dataframe needed for physical oversampling
    Xfeature_only: if set to True, only X-ray features will be used for training
    missingvalues_replace: if set to True, missing values will be replaced to -100
    '''

    (i, TD, field, random_state_sample, random_state_smote,oversample_switch, missingvalues_replace,ran_feature, used_cols) = args
    #  scaler_switch, 
    ran_factor = np.sqrt(2.)
    if random_state_sample is None:
        np.random.seed(randint(1, 999999999))
    else:
        np.random.seed(random_state_sample)

    for col in used_cols:
        TD[col] = np.random.randn(TD[col].size) * \
            TD['e_'+col] * ran_factor + TD[col]
        field[col] = np.random.randn(field[col].size) * \
            field['e_'+col] * ran_factor + field[col]
        TD = TD.drop(columns=['e_'+col])
        field = field.drop(columns=['e_'+col])
    
    # TD = postprocessing(TD, Xcat, gc=gc, distance_feature=distance_feature, distance_value=distance_value, add_cols=['Class', 'name'], apply_limit=apply_limit, mag2flux_switch=mag2flux_switch, stand_switch=standard_switch, Xfeature_only=Xfeature_only, color_select=color_select)
    # field = postprocessing(field, Xcat, gc=gc, distance_feature=distance_feature, distance_value=distance_value, add_cols=['name'], apply_limit=apply_limit, mag2flux_switch=mag2flux_switch, stand_switch=standard_switch, Xfeature_only=Xfeature_only, color_select=color_select)

    X_train, y_train, train_name = TD.drop(
        ['Class', 'name'], axis=1), TD.Class, TD.name
    X_test, test_name = field.drop('name', axis=1), field.name

    # if scaler_switch == True:
    #     X_train, [X_test] = scaling(ML_scaler, X_train, [X_test])

    if oversample_switch == True:
        ML_oversampler = SMOTE(
            random_state=random_state_smote, k_neighbors=4)  # , n_jobs=-1)
        X_train, y_train = oversampling(ML_oversampler, X_train, y_train)

    if missingvalues_replace == True:
        X_train = X_train.fillna(-100)
        X_test = X_test.fillna(-100)

    if ran_feature == 'normal':
        X_train['ran_fea'] = np.random.randn(X_train.shape[0])
        X_test['ran_fea'] = np.random.randn(X_test.shape[0])
    elif ran_feature == 'uniform':
        X_train['ran_fea'] = np.random.rand(X_train.shape[0])
        X_test['ran_fea'] = np.random.rand(X_test.shape[0])

    return [ X_train, y_train, X_test, test_name, train_name]



def mw_counterpart_flag(df, mw_cols=['Gmag', 'BPmag', 'RPmag', 'Jmag', 'Hmag', 'Kmag', 'W1mag_comb', 'W2mag_comb', 'W3mag_allwise']):

    df['mw_cp_flag'] = 0
    df = df.fillna(exnum)  # df.replace(np.nan, exnum, inplace=True)

    for i, mw_col in enumerate(mw_cols):
        df['mw_cp_flag'] = df.apply(
            lambda row: row.mw_cp_flag+2**i if row[mw_col] != exnum else row.mw_cp_flag, axis=1)

    df = df.replace(exnum, np.nan)

    return df



def convert_standard(df_convert, ra='RA',dec='DEC',ra_stack='RA_stack',dec_stack='DEC_stack',xi='xi',eta='eta',stack_col=False,inverse=False):
    
    if stack_col==False:
        if inverse==False:

            c1 = SkyCoord(ra=df_convert[ra]*u.deg, dec=df_convert[dec]*u.deg, frame='icrs')

            df_convert[xi] = 3600.*(180./np.pi)*(np.cos(c1.dec.rad)*np.sin(c1.ra.rad - (ra_stack*np.pi/180.))) \
                        /(np.sin(c1.dec.rad)*np.sin(dec_stack*np.pi/180.)+np.cos(c1.dec.rad)*np.cos(dec_stack*np.pi/180.)*np.cos(c1.ra.rad - (ra_stack*np.pi/180.)))

            df_convert[eta] = 3600.*(180./np.pi)*(np.sin(c1.dec.rad)*np.cos(dec_stack*np.pi/180.)-np.cos(c1.dec.rad)*np.sin(dec_stack*np.pi/180.)*np.cos(c1.ra.rad - (ra_stack*np.pi/180.))) \
                        /(np.sin(c1.dec.rad)*np.sin(dec_stack*np.pi/180.)+np.cos(c1.dec.rad)*np.cos(dec_stack*np.pi/180.)*np.cos(c1.ra.rad - (ra_stack*np.pi/180.)))

        elif inverse==True:

            df_convert[ra] = np.arctan((df_convert[xi]*np.pi/(3600.*180.))/(np.cos(dec_stack*np.pi/180.)-(df_convert[eta]*np.pi/(3600.*180.))*np.sin(dec_stack*np.pi/180.)))*180./np.pi + ra_stack

            df_convert[dec] = np.arcsin( (np.sin(dec_stack*np.pi/180.) + (df_convert[eta]*np.pi/(3600.*180.))*np.cos(dec_stack*np.pi/180.)) / np.sqrt(1.+(df_convert[xi]*np.pi/(3600.*180.))**2+(df_convert[eta]*np.pi/(3600.*180.))**2) )*180./np.pi
    
    elif stack_col==True:
        if inverse==False:

            c1 = SkyCoord(ra=df_convert[ra]*u.deg, dec=df_convert[dec]*u.deg, frame='icrs')
            c_stack = SkyCoord(ra=df_convert[ra_stack]*u.deg, dec=df_convert[dec_stack]*u.deg, frame='icrs')

            df_convert[xi] = 3600.*(180./np.pi)*(np.cos(c1.dec.rad)*np.sin(c1.ra.rad - (c_stack.ra.rad))) \
                        /(np.sin(c1.dec.rad)*np.sin(c_stack.dec.rad)+np.cos(c1.dec.rad)*np.cos(c_stack.dec.rad)*np.cos(c1.ra.rad - (c_stack.ra.rad)))

            df_convert[eta] = 3600.*(180./np.pi)*(np.sin(c1.dec.rad)*np.cos(c_stack.dec.rad)-np.cos(c1.dec.rad)*np.sin(c_stack.dec.rad)*np.cos(c1.ra.rad - (c_stack.ra.rad))) \
                        /(np.sin(c1.dec.rad)*np.sin(c_stack.dec.rad)+np.cos(c1.dec.rad)*np.cos(c_stack.dec.rad)*np.cos(c1.ra.rad - (c_stack.ra.rad)))

        elif inverse==True:

            df_convert[ra] = np.arctan((df_convert[xi]*np.pi/(3600.*180.))/(np.cos(df_convert[dec_stack]*np.pi/180.)-(df_convert[eta]*np.pi/(3600.*180.))*np.sin(df_convert[dec_stack]*np.pi/180.)))*180./np.pi + df_convert[ra_stack]

            df_convert[dec] = np.arcsin( (np.sin(df_convert[dec_stack]*np.pi/180.) + (df_convert[eta]*np.pi/(3600.*180.))*np.cos(df_convert[dec_stack]*np.pi/180.)) / np.sqrt(1.+(df_convert[xi]*np.pi/(3600.*180.))**2+(df_convert[eta]*np.pi/(3600.*180.))**2) )*180./np.pi

    return df_convert


# A function to perform the transformation as a matrix multiplication
def model(pars, X):
    return np.matmul(np.array([[1.0,0.0,pars[0]],
    [0.0,1.0,pars[1]],[0.0,0.0,1.0]]),[X[0], X[1], 1.0])

# A function to estimate the residual, multiplied by the weights
def fun_residual(pars, X, Y, weights):
    return np.dot(weights, (model(pars,X)[0]-Y[0])**2
    + (model(pars,X)[1]-Y[1])**2)


def count_dist_peaks(series, bins, hist_range, prominence=None, width=None):
    count, division = np.histogram(series, bins=bins, range=hist_range)
    peaks, props = find_peaks(count, prominence=prominence, width=width)
    return peaks

def cal_PU(df, OAA, counts, PU_name, ver='kim95', confidence=0.68):

    df[PU_name] = np.nan

    # Rayleigh distribution (2-D normal distribution)
    sigma = np.sqrt(2*np.log(1./(1-confidence)))

    if ver == 'kim95':

        sigma_this = np.sqrt(2*np.log(20.))

        s1 = np.where((df[counts] <= 10**2.1393) & (df[counts]> 1))[0]
        df.loc[s1, PU_name] =  10.**(0.1145 * df.loc[s1, OAA]-0.4958*np.log10( df.loc[s1, counts])+0.1932) *sigma/sigma_this

        #s2 = np.where((df[counts] > 10**2.1393) & (df[counts] <= 10**3.30))[0]
        s2 = np.where((df[counts] > 10**2.1393))[0]
        df.loc[s2, PU_name] =  10.**(0.0968 * df.loc[s2, OAA]-0.2064*np.log10(df.loc[s2, counts])-0.4260) *sigma/sigma_this

    elif ver == 'kim68':

        sigma_this = np.sqrt(2*np.log(1./0.32))

        s1 = np.where((df[counts] <= 10**2.1227) & (df[counts]> 1))[0]
        df.loc[s1, PU_name] =  10.**(0.1137 * df.loc[s1, OAA]-0.460*np.log10( df.loc[s1, counts])-0.2398)*sigma/sigma_this

        #s2 = np.where((df[counts] > 10**2.1227) & (df[counts] <= 10**3.30))[0]
        s2 = np.where((df[counts] > 10**2.1227))[0]
        df.loc[s2, PU_name] =  10.**(0.1031 * df.loc[s2, OAA]-0.1945*np.log10(df.loc[s2, counts])-0.8034)*sigma/sigma_this

    elif ver == 'csc90':

        sigma_this = np.sqrt(2*np.log(10.))

        df[PU_name] =  10.**(0.173 * df[OAA]-0.526*np.log10(df[counts])-0.023* df[OAA]*np.log10(df[counts])-0.031) * sigma/sigma_this

    elif ver == 'hong95': # https://iopscience.iop.org/article/10.1086/496966/pdf
        sigma_this = np.sqrt(2*np.log(20.))
        df[PU_name] = (0.25 + 0.1/np.log10( df[counts] + 1 ) * (1 + 1./np.log10( df[counts] + 1 )) + 0.03 * (df[OAA]/np.log10(df[counts] + 2))**2 + 0.0006 * (df[OAA]/np.log10(df[counts] + 3))**4) *sigma/sigma_this
    
    return df 

from pathlib import Path
from matplotlib.patches import Ellipse
from numpy import linalg as LA
import math

class ellipse_class():   
    def __init__(self):
        self.alpha = 0.
        self.delta = 0. 
        self.phi_major = 1. 
        self.phi_minor = 1. 
        self.theta_cel = 0. # celestial
        self.x = 0. 
        self.y = 0. 
        self.sigma_major = 0. 
        self.sigma_minor = 0. 
        self.theta = 0. # tangent plane
        self.p_hat = 0. 
        self.alpha_hat = 0. 
        self.delta_hat = 0. 
        self.p_major = 0.
        self.p_minor = 0.



def new_ellipse(alpha, delta, phi_major, phi_minor, theta_cel):
    
    e = ellipse_class()

    e.alpha     = alpha
    e.delta     = delta
    e.phi_major = phi_major
    e.phi_minor = phi_minor
    e.theta_cel = theta_cel
    ca = np.cos(alpha)
    cd = np.cos(delta)
    sa = np.sin(alpha)
    sd = np.sin(delta)
    e.p_hat = np.array([ca*cd, sa*cd, sd])
    e.alpha_hat = np.array([-sa, ca, 0.])
    e.delta_hat = np.array([-sd*ca, -sd*sa, cd])
    e.p_minor = e.p_hat*np.cos(e.phi_minor) + e.alpha_hat*np.sin(e.phi_minor)*np.cos(e.theta_cel)\
        - e.delta_hat*np.sin(e.phi_minor)*np.sin(e.theta_cel) # eq 35
    e.p_major = e.p_hat*np.cos(e.phi_major) + e.alpha_hat*np.sin(e.phi_major)*np.sin(e.theta_cel)\
        + e.delta_hat*np.sin(e.phi_major)*np.cos(e.theta_cel) # eq 36

    return e
   
      
def print_ellipse(e):
    
    print('ellipse')
    print(e.alpha)
    print(e.delta)
    print(e.phi_major)
    print(e.phi_minor)
    print(e.theta_cel)
    print(e.x)
    print(e.y)
    print(e.sigma_major)
    print(e.sigma_minor)
    print(e.theta)
    print(e.p_hat)
    print(e.alpha_hat)
    print(e.delta_hat)
    print(e.p_major)
    print(e.p_minor)
          
def get_tangent_plane_from_vector(p0):
    
    alpha0 = np.arctan2(p0[1], p0[0]) # eq 32
    delta0 = np.arcsin(p0[2]) # eq 32
    ex_hat = np.array([-np.sin(alpha0), np.cos(alpha0), 0.]) # eq 41
    ey_hat = np.array([-np.sin(delta0)*np.cos(alpha0), -np.sin(delta0)*np.sin(alpha0), np.cos(delta0)]) # eq 42
    return p0, ex_hat, ey_hat

def get_tangent_plane_from_ellipses(ellipses):
    
    p0 = 0
    for e in ellipses:
        p0 += e.p_hat
        
    p0 /= LA.norm(p0)
    
    return get_tangent_plane_from_vector(p0)

def project_ellipse(e, p0_hat, ex_hat, ey_hat):
    
    p = e.p_hat
    p_dot_p0 = np.dot(p, p0_hat)
    xa =  np.dot(p, ex_hat)/p_dot_p0 # eq 50
    ya =  np.dot(p, ey_hat)/p_dot_p0 # eq 51
    
    p = e.p_major
    p_dot_p0 = np.dot(p, p0_hat)
    xa_major =  np.dot(p, ex_hat)/p_dot_p0 # eq 50
    ya_major =  np.dot(p, ey_hat)/p_dot_p0 # eq 51
    
    p = e.p_minor
    p_dot_p0 = np.dot(p, p0_hat)
    xa_minor = np.dot(p, ex_hat)/p_dot_p0 # eq 50
    ya_minor = np.dot(p, ey_hat)/p_dot_p0 # eq 51
    
    e.x = xa
    e.y = ya
    #print('p_dot_p0, p, ex_hat, xa, e.x',p_dot_p0, p, ex_hat, xa, e.x)
    e.sigma_major = math.hypot(xa_major-xa, ya_major-ya) # eq 52
    e.sigma_minor = math.hypot(xa_minor-xa, ya_minor-ya) # eq 53
    e.theta = np.arctan2(xa_major-xa, ya_major-ya) # eq 54
    

def deproject_ellipse(e, p0, ex_hat, ey_hat):
    #print(p0, e.x, ex_hat)
    p = p0 + e.x*ex_hat + e.y*ey_hat
    p /= LA.norm(p) # eq 46
    e.p_hat = p
    e.alpha = np.arctan2(p[1], p[0]) # eq 32
    e.delta = np.arcsin(p[2])
    
    x_major = e.x + e.sigma_major*np.sin(e.theta) # eq 55
    y_major = e.y + e.sigma_major*np.cos(e.theta)
    p = p0 + x_major*ex_hat + y_major*ey_hat
    e.p_major = p/LA.norm(p) # eq 46
    
    x_minor = e.x + e.sigma_minor*np.cos(e.theta) # eq 56
    y_minor = e.y - e.sigma_minor*np.sin(e.theta)
    p = p0 + x_minor*ex_hat + y_minor*ey_hat
    e.p_minor = p/LA.norm(p) # eq 46
    
    ca, cd, sa, sd = np.cos(e.alpha), np.cos(e.delta), np.sin(e.alpha), np.sin(e.delta)
    e.alpha_hat = [-sa, ca, 0.] # eq 33
    e.delta_hat = [-sd*ca, -sd*sa, cd] # eq 34
    # Equations 37, 38, 39
    e.theta_cel = np.arctan(np.dot(e.p_major, e.alpha_hat) / np.dot(e.p_major, e.delta_hat))
    e.phi_major = np.arccos(np.dot(e.p_major, e.p_hat))
    e.phi_minor = np.arccos(np.dot(e.p_minor, e.p_hat))
    
# Implements eq 27
def ellipse_to_correlation_matrix(e):

    sigy2 = e.sigma_major**2
    sigx2 = e.sigma_minor**2
    c = np.cos(e.theta)
    s = np.sin(e.theta)
    c2,s2 = c*c, s*s
    sx2 = sigx2*c2 + sigy2*s2
    sy2 = sigx2*s2 + sigy2*c2
    rho_sxsy = c*s*(sigy2-sigx2)
    #print(sx2, rho_sxsy, rho_sxsy, sy2)
    #print(np.array([sx2, rho_sxsy, rho_sxsy, sy2]).reshape((2, 2)))
    return np.array([sx2, rho_sxsy, rho_sxsy, sy2]).reshape((2, 2))


# Implements equations 28, 29, 30
def correlation_matrix_to_ellipse(matrix, x0, y0):

    sx2 = matrix[0,0]
    sy2 = matrix[1,1]
    rho2_sxsy = 2.*matrix[0,1]
    sum_S = sy2+sx2
    diff = sy2-sx2
    e = ellipse_class()
    e.x, e.y = x0, y0
    #print('x0, y0',x0, y0)
    e.theta = 0.5*np.arctan2(rho2_sxsy, diff)
    diff = math.hypot(diff, rho2_sxsy)
    e.sigma_major = np.sqrt(0.5*(sum_S + diff))
    e.sigma_minor = np.sqrt(0.5*(sum_S - diff))
    return e

def inverse_2x2(a):

    det = a[0,0] * a[1,1] - a[0,1]*a[1,0]
    if (det == 0.0):
        print("matrix is singular")
        return None
    else:
        a1 = np.empty((2,2), dtype=np.float128)
        a1[0,0] = a[1,1]
        a1[0,1] = -a[0,1]
        a1[1,0] = -a[1,0]
        a1[1,1] = a[0,0]
        return a1/det

# Implememts eq 24
def combine_ellipses(es):

    num = len(es)
    Cinv = 0
    mu = 0
    for i in range(num):
        e = es[i]
        #print_ellipse(e)
        #print('check from here!!!')
        C_m = ellipse_to_correlation_matrix(e)
        Cinv_m = inverse_2x2(C_m)
        mu += np.matmul(Cinv_m, [e.x, e.y])
        Cinv += Cinv_m
    #print('mu',mu)
    #print('Cinv',Cinv)

    C = inverse_2x2(Cinv)
    mu = np.matmul(C, mu)
    #print(mu)
    #print(C)
    return correlation_matrix_to_ellipse(C, mu[0], mu[1]) ### here mu[0], mu[1]

def deg2hms(alpha, delta):
    
    cc2 = SkyCoord(alpha*u.deg,delta*u.deg, frame='icrs')
    #cc2_string = cc2.to_string('hmsdms')
    ra_hms = f'{int(cc2.ra.hms.h):02d} {int(cc2.ra.hms.m):02d} {cc2.ra.hms.s:4.2f}'
    dec_dms = f'{int(cc2.dec.dms.d):02d} {abs(int(cc2.dec.dms.m)):02d} {abs(cc2.dec.dms.s):4.2f}'
    #print(f'    {int(cc2.ra.hms.h):02d} {int(cc2.ra.hms.m):02d} {cc2.ra.hms.s:4.2f}  {int(cc2.dec.dms.d):02d} {abs(int(cc2.dec.dms.m)):02d} {abs(cc2.dec.dms.s):4.2f}            {major:.2f}            {minor:.2f}           {theta:.2f}')
    return ra_hms, dec_dms

def deg2hms_print(alpha, delta, major, minor, theta):
    
    cc2 = SkyCoord(alpha*u.deg,delta*u.deg, frame='icrs')
    #cc2_string = cc2.to_string('hmsdms')
    ra_hms = f'{int(cc2.ra.hms.h):02d} {int(cc2.ra.hms.m):02d} {cc2.ra.hms.s:4.2f}'
    dec_dms = f'{int(cc2.dec.dms.d):02d} {abs(int(cc2.dec.dms.m)):02d} {abs(cc2.dec.dms.s):4.2f}'
    #print(f'    {int(cc2.ra.hms.h):02d} {int(cc2.ra.hms.m):02d} {cc2.ra.hms.s:4.2f}  {int(cc2.dec.dms.d):02d} {abs(int(cc2.dec.dms.m)):02d} {abs(cc2.dec.dms.s):4.2f}            {major:.2f}            {minor:.2f}           {theta:.2f}')
    return f'RA:{int(cc2.ra.hms.h):02d} {int(cc2.ra.hms.m):02d} {cc2.ra.hms.s:4.2f} DEC:{int(cc2.dec.dms.d):02d} {abs(int(cc2.dec.dms.m)):02d} {abs(cc2.dec.dms.s):4.2f} major:{major:.2f} minor:{minor:.2f} ang:{theta:.2f}'


def slsh_main(alphas, deltas, phimajors, phiminors, thetas, plot_limit=1,err_unit='arcsec',add_sys=False,plotting=True):

    alphas, deltas, phimajors, phiminors, thetas = alphas.copy(), deltas.copy(), phimajors.copy(), phiminors.copy(), thetas.copy()
    # convert to radians
    rad_per_deg = np.pi/180.0
    if err_unit == 'arcsec':
        err_factor = 3600.
    elif err_unit == 'arcmin':
        err_factor = 60.
    alphas *= rad_per_deg
    deltas *= rad_per_deg
    phimajors *= rad_per_deg/err_factor
    #phimajors /= 60.0
    phiminors *= rad_per_deg/err_factor
    #phiminors /= 60.0
    thetas *= rad_per_deg
    
    ellipses = []
    for i in range(len(alphas)):
        
    
        e = new_ellipse(alphas[i], deltas[i], phimajors[i], phiminors[i], thetas[i])
        #print_ellipse(e)
        ellipses.append(e)
    
    if plotting:
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'aspect': 'equal'})
    #print(ellipses)
    (p0, ex_hat, ey_hat) = get_tangent_plane_from_ellipses(ellipses)
    #print('get_tangent',p0, ex_hat, ey_hat)
    for e in ellipses:
        project_ellipse(e, p0, ex_hat, ey_hat)
        #print(e.x, e.y, e.phi_major, e.phi_minor, e.theta)
        x, y, maj, minor, the = e.x/rad_per_deg*err_factor, e.y/rad_per_deg*err_factor, e.phi_major/rad_per_deg*2*err_factor, e.phi_minor/rad_per_deg*2*err_factor, 90-e.theta/rad_per_deg
        #print(x, y, maj, minor, the)
        if plotting:
            ellipse = Ellipse((x, y), maj, minor, angle=the, edgecolor='blue', facecolor="None",alpha=1)
            ax.add_artist(ellipse)
    
    
    
    new_e = combine_ellipses(ellipses)
    #print('new_e.x',new_e.x)
    deproject_ellipse(new_e, p0, ex_hat, ey_hat)
    alpha = new_e.alpha/rad_per_deg
    if alpha<0: 
          alpha += 360. 
    delta = new_e.delta/rad_per_deg
    theta = new_e.theta/rad_per_deg
    if theta<0:
        theta+=180.
    
    major = new_e.phi_major/rad_per_deg*err_factor
    minor = new_e.phi_minor/rad_per_deg*err_factor
    if add_sys:
        major = np.sqrt(major**2+0.71**2)
        minor = np.sqrt(minor**2+0.71**2)
    x, y, maj, mino, the = new_e.x/rad_per_deg*err_factor, new_e.y/rad_per_deg*err_factor, new_e.phi_major/rad_per_deg*2*err_factor, new_e.phi_minor/rad_per_deg*2*err_factor, 90-new_e.theta/rad_per_deg
    #print(x, y, maj, minor, the)
    if plotting:
        ellipse = Ellipse((x, y), maj, mino, angle=the, edgecolor='green', facecolor="None",alpha=1)
        ax.add_artist(ellipse)

    
        ax.set_xlim(-plot_limit, plot_limit)
        ax.set_ylim(-plot_limit, plot_limit)
        plt.show()
        
    cc2 = SkyCoord(alpha*u.deg,delta*u.deg, frame='icrs')
    #cc2_string = cc2.to_string('hmsdms')
    ra_hms = f'{int(cc2.ra.hms.h):02d} {int(cc2.ra.hms.m):02d} {cc2.ra.hms.s:4.2f}'
    dec_dms = f'{int(cc2.dec.dms.d):02d} {abs(int(cc2.dec.dms.m)):02d} {abs(cc2.dec.dms.s):4.2f}'
    # print(f'    {int(cc2.ra.hms.h):02d} {int(cc2.ra.hms.m):02d} {cc2.ra.hms.s:4.2f}  {int(cc2.dec.dms.d):02d} {abs(int(cc2.dec.dms.m)):02d} {abs(cc2.dec.dms.s):4.2f}            {major:.2f}            {minor:.2f}           {theta:.2f}')
    #print(cc2.ra.to_string(unit=u.degree, sep=':'))
    #print(f'alpha: {alpha:11.6f} degrees')
    #print(f'delta: {delta:11.6f} degrees')
    #print(f'theta: {theta:11.6f} degrees')
    #print(f'major: {major:11.6f} {err_unit}')
    #print(f'minor: {minor:11.6f} {err_unit}')
    #print('ha')
    #return alpha, delta, theta, major, minor
    return ra_hms, dec_dms, alpha, delta, theta, major, minor
    

def combine_error_ellipses(src, df_src, alphas, deltas, phimajors, phiminors, thetas,theta_means,field_name,query_dir,plotting):

    err_unit = 'arcsec'
    plot_limit = max(phimajors)*8
    add_sys = False
    if err_unit == 'arcsec':
        #err_factor = 1.#3600.
        coord_factor = 3600.
    if err_unit == 'arcmin':
        coord_factor = 60.

    if add_sys:
        phimajors = np.array([ np.sqrt(maj**2+0.71**2) for maj in phimajors])
        phiminors = np.array([ np.sqrt(minor**2+0.71**2) for minor in phiminors])

    ra_hms, dec_dms, alpha_ave, delta_ave, theta_ave, major_ave, minor_ave = slsh_main(alphas, deltas, phimajors, phiminors, thetas,plot_limit=plot_limit, err_unit=err_unit,add_sys=False,plotting=False)
    # print(alpha_ave, delta_ave)
    #print(alpha_ave)
    #if alpha_ave<0:
        #alpha_ave+=360

    #'''
    if plotting:
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'aspect': 'equal'})
    #print(alphas, deltas, thetas, phimajors, phiminors)
    
    

        for alpha, delta, theta, major, minor,theta,color,ii in zip(alphas, deltas, thetas, phimajors, phiminors,thetas,['blue','red'],['1', '2']):
            label = f's{ii}: {deg2hms_print(alpha, delta, major, minor, theta)}'
            ellip = Ellipse(((alpha-alpha_ave)*coord_factor, (delta-delta_ave)*coord_factor), major*2, minor*2, angle=90-theta, edgecolor=color, facecolor="None",alpha=1,label=label)
            ax.add_artist(ellip)

        label = f'cb: {deg2hms_print(alpha_ave, delta_ave, major_ave, minor_ave, theta_ave)}'
        ellip = Ellipse((0, 0), major_ave*2, minor_ave*2, angle=90-theta_ave, edgecolor='green', facecolor="None",alpha=1,label=label)
        ax.add_artist(ellip)

        ra_master = Angle(df_src['ra'].values[0], 'hourangle').degree
        dec_master = Angle(df_src['dec'].values[0], 'deg').degree
        r0_master = np.sqrt(df_src['err_ellipse_r0'].values[0]**2-0.71**2)
        r1_master = np.sqrt(df_src['err_ellipse_r1'].values[0]**2-0.71**2)
        ang_master = df_src['err_ellipse_ang'].values[0]
    
    
        print(ra_master, dec_master)
        label = f'ms: {deg2hms_print(ra_master, dec_master, r0_master, r1_master, ang_master)}'
        ellip = Ellipse(((ra_master-alpha_ave)*coord_factor, (dec_master-delta_ave)*coord_factor), r0_master*2, r1_master*2, angle=90-ang_master, edgecolor='None', facecolor="orange",alpha=0.5,label=label)
        ax.add_artist(ellip)

        ax.set_xlim(-plot_limit, plot_limit)
        ax.set_ylim(-plot_limit, plot_limit)
        ax.set_xlabel(f'{ra_hms} (arcsec)')
        ax.set_ylabel(f'{dec_dms} (arcsec)')
        ax.set_title(src)
        ax.legend(fontsize=8)
        plt.savefig(f'./{query_dir}/{field_name}/astrometry/{src}_comb_PU.png',bbox_inches='tight',dpi=400)
        plt.close()
            
    # plt.show()
    #'''
    #print(alpha_ave, delta_ave, theta_ave, major_ave, minor_ave)
    #c1 = SkyCoord(alpha_ave*u.deg,delta_ave*u.deg, frame='icrs')
    #print(c1.to_string('hmsdms'))
    #print((18+(4.+36.89/60)/60)*15,-(30+(5+38.98/60)/60))

    return ra_hms, dec_dms, alpha_ave, delta_ave,  theta_ave, major_ave, minor_ave

def stack_astrometry(field_name, RA, DEC, radius, query_dir,template_dir='./data', csc_version = '2.0', plotting = False):


    if glob.glob(f'./{query_dir}/{field_name}/astrometry/{field_name}_PU.csv') == []:
        

        # stack_lists = []

        Path(f'./{query_dir}/{field_name}/astrometry/').mkdir(parents=True, exist_ok=True)
        print(f'applying astrometry on stack-level positions of {field_name}')
        CSCviewsearch(field_name, RA, DEC, max(60,radius+30),query_dir,template_dir=template_dir,csc_version='2.0',engine='wget',adql_version='csc_astrometry_stack_flag_template',suffix='_stack')                   
        df_res = pd.read_csv(f'{query_dir}/{field_name}/{field_name}_stack_wget.txt', header=43, sep='\t')
        
        for col in df_res.columns:
            #print(col, df_res[col].dtypes)
            if df_res[col].dtypes=='object':
                #print(col)
                df_res[col] = df_res[col].str.lstrip()
                df_res[col] = df_res[col].str.rstrip()
                
        df_res = df_res[df_res['instrument'] == 'ACIS'].reset_index(drop=True)
        
        df = df_res[df_res['detect_stack_id'].isin(df_res.loc[df_res['separation']<radius*60, 'detect_stack_id'])].reset_index(drop=True)
        
        df['delta_xi'] = 0.
        df['delta_eta'] = 0.
        df['xi_old'] = np.nan
        df['eta_old'] = np.nan
        df['wrmsr_after'] = np.nan
        '''
        df['wrmsr_before'] = np.nan
        
        df['match_group'] = np.nan
        df['match_num'] = np.nan
        df['good_num'] = np.nan
        df['rho'] = np.nan
        df['comment'] = 0
        '''
        if len(df)>0:
            #print(df_res.columns)
            #print(src_i)
            if plotting:
                fig, ax = plt.subplots(figsize =(10, 10))
            df['RA'] = Angle(df['ra.1'], 'hourangle').degree
            df['DEC']= Angle(df['dec.1'], 'deg').degree
            df['RA_stack'] = Angle(df['ra_stack'], 'hourangle').degree
            df['DEC_stack']= Angle(df['dec_stack'], 'deg').degree
            #print(df[(df['theta_mean']=="         ") | (df['err_ellipse_r0.1']=="         ")])
            df['theta_mean'] = df['theta_mean'].replace({"":99})
            df['theta_mean'] = pd.to_numeric(df['theta_mean'])
            df['err_ellipse_r0.1'] = df['err_ellipse_r0.1'].replace({"":99})
            df['err_ellipse_r0.1'] = pd.to_numeric(df['err_ellipse_r0.1'])
            df['err_ellipse_r1.1'] = df['err_ellipse_r1.1'].replace({"":99})
            df['err_ellipse_r1.1'] = pd.to_numeric(df['err_ellipse_r1.1'])
            df['err_ellipse_ang.1'] = df['err_ellipse_ang.1'].replace({"":99})
            df['err_ellipse_ang.1'] = pd.to_numeric(df['err_ellipse_ang.1'])
            df = df.replace({'TRUE': True, 'False': False, 'FALSE':False})
            # print(field_name, RA, DEC, radius, fgl_unas.loc[src_i,'CSC2.0_N'],fgl_unas.loc[src_i,'CSC2.0_ObsIDs'],fgl_unas.loc[src_i,'CSC2.0_N_ObsIDs'] )

            
            #print(df.loc[200:, ['RA','DEC','RA_stack']])
            
            


            stacks_ids, n_stack = list(df['detect_stack_id'].unique()), len(df['detect_stack_id'].unique())
            #print(stacks_ids[1])
            #FGL_stacks[field_name] = stacks_ids
            print(n_stack)
            
            stack_colors = []
            cm = pylab.get_cmap('gist_rainbow')
            for i in range(n_stack):
                color = cm(1.*i/n_stack) 
                stack_colors.append(color)
                
            #df['colors'] = df.apply(lambda r: stack_colors[stacks_ids.index(r.detect_stack_id)] ,axis=1)
            #df[['detect_stack_id','colors']]
            if plotting:
                for stack_id,stack_color in zip(stacks_ids,stack_colors):
                    
                    ax.scatter(df.loc[df['detect_stack_id']==stack_id, 'RA'], df.loc[df['detect_stack_id']==stack_id, 'DEC'], s=100, c=stack_color, alpha=0.5, label=stack_id)
                #ax.scatter(RA, DEC, )
                circle1 = plt.Circle((RA, DEC), 1, fc='none',ec='pink', color = 'pink', linewidth=1,alpha=1)
                circle2 = plt.Circle((RA, DEC), radius/60, fc='none',ec='orange', color = 'orange', linewidth=2,alpha=1)
                ax.add_artist(circle1)  
                ax.add_artist(circle2)  
                
                ax.set(xlim=(min(df['RA'].min(), RA-1.1), max(df['RA'].max(), RA+1.1)), ylim=(min(df['DEC'].min(), DEC-1.1), max(df['DEC'].max(), DEC+1.1)))

                #ax.add_artist(circle2)  
                ax.legend()
                #plt.show()
                plt.savefig(f'./{query_dir}/{field_name}/astrometry/stacks.png',bbox_inches='tight',dpi=400)
                plt.close()
                
            
            for stack_id,stack_color in zip(stacks_ids,stack_colors):
                '''
                if stack_id in stack_lists:
                    print(f'{stack_id} already in other FGL sources!')
                    df.loc[df['detect_stack_id']==stack_id, 'comment'] = 1
                            
                            
                    pass
                '''
                stack_solution = {'field':field_name,'stack_id':stack_id,'delta_xi':0.,'delta_eta':0., \
                                'wrmsr_before':np.nan,'wrmsr_after':np.nan,'match_group':np.nan, \
                                'match_num':np.nan,'good_num':np.nan,'rho':np.nan,'comment': ''}
                #else:
                #stack_lists.append(stack_id)
                df_s = df[df['detect_stack_id']==stack_id].reset_index(drop=True)

                print(stack_id)
                
                ra_stack, dec_stack = df_s['RA_stack'].values[0], df_s['DEC_stack'].values[0]

                #print(df_s.columns)
                df_s = convert_standard(df_s, ra='RA',dec='DEC',ra_stack=ra_stack,dec_stack=dec_stack,xi='xi',eta='eta')

                #print(df_s[['theta_mean']])
                #print(df_s['likelihood_class.1'].unique(), df_s['man_inc_flag'].unique(),df_s['sat_src_flag.1'].unique(),df_s['extent_code'].unique())
                
            
                df_s_good = df_s[((df_s['likelihood_class.1']==True) | (df_s['likelihood_class.1']=='MARGINAL')) & (df_s['theta_mean']<10.) & (df_s['err_ellipse_r0.1']<=90) & ((df_s['man_inc_flag']==True) | ((df_s['sat_src_flag.1']==False) & (df_s['extent_code']<16)) )].reset_index(drop=True)

                #df_s_good = df_s[((df_s['likelihood_class.1']==True) | (df_s['likelihood_class.1']=='True') | (df_s['likelihood_class.1']=='MARGINAL')) & ((df_s['man_inc_flag']==True) | ((df_s['sat_src_flag.1']==False) & (df_s['extent_code']<16) & (df_s['theta_mean']<10))) & (df_s['err_ellipse_r0.1']<=20)].reset_index(drop=True)
                
                #df.loc[df['detect_stack_id']==stack_id, 'good_num'] = len(df_s_good)
                stack_solution['good_num'] = len(df_s_good)

                #print(len(df_s),len(df_s_good))
                if len(df_s_good)>3:
                    #ra_stack, dec_stack = df_s_good['RA_stack'].values[0], df_s_good['DEC_stack'].values[0]
                    print(f'stack coord: {ra_stack} {dec_stack}')
                    

    #                 viz = Vizier(row_limit=-1,  timeout=5000, columns=['DR3Name','Source','RA_ICRS', 'DE_ICRS', 'e_RA_ICRS', 'e_DE_ICRS','PM', 'pmRA',\
    #                        'e_pmRA', 'pmDE', 'e_pmDE','AllWISE', '+_r'], catalog='I/355/gaiadr3')
                    

    #                 query_res = viz.query_region(SkyCoord(ra=ra_stack, dec=dec_stack,unit=(u.deg, u.deg),frame='icrs'),
    #                                            radius=12*u.arcmin)

    #                 df_gaia = query_res[0].to_pandas()

                    coord = SkyCoord(ra=ra_stack ,  dec=dec_stack, unit=(u.degree, u.degree), frame='icrs')
                    j = Gaia.cone_search_async(coord, radius=u.Quantity(12, u.arcmin))
                    #INFO: Query finished. [astroquery.utils.tap.core]
                    query_res = j.get_results()
                    df_gaia = query_res.to_pandas()
                    df_gaia = df_gaia.rename(columns={'DESIGNATION':'DR3Name','source_id':'Source','ra':'RA_ICRS','dec':'DE_ICRS','pm':'PM'})
                    
                    
                    rho_gaia = len(df_gaia)/(np.pi*12**2)
                    df_gaia = df_gaia[~df_gaia['RA_ICRS'].isnull()].reset_index(drop=True)
                    print(len(df_gaia))
                    #print(df_gaia)
                    #print(df_gaia.columns)

                    df_gaia_good = df_gaia[(df_gaia['PM']<50) | (df_gaia['PM'].isnull())].reset_index(drop=True)
                    #print(len(df_gaia), len(df_gaia_good))
                    #print(df_gaia_good[['RA_ICRS','DE_ICRS']])


                    df_match = pd.DataFrame()

                    X_cat = SkyCoord(ra=df_s_good['RA']*u.degree,         dec=df_s_good['DEC']*u.degree)
                    G_cat = SkyCoord(ra=df_gaia_good['RA_ICRS']*u.degree, dec=df_gaia_good['DE_ICRS']*u.degree)

                    gaia_j = 1
                    #print(G_cat)
                    idxs, gaia_d2d, d3d = X_cat.match_to_catalog_sky(G_cat,nthneighbor=gaia_j)
                    df_s_good['gaia_sep_'+str(gaia_j)] = gaia_d2d.arcsec

                    #print(df_s_good[['name','RA','DEC','gaia_sep_'+str(gaia_j)]])
                    #print(df_gaia_good.loc[idxs,['DR3Name','RA_ICRS','DE_ICRS']])
                    df_gaia_sep = df_gaia_good.iloc[idxs]
                    df_gaia_sep = df_gaia_sep.reset_index(drop=True)
                    df_s_good_sep = df_s_good[['name','RA','DEC','err_ellipse_r0.1','err_ellipse_r1.1','theta_mean','xi','eta','RA_stack','DEC_stack','gaia_sep_'+str(gaia_j)]]
                    df_s_good_sep = df_s_good_sep.reset_index(drop=True).rename(columns={'gaia_sep_'+str(gaia_j):'gaia_sep'})
                    df_sep = pd.concat([df_s_good_sep, df_gaia_sep[['DR3Name','RA_ICRS','DE_ICRS']]], axis=1)#,ignore_index=True, sort=False)
                    #print(df_sep)
                    df_match = pd.concat([df_match, df_sep], ignore_index=True, sort=False)


                    while min(gaia_d2d.arcsec) <= 3:
                        #print(gaia_j)
                        gaia_j +=1
                        idxs, gaia_d2d, d3d = X_cat.match_to_catalog_sky(G_cat,nthneighbor=gaia_j)
                        df_s_good['gaia_sep_'+str(gaia_j)] = gaia_d2d.arcsec
                        df_gaia_sep = df_gaia_good.iloc[idxs]
                        df_gaia_sep = df_gaia_sep.reset_index(drop=True)
                        df_s_good_sep = df_s_good[['name','RA','DEC','xi','eta','RA_stack','DEC_stack','theta_mean','gaia_sep_'+str(gaia_j)]]
                        df_s_good_sep = df_s_good_sep.reset_index(drop=True).rename(columns={'gaia_sep_'+str(gaia_j):'gaia_sep'})
                        #df_gaia_sep = df_gaia_good.iloc[idxs].reset_index(drop=True)
                        df_sep = pd.concat([df_s_good_sep, df_gaia_sep[['DR3Name','RA_ICRS','DE_ICRS']]], axis=1)#,ignore_index=True, sort=False)

                        df_match = pd.concat([df_match, df_sep], ignore_index=True, sort=False)




                    #print(len(df_s_good[df_s_good['gaia_sep']<3]))


                    viz = Vizier(row_limit=-1,  timeout=5000, columns=['AllWISE', 'RAJ2000', 'DEJ2000', 'eeMaj', 'eeMin', 'eePA','ccf', '+_r'], catalog='II/328/allwise')


                    query_res = viz.query_region(SkyCoord(ra=ra_stack, dec=dec_stack,unit=(u.deg, u.deg),frame='icrs'),
                                            radius=12*u.arcmin)

                    df_allwise = query_res[0].to_pandas()




                    df_allwise['PM'] = np.nan

    #                 df_allwise.set_index('AllWISE')
    #                 df_gaia.set_index('AllWISE')

    #                 df_allwise.update(df_gaia[['PM']])

    #                 df_gaia.reset_index(inplace=True)
    #                 df_allwise.reset_index(inplace=True)

                    #df_allwise_good = df_allwise[((df_allwise['PM']<50) | (df_allwise['PM'].isnull())) & ((df_allwise['ccf'].str[0]=='0') |(df_allwise['ccf'].str[1]=='0') |(df_allwise['ccf'].str[2]=='0')  |(df_allwise['ccf'].str[3]=='0')) ].reset_index(drop=True)
                    df_allwise_good = df_allwise[((df_allwise['PM']<50) | (df_allwise['PM'].isnull())) ].reset_index(drop=True)

                    #print(len(df_allwise),len(df_allwise_good))



                    A_cat = SkyCoord(ra=df_allwise_good['RAJ2000']*u.degree, dec=df_allwise_good['DEJ2000']*u.degree)
                    #
                    allwise_j = 1
                    idxs, allwise_d2d, d3d = X_cat.match_to_catalog_sky(A_cat,nthneighbor=allwise_j)
                    df_s_good['allwise_sep_'+str(allwise_j)] = allwise_d2d.arcsec
                    while min(allwise_d2d.arcsec) < 3:
                        allwise_j +=1
                        idxs, allwise_d2d, d3d = X_cat.match_to_catalog_sky(A_cat,nthneighbor=allwise_j)
                        df_s_good['allwise_sep_'+str(allwise_j)] = allwise_d2d.arcsec



                    #print(len(idxs), allwise_d2d)
                    #df_s_good['neb_index'] = idxs 
                    #df_s_good['allwise_sep'] = allwise_d2d.arcsec
                    #print(gaia_j)
                    #print([len(df_s_good[df_s_good['gaia_sep_'+str(ii+1)]<3]) for ii in range(gaia_j)])
                    #print([len(df_s_good[df_s_good['allwise_sep_'+str(ii+1)]<3]) for ii in range(allwise_j)])

                    sum_gaia_match = sum([len(df_s_good[df_s_good['gaia_sep_'+str(ii+1)]<3]) for ii in range(gaia_j)])
                    sum_wise_match =sum([len(df_s_good[df_s_good['allwise_sep_'+str(ii+1)]<3]) for ii in range(allwise_j)])


                    #df_match['xi_gaia'] = 3600.*(180./np.pi)*(np.cos(df_match['DE_ICRS']*np.pi/180.)*np.sin((df_match['RA_ICRS'] - df_match['RA_stack'])*np.pi/180.))/(np.sin(df_match['DE_ICRS']*np.pi/180.)* np.sin(df_match['DEC_stack']*np.pi/180.)+ np.cos(df_match['DE_ICRS']*np.pi/180.)*np.cos(df_match['DEC_stack']*np.pi/180.)*np.cos((df_match['RA_ICRS']-df['RA_stack'])*np.pi/180.))
                    #df_match['xi_gaia'] = 3600.*(180./np.pi)*(np.cos(df_match['DE_ICRS']*np.pi/180.)*np.sin((df_match['RA_ICRS'] - ra_stack)*np.pi/180.))/(np.sin(df_match['DE_ICRS']*np.pi/180.)* np.sin(dec_stack*np.pi/180.)+ np.cos(df_match['DE_ICRS']*np.pi/180.)*np.cos(dec_stack*np.pi/180.)*np.cos((df_match['RA_ICRS']-ra_stack)*np.pi/180.))

                    #df_match['eta_gaia']= 3600.*(180./np.pi)*(np.sin(df_match['DE_ICRS']*np.pi/180.)*np.cos(df_match['DEC_stack']*np.pi/180.)- np.cos(df_match['DE_ICRS']*np.pi/180.)*np.sin(df_match['DEC_stack']*np.pi/180.)*np.cos((df_match['RA_ICRS']-df_match['RA_stack'])*np.pi/180.))/(np.sin(df_match['DE_ICRS']*np.pi/180.)*np.sin(df_match['DEC_stack']*np.pi/180.)+ np.cos(df_match['DE_ICRS']*np.pi/180.)*np.cos(df_match['DEC_stack']*np.pi/180.)*np.cos((df_match['RA_ICRS']-df_match['RA_stack'])*np.pi/180.))
                    #df_match['eta_gaia']= 3600.*(180./np.pi)*(np.sin(df_match['DE_ICRS']*np.pi/180.)*np.cos(dec_stack*np.pi/180.)- np.cos(df_match['DE_ICRS']*np.pi/180.)*np.sin(dec_stack*np.pi/180.)*np.cos((df_match['RA_ICRS']-ra_stack)*np.pi/180.))/(np.sin(df_match['DE_ICRS']*np.pi/180.)*np.sin(dec_stack*np.pi/180.)+ np.cos(df_match['DE_ICRS']*np.pi/180.)*np.cos(dec_stack*np.pi/180.)*np.cos((df_match['RA_ICRS']-ra_stack)*np.pi/180.))

                    df_match = convert_standard(df_match, ra='RA_ICRS',dec='DE_ICRS',ra_stack=ra_stack,dec_stack=dec_stack,xi='xi_gaia',eta='eta_gaia')

                    df_match['delta_xi'] =  df_match['xi'] -  df_match['xi_gaia'] 
                    df_match['delta_eta'] =  df_match['eta'] -  df_match['eta_gaia'] 

                    df_match['err_area.1'] = df_match['err_ellipse_r0.1']*df_match['err_ellipse_r1.1']

                    #print(df_match[['RA','DEC','xi','eta','RA_ICRS','DE_ICRS','xi_gaia','eta_gaia','RA_stack','DEC_stack','delta_xi','delta_eta']])


                    #for col in ['RA_ICRS','DE_ICRS','RA_stack','DEC_stack']:
                        #print(df_match[col].dtypes)



                    #group_colors = []
                    #cm = pylab.get_cmap('gist_rainbow')
                    #for i in range(3):
                        #color = cm(1.*i/3) 
                        #group_colors.append(color)
                    group_colors = ['red', 'green', 'cyan']
                    markers = ['o','P','D']

                    #'''
                    if plotting:
                        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9,6))    

                        bin_size = 50 

                        r_array = df_match.loc[df_match['gaia_sep']<=3, 'gaia_sep'].values
                        ax.hist(r_array, bins=bin_size,  linestyle='solid',  color='grey', histtype='step', fill=False)

                    #ax.set_xscale('log')
                    #ax.set_yscale('log')

                        count, division = np.histogram(r_array, bins=bin_size, range=[0, 3])
                        peak_dic = count_dist_peaks(r_array, bins=bin_size, hist_range=[0, 3],prominence=5) #prominence=100,width=[0,10],hist_range=[0, 10]) 
                        for p in peak_dic:
                            ax.axvline(x=(division[p]+division[p+1])/2,c='red',alpha=0.2)
                            ax.text((division[p]+division[p+1])/2,3,f'{(division[p]+division[p+1])/2:.2f}',rotation=90)
                            #print(f'peak:{(division[p]+division[p+1])/2:.2f}')


                        #r_arrays = [df_match.loc[(df_match['gaia_sep']>grp_k) & (df_match['gaia_sep']<=(grp_k+1)), 'gaia_sep'].values for grp_k in range(3)]
                        #ax.hist(r_arrays, bins=bin_size, color=group_colors, histtype='step', stacked=True, fill=False)


                        #plt.show() 
                        plt.savefig(f'./{query_dir}/{field_name}/astrometry/{stack_id}_seps.png',bbox_inches='tight',dpi=400)
                        plt.close()

                    #'''
                    if plotting:
                        fig, ax = plt.subplots(figsize =(10, 10))
                    sigmas_gaia = []
                    nmatchs_gaia = []
                    nmatchs2_gaia = []


                    for grp_k,grp_size in zip(range(3),[10, 100, 50]):


                        df_s_good['nmatch_gaia_'+str(grp_k+1)] = 0


                        df_match_group = df_match.loc[(df_match['gaia_sep']>grp_k) & (df_match['gaia_sep']<=(grp_k+1))].reset_index(drop=True)

                        sigma_x, sigma_y = df_match_group['delta_xi'].std(), df_match_group['delta_eta'].std()
                        sigma_gaia = np.sqrt(sigma_x*sigma_y)
                        sigmas_gaia.append(sigma_gaia)
                        nmatchs2_gaia.append(len(df_match_group))
                        nmatchs_gaia.append(len(df_match_group['name'].unique()))
                        #print(df_match_group[['delta_xi','delta_eta']])
                        #print(f'group {grp_k+1} {len(df_match_group)} {sigma_gaia}')#{sigma_x} {sigma_y} {sigma_gaia}')

                        #ax.scatter(df_match_group['delta_xi'], df_match_group['delta_eta'], s= grp_size, c=group_colors[grp_k], marker=markers[grp_k],alpha=0.4,label=f'{grp_k}<sep<{grp_k+1}')
                        if plotting:
                            ax.scatter(df_match_group['delta_xi'], df_match_group['delta_eta'], s= 10*df_match_group['err_area.1'], c=group_colors[grp_k], marker=markers[grp_k],alpha=0.4,label=f'{grp_k}<sep<{grp_k+1}')

                        for sep_j in range(gaia_j):

                            df_s_good.loc[ (df_s_good['nmatch_gaia_'+str(grp_k+1)]==0) & (df_s_good['gaia_sep_'+str(sep_j +1)]>grp_k) & (df_s_good['gaia_sep_'+str(sep_j +1)]<= (grp_k+1)), 'nmatch_gaia_'+str(grp_k+1)]= 1
                    #print(min(df_match[df_match['gaia_sep']<=3, 'delta_xi'].values))
                    if plotting:
                        ax.set(xlim=(df_match.loc[df_match['gaia_sep']<=3, 'delta_xi'].min()-1,df_match.loc[df_match['gaia_sep']<=3, 'delta_xi'].max()+1), ylim=(df_match.loc[df_match['gaia_sep']<=3, 'delta_eta'].min()-1,df_match.loc[df_match['gaia_sep']<=3, 'delta_eta'].max()+1)) 


                    #print(df_s_good.loc[:10, ['nmatch_gaia_1','nmatch_gaia_2','nmatch_gaia_3','gaia_sep_1','gaia_sep_2','gaia_sep_3','gaia_sep_4']])
                    #print(df_s_good['nmatch_gaia_1'].sum(),df_s_good['nmatch_gaia_2'].sum(),df_s_good['nmatch_gaia_3'].sum())

                    df_astro = pd.DataFrame({'grp':np.arange(1,4,1),'sigma':sigmas_gaia, 'nmatch':nmatchs_gaia,'nmatch2':nmatchs2_gaia}).sort_values(by=['sigma']).reset_index(drop=True)
                    grp_sel = 0
                    for i in range(3):
                        if df_astro.loc[i, 'nmatch']>=3:
                            grp_sel = df_astro.loc[i, 'grp']
                            break
                    if grp_sel == 0:
                        grp_sel = df_astro.loc[df_astro['nmatch']==max(df_astro['nmatch']), 'grp'].values[0]
                    print(df_astro)
                    print(grp_sel)

                    df_match_sel2 = df_match[(df_match['gaia_sep']>(grp_sel-1)) & (df_match['gaia_sep']<=grp_sel)].reset_index(drop=True)
                    df_match_sel = df_match_sel2.drop_duplicates(subset=['name']).reset_index(drop=True)
                    #print(len(df_match_sel2),len(df_match_sel))
                    #print(df_match_sel.loc[50:,:])
                    #df_match_sel2 = df_match[(df_match['gaia_sep']>(grp_sel-1)) & (df_match['gaia_sep']<=grp_sel)].reset_index(drop=True)
                    #print(len(df_match_sel2))

                    #df.loc[df['detect_stack_id']==stack_id, 'match_group'] = grp_sel
                    #df.loc[df['detect_stack_id']==stack_id, 'match_num'] = len(df_match_sel)
                    #df.loc[df['detect_stack_id']==stack_id, 'rho'] = rho_gaia
                    
                    if len(df_match_sel)>=3 and grp_sel==1:
                
                        stack_solution['match_group'] = grp_sel
                        stack_solution['match_num'] = len(df_match_sel)
                        stack_solution['rho'] = rho_gaia

                        xi_csc2 = df_match_sel['xi'].values
                        eta_csc2 = df_match_sel['eta'].values
                        xi_gaia = df_match_sel['xi_gaia'].values
                        eta_gaia = df_match_sel['eta_gaia'].values
                        #print(len(xi_csc2),len(eta_csc2),len(xi_gaia),len(eta_gaia))
                        w = [1./(df_match_sel['err_ellipse_r0.1'].values*df_match_sel['err_ellipse_r1.1'].values)]*2

                        # A function to perform the transformation as a matrix multiplication
                        #def model(pars, X):
                        #    return np.matmul(np.array([[1.0,0.0,pars[0]],
                        #    [0.0,1.0,pars[1]],[0.0,0.0,1.0]]),[X[0], X[1], 1.0])

                        # A function to estimate the residual, multiplied by the weights
                        #def fun_residual(pars, X, Y, weights):
                            #print(pars, X, Y, weights)
                        #    return np.dot(weights, (model(pars,X)[0]-Y[0])**2
                        #    + (model(pars,X)[1]-Y[1])**2)

                        WRMSR = fun_residual((0,0), [xi_csc2,eta_csc2], [xi_gaia,eta_gaia], w)

                        #w = [1./df_stack['err_ellipse_r0.1'].values*df_stack['err_ellipse_r1.1'].values]*2
                        #print(xi_csc2)
                        # Set starting point
                        x0 = (0.,  0.)
                        # The call to least_squares
                        print(f'gaia match: {sum_gaia_match}, wise match: {sum_wise_match}')
                        if sum_wise_match>sum_gaia_match:
                            print('Should use wise-table!!!!!')
                            stack_solution['comment'] = stack_solution['comment']+'4'
                        print(f'gaia density:{rho_gaia}, nmatch:{len(df_match_sel)}')
                        if (grp_sel != 1) or len(df_match_sel)<3 or ((len(df_match_sel)<10) and (rho_gaia>100)):
                            print('check!!!')
                            stack_solution['comment'] = stack_solution['comment']+'1'

                        try:
                            res = least_squares(fun_residual, x0, loss='cauchy',bounds=(-10, 10), args=([xi_csc2,eta_csc2], [xi_gaia,eta_gaia], w),verbose=0)


                            #print ("linear fit ",res.x)
                            WRMSR_before, WRMSR_after = np.sqrt(WRMSR[0]/sum(w[0])), np.sqrt(res.fun[0]/sum(w[0]))
                            #print(f'WRMSR before: { WRMSR_before:.3f}, after: {WRMSR_after:.3f}, {(np.sqrt(WRMSR[0]/sum(w[0]))-np.sqrt(res.fun[0]/sum(w[0])))*100/np.sqrt(WRMSR[0]/sum(w[0])):.1f}%')
                            if plotting:
                                ax.scatter(-res.x[0], -res.x[1], s=100, c='orange', marker='*',alpha=0.7,label='correction')
                            df.loc[df['detect_stack_id']==stack_id, 'delta_xi'] = res.x[0]
                            df.loc[df['detect_stack_id']==stack_id, 'delta_eta'] = res.x[1]
                            #df.loc[df['detect_stack_id']==stack_id, 'wrmsr_before'] = WRMSR_before
                            df.loc[df['detect_stack_id']==stack_id, 'wrmsr_after'] = WRMSR_after
                            stack_solution['delta_xi'] = res.x[0]
                            stack_solution['delta_eta'] = res.x[1]
                            stack_solution['wrmsr_before'] = WRMSR_before
                            stack_solution['wrmsr_after'] = WRMSR_after


                        except:
                            print('least_squares fails!')
                            stack_solution['comment'] = stack_solution['comment']+'2'
                            
                    elif grp_sel==2 or grp_sel==3:
                        stack_solution['comment'] = stack_solution['comment']+'5'
                    else:
                        stack_solution['comment'] = stack_solution['comment']+'6'
                    if plotting:
                        plt.legend(fontsize=15)


                        #plt.show()  
                        plt.savefig(f'./{query_dir}/{field_name}/astrometry/{stack_id}_solution.png',bbox_inches='tight',dpi=400)
                        plt.close()

                    
                else:
                    print('Less than 3 good source!!!')
                    #df.loc[df['detect_stack_id']==stack_id, 'comment'] = 1
                    #stack_solution['comment'] = 1
                    stack_solution['comment'] = stack_solution['comment']+'3'
                
                
                # solutions = pd.concat([solutions,pd.DataFrame(stack_solution, index=[0])], ignore_index=True)
                pd.DataFrame(stack_solution, index=[0]).to_csv(f'./{query_dir}/{field_name}/astrometry/{field_name}_solution.csv',index=False)
            
            df =  convert_standard(df, ra='RA',dec='DEC',ra_stack='RA_stack',dec_stack='DEC_stack',xi='xi_old',eta='eta_old',stack_col=True,inverse=False)
            df['xi_new'] = df['xi_old'] + df['delta_xi']
            df['eta_new'] = df['eta_old'] + df['delta_eta']
            df =  convert_standard(df, ra='RA_new',dec='DEC_new',ra_stack='RA_stack',dec_stack='DEC_stack',xi='xi_new',eta='eta_new',stack_col=True,inverse=True)
            df['r0_new'] = np.sqrt(df['err_ellipse_r0.1']**2+(df['wrmsr_after'].fillna(0.71/np.sqrt(np.log(20)*2))*np.sqrt(np.log(20)*2))**2)
            df['r1_new'] = np.sqrt(df['err_ellipse_r1.1']**2+(df['wrmsr_after'].fillna(0.71/np.sqrt(np.log(20)*2))*np.sqrt(np.log(20)*2))**2)
                    
            df.to_csv(f'./{query_dir}/{field_name}/astrometry/{field_name}_astro.csv',index=False)
        #else:
            #df['comment'] = 2

        else:
            print(f'{field_name} less than 1 source!')

        df_err = pd.DataFrame()
        
        df_good = df[df['err_ellipse_r0.1']!=99.].reset_index(drop=True)
        df_good = df_good[(df_good['match_type']=='u') & (df_good['sat_src_flag.1']!=True)].reset_index(drop=True)
        df_dup = df_good[df_good.duplicated(subset=['name'],keep=False)].reset_index(drop=True)
        df_dup = df_dup.sort_values(by='name')
        #print(df_dup['match_type'].value_counts())
        # print(df_dup)
        for src in df_dup['name'].unique():
            # print(src)
            df_src = df_dup[df_dup['name']==src].reset_index(drop=True)
            #print(df_src)
            sep = df_src['separation'].values[0]
            if len(df_src[df_src['sat_src_flag.1']==True])>0:
                print('sat_src_flag!!!')
            
            alphas2, deltas2, phimajors2, phiminors2, thetas2,theta_means2 = df_src['RA_new'].values, df_src['DEC_new'].values, df_src['r0_new'].values, df_src['r1_new'].values, df_src['err_ellipse_ang.1'].values, df_src['theta_mean'].values
            
            ra_ave, dec_ave, alpha_ave, delta_ave, theta_ave, r0_ave, r1_ave = combine_error_ellipses(src, df_src,alphas2, deltas2, phimajors2, phiminors2, thetas2,theta_means2,field_name,query_dir,plotting=False)
            
            df_err = pd.concat([df_err, pd.DataFrame({'separation':sep,'field':field_name,'name':src,'RA_new':alpha_ave,'DEC_new':delta_ave,'r0_new':r0_ave,'r1_new':r1_ave,'ang_new':theta_ave}, index=[0])], ignore_index=True)
        
            #print(df_err)
        if len(df_err)==0:
            df_single = df_good
        else:
            df_single = df_good[~df_good['name'].isin(df_err['name'])].reset_index(drop=True)
        
        df_single['field'] = field_name
        df_err = pd.concat([df_err,df_single[['separation', 'field','name','RA_new','DEC_new','r0_new','r1_new','err_ellipse_ang.1']].rename(columns={'err_ellipse_ang.1':'ang_new'})], ignore_index=True)
        
        df_master = df_good.drop_duplicates(subset=['name']).reset_index(drop=True)
        df_master['RA_csc'] = Angle(df_master['ra'], 'hourangle').degree
        df_master['DEC_csc']= Angle(df_master['dec'], 'deg').degree
        df_err = pd.merge(df_err, df_master[['name','RA_csc','DEC_csc','err_ellipse_r0','err_ellipse_r1','err_ellipse_ang']], on='name',how='inner')
        # df_err = df_err.rename(columns={'RA_new':'ra','DEC_new':'dec','r0_new':'err_ellipse_r0','r1_new':'err_ellipse_r1',	err_ellipse_ang
        df_err.to_csv(f'./{query_dir}/{field_name}/astrometry/{field_name}_PU.csv',index=False)
        
        # df_err = df_err[df_err['separation']<=radius*60.]
        
        # df_comb = pd.concat([df_comb, df_err], ignore_index=True)


from os import path
from astropy.table import Table

def cal_CT_match(df_X, df_MW, csc='name'):
    df_X['match_CT'] = np.nan
    for csc_name in df_MW[csc].unique():
        #print(df_MW[df_MW[csc]==csc_name])
        pis = df_MW.loc[df_MW[csc]==csc_name, 'p_i']#.values()
        max_pi = max(pis)# df_MW.loc[df_MW[csc]==csc_name, 'p_i'].max()
        CTs = sorted([max_pi / pi for pi in pis if pi!=0])
        #print(CTs)
        CT = CTs[1]
        
        
        df_X.loc[(df_X[csc]==csc_name) , 'match_CT'] = CT
        
    return df_X

def process_crossmatching(df, data_dir, field_name):

    nwaydata_dir =f'{data_dir}/nway'
    df_match = pd.DataFrame()
    for i in range(len(df)):  
        
        X_name = df.loc[i, 'name'][5:]

        if path.exists(f'{nwaydata_dir}/{X_name}_MW_match.fits'):
            df_m= Table.read(f'{nwaydata_dir}/{X_name}_MW_match.fits', format='fits').to_pandas()
            # df_m['MW_counts'] = len(df_m)-1
            df_mw= pd.read_csv(f'{nwaydata_dir}/{X_name}_MW_crossmatch.csv')
            df_m = pd.merge(df_m, df_mw, left_on='MW_id', right_on='id', how='left')
            df_match = pd.concat([df_match, df_m], ignore_index=True, sort=False)
        else:
            print(f'{nwaydata_dir}/{X_name}_MW_match.fits not exsit!')

    df_match = df_match.applymap(lambda x: x.decode() if isinstance(x, bytes) else x)

    df_match = df_match.rename(columns={'CSC__2CXO':'CSC_name'})

    # number of MW associations for each x-ray source
    df_match['MW_counts'] = df_match.groupby('CSC_name')['CSC_name'].transform('count')-1

    # multiple possible multiwavelength associations for each x-ray source
    # groupby 'name', create 'MW_name' column by extending with association number, with associations ordered by 'p_i' column
    df_match['name'] = df_match.groupby('CSC_name').apply(lambda x: x['CSC_name'] + '-' + x['p_i'].rank(ascending=False).astype(int).astype(str)).reset_index(level=0)[0]
    df_match.loc[df_match['p_i']==0, 'name'] = df_match.loc[df_match['p_i']==0, 'CSC_name'] + '-0'

    # print(len(df_match))

    # remove extended sources based on Gaia E_BP_RP_ and AllWISE ex flag
    df_ext_remove = df_match[df_match['name'].isin(df_match.loc[((df_match['GAIA_E_BP_RP_']>20) | (df_match['ALLWISE_ex']==5)), 'name'])].reset_index(drop=True)
    df_match_clean = df_match[~(df_match['CSC_name'].isin(df_ext_remove['CSC_name']))].reset_index(drop=True)
    # print(len(df_match), len(df_ext_remove))
    # print(df_match.loc[(df_match['CSC_name'].isin(df_ext_remove['CSC_name'])) & (df_match['match_flag']==1), 'MW_counts'].value_counts())


    df_match_c = df_match_clean
    df_cat = df_match_c #pd.merge(df_match, TD[['name','Class']], on='name')
    # print(len(df_cat))

    df_confused = df_cat[df_cat['MW_counts']>=2].reset_index(drop=True)
    
    df_confused_matchCT = cal_CT_match(df_confused, df_confused, csc='CSC_name')
    # print(df_confused_matchCT.loc[df_confused_matchCT['match_flag']==1, 'match_CT'].describe())
    # print(len(df_confused_matchCT[df_confused_matchCT['match_flag']==1])-df_confused_matchCT.loc[df_confused_matchCT['match_flag']==1, 'p_any'].sum())
    df_nonconfused = df_cat[~(df_cat['CSC_name'].isin(df_confused_matchCT['CSC_name']))].reset_index(drop=True)
    # print(df_nonconfused['match_flag'].value_counts())
    df_nonconfused = df_nonconfused[(df_nonconfused['match_flag']==1) & (df_nonconfused['ncat']==2)]
    #print(df_nonconfused['p_any'].describe())
    #print(len(df_nonconfused)-df_nonconfused['p_any'].sum())

    matchCT_remove = np.where((df_match_c['CSC_name'].isin(df_confused_matchCT.loc[df_confused_matchCT['match_CT']>10,'CSC_name'])) & (df_match_c['match_flag']==0) & (df_match_c['ncat']==2))[0]
    df_match_c.loc[df_match_c['CSC_name'].isin(df_match_c.loc[matchCT_remove, 'CSC_name']), 'MW_counts'] = 1
    df_match_c2 = df_match_c.drop(matchCT_remove).reset_index(drop=True)

    # print(len(df_match_c), len(matchCT_remove), len(df_match_c2))


    # df = df_match_c2.copy()


    # Process different crossmatching associations for each X-ray source

    df_nocp  = df_match_c2[(df_match_c2['MW_counts']==0) ].reset_index(drop=True)
    df_1cp   = df_match_c2[(df_match_c2['MW_counts']==1)].reset_index(drop=True)
    df_mutcp = df_match_c2[df_match_c2['MW_counts']>2].reset_index(drop=True)
    df_mutcp = cal_CT_match(df_mutcp, df_mutcp, csc='CSC_name')

    df_2cp  = df_match_c2[df_match_c2['MW_counts']==2].reset_index(drop=True)
    df_3cp  = df_mutcp[df_mutcp['MW_counts']==3].reset_index(drop=True)
    df_4cp  = df_mutcp[df_mutcp['MW_counts']==4].reset_index(drop=True)
    df_mcp  = df_mutcp[df_mutcp['MW_counts']>4].reset_index(drop=True)

    mw_dict = {'g':[col for col in df_match_c2.columns if col[:4]=='GAIA'],
            'c':[col for col in df_match_c2.columns if col[:7]=='CATWISE'],
            'a':[col for col in df_match_c2.columns if col[:7]=='ALLWISE'],
            't':[col for col in df_match_c2.columns if col[:5]=='TMASS']}
    #print(mw_dict)

    # print(len(df_2cp)/3, len(df_3cp)/4, len(df_4cp)/5, len(df_mcp))

    df_match_c3 = pd.concat([df_nocp, df_1cp, df_2cp, df_mutcp], ignore_index=True, sort=False)
    # print(len(df_match_c3), len(df_match_c3['CSC_name'].unique()))
    #print(df_match_c3['Class'].value_counts())
    # print(print(df_match_c3['MW_counts'].value_counts()))

    # keep only those X-ray sources with no more than 5 counterparts
    df_match_c3 = df_match_c3[df_match_c3['MW_counts']<=5]
    df_match_c3.to_csv(f'{data_dir}/{field_name}_MW.csv',index=False)

    # Updating variability information, from Update_var.ipynb
    # Merge MW columns with CSC columns, rename columns

    df_csc = pd.read_csv(f'{data_dir}/{field_name}_wget.txt', comment='#', delimiter='\t')
    # drop per observation level rows
    df_csc = df_csc.drop_duplicates(subset=['name']).reset_index(drop=True)
    # strip whitespace from all columns
    df_csc = df_csc.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    # print(len(df_csc))
    # df_obs = df_csc[df_csc['name'].isin(df_mw['name'])].reset_index(drop=True)
    # print(len(df_obs))

    df_csc['usrid'] = df_csc['name']

    df_csc['per_remove_code'] = 0

    df_mw_ave, df_mw_obs = cal_ave_v2(df_csc, f'{data_dir}', dtype='field',Chandratype='CSC',verb=0)

    #df_ave[['name','var_inter_prob','kp_prob_b_max','significance_max']].to_csv(f'./data/from_allCSC/CSCv2_var.csv', index=False)

    df_mw_ave = df_mw_ave.rename(columns={'name':'CSC_name'})
    df_fields = df_mw_ave.merge(df_match_c3, left_on='CSC_name', right_on='CSC_name', how='right')

    # df_fields = df_fields[~df_fields['FGL'].isin(['J1900.8+0118','J1211.8-6021c','J1834.9-0800','J1759.4-3103','J1649.2-4513c'])].reset_index(drop=True)

    df_fields = df_fields[df_fields['significance']>=3].reset_index(drop=True)
    # df_fields['Field'] = field_name

    df_fields = df_fields.rename(columns={'flux_aper90_avg_s':'Fcsc_s','flux_aper90_avg_m':'Fcsc_m','flux_aper90_avg_h':'Fcsc_h','flux_aper90_avg_b_manual':'Fcsc_b', 'e_flux_aper90_avg_s':'e_Fcsc_s','e_flux_aper90_avg_m':'e_Fcsc_m','e_flux_aper90_avg_h':'e_Fcsc_h','e_flux_aper90_avg_b_manual':'e_Fcsc_b', 'kp_prob_b_max':'var_intra_prob','var_inter_prob':'var_inter_prob','Signif.':'significance', 
                                        'GAIA_DR3Name': 'DR3Name_gaia', 'GAIA_Gmag':'Gmag','GAIA_BPmag':'BPmag','GAIA_RPmag':'RPmag', 'GAIA_e_Gmag':'e_Gmag','GAIA_e_BPmag':'e_BPmag','GAIA_e_RPmag':'e_RPmag','GAIA_RPlx':'RPlx','GAIA_PM':'PM','GAIA_epsi':'epsi','GAIA_sepsi':'sepsi','GAIA_RUWE':'ruwe', 'GAIA_rgeo': 'rgeo', 'GAIA_B_rgeo': 'B_rgeo', 'GAIA_b_rgeo': 'b_rgeo', 'GAIA_rpgeo': 'rpgeo', 'GAIA_B_rpgeo': 'B_rpgeo', 'GAIA_b_rpgeo': 'b_rpgeo',
                                        'TMASS_Jmag':'Jmag','TMASS_Hmag':'Hmag','TMASS_Kmag':'Kmag', 'ALLWISE_W3mag':'W3mag','TMASS_e_Jmag':'e_Jmag','TMASS_e_Hmag':'e_Hmag','TMASS_e_Kmag':'e_Kmag', 'ALLWISE_e_W3mag':'e_W3mag'})

    for w in ['W1', 'W2']:
        df_fields[w+'mag'] = df_fields['CATWISE_'+w+'mproPM']
        df_fields['e_'+w+'mag'] = df_fields['CATWISE_'+'e_'+w+'mproPM']
        df_fields.loc[df_fields[w+'mag'].isnull(), 'e_'+w+'mag'] = df_fields.loc[df_fields[w+'mag'].isnull(), 'ALLWISE_'+'e_'+w+'mag']
        df_fields.loc[df_fields[w+'mag'].isnull(), w+'mag'] = df_fields.loc[df_fields[w+'mag'].isnull(), 'ALLWISE_'+w+'mag']
        
    return df_fields

from muwclass_library import confident_sigma

def combine_class_result(field_name, dir_out, class_labels,weight_CM=False):

    df_all = pd.read_csv(f'{dir_out}/classes.csv')
    df_mean = df_all.groupby('name').mean(numeric_only=True).iloc[:,:len(class_labels)]

    df_std = df_all.groupby('name').std(numeric_only=True).iloc[:,:len(class_labels)]

    df_class = df_mean.idxmax(axis=1)
    df_prob = df_mean.max(axis=1)
    df_prob_e = pd.DataFrame(data=[df_std.values[i][np.argmax(np.array(df_mean), axis=1)[i]]  for i in range(len(df_std))], columns=['Class_prob_e'])
    df_mean = df_mean.add_prefix('P_')
    df_std  = df_std.add_prefix('e_P_')

    df = pd.concat([pd.concat([df_mean, df_std, df_class, df_prob], axis=1).rename(columns={0:'Class',1:'Class_prob'}).rename_axis('name').reset_index(), df_prob_e], axis=1)
    # print(df.head())

    if weight_CM == True:
        TD_evaluation = pd.read_csv('../files/LOO_classes.csv')
        # calculate weighted probability by accounting the confusion matrix 

        cm_precision = confusion_matrix(TD_evaluation.Class, TD_evaluation.true_Class, labels=class_labels)
        cm_precision = cm_precision / cm_precision.sum(axis=1)[:,None]

        for c in class_labels:
            df = df.rename(columns={'P_'+c:'P_uw_'+c, 'e_P_'+c:'e_P_uw_'+c})
        df = df.rename(columns={'Class':'Class_uw', 'Class_prob':'Class_prob_uw', 'Class_prob_e':'Class_prob_e_uw'})

        # calculate weighted probabilities and errors using error propagation
        df[['P_' + c for c in class_labels]] = df[['P_uw_' + c for c in class_labels]].dot(cm_precision)

        # consider propagating errors using uncertainty in precision matrix too
        df[['e_P_' + c for c in class_labels]] = np.sqrt(np.square(df[['e_P_uw_' + c for c in class_labels]]).dot(np.square(cm_precision)))

        # get most probable class based on weighted probabilities
        df['Class'] = df[['P_' + c for c in class_labels]].idxmax(axis="columns").str.strip('P_')

        df['Class_prob'] = df[['P_' + c for c in class_labels]].max(axis=1)

        # get most probable class errors
        idx, cols = pd.factorize(df['Class'])
        cols = 'e_P_' + cols
        df['Class_prob_e'] = df.reindex(cols, axis=1).to_numpy()[np.arange(len(df)), idx]

        #print(1-(df[['P_w_' + c for c in class_labels]].sum(1)))
    #df = confident_flag(df, class_cols=class_labels)
    df = confident_sigma(df, class_cols=class_labels)

    df_MW = pd.read_csv(f'{dir_out}/{field_name}_MW.csv')
    # print(df_MW.columns)
    
    #df_MW = df_MW.rename(columns={'Fcsc_b':'F_b'})
    #df_MW = prepare_cols(df_MW, cp_thres=0, vphas=False,gaiadata=False)

    df_comb = pd.merge(df, df_MW, left_on='name', right_on='name', how='left')
    df_save = df_comb
    # df_save = df_save.rename(columns={'CSC_RA':'ra', 'CSC_DEC':'dec','CSC_err_r0':'r0'})
    #print(df_save.columns)
    
    #df_save['PU_TeV'] = region_size
    #df_save['TeV_extent'] = 'p'
    df_save = df_save.sort_values(by=['significance','name'],ascending=[False,True]).reset_index(drop=True)
    
    
    mwbands = ['Gmag','BPmag', 'RPmag', 'Jmag','Hmag', 'Kmag', 'W1mag', 'W2mag', 'W3mag']
    df_save['cp_flag'] = 1
    df_save.loc[df_save[mwbands].isna().all(axis=1), 'cp_flag']=0
    df_save['cp_counts'] = 0
    df_save['ra_X'], df_save['dec_X'] = df_save['ra'], df_save['dec']
                
    for band in mwbands:
        df_save.loc[~df_save[band].isna(), 'cp_counts']=df_save.loc[~df_save[band].isna(), 'cp_counts']+1
    df_save['HR_hms'] = (df_save['Fcsc_h']-df_save['Fcsc_m']-df_save['Fcsc_s'])/(df_save['Fcsc_h']+df_save['Fcsc_m']+df_save['Fcsc_s'])
    df_save['F_b'] = df_save['Fcsc_h']+df_save['Fcsc_m']+df_save['Fcsc_s']
    df_save['e_F_b'] = np.sqrt(df_save['e_Fcsc_h']**2+df_save['e_Fcsc_m']**2+df_save['e_Fcsc_s']**2)
    class_prob_columns = [ 'P_'+c for c in class_labels]+[  'e_P_'+c for c in class_labels]
    
    # df_save['true_Class'] = np.nan

    df_save.to_csv(f'{dir_out}/{field_name}_class.csv',index=False) #

    return df_save#field_mw_class

import pandas.io.formats.style

def write_to_html_file(df, clusters_dict, field_name, filename='out.html'):
    '''
    Write an entire dataframe to an HTML file with nice formatting.
    '''

    result = '''

<HTML>
<HEAD>

    <meta name="viewport" content="width=device-width, height=device-height, initial-scale=1.0, user-scalable=no">
    <script type="text/javascript">
    var hipsDir=null;</script>

    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible"
        content="IE=edge">
    <!-- <meta name="viewport"
        content="width=device-width, 
                initial-scale=1.0"> -->
    <style>
        .popup {
            position: absolute;
            z-index: 1;
            left: 60%;
            top: 0;
            width: 40%;
            height: 100%;
            overflow: auto;
            background-color: rgba(0, 0, 0, 0.4);
            display: none;
        }
        .popup-content {
            background-color: white;
            margin: 10% auto;
            padding: 20px;
            border: 1px solid #888888;
            width: 90%;
            font-weight: bolder;
        }
        .popup-content button {
            display: block;
            margin: 0 auto;
        }
        .show {
            display: block;
        }
        /* h1 {
            color: green;
        } */
    </style>

</HEAD>
'''

    result += f'<H1> Multiwavelength Classification of {field_name} Field using MUWCLASS <button id="myButton"> Project Overview </button> </H1>'

    result += '''

<!-- This Web resource contains HiPS(*) components for <B>ESO EPO</B> progressive survey. -->


<div id="myPopup" class="popup">
  <div class="popup-content">
    <h2>Project goals</h2>
      <p>We plan to develop an automated multiwavelength machine-learning classification pipeline (MUWCLASS) to identify the nature of X-ray sources 
    that have been observed by <a href='https://cxc.cfa.harvard.edu/csc/'>Chandra Source Catalog v2.0</a>. 

      <h2>Work performed</h2>
      <p>We collected  multwavelength properties of all X-ray sources in these fields using the Chandra X-ray Source catalog and multiwavelength properties from  Gaia-DR3, 2MASS,
        and WISE catalogs. We used the machine-learning approach implemented in our MUWCLASS pipeline (see e.g., <a href='https://ui.adsabs.harvard.edu/abs/2022ApJ...941..104Y/abstract'>Yang et al., 2022</a>).
        The pipeline relies on the training dataset of ~3,000 X-ray sources that have known astrophysical types (currently these are LM-STAR, HM-STAR, YSO, AGN, LMXB, HMXB, CV, NS).
        We used this pipeline to classify X-ray sources whose classifications are provided in the table below.</p>
      <h2>Funding support</h2>
      <p>This work is supported by the National Aeronautics and Space Administration (NASA) through the Astrophysics Data Analysis Program (ADAP) award 80NSSC19K0576.</p>

      <button id="closePopup">
            Close
        </button>
  </div>
</div>



<script>
    myButton.addEventListener("click", function () {
        myPopup.classList.add("show");
    });
    closePopup.addEventListener("click", function () {
        myPopup.classList.remove("show");
    });
    window.addEventListener("click", function (event) {
        if (event.target == myPopup) {
            myPopup.classList.remove("show");
        }
    });
</script>




<TABLE>
<TR>
<TD>
<script src="https://code.jquery.com/jquery-1.10.1.min.js"></script>
<script type="text/javascript" src="../../../files/aladin.js" charset="utf-8"></script>

<div id="aladin-lite-div" style="width:900px;height:600px"> 
  <div style="background-color: rgba(255, 255, 255, 0.6); z-index: 20; position: absolute; left: 10px;bottom: 30%;">&nbsp;&nbsp;Overlay opacity:<br><input id="opacity" type="range" min="0" max="1" step="0.05" value="0.5"></div>
</div>


Show sources with CT greater than:
<input id='slider' style='vertical-align:middle;width:40vw;' step='0.1' min='0' max='10' type='range' value='0'>
<span id='pmVal'  >0 </span><br><br><div id='aladin-lite-div' style='width: 400px;height: 5px;'></div>

and with Significance greater than:
<input id='slider2' style='vertical-align:middle;width:40vw;' step='0.1' min='3' max='10' type='range' value='0'>
<span id='sigVal'  >0 </span><br><br><div id='aladin-lite-div' style='width: 400px;height: 5px;'></div>



<script type="text/javascript">

    const slider = document.getElementById('opacity');
    slider.oninput = function() {
        aladin.getOverlayImageLayer().setAlpha(slider.value);
    };


    let aladin;
    A.init.then(() => {

        var pmThreshold = 0;
        var sigThreshold = 0;

        var slider = document.getElementById('slider');
        slider.oninput = function() {
            pmThreshold = this.value;
            $('#pmVal').html(pmThreshold);
            hips.reportChange();
        }

        var sigThreshold = 0;

        var slider = document.getElementById('slider2');
        slider.oninput = function() {
            sigThreshold = this.value;
            $('#sigVal').html(sigThreshold);
            hips.reportChange();
        }
        
        var myFilterFunction = function(source) {
            var totalPm  = parseFloat(source.data['CT']);
            var signif  = parseFloat(source.data['significance']);
            if (isNaN(totalPm) || isNaN(signif)) {
                return false;
            }
            return totalPm>pmThreshold && signif>sigThreshold;
        }

        // define custom draw function
        var drawFunction = function(source, canvasCtx, viewParams) {
            canvasCtx.beginPath();
            canvasCtx.arc(source.x, source.y,6, 0, 2 * Math.PI, false);
            canvasCtx.closePath();
            // canvasCtx.strokeStyle = '#c38';
            canvasCtx.strokeStyle = source.data['color'];
            canvasCtx.lineWidth = 3;
            canvasCtx.globalAlpha = 0.7,
            canvasCtx.stroke();
            var fov = Math.max(viewParams['fov'][0], viewParams['fov'][1]);

            // object name is displayed only if fov<10°
            if (fov>10) {
                return;
            }

            canvasCtx.globalAlpha = 0.9;
            canvasCtx.globalAlpha = 1;

            var xShift = 20;

            canvasCtx.font = '15px Arial'
            canvasCtx.fillStyle = '#eee';
            // canvasCtx.fillText(source.data['name'], source.x + xShift, source.y -4);

            // object type is displayed only if fov<2°
            if (fov>0.05) {
                return;
            }
            canvasCtx.font = '12px Arial'
            canvasCtx.fillStyle = '#abc';
            canvasCtx.fillText(source.data['Class'], source.x + 2 + xShift, source.y + 10);
        };

        // define custom draw function
        var drawFunction2 = function(source, canvasCtx, viewParams) {
            canvasCtx.beginPath();
            // canvasCtx.arc(source.x, source.y,10, 0, 2 * Math.PI, false);
            canvasCtx.rect(source.x-8, source.y-8,16,16)
            canvasCtx.closePath();
            // canvasCtx.strokeStyle = '#c38';
            canvasCtx.strokeStyle = source.data['color'];
            canvasCtx.lineWidth = 3;
            canvasCtx.globalAlpha = 0.7,
            canvasCtx.stroke();
            var fov = Math.max(viewParams['fov'][0], viewParams['fov'][1]);

            // object name is displayed only if fov<10°
            if (fov>10) {
                return;
            }

            canvasCtx.globalAlpha = 0.9;
            canvasCtx.globalAlpha = 1;

            var xShift = 20;

            canvasCtx.font = '15px Arial'
            canvasCtx.fillStyle = '#eee';
            // canvasCtx.fillText(source.data['name'], source.x + xShift, source.y -4);

            // object type is displayed only if fov<2°
            if (fov>0.05) {
                return;
            }
            canvasCtx.font = '12px Arial'
            canvasCtx.fillStyle = '#abc';
            canvasCtx.fillText(source.data['Class'], source.x + 2 + xShift, source.y + 10);
        };
'''

    result += f'    aladin = A.aladin("#aladin-lite-div", \u007bsurvey: "https://cdaftp.cfa.harvard.edu/cxc-hips",showSimbadPointerControl: true, name:"Chandra",  target: "{clusters_dict[field_name]["ra"]} {clusters_dict[field_name]["dec"]}", fov: 12 / 60. \u007d);'

    result += '''
    //aladin.toggleFullscreen();

    aladin.setOverlayImageLayer('https://alasky.cds.unistra.fr/pub/10.1051_0004-6361_201732098flux/'); // CDS/P/HGPS/Flux
    aladin.getOverlayImageLayer().setAlpha(0.5);

    aladin.setOverlayImageLayer('CDS/P/DECaPS/DR2/color'); // https://alasky.cds.unistra.fr/DECaPS/DR2/CDS_P_DECaPS_DR2_color/
    aladin.getOverlayImageLayer().setAlpha(0.5);

    aladin.setOverlayImageLayer('CSIRO/P/RACS/low/I'); // https://casda.csiro.au/hips/RACS/low/I/
    aladin.getOverlayImageLayer().setAlpha(0.5);

    aladin.setOverlayImageLayer('CSIRO/P/RACS/mid/I');
    aladin.getOverlayImageLayer().setAlpha(0.5);

    aladin.setOverlayImageLayer('http://archive-new.nrao.edu/vlass/HiPS/VLASS_Epoch1/Quicklook/');
    aladin.getOverlayImageLayer().setAlpha(0.5);

    // aladin.setOverlayImageLayer('http://cade.irap.omp.eu/documents/Ancillary/4Aladin/CGPS_VGPS/');
    // aladin.getOverlayImageLayer().setAlpha(0.5);

    aladin.setOverlayImageLayer('CDS/P/NVSS'); // https://alasky.cds.unistra.fr/NVSS/intensity/
    aladin.getOverlayImageLayer().setAlpha(0.5);
    
    var overlay = A.graphicOverlay({color: 'white', lineWidth: 3});
        aladin.addOverlay(overlay);
        overlay.addFootprints([
'''
    result += f'            A.ellipse({clusters_dict[field_name]["ra"]}, {clusters_dict[field_name]["dec"]}, {clusters_dict[field_name]["radius"]/(60.*np.cos(clusters_dict[field_name]["dec"]*np.pi/180))},{clusters_dict[field_name]["radius"]/60.},0, \u007bcolor: "cyan"\u007d),'
    
    result += '''
            ]);
        // overlay.add(); // radius in degrees , use the polygon to use the calculated coordinates of ellipses

        var cat = A.catalog({name:'popup', sourceSize: 100, onClick: 'showTable', shape: drawFunction, filter: myFilterFunction}); 
        aladin.addCatalog(cat);
'''
    
    df['color'] = 'red'

    for clas, color in zip(['HM-STAR','AGN','YSO','LMXB','CV','HMXB','LM-STAR','NS'], ['deepskyblue', 'cyan','lime','orange','blue','peru','yellow','magenta']):
        df.loc[df['Class']==clas, 'color'] = color 


    for i, df_s in df.iterrows():
        # print(df_s)
        if df_s['name'][-1]=='0':
            
            result += f'        cat.addSources([A.source({df_s["CSC_RA"]}, {df_s["CSC_DEC"]}, \u007bname:"{df_s["name"]}",CT:{df_s["CT"]},significance:{df_s["significance"]},Class:"{df_s["Class"]}",color:"{df_s["color"]}"\u007d)]);\n'
        else:
            
            result += f"        cat.addSources([A.source({df_s['MW_RA']}, {df_s['MW_DEC']}, \u007bname:'{df_s['name']}',CT:{df_s['CT']},significance:{df_s['significance']},Class:'{df_s['Class']}',color:'{df_s['color']}'\u007d)]);\n"
    
    result += '''
        // cat.addSources([A.source(134.782442, -43.707852, {name:'M 86',CT:5,significance: 8,Class: 'AGN',color: 'lime'})]);

        var hips = A.catalogFromURL('https://raw.githubusercontent.com/huiyang-astro/FGL-aladin-lite-test/main/FGL_11152023_all_class_color.vot', {onClick: 'showTable', sourceSize: 100,  name: 'CSC',shape: drawFunction,filter: myFilterFunction}); //,displayLabel: true, labelColumn: 'Class', labelColor: 'cyan', labelFont: '20px sans-serif', onClick: 'showTable',
        hips.hide();  
        aladin.addCatalog(hips);
        
        $('input[type=radio][name=otype]').change(function() {
            requestedOtype = this.value;
            hips.reportChange();
        });

        var hips2 = A.catalogFromURL('https://raw.githubusercontent.com/huiyang-astro/FGL-aladin-lite-test/main/CSCv2_TD.vot', {onClick: 'showPopup', sourceSize: 200,  name:'TD',shape:drawFunction2}); //,displayLabel: true, labelColumn: 'Class', labelColor: 'cyan', labelFont: '20px sans-serif', onClick: 'showTable',
         aladin.addCatalog(hips2);
        
        $('input[type=radio][name=otype]').change(function() {
            requestedOtype = this.value;
            hips2.reportChange();
        });

        var hips9 = A.catalogFromURL('https://raw.githubusercontent.com/huiyang-astro/FGL-aladin-lite-test/main/ATNF_NS.vot', {onClick: 'showPopup', sourceSize: 20,  name:'ATNF',shape:'plus', color:'magenta'}); //,displayLabel: true, labelColumn: 'Class', labelColor: 'cyan', labelFont: '20px sans-serif', onClick: 'showTable',
         aladin.addCatalog(hips9);
        
        var hips10 = A.catalogFromURL('https://raw.githubusercontent.com/huiyang-astro/FGL-aladin-lite-test/main/TeVCat.vot', {onClick: 'showPopup', sourceSize: 20,  name:'TeVCat',shape:'plus', color:'gold'}); //,displayLabel: true, labelColumn: 'Class', labelColor: 'cyan', labelFont: '20px sans-serif', onClick: 'showTable',
         aladin.addCatalog(hips10);

        // var hips11 = A.catalogFromURL('https://raw.githubusercontent.com/huiyang-astro/FGL-aladin-lite-test/main/BeStar.vot', {onClick: 'showPopup', sourceSize: 20,  name:'BeStar',shape:'plus', color:'rgb(0,255,0)'}); //,displayLabel: true, labelColumn: 'Class', labelColor: 'cyan', labelFont: '20px sans-serif', onClick: 'showTable',
        //  aladin.addCatalog(hips11);

        
    });

    
</script>    
</TD>
<TD>
    

    <button id="readmeButton"> README </button>

    <div id="readmePopup" class="popup">
        <div class="popup-content">
            <h2>Aladin-Lite Visualization Window</h2>
                <!-- <p> -->
            <ol>
                <li>You can filter on the classification confidence threshold (CT, the higher the value is the more confidence is the classification, a default value CT=2 is used for confident classifications) and the X-ray source significance by moving the slidebars below the visualization plot. </li>
                <li>You can select image layers and catalog markers from the Manage Layers Button. You can also use the Simbad pointer Button to search for nearby Simbad sources.  More details on <a href='https://aladin.cds.unistra.fr/AladinLite/doc/'>Aladin Lite documentation page</a>.,</li>
                <!-- <li></li> -->
            </ol> 
            <!-- </p> -->
            <h2>Summary Panel</h2>
            <p>The (Classification) Summary panel shows a summarized information of classifications for all sources and confident classifications of significant sources.</p>
            
            <h2>Interactive Table</h2>
            <ol>
                <li> All X-ray sources without significance or CT cuts will be present in the table.</li>
                <li> You can choose to show more columns by selecting the column names from the Column visibillity button.</li>
            </ol>
            Column definitions:
            <ul>
                <li>name: X-ray name from CSCv2.0, -0 indicates no counterpart, -i indicates different counterparts</li>
                <li>Class: classification from MUWCLASS</li>
                <li>Class_prob,Class_prob_e: classification probability and its uncertainty</li>
                <li>CT: classification confidence threshold</li>
                <li>CSC_RA,CSC_DEC,CSC_err_r0,CSC_err_r1,CSC_PA: X-ray coordinate and its error ellipse in arcsec</li>
                <li>significance: X-ray significance</li>
                <li>Fcsc_{b,s,m,h}: X-ray band fluxes at the broad (0.5-7 keV), soft (0.5-1.2 keV), medium (1.2-2 keV), hard (2-7 keV) bands</li>
                <li>HR_hms: X-ray hardness ratios</li>
                <li>var_intra_prob, var_inter_prob: X-ray intra-observation (within one observation) and inter-observation (between observation) variability probability </li>
                <li>p_any,p_i: NWAY association probability and the probability of each ith counterpart </li>
                <li>MW_RA,MW_DEC,MW_err0,MW_sep:counterpart coordinate, its positional uncertainty (in arcsec), and separation to the X-ray position</li>
                <li>Gaia_DR3Name,CATWISE_Name,AllWISE_Name,TMASS_Name</li>
                <li>Gmag,BPmag,RPmag: Gaia DR3 magnitudes </li>
                <li>RPlx: Gaia DR3 Parallax divided by its standard error </li>
                <li>PM: Gaia DR3 Total proper motion, in arcsec/yr</li>
                <li>rgeo: Gaia EDR3 geometric distance posterior, in pc</li>
                <li>Jmag,Hmag,Kmag: 2MASS magnitudes </li>
                <li>W1mag,W2mag,W3mag: WISE magnitudes </li>
                <li>color:marker color of different classification in the interactive plot</li>
            </ul>
            <!-- <p>This work is supported by the National Aeronautics and Space Administration (NASA) through the Astrophysics Data Analysis Program (ADAP) award 80NSSC19K0576.</p>
     -->
            <button id="closereadmePopup">
                Close
            </button>
        </div>
    </div>

    <script>
        readmeButton.addEventListener("click", function () {
            readmePopup.classList.add("show");
        });
        closereadmePopup.addEventListener("click", function () {
            readmePopup.classList.remove("show");
        });
        window.addEventListener("click", function (event) {
            if (event.target == readmePopup) {
                readmePopup.classList.remove("show");
            }
        });
    </script>

    <div>
        <h3><u>(Classification) Summary</u></h3>    
'''
    class_dict = dict(df['Class'].value_counts())
    df_conf = df[(df['significance']>=5.) & (df['CT']>=2.)]
    class_conf_dict = dict(df_conf['Class'].value_counts())
    class_all = ''
    class_conf = ''
    for clas in df['Class'].unique():
        class_all += f'{class_dict[clas]}{clas} '
    for clas in df_conf['Class'].unique():
        class_conf += f'{class_conf_dict[clas]}{clas} '
    result += f'        <li id="all_clas">All classifications: {class_all}</li>\n'
    result += f'        <li id="conf_clas">Classifications of significant sources (S/N>=5) with CT>=2: {class_conf}</li>'

    result += '''
        <li id="comments">Comment: </li>
      </div>



</TD>
</TR>
</TABLE>


<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Interactive Table with Toggleable Columns</title>

<!-- Include jQuery -->
<script src="https://code.jquery.com/jquery-3.6.4.min.js"></script>

<!-- Include DataTables CSS and JS -->
<link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.10.25/css/jquery.dataTables.css">
<script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/1.10.25/js/jquery.dataTables.js"></script>

<!-- Include DataTables ColumnVisibility extension CSS and JS -->
<link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/buttons/1.7.1/css/buttons.dataTables.min.css">
<script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/buttons/1.7.1/js/dataTables.buttons.min.js"></script>
<script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/buttons/1.7.1/js/buttons.colVis.min.js"></script>


    '''
    
    df = df.replace({-99.:np.nan,'nan':'','NaN':''})
    # print(df_html.to_html())


    formats = {'Class_prob': '{:.2f}','Class_prob_e':'{:.2f}','CT':'{:.2f}','CSC_err_r0':'{:.2f}','Fcsc_s':'{:.2e}','Fcsc_m':'{:.2e}','Fcsc_h':'{:.2e}','Fcsc_b':'{:.2e}','HR_hms':'{:.2f}','p_any':'{:.2f}','p_i':'{:.2f}','MW_err0':'{:.2f}','MW_sep':'{:.2f}','Gmag':'{:.3f}','BPmag':'{:.3f}','RPmag':'{:.3f}','RPlx':'{:.1f}','PM':'{:.5f}','rgeo':'{:.0f}'}

    for col, f in formats.items():
        df[col] = df[col].map(lambda x: f.format(x))

    if type(df) == pd.io.formats.style.Styler:
        result += df.render()
    else:
        result += df.to_html(classes='wide',escape=False,index=False)
    result += '''

<script>
  $(document).ready(function() {
      // Initialize DataTable
      var table = $('#interactiveTable').DataTable({
          dom: 'Bflrtip',
          columnDefs: [ //{targets:[1,3], visible: false}],
                    {targets: [2,3,5,6,7,8,9,12,13,14,18,19,20,21,22,23,23,24,25,26,27,29,30,31,32,33,35,36,38,39,40], visible: false }, // Hide the Country column by default (index 2)
                ],
          buttons: ['colvis']
      });
  
      // Define column visibility settings
      // table.column(0).visible(true);

  });
  </script>


</body>
</html>

<HR>
Any comments are welcome by <A HREF="https://github.com/muwclass/MUWCLASS/issues/new">submitting an issue report</A>. 
This visualization tool is displayed by <A HREF="https://aladin.u-strasbg.fr/AladinLite">Aladin Lite</A>. 
<br>
<!-- Funding support: This work is supported by the National Aeronautics and Space Administration (NASA) through the Astrophysics Data Analysis Program (ADAP) award 80NSSC19K0576. -->



</HTML>


'''
    result = result.replace('    <table border="1" class="dataframe wide">', '<table id="interactiveTable" class="display" style="width:100%">')
    with open(filename, 'w') as f:
        f.write(result)


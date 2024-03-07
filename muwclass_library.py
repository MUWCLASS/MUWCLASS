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

import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

import shap

from physical_oversampling import physical_oversample_csv, test_reddening_grid_csv

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
        df.loc[s, 'temp_'+col] = np.random.randn(df.loc[s, col].size) * df.loc[s, 'e_'+col] * factor + df.loc[s, col]
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
                data.loc[data['Class'].isin(red_class), band] = data.loc[data['Class'].isin(red_class), band]*red_fact
    return data


def apply_red2mw(data, ebv, red_class='AGN', deredden=False, self_unred=False, gc=False):
    # extinction.fitzpatrick99 https://extinction.readthedocs.io/en/latest/
    # wavelengths of B, R, I (in USNO-B1), J, H, K (in 2MASS), W1, W2, W3 (in WISE) bands in Angstroms
    # wavelengths of G, Gbp, Grp (in Gaia), J, H, K (in 2MASS), W1, W2, W3 (in WISE) bands in Angstroms

    # ebv is constant ebv applied to whole dataframe
    # deredden requires ebv column in the dataframe

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
        # HUGS luminosities, not absolute magnitudes
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
    if distance_feature=='nodist':
        standidize_by = 'Fcsc_b'
    else:
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
                features = [feature for feature in features if feature not in CSC_flux_features + hugs_features]
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

    # field.to_csv(f'field_processed.csv', index=False)

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

def class_train_model_and_classify_shap(arr):
    
    [i, X_train, y_train, X_test, test_name, X_train_name], ML_model, opts = arr

    if ML_model == 'RF':
        
        clf = RandomForestClassifier(**opts)

        # print(np.shape(X_train),np.shape(X_test))
        # print(X_test.columns)
        clf.fit(X_train, y_train)

        classes = clf.classes_ 

        # compute SHAP values
        explainer = shap.TreeExplainer(clf)
        # shap_values = explainer.shap_values(X_test)
        shap_values = explainer(X_test)
        # print(shap_values)

        pred = clf.predict(X_test)
        prob = clf.predict_proba(X_test)

        imp = clf.feature_importances_

        df_test = pd.DataFrame(prob, columns=classes)
        df_test['Class'] = pred
        df_test['Class_prob'] = prob.max(axis=1)
        df_test['name'] = test_name.tolist()

        df_imp = pd.DataFrame(columns=X_test.columns)
        df_imp.loc[len(df_imp)] = np.array(imp)


        return i, shap_values, classes, test_name, df_test, df_imp



def get_classification_path(clf, X_test, sample_id=0, verb=True):
    '''
    processes X_test.iloc[sample_id] 
    '''
    print('get_classification_path')

    feature_names = clf.feature_names_in_
    classes = clf.classes_
    out_path = []
    out_pred = []

    for est in clf.estimators_:

        n_nodes = est.tree_.node_count
        children_left = est.tree_.children_left
        children_right = est.tree_.children_right
        feature = est.tree_.feature
        threshold = est.tree_.threshold

        weighted_n_node_samples = est.tree_.weighted_n_node_samples

        node_indicator = est.decision_path(X_test.to_numpy())
        leaf_id = est.apply(X_test.to_numpy())

        # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
        node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                            node_indicator.indptr[sample_id + 1]]

        if verb:
            print('Rules used to predict sample {id}:\n'.format(id=sample_id))

        out_path.append([])
        for node_id in node_index:
            # continue to the next node if it is a leaf node
            if leaf_id[sample_id] == node_id:
                continue

            out_path[-1].append([weighted_n_node_samples[node_id],
                                feature_names[feature[node_id]]])

            if verb:
                # check if value of the split feature for sample 0 is below threshold
                if (X_test.iloc[sample_id, feature[node_id]] <= threshold[node_id]):
                    threshold_sign = "<="
                else:
                    threshold_sign = ">"

                print("node {node}, {samples} samples : {feature} = {value:.2f} "
                      "{inequality} {threshold:.2f}".format(
                          node=node_id,
                          samples=str(weighted_n_node_samples[node_id]).rstrip(
                              '0').rstrip('.'),
                          sample=sample_id,
                          feature=feature_names[feature[node_id]],
                          value=X_test.iloc[sample_id, feature[node_id]],
                          inequality=threshold_sign,
                          threshold=threshold[node_id]))

        pred = int(est.predict(X_test.to_numpy())[sample_id])
        out_pred.append(classes[pred])

    return out_path, out_pred


def plot_tree(clf, dir_out, sample_id=0):

    feature_names = clf.feature_names_in_
    class_names = clf.classes_
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(35, 35), dpi=800)
    tree.plot_tree(clf.estimators_[sample_id],
                   feature_names=feature_names,
                   class_names=class_names,
                   filled=True, fontsize=1)
    fig.savefig(f'{dir_out}/rf_individualtree_{sample_id}.png')
    plt.close()
    return clf


def loo_train_and_classify(arr):

    [i, X_train, y_train, X_test, y_test, test_name], opts, return_clf = arr

    print(i)

    clf = RandomForestClassifier(**opts)

    clf.fit(X_train, y_train)

    classes = clf.classes_

    pred = clf.predict(X_test)
    prob = clf.predict_proba(X_test)

    imp = clf.feature_importances_

    df_test = pd.DataFrame(prob, columns=classes)
    df_test['true_Class'] = y_test.tolist()
    df_test['Class'] = pred
    df_test['Class_prob'] = prob.max(axis=1)
    df_test['name'] = test_name.tolist()

    df_imp = pd.DataFrame(columns=X_test.columns)
    df_imp.loc[len(df_imp)] = np.array(imp)

    # test_path, test_pred = get_classification_path(clf, X_test, sample_id=0, verb=False)

    if return_clf:
        return i, df_test, df_imp, clf
    else:
        return i, df_test, df_imp  # , test_path, test_pred


def class_train_model_and_classify(arr):

    [X_train, y_train, X_test, test_name,
        X_train_name], ML_model, opts, return_clf = arr

    # if ML_model == 'RF':

    clf = RandomForestClassifier(**opts)

    clf.fit(X_train, y_train)

    classes = clf.classes_

    pred = clf.predict(X_test)
    prob = clf.predict_proba(X_test)

    

    df_test = pd.DataFrame(prob, columns=classes)
    df_test['Class'] = pred
    df_test['Class_prob'] = prob.max(axis=1)
    df_test['name'] = test_name.tolist()

    # df_test.to_csv(f'{dir_out}class_{i}.csv',index=False)

    imp = clf.feature_importances_

    df_imp = pd.DataFrame(columns=X_test.columns)
    df_imp.loc[len(df_imp)] = np.array(imp)

    # return i, df_test #, df_imp  # , test_path, test_pred

    # elif ML_model == 'lightGBM':

    #     class_labels_lgb = {'AGN': 0, 'NS': 1, 'CV': 2, 'HMXB': 3,
    #                         'LMXB': 4, 'HM-STAR': 5, 'LM-STAR': 6, 'YSO': 7}

    #     y_train_lgb = [class_labels_lgb[y] for y in y_train]

    #     train_data = lgb.Dataset(X_train, label=y_train_lgb)
    #     # , valid_sets=[validation_data])
    #     bst = lgb.train(opts, train_data, num_boost_round=10)
    #     # bst.save_model('model_zeroasmissing.txt')

    #     prob = bst.predict(X_test)

    #     df_test = pd.DataFrame(
    #         prob, columns=['AGN', 'NS', 'CV', 'HMXB', 'LMXB', 'HM-STAR', 'LM-STAR', 'YSO'])

    #     df_test['Class'] = df_test.idxmax(axis=1)
    #     df_test['Class_prob'] = prob.max(axis=1)
    #     df_test['name'] = test_name.tolist()

    #     df_imp = pd.DataFrame(columns=bst.feature_name())
    #     df_imp.loc[len(df_imp)] = np.array(bst.feature_importance())


    if return_clf:
        return df_test, df_imp, clf
    else:
        return df_test, df_imp  # , test_path, test_pred


def loo_save_res(res, dir_out, save_imp=False, suffix=''):

    ii = []

    df_classes = []
    df_imps = []
    paths = []

    for r in res:
        # i, df_test, df_imp, test_path, test_pred = r
        i, df_test, df_imp = r

        ii.append(i)

        df_classes.append(df_test)
        df_imps.append(df_imp)
        # paths.append([test_path, test_pred])

    ii = np.argsort(ii)

    df_classes = [df_classes[i] for i in ii]
    df_classes = pd.concat(df_classes).reset_index(drop=True)

    df_classes.to_csv(f'{dir_out}/loo_classes{suffix}.csv', index=False)

    if save_imp:
        df_imps = [df_imps[i] for i in ii]
        df_imps = pd.concat(df_imps).reset_index(drop=True)
        df_imps.to_csv(f'{dir_out}/loo_imps{suffix}.csv', index=False)

    # paths = [paths[i] for i in ii]
    # json.dump(paths, open(f'{dir_out}/paths.json', 'wt'))

    # print(f'output files in {dir_out}:\nclasses.csv\nimps.csv\npaths.json')


def class_save_res(res, dir_out):

    ii = []

    df_classes = []
    df_imps = []
    paths = []

    for i, r in enumerate(res):
        df_test, df_imp = r  # , test_path, test_pred = r

        ii.append(i)

        df_classes.append(df_test)
        df_imps.append(df_imp)
        # paths.append([test_path, test_pred])

    ii = np.argsort(ii)

    df_classes = [df_classes[i] for i in ii]
    df_classes = pd.concat(df_classes).reset_index(drop=True)
    df_classes.to_csv(f'{dir_out}/classes.csv', index=False)

    # '''
    df_imps = [df_imps[i] for i in ii]
    df_imps = pd.concat(df_imps).reset_index(drop=True)
    df_imps.to_csv(f'{dir_out}/imps.csv', index=False)
    # '''
    # paths = [paths[i] for i in ii]
    # json.dump(paths, open(f'{dir_out}/paths.json', 'wt'))

    # print(f'output files in {dir_out}:\nclasses.csv\nimps.csv\npaths.json')

def plot_confusion_matrix(df, 
                          title='Normalized confusion matrix (%)',
                          cm_type='recall',
                          classes = ['AGN','CV','HM-STAR','LM-STAR','HMXB','LMXB','NS','YSO'],
                          true_class = 'true_Class',
                          pred_class = 'Class',
                          normalize=True,
                          count_fraction=False,
                          df_all = None,                           
                          pallete=cc.fire[::-1],
                          fill_alpha=0.6,
                          width=600, 
                          height=600,
                          plot_zeroes=True
                         ):

    if cm_type=='recall':
        xlabel='Predicted Class'
        ylabel='True Class'
    elif cm_type=='precision':
        xlabel='True Class'
        ylabel='Predicted Class'
    else:
        raise ValueError("Type must be recall or precision!") 

    #classes = np.sort(df.true_Class.unique())
    if cm_type=='recall':
        cm = confusion_matrix(df[true_class], df[pred_class], labels=classes)
    if cm_type=='precision':
        cm = confusion_matrix(df[pred_class], df[true_class], labels=classes)
    
    if normalize:
        cm = 100 * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    cm = np.nan_to_num(cm, nan=0, posinf=0, neginf=0)
    _ = []
    for i in range(len(cm)):
        for j in range(len(cm[0])):        
            f = format(np.round(cm[i, j]).astype(int), 'd')  
            if not plot_zeroes and f == '0': continue # f = ''
            _.append([classes[i], classes[j], f])        
    _ =  pd.DataFrame(dict(zip(['y_classes', 'x_classes', 'counts'], np.transpose(_))))         
    source = ColumnDataSource(_)

    p = figure(width=width, 
               height=height, 
               title=title,
               x_range=classes, 
               match_aspect=True,
               # aspect_scale=2,
               y_range=classes[::-1], 
               toolbar_location=None, 
               # tools='hover'
              )

    p.rect('x_classes', 
           'y_classes', 
           1, 
           1, 
           source=source, 
           fill_alpha=fill_alpha, 
           line_color=None,
           color=linear_cmap('counts', pallete, 0, 2 * cm.max())
          )

    text_props = {'source': source, 'text_align': 'center', 'text_baseline': 'middle'}
    x = dodge('x_classes', 0, range=p.x_range)
    r = p.text(x=x, y='y_classes', text='counts', **text_props)
    r.glyph.text_font_style='bold'

    p.outline_line_color = None
    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_label_text_font_size = '10pt'
    p.xaxis.major_label_orientation = np.pi/4

    p.yaxis.axis_label = ylabel#'True class'
    p.xaxis.axis_label = xlabel#'Predicted class'

    p.axis.axis_label_text_font_size = '18pt'
    p.axis.axis_label_text_font_style = 'normal'

    p.title.text_font_size = '18pt'
    p.title.text_font_style = 'normal'

    p.axis.major_label_standoff = 5
        
    if cm_type == 'recall':
        class_abun = df[true_class].value_counts().to_dict()  
    elif cm_type=='precision':
        class_abun = df[pred_class].value_counts().to_dict()  
    y_labels = {_: f'{_}\n{class_abun[_] if _ in class_abun else 0}' for _ in p.y_range.factors}   

    if count_fraction == True:
        if cm_type == 'recall':
            class_abun_all = df_all[true_class].value_counts().to_dict()  
        elif cm_type=='precision':
            class_abun_all = df_all[pred_class].value_counts().to_dict()  
    
        y_labels = {_: f'{_}\n{class_abun[_]/class_abun_all[_] if _ in class_abun else 0:.2f}' for _ in p.y_range.factors}    
    p.yaxis.formatter = FuncTickFormatter(code=f'''
            var labels = {y_labels}
            return labels[tick]
        ''') 
        
    return(p)


def plot_confusion_matrix_v1(df,
                          title='Normalized confusion matrix (%)',
                          normalize=True,
                          pallete=cc.fire[::-1],
                          fill_alpha=0.6,
                          width=600,
                          height=600,
                          plot_zeroes=True,
                          completeness_values=False,
                          total_dict = None,
                          ):

    classes = np.sort(df.true_Class.unique())

    cm = confusion_matrix(df.true_Class, df.Class, labels=classes)

    if normalize:
        cm = 100 * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    _ = []
    for i in range(len(cm)):
        for j in range(len(cm[0])):
            f = format(np.round(cm[i, j]).astype(int), 'd')
            if not plot_zeroes and f == '0':
                continue  # f = ''
            _.append([classes[i], classes[j], f])
    _ = pd.DataFrame(
        dict(zip(['true_classes', 'classes', 'counts'], np.transpose(_))))
    source = ColumnDataSource(_)

    p = figure(width=width,
               height=height,
               title=title,
               x_range=classes,
               match_aspect=True,
               # aspect_scale=2,
               y_range=classes[::-1],
               toolbar_location=None,
               # tools='hover'
               )

    p.rect('classes',
           'true_classes',
           1,
           1,
           source=source,
           fill_alpha=fill_alpha,
           line_color=None,
           color=linear_cmap('counts', pallete, 0, 2 * cm.max())
           )

    text_props = {'source': source,
                  'text_align': 'center', 'text_baseline': 'middle'}
    x = dodge('classes', 0, range=p.x_range)
    r = p.text(x=x, y='true_classes', text='counts', **text_props)
    r.glyph.text_font_style = 'bold'

    p.outline_line_color = None
    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_label_text_font_size = '10pt'
    p.xaxis.major_label_orientation = np.pi/4

    p.yaxis.axis_label = 'True class'
    p.xaxis.axis_label = 'Predicted class'

    p.axis.axis_label_text_font_size = '18pt'
    p.axis.axis_label_text_font_style = 'normal'

    p.title.text_font_size = '18pt'
    p.title.text_font_style = 'normal'

    p.axis.major_label_standoff = 5

    class_abun = df.true_Class.value_counts().to_dict()
    if completeness_values:
        y_labels = {_: f'{_}\n{class_abun[_]}\n({class_abun[_]*100/total_dict[_]:.1f}%)' for _ in p.y_range.factors}
    else:
        y_labels = {_: f'{_}\n{class_abun[_]}' for _ in p.y_range.factors}
    p.yaxis.formatter = FuncTickFormatter(code=f'''
            var labels = {y_labels}
            return labels[tick]
        ''')

    return (p)


def plot_CM_withSTD(df,
                    normalize=False,
                    cm_type='recall',
                    title=None,
                    cmap=plt.cm.Blues):
    """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if not title:
        if normalize:
            title = f'Normalized {cm_type} confusion matrix'
        else:
            title = f'{cm_type.capitalize()} confusion matrix, without normalization'

    classes = np.sort(df.true_Class.unique())
    if cm_type == 'recall':
        cm = confusion_matrix(df.true_Class, df.Class, labels=classes)
    elif cm_type == 'precision':
        cm = confusion_matrix(df.Class, df.true_Class, labels=classes)
    stds = np.zeros((len(classes), len(classes)))
    total = np.sum(cm, axis=1)

    # old code
    # class_with_num = [class_labels[classes[i]] + '\n' + str(round(total[i])) for i in range(len(total))]

    class_with_num = [classes[i] + '\n' +
                      str(round(total[i])) for i in range(len(classes))]

    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm_copy = cm.copy()
        cm = cm.astype('float') / cm_copy.sum(axis=1)[:, np.newaxis]
        stds = stds.astype('float') / cm_copy.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    else:
        pass
        # print('Confusion matrix, without normalization')

    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # set colorbar font size
    ax.figure.axes[-1].tick_params(labelsize=15)
    # We want to show all ticks...
    if cm_type == 'recall':
        xlabel = 'Predicted label'
        ylabel = 'True label'
    elif cm_type == 'precision':
        xlabel = 'True label'
        ylabel = 'Predicted label'
    else:
        raise ValueError("Type must be recall or precision!")

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes,
           yticklabels=class_with_num,
           title=title,
           xlabel=xlabel,
           ylabel=ylabel,)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # set axes tick label font size
    ax.tick_params(axis='both', which='major', labelsize=15)

    # set axes label font size
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)

    # set title font size
    ax.title.set_fontsize(20)

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    # print(thresh)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, "{:.2f} \n".format(cm[i, j])+r"$\pm$"+"{:.2f}".format(stds[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=10)
    fig.tight_layout()
    return fig


def confident_sigma(df, class_cols=['AGN', 'CV', 'HM-STAR', 'LM-STAR', 'HMXB', 'LMXB', 'NS', 'YSO'], class_prob='Class_prob', class_prob_e='Class_prob_e'):

    df['CT'] = np.nan
    df['CT'] = df.apply(lambda row: sorted([(row[class_prob] - row['P_'+clas])/(row[class_prob_e]+row['e_P_'+clas]
                        if row[class_prob_e]+row['e_P_'+clas] != 0 else 1e-5) for clas in class_cols])[1], axis=1)

    return df


def mw_counterpart_flag(df, mw_cols=['Gmag', 'BPmag', 'RPmag', 'Jmag', 'Hmag', 'Kmag', 'W1mag_comb', 'W2mag_comb', 'W3mag_allwise']):

    df['mw_cp_flag'] = 0
    df = df.fillna(exnum)  # df.replace(np.nan, exnum, inplace=True)

    for i, mw_col in enumerate(mw_cols):
        df['mw_cp_flag'] = df.apply(
            lambda row: row.mw_cp_flag+2**i if row[mw_col] != exnum else row.mw_cp_flag, axis=1)

    df = df.replace(exnum, np.nan)

    return df


def find_confident(df, method='70', thres=2.):

    df_conf = pd.DataFrame()

    if method == 'sigma-qt':

        for i, df_s in df.iterrows():
            # print(i, df_s)
            prob1 = df_s['Class_prob_2sig_low']  # .values[0]
            upp_prob_cols = [c for c in df.columns if 'P_2sig_upp' in c]
            upp_prob_cols.remove('P_2sig_upp_'+df_s['Class'])
            prob2 = max([df_s[upp_prob_cols[i]]
                        for i in range(len(upp_prob_cols))])
            if prob1 > prob2:

                df_conf = df_conf.append(df_s)

    if method == '70':

        for source, i in zip(df.name.unique(), range(len(df.name.unique()))):
            df_source = df[df.name == source]
            if df_source.loc[df['name'] == source, 'Class_prob'].values > 0.7:
                df_conf = df_conf.append(df_source)

    if method == 'sigma':

        for source, i in zip(df.name.unique(), range(len(df.name.unique()))):
            df_s = df[df.name == source]

            prob1 = df_s['Class_prob'].values[0] - \
                thres*df_s['Class_prob_e'].values[0]

            classes = [c.strip('e_P_') for c in df.columns if 'e_P' in c]

            prob_cols = ['P_' + c for c in classes]
            e_prob_cols = ['e_P_' + c for c in classes]

            prob_cols.remove('P_'+df_s['Class'].values)
            e_prob_cols.remove('e_P_'+df_s['Class'].values)

            # print(prob_cols)
            # print(e_prob_cols)

            # prob2 = max([df_s[prob_cols[i]].values[0] for i in range(len(prob_cols))])
            prob2 = max([df_s[prob_cols[i]].values[0]+thres *
                        df_s[e_prob_cols[i]].values[0] for i in range(len(prob_cols))])

            if prob1 > prob2:

                df_conf = df_conf.append(df_s)

    if method == 'both':

        for source, i in zip(df.name.unique(), range(len(df.name.unique()))):
            df_s = df[df.name == source]

            prob1 = df_s['Class_prob'].values[0] - \
                thres*df_s['Class_prob_e'].values[0]

            prob_cols = ['P_AGN', 'P_NS', 'P_CV', 'P_HMXB',
                         'P_LMXB', 'P_LM-STAR', 'P_HM-STAR', 'P_YSO']
            e_prob_cols = ['e_P_AGN', 'e_P_NS', 'e_P_CV', 'e_P_HMXB',
                           'e_P_LMXB', 'e_P_LM-STAR', 'e_P_HM-STAR', 'e_P_YSO']

            prob_cols.remove('P_'+df_s['Class'].values)
            e_prob_cols.remove('e_P_'+df_s['Class'].values)

            # print(prob_cols)
            # print(e_prob_cols)

            # prob2 = max([df_s[prob_cols[i]].values[0] for i in range(len(prob_cols))])
            prob2 = max([df_s[prob_cols[i]].values[0]+thres *
                        df_s[e_prob_cols[i]].values[0] for i in range(len(prob_cols))])

            if prob1 > prob2 and df_s['Class_prob'].values > 0.7:

                df_conf = df_conf.append(df_s)

    if method == 'previous':

        for source, i in zip(df.name.unique(), range(len(df.name.unique()))):
            df_s = df[df.name == source]

            prob1 = df_s['Class_prob'].values[0] - \
                thres*df_s['Class_prob_e'].values[0]

            prob_cols = ['P_AGN', 'P_NS', 'P_CV', 'P_HMXB',
                         'P_LMXB', 'P_LM-STAR', 'P_HM-STAR', 'P_YSO']
            e_prob_cols = ['e_P_AGN', 'e_P_NS', 'e_P_CV', 'e_P_HMXB',
                           'e_P_LMXB', 'e_P_LM-STAR', 'e_P_HM-STAR', 'e_P_YSO']

            prob_cols.remove('P_'+df_s['Class'].values)
            e_prob_cols.remove('e_P_'+df_s['Class'].values)

            # print(prob_cols)
            # print(e_prob_cols)

            prob2 = max([df_s[prob_cols[i]].values[0] for i in range(8)])
            # print(prob2)

            if prob1 > prob2:

                df_conf = df_conf.append(df_s)

    return df_conf


def confidence(df_class, weighted=False, cut='sigma', sigma=2):

    df_class = df_class.copy()

    # get list of classes from probability columns
    classes = [c.strip('e_P_')
               for c in df_class.columns if 'e_P' in c and 'e_P_w' not in c]

    # define stellar compact object classes
    CO_classes = ['LMXB', 'HMXB', 'CV', 'NS']
    nonCO_classes = list(set(classes)-set(CO_classes))

    # define probability and error of stellar compact object

    if weighted:
        df_class['P_CO'] = df_class[['P_w_' + c for c in CO_classes]].sum(1)
        df_class['e_P_CO'] = np.sqrt(
            np.sum(np.square(df_class[['e_P_w_' + c for c in CO_classes]]), 1))

    else:
        df_class['P_CO'] = df_class[['P_' + c for c in CO_classes]].sum(1)
        df_class['e_P_CO'] = np.sqrt(
            np.sum(np.square(df_class[['e_P_' + c for c in CO_classes]]), 1))

    # candidate CO if P_CO - sigma stdev greater than max of non CO classes' sigma * stdev upper probability limit
    df_class['Candidate_CO'] = (df_class['P_CO'] - sigma*df_class['e_P_CO'] >= (df_class.loc[:, [
                                'P_' + c for c in nonCO_classes]].values + sigma*df_class.loc[:, ['e_P_' + c for c in nonCO_classes]].values).max(1))

    # df_class['Candidate_CO'] = df_class['P_CO']>=0.7

    if cut == "simple":
        df_class["Class"] = np.where(
            df_class["Class_prob"] >= 0.70, df_class["Class"], "Unconfident Classification")

    if cut == "sigma":
        # select sources whose most confident classification is sigma stdev greater than all other classifications' P + sigma*e_P.

        classes = [c.strip('e_P_')
                   for c in df_class.columns if 'e_P' in c and 'e_P_w' not in c]
        classes.remove('CO')
        ps = df_class.loc[:, ['P_' + c for c in classes]]
        pes = df_class.loc[:, ['e_P_' + c for c in classes]]

        # find second largest probability
        # second = ps.apply(lambda row: row.nlargest(n).values[-1], axis=1)
        # second_e = pes.apply(lambda row: row.nlargest(2).values[-1], axis=1)

        # p + sigma*e_p
        second = ps.values+sigma*pes.values

        # remove most confident classification from second
        idx = ps.values.argmax(1)[:, None]
        second = second[np.arange(ps.shape[1]) != idx].reshape(ps.shape[0], -1)

        if weighted:
            df_class["Class_w"] = np.where(df_class["Class_prob_w"]-sigma*df_class["Class_prob_e_w"]
                                           >= second.max(1), df_class["Class_w"], "Unconfident Classification")
        else:
            df_class["Class"] = np.where(df_class["Class_prob"]-sigma*df_class["Class_prob_e"]
                                         >= second.max(1), df_class["Class"], "Unconfident Classification")

    return df_class


def classification_probs(data):
    probs_median = []
    probs_std = []
    preds = []
    sources = []
    pred_p = []
    pred_e_p = []
    true_classes = []

    for source, i in zip(data.name.unique(), range(len(data.name.unique()))):
        df_source = data[data.name == source]
        src_probs = np.array(df_source.iloc[:, :8])

        prob_median = np.median(src_probs, axis=0) / \
            np.median(src_probs, axis=0).sum()

        # median of probability vectors no longer add to 1, normaize

        prob_std = (np.percentile(src_probs, 84, axis=0) -
                    np.percentile(src_probs, 16, axis=0))/2.
        # print(prob_ave,'\n',prob_std)
        probs_median.append(prob_median)
        probs_std.append(prob_std)
        ind_pred = np.argmax(prob_median)
        preds.append(data.columns[ind_pred])
        if 'true_Class' in data.columns:
            true_classes.append(df_source['true_Class'].values[0])
        sources.append(source)
        pred_p.append(prob_median[ind_pred])
        pred_e_p.append(prob_std[ind_pred])

    if 'true_Class' in data.columns:
        df_save = pd.DataFrame({'name': sources,
                                'Class': preds,
                                'true_Class': true_classes,
                                'Class_prob': pred_p,
                                'Class_prob_e': pred_e_p}
                               )
    else:
        df_save = pd.DataFrame({'name': sources,
                                'Class': preds,
                                'Class_prob': pred_p,
                                'Class_prob_e': pred_e_p}
                               )

    for i in range(8):
        df_save['P_'+data.columns[i]] = np.array(probs_median)[:, i]
        df_save['e_P_'+data.columns[i]] = np.array(probs_std)[:, i]

    return df_save


plt.rcParams.update({'font.size': 40})
params = {'legend.fontsize': 'large',
          # 'figure.figsize': (15, 5),
          'axes.labelsize': 'large',
          'axes.titlesize': 'large',
          'xtick.labelsize': 'large',
          'ytick.labelsize': 'large'}
plt.rcParams.update(params)


def plot_classifier_matrix_withSTD(probs, stds, pred, yaxis, classes, normalize=False,
                                   title=False, nocmap=False,
                                   cmap=plt.cm.Blues):
    if not title:
        title = 'Classifier matrix'
    length = len(pred)
    '''
    if length <=2:
        fig, ax = plt.subplots(figsize=(21, length+3))
    if length >2:
        #fig, ax = plt.subplots(figsize=(21, length*1.5+3))
        fig, ax = plt.subplots(figsize=(21, length+3))
    '''
    # fig, ax = plt.subplots(figsize=(12, length+2))
    fig, ax = plt.subplots(figsize=(21, length+3))
    im = ax.imshow(probs, interpolation='nearest', cmap=cmap)
    if nocmap == False:
        ax.figure.colorbar(im, ax=ax)
    probs = np.array(probs)
    # We want to show all ticks...
    ax.set(xticks=np.arange(probs.shape[1]),
           yticks=np.arange(probs.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=yaxis,
           title=title,
           # ylabel='True label',
           xlabel='Class')
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.xaxis.label.set_size(25)
    ax.set_title(title, fontsize=30)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=20,
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    # fmt = '.2f' if normalize else 'd'
    thresh = probs.max() / 2.
    for i in range(probs.shape[0]):
        for j in range(probs.shape[1]):
            ax.text(j, i, "{:.2f} \n".format(probs[i, j])+r"$\pm$"+"{:.2f}".format(stds[i, j]),
                    ha="center", va="center", fontsize=20,
                    color="white" if probs[i, j] > thresh else "black")

    fig.tight_layout()
    return fig


def plot_Feature_Importance_withSTD(imp, std, features, fig_width, fig_height, fontsize=15):
    # sbn.set_style("white")
    N = len(imp)

    ind = np.arange(N)  # the x locations for the groups

    width = 0.7  # 16./N *2.      # the width of the bars

    fig, ax = plt.subplots(figsize=(fig_width/80, fig_height/80))
    rects1 = ax.barh(ind, imp*100, width, xerr=std*100, ecolor='orange')
    # rects2 = ax.barh(ind, imp_noran*100, width, xerr=std_noran*100, alpha=0, ecolor='red')
    # rects2 = ax.bar(ind + width, lassoo*100, width, color='g')
    # add some text for labels, title and axes ticks
    ax.set_xlabel('Importance (in % usage)',
                  fontweight='bold', fontsize=fontsize)
    ax.set_xticks(range(20))
    ax.set_xticklabels((str(i) for i in range(20)),
                       fontweight='bold', fontsize=fontsize)
    ax.set_title('Feature Importance', fontweight='bold',
                 fontsize=fontsize*1.2)
    ax.set_yticks(ind)
    ax.set_yticklabels(features, fontweight='bold', fontsize=fontsize/1.5)

    # ax.legend(rects1[0], ('Random Forest Regressor'))
    ax.set_xlim(0,)
    # ax.set_ylim(-0.5,N+0.5)

    # print('There are ',len(features), ' features')
    # print(features)
    # print(imp)
    # zip([0.8, 1., 1.5], ['orange','red','green']):
    for threshold, color in zip([1.], ['red']):

        thres = threshold/100.
        plt.axvline(threshold, color=color)

        index_feature_select = np.where(imp > thres)[0]
        features_selected = np.array(features)[index_feature_select]
        features_selected_imp = imp[index_feature_select]
        # print('There are ',len(features_selected), ' features selected with thres at', str(threshold))
        # print( features_selected, ' as Selected features')
        # print(features_selected_imp, ' as Selected features imps')
    # plt.show()
    # fig.tight_layout()
    return fig

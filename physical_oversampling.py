import pandas as pd
import numpy as np
from gdpyc import GasMap, DustMap
from astropy.coordinates import SkyCoord
from scipy.interpolate import InterpolatedUnivariateSpline
import extinction
from random import randint


from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler 
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

class_labels = ['AGN','YSO','LM-STAR', 'HM-STAR', 'NS','CV',   'HMXB', 'LMXB']
class_color = {'YSO':'grey', 'LMXB':'magenta', 'HM-STAR':'green', 'LM-STAR':'orange', 'CV':'blue', 'NS':'brown', 'AGN':'red', 'HMXB':'purple'}

gaia_features = ['Gmag','BPmag', 'RPmag']
gaia_limits   = [21.5,   21.5,   21.]
gaia_zeros    = [2.5e-9, 4.08e-9, 1.27e-9]#[3228.75, 3552.01, 2554.95]
gaia_eff_waves   = [5822.39, 5035.75, 7619.96]
gaia_width_waves = [4052.97, 2157.50, 2924.44]
twomass_features = ['Jmag','Hmag','Kmag']
twomass_limits   = [18.5,   18.0,  17.0]
twomass_zeros    = [3.13e-10, 1.13e-10, 4.28e-11]#[1594.,  1024., 666.7]
twomass_eff_waves   = [12350., 16620., 21590.]
twomass_width_waves = [1624.32, 2509.40, 2618.87]
wise_features = ['W1mag','W2mag','W3mag']
wise_limits   = [18.5,   17.5   , 14.5]
wise_zeros    = [8.18e-12, 2.42e-12, 6.52e-14] #[309.54, 171.787, 31.674]
wise_eff_waves      = [33526, 46028, 115608]
wise_width_waves    = [6626.42, 10422.66, 55055.71] #[34000., 46000., 120000.]

hugs_features = ['F275W', 'F336W', 'F438W', 'F606W', 'F814W']

MW_features = gaia_features + twomass_features + wise_features
MW_limits = gaia_limits + twomass_limits + wise_limits  # limiting magnitudes
MW_zeros  = gaia_zeros + twomass_zeros + wise_zeros    # zero points to convert magnitude to flux in wavelength space
MW_width_waves  = gaia_width_waves + twomass_width_waves + wise_width_waves    # effective wavelength widths
MW_eff_waves  = gaia_eff_waves + twomass_eff_waves + wise_eff_waves    # effective wavelength 

Gammas = {'AGN':1.94,'YSO':2.95,'LM-STAR':5.14,'HM-STAR':3.09,'NS':1.94,'CV':1.61,'HMXB':1.28,'LMXB':1.97}
df_Gamma = pd.DataFrame.from_dict(data=Gammas, orient='index',columns=['Gamma'])

ebv_uniform_highest = 3.

def red_factor(ene, nH, Gamma, tbabs_ene, tbabs_cross): # energy in keV, nH in cm^-2

    if Gamma == 2:
        flux_unred_int = np.log(ene[1]) - np.log(ene[0])
    else:
        flux_unred_int   = (ene[1]**(2.-Gamma)-ene[0]**(2.-Gamma))/(2.-Gamma)
            
    _ = np.array([_**(1 - Gamma) for _ in tbabs_ene]) # pseudo spectrum            
    tbabs_flux_red = _ * np.exp(-nH * 1e-3 * tbabs_cross) # extincted spectrum
    
    finterp = InterpolatedUnivariateSpline(tbabs_ene, tbabs_flux_red, k=1)
    
    flux_red_int = finterp.integral(*ene)
        
    return flux_red_int / flux_unred_int


def apply_red2mw(data, ebv, red_class='AGN', self_unred = False):
    # extinction.fitzpatrick99 https://extinction.readthedocs.io/en/latest/
    ### wavelengths of B, R, I (in USNO-B1), J, H, K (in 2MASS), W1, W2, W3 (in WISE) bands in Angstroms
    # wavelengths of G, Gbp, Grp (in Gaia), J, H, K (in 2MASS), W1, W2, W3 (in WISE) bands in Angstroms
    
    #waves = gaia_eff_waves + twomass_eff_waves + wise_eff_waves 
    #bands = gaia_features + twomass_features + wise_features
       
    for wave, band in zip(MW_eff_waves, MW_features):
        if red_class != 'all':
            if self_unred == True:
                for idx in data.loc[data['Class'] == red_class].index.tolist():
                    data[band][idx] = data[band][idx] + extinction.fitzpatrick99(np.array([wave]), 3.1*(ebv-data['ebv_apply'][idx]))
            if self_unred == False:
                data.loc[data.Class == red_class, band] = data.loc[data.Class == red_class, band] + extinction.fitzpatrick99(np.array([wave]), 3.1*ebv)
        else:
            if self_unred == True:
                #for idx in data.loc[data['Class'] == red_class].index.tolist():
                #data[band] = data[band] + extinction.fitzpatrick99(np.array([wave]), 3.1*(ebv-data['ebv_unred']))
                data[band] = data.apply(lambda r: r[band] + extinction.fitzpatrick99(np.array([wave]), 3.1*(ebv-r['ebv_apply'])),axis=1) 
                data[band] = data.apply(lambda r: r[band][0],axis=1)
            if self_unred == False:
                data[band] = data[band] + extinction.fitzpatrick99(np.array([wave]), 3.1*ebv)
     
    return data

def apply_red2csc(data, nh, tbabs_ene, tbabs_cross, red_class='AGN', self_unred=False, Gamma=2):
    bands = ['Fcsc_s', 'Fcsc_m', 'Fcsc_h', 'Fcsc_b']
    enes  = [[0.5,1.2], [1.2,2.0], [2.0,7.0], [0.5, 7.0]]
    for ene, band in zip(enes, bands):
        red_fact = red_factor(ene, nh, Gamma, tbabs_ene, tbabs_cross)
        if red_class != 'all':
            if self_unred == True:
                for idx in data.loc[data['Class'] == red_class].index.tolist():
                    data[band][idx] = data[band][idx] * red_factor(ene, nh - data['nH_apply'][idx], Gamma, tbabs_ene, tbabs_cross)
            if self_unred == False:
                data.loc[data.Class == red_class, band] = data[band]*red_fact
        else:
            if self_unred == True:
                #for idx in data.loc[data['Class'] == red_class].index.tolist():
                #data[band] = data[band] * red_factor(ene, nh - data['nH_unred'], Gamma, tbabs_ene, tbabs_cross)
                data[band] = data.apply(lambda r: r[band] * red_factor(ene, nh - r.nH_apply, r.Gamma, tbabs_ene, tbabs_cross),axis=1)
            if self_unred == False:
                data[ band] = data[band]*red_fact
    return data


def physical_oversample(TD, tbabs_ene, tbabs_cross,random_state=None,ebv_pdf='uniform' ):
    
    if random_state is None:
        np.random.seed(randint(1,999999999)) 
    else:
        np.random.seed(random_state)
        
    TD_X, TD_y = TD, TD['Class']
    ros = RandomOverSampler(random_state=random_state)
    X_res, y_res = ros.fit_resample(TD_X, TD_y)
    
    X_res = X_res[X_res.duplicated(subset=['name'])].reset_index(drop=True)
    
    if ebv_pdf == 'uniform':
        X_res['ebv_apply'] = -1.*np.random.uniform(low=0.0, high=ebv_uniform_highest, size=len(X_res))
    elif ebv_pdf == 'poisson':
        X_res['ebv_apply'] = -1.*np.random.poisson(lam=3, size=len(X_res))
        
    X_res['nH_apply'] = 2.21 * 3.1 * X_res['ebv_apply'] + np.random.normal(loc=0.0, scale=1.0, size=len(X_res))*0.09*np.sqrt(2) * 3.1 * X_res['ebv_apply']

    #Gammas = {}

    #for clas in class_labels:
        #Gammas[clas] = TD.loc[TD.Class==clas, 'powlaw_gamma_mean'].median()
        
    X_res['Gamma'] = X_res.apply(lambda r: np.random.randn()*r['e_powlaw_gamma_mean']*np.sqrt(2.)+ r['powlaw_gamma_mean'] if r['powlaw_gamma_mean']>-100 else Gammas[r['Class']],axis=1)

    data_red = apply_red2mw(X_res.copy(), 0,  red_class='all', self_unred=True)

    df_oversample = apply_red2csc(data_red, 0, tbabs_ene, tbabs_cross,  red_class='all', self_unred=True)
    
    df_oversample = pd.concat([TD, df_oversample], ignore_index=True, sort=False)

    
    return df_oversample

def test_reddening_grid(df, TD, tbabs_ene, tbabs_cross,random_state=None):

    if random_state is None:
        np.random.seed(randint(1,999999999)) 
    else:
        np.random.seed(random_state)
        
    dfs = pd.concat([df]*int(ebv_uniform_highest+1), ignore_index=True)
    #print(dfs)
    dfs['ebv_apply'] = dfs.index
    dfs['ebv_apply'] = pd.to_numeric(dfs['ebv_apply'])*-1.
    #print(dfs['ebv_apply'])
    dfs['nH_apply'] = 2.21 * 3.1 * dfs['ebv_apply'] + np.random.normal(loc=0.0, scale=1.0, size=len(dfs))*0.09*np.sqrt(2) * 3.1 * dfs['ebv_apply']
    #Gammas = {}

    #for clas in class_labels:
        #Gammas[clas] = TD.loc[TD.Class==clas, 'powlaw_gamma_mean'].median()
    
    dfs['Gamma'] = dfs.apply(lambda r: np.random.randn()*r['e_powlaw_gamma_mean']*np.sqrt(2.)+ r['powlaw_gamma_mean'] if r['powlaw_gamma_mean']>-100 else Gammas[r['Class']],axis=1)

    data_red = apply_red2mw(dfs.copy(), 0,  red_class='all', self_unred=True)

    dfs = apply_red2csc(data_red, 0, tbabs_ene, tbabs_cross,  red_class='all', self_unred=True)
    
    return dfs
   
    

def physical_oversample_hpy(TD, hpy, random_state=None,ebv_pdf='uniform'):
    
    if random_state is None:
        np.random.seed(randint(1,999999999)) 
    else:
        np.random.seed(random_state)
        
    TD_X, TD_y = TD, TD['Class']
    ros = RandomOverSampler(random_state=random_state)
    X_res, y_res = ros.fit_resample(TD_X, TD_y)
    
    X_res = X_res[X_res.duplicated(subset=['name'])].reset_index(drop=True)
    
    if ebv_pdf == 'uniform':
        X_res['ebv_apply'] = np.random.uniform(low=0.0, high=10, size=len(X_res))
    elif ebv_pdf == 'poisson':
        X_res['ebv_apply'] = np.random.poisson(lam=3, size=len(X_res))

    X_res['ebv_index'] = np.round(X_res['ebv_apply']*10,0)
    X_res['ebv_index'] = X_res['ebv_index'].astype(int)

    X_res['nH_apply'] = 2.21 * 3.1 * X_res['ebv_apply'] + np.random.normal(loc=0.0, scale=1.0, size=len(X_res))*0.09*np.sqrt(2) * 3.1 * X_res['ebv_apply']
    X_res.loc[X_res['nH_apply']<0, 'nH_apply'] = 0
    X_res.loc[X_res['nH_apply']>500, 'nH_apply'] = 500
    X_res['nH_index'] = np.round(X_res['nH_apply'],0)
    X_res['nH_index'] = X_res['nH_index'].astype(int)
       
    X_res['Gamma'] = X_res.apply(lambda r: np.random.randn()*r['e_powlaw_gamma_mean']*np.sqrt(2.)+ r['powlaw_gamma_mean'] if r['powlaw_gamma_mean']>-100 else Gammas[r['Class']],axis=1)
    X_res.loc[X_res['Gamma']<-3.8, 'Gamma'] = -3.8
    X_res.loc[X_res['Gamma']>14.4, 'Gamma'] = 14.4
    X_res['Gamma'] = np.round((X_res['Gamma']),1)
    X_res.loc[X_res['Gamma']==-0.0, 'Gamma'] = 0.0
    
    #data_red = apply_red2mw(X_res.copy(), 0,  red_class='all', self_unred=True)

    group_extinction = hpy.get('extinction')
    
    for band in MW_features:

        X_res[band] = X_res.apply(lambda r: r[band] - group_extinction[band][r.ebv_index],axis=1) 
    
    #df_oversample = apply_red2csc(data_red, 0, tbabs_ene, tbabs_cross,  red_class='all', self_unred=True)
    
    group_abs = hpy.get('absorption')

    bands = ['Fcsc_s', 'Fcsc_m', 'Fcsc_h', 'Fcsc_b']
    for band in bands:
        X_res[band] = X_res.apply(lambda r: r[band] * group_abs[f'{band}/{str(r.Gamma)}'][r.nH_index],axis=1) 
        
        #print(X_res[[band,band+'_red']])
    
    df_oversample = pd.concat([TD, X_res], ignore_index=True, sort=False)
       
    return df_oversample

def test_reddening_grid_hpy(df, hpy,random_state=None):

    if random_state is None:
        np.random.seed(randint(1,999999999)) 
    else:
        np.random.seed(random_state)
        
    dfs = pd.concat([df]*101, ignore_index=True)
    #print(dfs)
    dfs['ebv_apply'] = dfs.index
    dfs['ebv_apply'] = pd.to_numeric(dfs['ebv_apply'])*0.1
    dfs['ebv_index'] = np.round(dfs['ebv_apply']*10,0)
    dfs['ebv_index'] = dfs['ebv_index'].astype(int)
    #print(dfs['ebv_apply'])
    dfs['nH_apply'] = 2.21 * 3.1 * dfs['ebv_apply'] + np.random.normal(loc=0.0, scale=1.0, size=len(dfs))*0.09*np.sqrt(2) * 3.1 * dfs['ebv_apply']
    dfs.loc[dfs['nH_apply']<0, 'nH_apply'] = 0
    dfs.loc[dfs['nH_apply']>500, 'nH_apply'] = 500
    dfs['nH_index'] = np.round(dfs['nH_apply'],0)
    dfs['nH_index'] = dfs['nH_index'].astype(int)
    
     
    dfs['Gamma'] = dfs.apply(lambda r: np.random.randn()*r['e_powlaw_gamma_mean']*np.sqrt(2.)+ r['powlaw_gamma_mean'] if r['powlaw_gamma_mean']>-100 else Gammas[r['Class']],axis=1)
    dfs.loc[dfs['Gamma']<-3.8, 'Gamma'] = -3.8
    dfs.loc[dfs['Gamma']>14.4, 'Gamma'] = 14.4
    dfs['Gamma'] = np.round((dfs['Gamma']),1)
    dfs.loc[dfs['Gamma']==-0.0, 'Gamma'] = 0.0
    
    group_extinction = hpy.get('extinction')

    for band in MW_features:

        dfs[band] = dfs.apply(lambda r: r[band] - group_extinction[band][r.ebv_index],axis=1) 

    #data_red = apply_red2mw(dfs.copy(), 0,  red_class='all', self_unred=True)

    group_abs = hpy.get('absorption')

    bands = ['Fcsc_s', 'Fcsc_m', 'Fcsc_h', 'Fcsc_b']
    for band in bands:

        dfs[band] = dfs.apply(lambda r: r[band] * group_abs[f'{band}/{str(r.Gamma)}'][r.nH_index],axis=1) 
        
    #dfs = apply_red2csc(data_red, 0, tbabs_ene, tbabs_cross,  red_class='all', self_unred=True)
    
    return dfs
   

def physical_oversample_csv(TD, df_reds, random_state=None,ebv_pdf='uniform'):
    
    if random_state is None:
        np.random.seed(randint(1,999999999)) 
    else:
        np.random.seed(random_state)

    TD_X, TD_y = TD, TD['Class']
    ros = RandomOverSampler(random_state=random_state)
    X_res, y_res = ros.fit_resample(TD_X, TD_y)

    X_res = X_res[X_res.duplicated(subset=['name'])].reset_index(drop=True)

    if ebv_pdf == 'uniform':
        X_res['ebv_apply'] = np.random.uniform(low=0.0, high=ebv_uniform_highest, size=len(X_res))
    elif ebv_pdf == 'poisson':
        X_res['ebv_apply'] = np.random.poisson(lam=3, size=len(X_res))
    elif ebv_pdf == 'gamma':
        X_res['ebv_apply'] = np.random.gamma(0.5, scale=1.5, size=len(X_res))
    
    X_res.loc[X_res['ebv_apply']>50., 'ebv_apply'] = 50.
    X_res['ebv_index'] = np.round(X_res['ebv_apply']*10,0)
    X_res['ebv_index'] = X_res['ebv_index'].astype(int)

    X_res['nH_apply'] = 2.21 * 3.1 * X_res['ebv_apply'] + np.random.normal(loc=0.0, scale=1.0, size=len(X_res))*0.09*np.sqrt(2) * 3.1 * X_res['ebv_apply']
    X_res.loc[X_res['nH_apply']<0, 'nH_apply'] = 0
    X_res.loc[X_res['nH_apply']>500, 'nH_apply'] = 500
    X_res['nH_index'] = np.round(X_res['nH_apply'],0)
    X_res['nH_index'] = X_res['nH_index'].astype(int)
    #print(X_res[['ebv_apply','ebv_index']])

    X_res = X_res.set_index(['Class']) 
    X_res['Gamma'] = np.nan
    X_res.update(df_Gamma)
    X_res = X_res.reset_index()
    mask = X_res['powlaw_gamma_mean']>-100
    X_res.loc[mask, 'Gamma'] = np.random.randn(mask.size)*X_res['e_powlaw_gamma_mean']*np.sqrt(2.)+ X_res['powlaw_gamma_mean']


    #X_res['Gamma'] = X_res.apply(lambda r: np.random.randn()*r['e_powlaw_gamma_mean']*np.sqrt(2.)+ r['powlaw_gamma_mean'] if r['powlaw_gamma_mean']>-100 else Gammas[r['Class']],axis=1)
    X_res.loc[X_res['Gamma']<-3.8, 'Gamma'] = -3.8
    X_res.loc[X_res['Gamma']>14.4, 'Gamma'] = 14.4
    X_res['Gamma'] = np.round((X_res['Gamma']),1)
    X_res.loc[X_res['Gamma']==-0.0, 'Gamma'] = 0.0
    X_res['Gamma_col'] = X_res.apply(lambda r: str(np.around(r.Gamma, 1)),axis=1)

    df_extinction = df_reds[0] 


    X_res = X_res.set_index(['ebv_index'])

    for band in MW_features:
        #print(band)
        
        X_res[band+'_cor'] = np.nan
        X_res.update(df_extinction)
        
        X_res[band] = X_res[band] - X_res[band+'_cor']
    X_res = X_res.reset_index()


    bands = ['Fcsc_s', 'Fcsc_m', 'Fcsc_h', 'Fcsc_b']
    X_res = X_res.set_index(['nH_index','Gamma_col'])

    for band, df_abs_band in zip(bands, df_reds[1]):
        #print(band)
        
        X_res[band+'_cor'] = np.nan
        
        X_res.update(df_abs_band)
        
        X_res[band] = X_res[band]*X_res[band+'_cor']

    X_res = X_res.reset_index()
    
    df_oversample = pd.concat([TD, X_res], ignore_index=True, sort=False)

    return df_oversample

def test_reddening_grid_csv(df, df_reds,random_state=None):

    if random_state is None:
        np.random.seed(randint(1,999999999)) 
    else:
        np.random.seed(random_state)
        
    dfs = pd.concat([df]*int(ebv_uniform_highest+1), ignore_index=True)
    #print(dfs)
    dfs['ebv_apply'] = dfs.index
    dfs['ebv_apply'] = pd.to_numeric(dfs['ebv_apply'])*1.
    dfs['ebv_index'] = np.round(dfs['ebv_apply']*10,0)
    dfs['ebv_index'] = dfs['ebv_index'].astype(int)
    #print(dfs['ebv_apply'])
    dfs['nH_apply'] = 2.21 * 3.1 * dfs['ebv_apply'] + np.random.normal(loc=0.0, scale=1.0, size=len(dfs))*0.09*np.sqrt(2) * 3.1 * dfs['ebv_apply']
    dfs.loc[dfs['nH_apply']<0, 'nH_apply'] = 0
    dfs.loc[dfs['nH_apply']>500, 'nH_apply'] = 500
    dfs['nH_index'] = np.round(dfs['nH_apply'],0)
    dfs['nH_index'] = dfs['nH_index'].astype(int)
    
    #dfs['Gamma'] = dfs.apply(lambda r: np.random.randn()*r['e_powlaw_gamma_mean']*np.sqrt(2.)+ r['powlaw_gamma_mean'] if r['powlaw_gamma_mean']>-100 else Gammas[r['Class']],axis=1)
    dfs = dfs.set_index(['Class']) 
    dfs['Gamma'] = np.nan
    dfs.update(df_Gamma)
    dfs = dfs.reset_index()
    mask = dfs['powlaw_gamma_mean']>-100
    dfs.loc[mask, 'Gamma'] = np.random.randn(mask.size)*dfs['e_powlaw_gamma_mean']*np.sqrt(2.)+ dfs['powlaw_gamma_mean']

    
    dfs.loc[dfs['Gamma']<-3.8, 'Gamma'] = -3.8
    dfs.loc[dfs['Gamma']>14.4, 'Gamma'] = 14.4
    dfs['Gamma'] = np.round((dfs['Gamma']),1)
    dfs.loc[dfs['Gamma']==-0.0, 'Gamma'] = 0.0
    dfs['Gamma_col'] = dfs.apply(lambda r: str(np.around(r.Gamma, 1)),axis=1)
    
    df_extinction = df_reds[0] #pd.read_csv(f'{csv_dir}/extinction_MWbands.csv',index_col='ebv_index')
    
    dfs = dfs.set_index(['ebv_index'])
    

    for band in MW_features:
        
        dfs[band+'_cor'] = np.nan
        dfs.update(df_extinction)
        
        dfs[band] = dfs[band] - dfs[band+'_cor']
        #X_res[band] = X_res.apply(lambda r: r[band] - df_extinction.loc[r.ebv_index, band],axis=1) 
    dfs = dfs.reset_index()
    
    
    bands = ['Fcsc_s', 'Fcsc_m', 'Fcsc_h', 'Fcsc_b']
    dfs = dfs.set_index(['nH_index','Gamma_col'])
    #print(X_res)
    for band, df_abs_band in zip(bands, df_reds[1]):
        #print(band)

        #df_abs_band = pd.read_csv(f'{csv_dir}/abs_{band}.csv',index_col='nH')
        #df_abs_band = df_abs_band.stack()
        #df_abs_band= df_abs_band.reset_index().rename(columns={'nH':'nH_index','level_1':'Gamma_col',0:band+'_cor'})
        #df_abs_band = df_abs_band.set_index(['nH_index','Gamma_col'])
        dfs[band+'_cor'] = np.nan
        
        dfs.update(df_abs_band)
        
        #X_res[band] = X_res.apply(lambda r: r[band] * df_abs_band.loc[r.nH_index, r.Gamma_col],axis=1) 
        dfs[band] = dfs[band]*dfs[band+'_cor']

    dfs = dfs.reset_index()
    

    return dfs
   
    
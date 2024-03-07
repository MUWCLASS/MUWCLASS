import numpy as np
import pandas as pd
from collections import Counter
from astropy.io.fits import getdata
from astropy import units as u
from astropy.coordinates import SkyCoord, Angle
from astroquery.vizier import Vizier
from astropy.table import Table
from astroquery.xmatch import XMatch
from astroquery.simbad import Simbad
import time
from astropy.time import Time
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, draw, show
import sys  
import os
from os import path
from io import StringIO
import multiprocessing as mp
from pathlib import Path
import glob
import requests

gaia_search_radius = 10 # arcsec
nway_dir = '/Users/huiyang/Softwarses/nway-master/'

def CSCviewsearch(field_name, ra, dec, radius, query_dir, template_dir='./data',csc_version='2.0', engine='curl',adql_version='csc_query_cnt_template',suffix=''):
    # csc_version = '2.0' or 'current' for 2.1
    # engine = 'curl' or 'wget'
    ra_low  = ra - radius/(60.*np.cos(dec*np.pi/180.))
    ra_upp  = ra + radius/(60.*np.cos(dec*np.pi/180.))
    dec_low = dec - radius/60
    dec_upp = dec + radius/60
    rad_cone = radius
    
    f = open(f'{template_dir}/template/{adql_version}.adql', "r")
    adql = f.readline()

    ra_temp = '266.599396'
    dec_temp = '-28.87594'  
    ra_low_temp = '266.5898794490786'
    ra_upp_temp = '266.60891255092145'
    dec_low_temp = '-28.884273333333333'
    dec_upp_temp = '-28.867606666666667'
    rad_cone_temp = '0.543215'
    #'''
    for [str1, str2] in [[rad_cone, rad_cone_temp], [ra, ra_temp], [dec, dec_temp], [ra_low, ra_low_temp], [ra_upp, ra_upp_temp], [dec_low, dec_low_temp], [dec_upp, dec_upp_temp]]:
        adql = adql.replace(str2, str(str1))

    text_file = open(f'{query_dir}/{field_name}/{field_name}{suffix}_{engine}.adql', "w")
    if engine == 'wget':
        adql = 'http://cda.cfa.harvard.edu/csccli/getProperties?query='+adql
    text_file.write(adql)
    text_file.close()
    
    if engine == 'curl':

        os.system(f"curl -o {query_dir}/{field_name}/{field_name}{suffix}_{engine}.txt \
                --form version={csc_version}  \
                --form query=@{query_dir}/{field_name}/{field_name}{suffix}_{engine}.adql \
                http://cda.cfa.harvard.edu/csccli/getProperties")
    elif engine == 'wget':

        # if operatin system is linux
        if os.name == 'posix':
            os.system(f"wget -O {query_dir}/{field_name}/{field_name}{suffix}_{engine}.txt -i \
                      {query_dir}/{field_name}/{field_name}{suffix}_{engine}.adql")
        # elif os.name == 'nt':
        #     # read uri from the first line of the file, strip the newline character
        #     uri = open(query_dir+'/'+field_name+"_wget.adql", "r").readline().strip()
        #     # use requests to get the data
        #     r = requests.get(uri)
        #     # write the data to the file
        #     open(query_dir+'/'+field_name+".txt", "w").write(r.text)



    
    #df = pd.read_csv(f'{data_dir}/{field_name}_curl.txt', comment='#', sep='\t', na_values=' '*9)

    return None

vizier_cols_dict = {
    'catalogs': {'CSC':'IX/57/csc2master','gaia':'I/355/gaiadr3','gaiadist':'I/352/gedr3dis', 'tmass':'II/246/out','allwise':'II/328/allwise','catwise':'II/365/catwise'},\
    #'search_radius':{'CSC':0.05/60,'gaia':1., 'tmass':1.,'allwise':1.,'catwise':1.},\
    'search_radius':{'CSC':0.05/60,'gaia':0.5, 'tmass':0.5,'allwise':0.5,'catwise':0.5},\
    #'search_radius':{'CSC':0.05/60,'gaia':3, 'tmass':3,'allwise':3,'catwise':3},\
    #'search_radius':{'CSC':0.01/60,'gaia':5., 'tmass':5.,'allwise':5.,'catwise':5.},
    #'search_radius':{'CSC':0.01/60,'gaia':10./60, 'tmass':10./60,'allwise':10./60,'catwise':10./60},
    'IX/57/csc2master':['_r','_RAJ2000','_DEJ2000','2CXO','RAICRS','DEICRS','r0','r1','PA','fe','fc',\
                       #'fp','fv','fst','fs','fa','fi','fr','fm','ff','fVi',\
                       #'F90b','F9h','F90m','F90s','F90u']
                       ],          
    'I/355/gaiadr3':['_r','_RAJ2000','_DEJ2000','Source','RA_ICRS','DE_ICRS','e_RA_ICRS','e_DE_ICRS','RADEcor',\
              'Plx','e_Plx','PM','pmRA','e_pmRA','pmDE','e_pmDE','pmRApmDEcor','epsi','amax','RUWE','Gmag','BPmag','RPmag','e_Gmag','e_BPmag','e_RPmag',  \
              #'BP-RP','BP-G','G-RP',\
              'AllWISE','dAllWISE','f_AllWISE','AllWISEoid','2MASS','d2MASS','f_2MASS','2MASScoid'
              ],
    'II/246/out': ['_r','_RAJ2000','_DEJ2000','Date','JD','2MASS','RAJ2000','DEJ2000','errMaj','errMin','errPA',\
              'Jmag','Hmag','Kmag','e_Jmag','e_Hmag','e_Kmag'
              ],
    'II/328/allwise': ['_r','_RAJ2000','_DEJ2000','AllWISE','ID','RAJ2000','DEJ2000','eeMaj','eeMin','eePA',\
            'RA_pm','e_RA_pm','DE_pm','e_DE_pm','cosig_pm','pmRA','e_pmRA','pmDE','e_pmDE',\
              'W1mag','W2mag','W3mag','W4mag','e_W1mag','e_W2mag','e_W3mag','e_W4mag',\
              '2Mkey','d2M','2M','Jmag','e_Jmag','Hmag','e_Hmag','Kmag','e_Kmag'
              ],
    'II/365/catwise': ['_r','_RAJ2000','_DEJ2000','Name','objID','RA_ICRS','e_RA_ICRS','DE_ICRS','e_DE_ICRS','ePos',\
            'MJD','RAPMdeg','e_RAPMdeg','DEPMdeg','e_DEPMdeg','ePosPM','pmRA','e_pmRA','pmDE','e_pmDE','plx1','e_plx1',\
              'W1mproPM','W2mproPM','e_W1mproPM','e_W2mproPM',
              ],
}
def newcsc_prepare(df_q,X_name,name_col='name',ra_col='ra', dec_col='dec',r0_col='r0',r1_col='r1',PA_col='PA',data_dir='data',sigma=2):

    if sigma == 2: # 95%
        sigma_factor = 1.#np.sqrt(np.log(20)*2)
    elif sigma == 1: # ≈ 39.347% in 2-D   
        sigma_factor = 1./np.sqrt(np.log(20)*2)
    elif sigma == 3: # ≈ 39.347% in 2-D   
        sigma_factor = np.sqrt(np.log(1./0.0027)*2)/np.sqrt(np.log(20)*2)
   
    df_q['_2CXO'] = df_q[name_col]
    df_q['RA']  = df_q[ra_col]
    df_q['DEC'] = df_q[dec_col]
    
    df_q['ID'] = df_q.index + 1
    df_q['err_r0'] = df_q[r0_col]*sigma_factor
    df_q['err_r1'] = df_q[r1_col]*sigma_factor
    df_q['PA'] = df_q[PA_col]
    
    new_t = Table.from_pandas(df_q[['ID','RA','DEC','err_r0','err_r1','PA','_2CXO']]) # r0 is 95%, should be consistent with other PUs, 

    os.makedirs(data_dir, exist_ok=True)
    new_t.write(f'{data_dir}/{X_name}_CSC.fits', overwrite=True)

    area = 550./317000
    
    os.system(f'python {nway_dir}nway-write-header.py {data_dir}/{X_name}_CSC.fits CSC {area}')
    
    return None


def nway_mw_prepare_hierarchical_v3(ra_x, dec_x, X_name, mode='individual', search_radius=12, ref_mjd=np.array([57388.]),pmra=0., pmde=0.,e_pmra=0., e_pmde=0., e_pmrade=0., catalog='gaia',data_dir='data',plot_density_curve=False,sigma=2,r0_in=3.):
    
    #catalog = 'I/355/gaiadr3'
    '''
    mjd_difs = {'gaia':X_mjd-57388.,'gaiadist':X_mjd-57388.,'2mass':max(abs(X_mjd-50600),(X_mjd-51955)),'catwise':X_mjd-57170.0,
              'unwise':max(abs(X_mjd-55203.),abs(X_mjd-55593.),abs(X_mjd-56627),abs(X_mjd-58088)),
              'allwise':max(abs(X_mjd-55203.),abs(X_mjd-55414.),abs(X_mjd-55468),abs(X_mjd-55593)),
              'vphas':max(abs(X_mjd-55923),abs(X_mjd-56536))
            }
    '''    

    #viz = Vizier(row_limit=-1,  timeout=5000, columns=vizier_cols_dict[vizier_cols_dict['catalogs'][catalog]],catalog=vizier_cols_dict['catalogs'][catalog])
    viz = Vizier(row_limit=-1,  timeout=5000, columns=["**","_r"],catalog=vizier_cols_dict['catalogs'][catalog])

    if mode == 'individual':
        search_radius = vizier_cols_dict['search_radius'][catalog] # arcmin, we plot the density vs radius and see it starts to converge at around 4'

    elif mode == 'cluster':
        search_radius = search_radius

    #print(f'radec:{ra_x} {dec_x},radius:{search_radius*60}arcsec')
    query = viz.query_region(SkyCoord(ra=ra_x, dec=dec_x,
                        unit=(u.deg, u.deg),frame='icrs'),
                        radius=search_radius*u.arcmin)
                        #,column_filters={'Gmag': '<19'}
    #print(catalog, ra_x, dec_x, search_radius*60)
    try:
        query_res = query[0]
    except:
        print("No source matched")
        return pd.DataFrame()
    

    df_q = query_res.to_pandas()
    df_q = df_q.sort_values(by='_r').reset_index(drop=True)
    df_q = df_q[~np.isnan(df_q['_r'])].reset_index(drop=True)
    #print(df_q)
    #print(df_q.dtypes)
    #print(df_q.columns)
    #num = len(df_q)

    #'''
    #print(df_q.columns)
    if sigma == 2: # 95%
        sigma_factor = np.sqrt(np.log(20)*2)
    elif sigma == 1: # ≈ 39.347% in 2-D   
        sigma_factor = 1
    elif sigma == 3: # ≈ 39.347% in 2-D   
        sigma_factor = np.sqrt(np.log(1./0.0027)*2)
    
        
    if catalog == 'CSC':
        
        df_q['RA']  = df_q['RAICRS']
        df_q['DEC'] = df_q['DEICRS']
         
        df_q['ID'] = df_q.index + 1
        df_q['err_r0'] = df_q['r0']#*sigma_factor/np.sqrt(np.log(20)*2)
        df_q['err_r1'] = df_q['r1']#*sigma_factor/np.sqrt(np.log(20)*2)

        new_t = Table.from_pandas(df_q[['ID','RA','DEC','err_r0','err_r1','PA','_2CXO','_r','fe','fc']]) # r0 is 95%, should be consistent with other PUs, 

    
    elif catalog == 'gaia':
        
        gaia_ref_mjd = 57388.
        mean_mjd = ref_mjd.mean()
        delta_yr = (mean_mjd - gaia_ref_mjd)/365.
        delta_mean_mjd = max(abs((ref_mjd - mean_mjd)/365.))
        delta_max_mjd = max(abs((ref_mjd - gaia_ref_mjd)/365.))
        
        df_q['RA']  = df_q['RA_ICRS']
        df_q['DEC'] = df_q['DE_ICRS']
        
        #df_q['RA']  = df_q.apply(lambda row:row.RA_ICRS+delta_yr*row.pmRA/(np.cos(row.DE_ICRS*np.pi/180.)*3.6e6),axis=1)
        #df_q['DEC'] = df_q.apply(lambda row:row.DE_ICRS+delta_yr*row.pmDE/3.6e6,axis=1)
        
        df_q['RA']  = df_q['RA_ICRS']+delta_yr*df_q['pmRA']/(np.cos(df_q['DE_ICRS']*np.pi/180.)*3.6e6)
        df_q['DEC'] = df_q['DE_ICRS']+delta_yr*df_q['pmDE']/3.6e6
        
        df_q.loc[df_q['RA'].isnull(),'RA']   = df_q.loc[df_q['RA'].isnull(),'RA_ICRS']
        df_q.loc[df_q['DEC'].isnull(),'DEC'] = df_q.loc[df_q['DEC'].isnull(),'DE_ICRS']
        
       
          
        # https://www.aanda.org/articles/aa/pdf/2018/08/aa32727-18.pdf eq. B.1-B.3
        
        df_q['C00'] = df_q['e_RA_ICRS'] * df_q['e_RA_ICRS']
        df_q['C01'] = df_q['e_RA_ICRS'] * df_q['e_DE_ICRS'] * df_q['RADEcor']
        df_q['C11'] = df_q['e_DE_ICRS'] * df_q['e_DE_ICRS']
        df_q['C33'] = df_q['e_pmRA']    * df_q['e_pmRA']
        df_q['C34'] = df_q['e_pmRA']    * df_q['e_pmDE'] * df_q['pmRApmDEcor']
        df_q['C44'] = df_q['e_pmDE']    * df_q['e_pmDE']
        df_q['sigma_pos'] = np.sqrt(0.5*(df_q.C00+df_q.C11) + 0.5*np.sqrt((df_q.C11-df_q.C00)**2+4*df_q.C01**2)) 
        df_q['e_Pos'] = df_q['sigma_pos'].fillna(0.)/1e3
        df_q['sigma_pm']  = np.sqrt(0.5*(df_q.C33+df_q.C44) + 0.5*np.sqrt((df_q.C44-df_q.C33)**2+4*df_q.C34**2))
        df_q['e_PM'] = df_q['sigma_pm'].fillna(0.)/1e3
        df_q['PM'] = df_q['PM'].fillna(0.)/1e3
        df_q['epsi'] = df_q['epsi'].fillna(0.)/1e3
        df_q['Plx'] = df_q['Plx'].fillna(0.)/1e3
        df_q['e_Plx'] = df_q['e_Plx'].fillna(0.)/1e3

        df_q['PU'] = sigma_factor * np.sqrt(df_q['e_Pos']**2+df_q['Plx']**2+df_q['e_Plx']**2 + (df_q['PM']*delta_mean_mjd)**2+(df_q['e_PM']*delta_max_mjd)**2+df_q['epsi']**2)
        #print(Table.from_pandas(df_q).columns)
        df_q['n_srcs'] = len(df_q)
        
        c = SkyCoord(ra=ra_x*u.degree, dec=dec_x*u.degree)
        df_q['sep'] = SkyCoord(ra=df_q['RA']*u.degree, dec=df_q['DEC']*u.degree).separation(c).arcsec
        
        new_t = Table.from_pandas(df_q[['RA','DEC','PU','sep','n_srcs','Source','RA_ICRS','DE_ICRS','_r','e_Pos','Plx','e_Plx','PM','pmRA','pmDE','e_PM','epsi','Gmag','BPmag','RPmag','e_Gmag','e_BPmag','e_RPmag','AllWISE','dAllWISE','f_AllWISE','AllWISEoid','_2MASS','d2MASS','f_2MASS','_2MASScoid']])
        #print(df_q[~df_q['RA'].isnull()][['RA','DEC','PU','Source','_r','e_Pos','Plx','e_Plx','PM','e_PM','epsi']])
        #df_q[['RA','DEC','PU','sep','n_srcs','Source','RA_ICRS','DE_ICRS','_r','e_Pos','Plx','e_Plx','PM','pmRA','pmDE','e_PM','epsi','Gmag','BPmag','RPmag','e_Gmag','e_BPmag','e_RPmag','AllWISE','dAllWISE','f_AllWISE','AllWISEoid','_2MASS','d2MASS','f_2MASS','_2MASScoid']].to_csv(f'{data_dir}/{X_name}_gaia.csv',index=False)
        #print(new_t)
    elif catalog == 'tmass':

        df_q['MJD'] = df_q.apply(lambda r: Time(r.Date, format='isot').to_value('mjd', 'long') if pd.notnull(r.Date) else r, axis=1)
        
        df_q['RA'] = df_q['RAJ2000']
        df_q['DEC'] = df_q['DEJ2000']
        df_q = df_q[~df_q['RA'].isnull()].reset_index(drop=True)
            
        df_q['err0'] = df_q['errMaj'] * sigma_factor
        df_q['err1'] = df_q['errMin'] * sigma_factor
        df_q = df_q.astype({'MJD':'int'})
        #print(df_q[['err0','err1']].describe())
        df_q['n_srcs'] = len(df_q)
        
        c = SkyCoord(ra=ra_x*u.degree, dec=dec_x*u.degree)
        df_q['sep'] = SkyCoord(ra=df_q['RA']*u.degree, dec=df_q['DEC']*u.degree).separation(c).arcsec

        new_t = Table.from_pandas(df_q[['RA','DEC','err0','err1','errPA','sep','n_srcs','MJD','_2MASS','_r','Jmag','Hmag','Kmag','e_Jmag','e_Hmag','e_Kmag']])
        #df_q[['RA','DEC','MJD','errMaj','errMin','errPA','_2MASS','_r']].to_csv(f'{data_dir}/{X_name}_2mass.csv',index=False)
    
    elif catalog == 'allwise':
        
        df_q['RA'] = df_q['RA_pm']
        df_q['DEC'] = df_q['DE_pm']
        df_q = df_q[~df_q['RA'].isnull()].reset_index(drop=True)
        
        # we don't use the proper motion measurements from allwise as they as affected by parallax and not reliable
        df_q['err0'] = df_q['eeMaj'] * sigma_factor
        df_q['err1'] = df_q['eeMin'] * sigma_factor

        df_q = df_q.rename(columns={'eePA':'errPA'})    
        #print(df_q[['eeMaj','eeMin','errPA','e_RA_pm','e_DE_pm']].describe())
        #print(Table.from_pandas(df_q).columns)
        df_q['n_srcs'] = len(df_q)
        
        c = SkyCoord(ra=ra_x*u.degree, dec=dec_x*u.degree)
        df_q['sep'] = SkyCoord(ra=df_q['RA']*u.degree, dec=df_q['DEC']*u.degree).separation(c).arcsec

        new_t = Table.from_pandas(df_q[['RA','DEC','err0','err1','errPA','RAJ2000','DEJ2000','e_RA_pm','e_DE_pm','sep','n_srcs','AllWISE','_r','W1mag','W2mag','W3mag','W4mag','e_W1mag','e_W2mag','e_W3mag','e_W4mag','_2Mkey','d2M','_2M','Jmag','e_Jmag','Hmag','e_Hmag','Kmag','e_Kmag']])
    
    elif catalog == 'catwise':
        
            
        df_q = df_q.rename(columns={'_tab1_20':'MJD'})
        mean_mjd = ref_mjd.mean()
        delta_mean_mjd = max(abs((ref_mjd - mean_mjd)/365.))
        df_q['delta_max_mjd'] = 0.
        df_q['delta_max_mjd'] = df_q.apply(lambda r: max(abs((ref_mjd - r.MJD)/365.)), axis=1)
        df_q.loc[df_q['delta_max_mjd'].isnull(),'delta_max_mjd'] = 0.

        
        df_q['RA']  = df_q['RA_ICRS']
        df_q['DEC'] = df_q['DE_ICRS']
        df_q = df_q[~df_q['RA'].isnull()].reset_index(drop=True)
        
        df_q['pmRA_ori'] = df_q['pmRA']
        df_q['pmDE_ori'] = df_q['pmDE']
        
        
        # https://www.aanda.org/articles/aa/pdf/2018/08/aa32727-18.pdf eq. B.1-B.3
        
        df_q['C00'] = df_q['e_RA_ICRS'] * df_q['e_RA_ICRS']
        df_q['C01'] = df_q['ePos'] #df_q['e_RA_ICRS'] * df_q['e_DE_ICRS'] * df_q['ePos']
        df_q['C11'] = df_q['e_DE_ICRS'] * df_q['e_DE_ICRS']
        df_q['C33'] = df_q['e_pmRA']    * df_q['e_pmRA']
        df_q['C44'] = df_q['e_pmDE']    * df_q['e_pmDE']
        df_q['sigma_pos'] = np.sqrt(0.5*(df_q.C00+df_q.C11) + 0.5*np.sqrt((df_q.C11-df_q.C00)**2+4*df_q.C01**2)) 
        df_q['e_Pos'] = df_q['sigma_pos'].fillna(0.)
        df_q['sigma_pm']  = np.sqrt(0.5*(df_q.C33+df_q.C44) + 0.5*np.sqrt((df_q.C44-df_q.C33)**2))
        df_q['e_PM_ori'] = df_q['sigma_pm'].fillna(0.)
        df_q['PM_ori'] = np.sqrt(df_q['pmRA'].fillna(0.)**2+df_q['pmDE'].fillna(0.)**2)
        df_q['e_PM'] = df_q['e_PM_ori']
        
        df_q.loc[(df_q['PM_ori']<5*df_q['e_PM_ori']) | (df_q['chi2pm']>1.5), 'pmRA'] = 0.
        df_q.loc[(df_q['PM_ori']<5*df_q['e_PM_ori']) | (df_q['chi2pm']>1.5), 'pmDE'] = 0.
        df_q.loc[(df_q['PM_ori']<5*df_q['e_PM_ori']) | (df_q['chi2pm']>1.5), 'e_PM'] = 0.
        
        df_q['PM'] = np.sqrt(df_q['pmRA'].fillna(0.)**2+df_q['pmDE'].fillna(0.)**2)
        
        df_q['RA']  = df_q['RA_ICRS']+(mean_mjd - df_q['MJD'])/365.*df_q['pmRA']/(np.cos(df_q['DE_ICRS']*np.pi/180.)*3.6e3)
        df_q['DEC'] = df_q['DE_ICRS']+(mean_mjd - df_q['MJD'])/365.*df_q['pmDE']/3.6e3
        
        df_q.loc[df_q['RA'].isnull(),'RA']   = df_q.loc[df_q['RA'].isnull(),'RA_ICRS']
        df_q.loc[df_q['DEC'].isnull(),'DEC'] = df_q.loc[df_q['DEC'].isnull(),'DE_ICRS']


        
        df_q['Plx'] = df_q['plx1'].fillna(0.)
        df_q['e_Plx'] = df_q['e_plx1'].fillna(0.)

        #df_q['PU'] = sigma_factor * np.sqrt(df_q['e_Pos']**2+df_q['Plx']**2+df_q['e_Plx']**2 + (df_q['PM']*delta_mean_mjd)**2+(df_q['e_PM']*df_q['delta_max_mjd'])**2)
        df_q['PU'] = sigma_factor * np.sqrt(df_q['e_Pos']**2 + (df_q['PM']*delta_mean_mjd)**2+(df_q['e_PM']*df_q['delta_max_mjd'])**2)
        
        df_q['n_srcs'] = len(df_q)
        
        c = SkyCoord(ra=ra_x*u.degree, dec=dec_x*u.degree)
        df_q['sep'] = SkyCoord(ra=df_q['RA']*u.degree, dec=df_q['DEC']*u.degree).separation(c).arcsec

        new_t = Table.from_pandas(df_q[['RA','DEC','PU','sep','n_srcs','Name','MJD','RA_ICRS','DE_ICRS','_r','e_Pos','Plx','e_Plx','chi2pm','PM','PM_ori','pmRA_ori','pmDE_ori','pmRA','pmDE','e_PM','e_PM_ori','e_pmRA','e_pmDE','W1mproPM','W2mproPM','e_W1mproPM','e_W2mproPM']])
    
    
    #print(df_q['PU'].describe())
    #'''
    #new_t = new_t[~(new_t['RA'].isnull)].reset_index(drop=True)
    #df_q.to_csv(f'{data_dir}/{X_name}_{catalog}.csv',index=False)
    #print(df_q)

    #print(new_t)
    #print(f'{data_dir}/{X_name}_{catalog}.fits')
    try:
        # create folder
        os.makedirs(data_dir, exist_ok=True)
        new_t.write(f'{data_dir}/{X_name}_{catalog}.fits', overwrite=True)
    except:
        print(f'{data_dir}/{X_name}_{catalog}.fits can not be produced at the first time.')
        if (catalog == 'gaia'):
            df_q.to_csv(f'{data_dir}/{X_name}_{catalog}.csv',index=False)
            df_q_csv = pd.read_csv(f'{data_dir}/{X_name}_{catalog}.csv')
            new_t = Table.from_pandas(df_q_csv[['RA','DEC','PU','sep','n_srcs','Source','RA_ICRS','DE_ICRS','_r','e_Pos','Plx','e_Plx','PM','pmRA','pmDE','e_PM','epsi','Gmag','BPmag','RPmag','e_Gmag','e_BPmag','e_RPmag','AllWISE','dAllWISE','f_AllWISE','AllWISEoid','_2MASS','d2MASS','f_2MASS','_2MASScoid']])
            new_t.write(f'{data_dir}/{X_name}_{catalog}.fits', overwrite=True)
        elif (catalog == 'allwise'):
            df_q.to_csv(f'{data_dir}/{X_name}_{catalog}.csv',index=False)
            df_q_csv = pd.read_csv(f'{data_dir}/{X_name}_{catalog}.csv')
            new_t = Table.from_pandas(df_q_csv[['RA','DEC','err0','err1','errPA','RAJ2000','DEJ2000','e_RA_pm','e_DE_pm','sep','n_srcs','AllWISE','_r','W1mag','W2mag','W3mag','W4mag','e_W1mag','e_W2mag','e_W3mag','e_W4mag','_2Mkey','d2M','_2M','Jmag','e_Jmag','Hmag','e_Hmag','Kmag','e_Kmag']])
            new_t.write(f'{data_dir}/{X_name}_{catalog}.fits', overwrite=True)
    
    
    #print(f'{data_dir}/{X_name}_{catalog}.fits')
    #os.system('pwd')
    if plot_density_curve:
        df_q['n_match'] = df_q.index+1
        df_q['rho'] = df_q['n_match']/(np.pi*df_q['_r']**2)
        plt.plot(df_q['_r'], df_q['rho'])
        plt.yscale("log")
        #print(df_q[['n_match','_r','rho']])
        
    
    if catalog == 'CSC':
        area = 550./317000
    else:
        area = np.pi * (search_radius/60)**2
        
        # excluding the center radius of 3"
        #print(len(df_q),len(df_q[df_q['_r']>r0_in]))
        #if len(df_q[df_q['_r']>r0_in])>0:
            #area = np.pi * ((search_radius/60)**2 - (r0_in/3600)**2)*len(df_q)/(len(df_q[df_q['_r']>r0_in]))
        #else:
            #area = np.pi * (search_radius/60)**2
        #print(f'area:{area},area2:{area_old}')
    #area = 550./317000
    os.system(f'python {nway_dir}nway-write-header.py {data_dir}/{X_name}_{catalog}.fits {catalog} {area}')
    
    return df_q
         
vizier_cols_dict = {
    'catalogs': {'CSC':'IX/57/csc2master','gaia':'I/355/gaiadr3','gaiadist':'I/352/gedr3dis', 'tmass':'II/246/out','allwise':'II/328/allwise','catwise':'II/365/catwise'},\
}

gaia_ref_mjd = 57388.

gaia_cols = ['GAIA_RA_ICRS','GAIA_DE_ICRS','GAIA_pmRA','GAIA_pmDE','GAIA_e_Pos','GAIA_Plx','GAIA_e_Plx','GAIA_e_PM','GAIA_epsi']
gaia_cols2 = ['GAIA_DR3Name'] + gaia_cols 
catwise_cols  = ['CATWISE_RA_ICRS','CATWISE_DE_ICRS','CATWISE_MJD','CATWISE_pmRA','CATWISE_pmDE','CATWISE_e_Pos','CATWISE_e_PM']
catwise_cols2 = ['CATWISE_Name'] + catwise_cols 

# Hubble Source Catalog API functions
hscapiurl = "https://catalogs.mast.stsci.edu/api/v0.1/hsc"
def hsccone(ra, dec, radius, table="summary", release="v3", format="csv", magtype="magaper2",
            columns=None, baseurl=hscapiurl, verbose=False, **kw):
    """Do a cone search of the HSC catalog

    Parameters
    ----------
    ra (float): (degrees) J2000 Right Ascension
    dec (float): (degrees) J2000 Declination
    radius (float): (degrees) Search radius (<= 0.5 degrees)
    table (string): summary, detailed, propermotions, or sourcepositions
    release (string): v3 or v2
    magtype (string): magaper2 or magauto (only applies to summary table)
    format: csv, votable, json, table
    columns: list of column names to include (None means use defaults)
    baseurl: base URL for the request
    verbose: print info about request
    **kw: other parameters (e.g., 'numimages.gte':2)
    """

    data = kw.copy()
    data['ra'] = ra
    data['dec'] = dec
    data['radius'] = radius
    return hscsearch(table=table, release=release, format=format, magtype=magtype,
                     columns=columns, baseurl=baseurl, verbose=verbose, **data)


def hscsearch(table="summary", release="v3", magtype="magaper2", format="csv",
              columns=None, baseurl=hscapiurl, verbose=False, **kw):
    """Do a general search of the HSC catalog (possibly without ra/dec/radius)

    Parameters
    ----------
    table (string): summary, detailed, propermotions, or sourcepositions
    release (string): v3 or v2
    magtype (string): magaper2 or magauto (only applies to summary table)
    format: csv, votable, json, table
    columns: list of column names to include (None means use defaults)
    baseurl: base URL for the request
    verbose: print info about request
    **kw: other parameters (e.g., 'numimages.gte':2).  Note this is required!
    """

    data = kw.copy()
    if not data:
        raise ValueError("You must specify some parameters for search")
    if format not in ("csv", "votable", "json", 'table'):
        raise ValueError("Bad value for format")
    if format == "table":
        rformat = "csv"
    else:
        rformat = format
    url = f"{cat2url(table, release, magtype, baseurl=baseurl)}.{rformat}"
    if columns:
        # check that column values are legal
        # create a dictionary to speed this up
        dcols = {}
        for col in hscmetadata(table, release, magtype)['name']:
            dcols[col.lower()] = 1
        badcols = []
        for col in columns:
            if col.lower().strip() not in dcols:
                badcols.append(col)
        if badcols:
            raise ValueError(f"Some columns not found in table: {', '.join(badcols)}")
        # two different ways to specify a list of column values in the API
        # data['columns'] = columns
        data['columns'] = f"[{','.join(columns)}]"

    # either get or post works
    # r = requests.post(url, data=data)
    r = requests.get(url, params=data)

    if verbose:
        print(r.url)
    r.raise_for_status()
    if format == "json":
        return r.json()
    elif format == "table":
        # use pandas to work around bug in Windows for ascii.read
        return Table.from_pandas(pd.read_csv(StringIO(r.text)))
    else:
        return r.text


def hscmetadata(table="summary", release="v3", magtype="magaper2", baseurl=hscapiurl):
    """Return metadata for the specified catalog and table
    
    Parameters
    ----------
    table (string): summary, detailed, propermotions, or sourcepositions
    release (string): v3 or v2
    magtype (string): magaper2 or magauto (only applies to summary table)
    baseurl: base URL for the request
    
    Returns an astropy table with columns name, type, description
    """
    url = f"{cat2url(table, release, magtype, baseurl=baseurl)}/metadata"
    r = requests.get(url)
    r.raise_for_status()
    v = r.json()
    # convert to astropy table
    tab = Table(rows=[(x['name'], x['type'], x['description']) for x in v],
                names=('name', 'type', 'description'))
    return tab


def cat2url(table="summary", release="v3", magtype="magaper2", baseurl=hscapiurl):
    """Return URL for the specified catalog and table
    
    Parameters
    ----------
    table (string): summary, detailed, propermotions, or sourcepositions
    release (string): v3 or v2
    magtype (string): magaper2 or magauto (only applies to summary table)
    baseurl: base URL for the request
    
    Returns a string with the base URL for this request
    """
    checklegal(table, release, magtype)
    if table == "summary":
        url = f"{baseurl}/{release}/{table}/{magtype}"
    else:
        url = f"{baseurl}/{release}/{table}"
    return url


def checklegal(table, release, magtype):
    """Checks if this combination of table, release and magtype is acceptable
    
    Raises a ValueError exception if there is problem
    """
    
    releaselist = ("v2", "v3")
    if release not in releaselist:
        raise ValueError(f"Bad value for release (must be one of {', '.join(releaselist)})")
    if release == "v2":
        tablelist = ("summary", "detailed")
    else:
        tablelist = ("summary", "detailed", "propermotions", "sourcepositions")
    if table not in tablelist:
        raise ValueError(f"Bad value for table (for {release} must be one of {', '.join(tablelist)})")
    if table == "summary":
        magtypelist = ("magaper2", "magauto")
        if magtype not in magtypelist:
            raise ValueError(f"Bad value for magtype (must be one of {', '.join(magtypelist)})")

def hscsearch_subfield(ra, dec, field_size, subfield_size, table="summary", release="v3", magtype="magauto", verbose=False, **kw):
    """
    If over 1 million sources in field, divide field into multiple subfields and concat
    field_size: half of angular size of field in degrees
    subfield_size: angular size of subfield in degrees
    """

    # define subfield centers
    subfield_center_decs = dec + np.arange(-field_size, field_size, subfield_size)
    subfield_center_ras = ra + np.arange(-field_size/np.cos(np.deg2rad(dec)), field_size/np.cos(np.deg2rad(dec)), subfield_size/np.cos(np.deg2rad(dec)))

    dfs = []

    for i in range(len(subfield_center_ras)):
        for j in range(len(subfield_center_decs)):
            print(subfield_center_ras[i], subfield_center_decs[j])
            constraints = {'MatchDec.gt': subfield_center_decs[j] - subfield_size/2., 
                        'MatchDec.lt': subfield_center_decs[j] + subfield_size/2., 
                        'MatchRa.gt': subfield_center_ras[i] - subfield_size/np.cos(np.deg2rad(subfield_center_decs[j]))/2., 
                        'MatchRa.lt': subfield_center_ras[i] + subfield_size/np.cos(np.deg2rad(subfield_center_decs[j]))/2.}
            
            # plot constraints
            # fig = plt.figure(figsize=(10,10))
            # plt.scatter(ra, dec, s=1, color='k', label='Field Center')
            # plt.scatter(subfield_center_ras[i], subfield_center_decs[j], s=100, color='r', label='Subfield Center')
            # plt.scatter(constraints['MatchRa.gt'], constraints['MatchDec.gt'], s=10, color='b', label='Subfield Corners')
            # plt.scatter(constraints['MatchRa.lt'], constraints['MatchDec.lt'], s=10, color='b')
            # plt.scatter(constraints['MatchRa.gt'], constraints['MatchDec.lt'], s=10, color='b')
            # plt.scatter(constraints['MatchRa.lt'], constraints['MatchDec.gt'], s=10, color='b')
            # plt.xlabel('RA')
            # plt.ylabel('Dec')

            t0 = time.time()
            try:
                tab = hscsearch(table=table, release=release, magtype=magtype, verbose=True, format='table', **constraints)
            except Exception as e:
                print(f'Error: {e}')
                continue
            
            print("{:.1f} s: retrieved data and converted to {}-row astropy table".format(time.time()-t0, len(tab)))
            df = tab.to_pandas()
            dfs.append(df)

    # concatenate all subfields into one table
    df_hugs = pd.concat(dfs)

    return df_hugs


def nway_merged_mw_prepare(ra_x, dec_x, X_name, search_radius=0.5, ref_mjd=np.array([57388.]),data_dir='data',sigma=2, verbose=False, rerun=False):
#def nway_merged_mw_prepare(ra_x, dec_x, X_name, ref_mjd=np.array([57388.]),pmra=0., pmde=0.,e_pmra=0., e_pmde=0., e_pmrade=0., catalog='gaia',data_dir='data',plot_density_curve=False,sigma=2,r0_in=3., verbose=False):
    
    if sigma == 2: # 95%
        sigma_factor_X = np.sqrt(np.log(20)*2)
    elif sigma == 1: # ≈ 39.347% in 2-D  
        sigma_factor_X = 1
    elif sigma == 3: # ≈ 99.73% in 2-D   
        sigma_factor_X = np.sqrt(np.log(1./0.0027)*2)

    #sigma = 5. 
    sigma_percentage = 0.9999994267 # 5-sigma
    #sigma_percentage = 0.999937 # 4-sigma
    #sigma_percentage = 0.9972 # 3-sigma
    sigma_factor = np.sqrt(-2.*np.log(1.-sigma_percentage))
    
    df_list = []

    for cat in ['gaia','gaiadist','tmass','allwise','catwise']:
    # matching to gaia 

        if path.exists(f'{data_dir}/{X_name}_{cat}.csv') == False or rerun==True:
    
            viz = Vizier(row_limit=-1,  timeout=5000, columns=["**","_r"],catalog=vizier_cols_dict['catalogs'][cat])
                
            # search_radius = 0.5 # arcmin, we plot the density vs radius and see it starts to converge at around 4'
            # print(f'radec:{ra_x} {dec_x},radius:{search_radius*60}arcsec')

            # print(f'matching to {cat}..........')
            query = viz.query_region(SkyCoord(ra=ra_x, dec=dec_x,
                                unit=(u.deg, u.deg),frame='icrs'),
                                radius=search_radius*u.arcmin)
                                
            #print(catalog, ra_x, dec_x, search_radius*60)
            try:
                query_res = query[0]
                df_q = query_res.to_pandas()
                df_q = df_q.sort_values(by='_r').reset_index(drop=True)
                df_q = df_q[~np.isnan(df_q['_r'])].reset_index(drop=True)
                df_list.append(df_q)
                if verbose:
                    print(f'{cat} match {len(df_q)}')
                df_q.to_csv(f'{data_dir}/{X_name}_{cat}.csv', index=False)
            except:
                if verbose:
                    print(f"{cat} No source matched")
                df_list.append(pd.DataFrame())
        else:
            df_q = pd.read_csv(f'{data_dir}/{X_name}_{cat}.csv')
            df_list.append(df_q)
    
    df_g,df_gaiadist,df_t,df_a,df_c = df_list[0],df_list[1],df_list[2],df_list[3],df_list[4]
    #print(len(df_g), len(df_gaiadist))
    if len(df_gaiadist)>0:
        #df_gaiadist['DR3Name'] = 'Gaia DR3 '+ str(df_gaiadist['Source'])
        df_g = pd.merge(df_g,df_gaiadist[['Source','rgeo','b_rgeo','B_rgeo','rpgeo','b_rpgeo','B_rpgeo','Flag']], on='Source', how='outer')
    # print(len(df_g))
    #print(df_g.columns)
    
    if len(df_g)>0:
        df_g['C00'] = df_g['e_RA_ICRS'] * df_g['e_RA_ICRS']
        df_g['C01'] = df_g['e_RA_ICRS'] * df_g['e_DE_ICRS'] * df_g['RADEcor']
        df_g['C11'] = df_g['e_DE_ICRS'] * df_g['e_DE_ICRS']
        df_g['C33'] = df_g['e_pmRA']    * df_g['e_pmRA']
        df_g['C34'] = df_g['e_pmRA']    * df_g['e_pmDE'] * df_g['pmRApmDEcor']
        df_g['C44'] = df_g['e_pmDE']    * df_g['e_pmDE']
        df_g['sigma_pos'] = np.sqrt(0.5*(df_g.C00+df_g.C11) + 0.5*np.sqrt((df_g.C11-df_g.C00)**2+4*df_g.C01**2)) 
        df_g['e_Pos'] = df_g['sigma_pos'].fillna(0.)/1e3
        df_g['sigma_pm']  = np.sqrt(0.5*(df_g.C33+df_g.C44) + 0.5*np.sqrt((df_g.C44-df_g.C33)**2+4*df_g.C34**2))
        df_g['e_PM'] = df_g['sigma_pm'].fillna(0.)/1e3
        df_g['PM'] = df_g['PM'].fillna(0.)/1e3
        df_g['epsi'] = df_g['epsi'].fillna(0.)/1e3
        df_g['Plx'] = df_g['Plx'].fillna(0.)/1e3
        df_g['e_Plx'] = df_g['e_Plx'].fillna(0.)/1e3  

        df_g = df_g.add_prefix('GAIA_')
              
    if len(df_a)>0:
        df_a['RA'] = df_a['RA_pm']
        df_a['DEC'] = df_a['DE_pm']
        df_a = df_a[~df_a['RA'].isnull()].reset_index(drop=True)
        
        # we don't use the proper motion measurements from allwise as they as affected by parallax and not reliable
        df_a['err0'] = df_a['eeMaj'] # * sigma_factor
        df_a['err1'] = df_a['eeMin'] # * sigma_factor

        df_a = df_a.rename(columns={'eePA':'errPA'}) 
        df_a = df_a.add_prefix('ALLWISE_') 

    if len(df_t)>0:
        df_t['MJD'] = df_t.apply(lambda r: Time(r.Date, format='isot').to_value('mjd', 'long') if pd.notnull(r.Date) else r, axis=1)
        
        df_t['RA'] = df_t['RAJ2000']
        df_t['DEC'] = df_t['DEJ2000']
        df_t = df_t[~df_t['RA'].isnull()].reset_index(drop=True)
            
        df_t['err0'] = df_t['errMaj'] # * sigma_factor
        df_t['err1'] = df_t['errMin'] # * sigma_factor
        df_t = df_t.astype({'MJD':'int'})
        df_t = df_t.add_prefix('TMASS_')

    if len(df_c)>0:

        df_c = df_c.rename(columns={'_tab1_20':'MJD'})
        df_c = df_c[~df_c['RA_ICRS'].isnull()].reset_index(drop=True)
            
        df_c['pmRA_ori'] = df_c['pmRA']
        df_c['pmDE_ori'] = df_c['pmDE']
        
        # https://www.aanda.org/articles/aa/pdf/2018/08/aa32727-18.pdf eq. B.1-B.3
        
        df_c['C00'] = df_c['e_RA_ICRS'] * df_c['e_RA_ICRS']
        df_c['C01'] = df_c['ePos'] #df_c['e_RA_ICRS'] * df_c['e_DE_ICRS'] * df_c['ePos']
        df_c['C11'] = df_c['e_DE_ICRS'] * df_c['e_DE_ICRS']
        df_c['C33'] = df_c['e_pmRA']    * df_c['e_pmRA']
        df_c['C44'] = df_c['e_pmDE']    * df_c['e_pmDE']
        df_c['sigma_pos'] = np.sqrt(0.5*(df_c.C00+df_c.C11) + 0.5*np.sqrt((df_c.C11-df_c.C00)**2+4*df_c.C01**2)) 
        df_c['e_Pos'] = df_c['sigma_pos'].fillna(0.)
        df_c['sigma_pm']  = np.sqrt(0.5*(df_c.C33+df_c.C44) + 0.5*np.sqrt((df_c.C44-df_c.C33)**2))
        df_c['e_PM_ori'] = df_c['sigma_pm'].fillna(0.)
        df_c['PM_ori'] = np.sqrt(df_c['pmRA'].fillna(0.)**2+df_c['pmDE'].fillna(0.)**2)
        df_c['e_PM'] = df_c['e_PM_ori']
        
        df_c.loc[(df_c['PM_ori']<5*df_c['e_PM_ori']) | (df_c['chi2pm']>1.5), 'pmRA'] = 0.
        df_c.loc[(df_c['PM_ori']<5*df_c['e_PM_ori']) | (df_c['chi2pm']>1.5), 'pmDE'] = 0.
        df_c.loc[(df_c['PM_ori']<5*df_c['e_PM_ori']) | (df_c['chi2pm']>1.5), 'e_PM'] = 0.
        
        df_c['PM'] = np.sqrt(df_c['pmRA'].fillna(0.)**2+df_c['pmDE'].fillna(0.)**2)

        df_c['Plx'] = df_c['plx1'].fillna(0.)
        df_c['e_Plx'] = df_c['e_plx1'].fillna(0.)
        df_c = df_c.add_prefix('CATWISE_')

    def get_plausible_associations(df1, df2, df1_ra, df1_dec, df2_ra, df2_dec, association_radius=5):
        '''
        df1: first catalog
        df2: second catalog
        df1_ra: ra column name of df1
        df1_dec: dec column name of df1
        df2_ra: ra column name of df2
        df2_dec: dec column name of df2
        association_radius: radius within which two sources are potentially associated, pending proper motion corrections and position uncertainties
        '''
        coords1 = SkyCoord(df1[df1_ra], df1[df1_dec], unit=u.deg, frame='icrs')
        coords2 = SkyCoord(df2[df2_ra], df2[df2_dec], unit=u.deg, frame='icrs')

        idx2, idx1, d2d, d3d = coords1.search_around_sky(coords2, association_radius*u.arcsec)

        df = pd.merge(df1.iloc[idx1].reset_index(drop=True), df2.iloc[idx2].reset_index(drop=True), left_index=True, right_index=True, how='outer')

        return df
        
    # begin hierarchical crossmatching of gaia, allwise, tmass, catwise, without considering X-ray information
    # future recursive version: begin with list of non empty catalogs, and then recursively crossmatch with the next catalog in the list
    if len(df_g)>0:      
        if verbose:  
            print('len(df_g)>0')

        
        if len(df_c)>0:
            if verbose:  
                print('len(df_c)>0')
            
            # df_g_copy = pd.concat([df_g[gaia_cols2]]*len(df_c), ignore_index=True).sort_values(by=['GAIA_DR3Name']).reset_index(drop=True)
            # df_c_copy = pd.concat([df_c]*len(df_g), ignore_index=True)

            # df_gc = pd.concat([df_g_copy, df_c_copy],axis=1)

            # df_gc = pd.merge(df_g[gaia_cols2], df_c[catwise_cols2], left_index=True, right_index=True, how='cross')

            df_gc = get_plausible_associations(df_g[gaia_cols2], df_c[catwise_cols2], 'GAIA_RA_ICRS', 'GAIA_DE_ICRS', 'CATWISE_RA_ICRS', 'CATWISE_DE_ICRS', association_radius=5)
            
            df_gc['GC_GAIA_RA'] = df_gc['GAIA_RA_ICRS']+(df_gc['CATWISE_MJD']-gaia_ref_mjd)/365.*df_gc['GAIA_pmRA']/3.6e6/(np.cos(df_gc['GAIA_DE_ICRS']*np.pi/180.))
            df_gc['GC_GAIA_DE'] = df_gc['GAIA_DE_ICRS']+(df_gc['CATWISE_MJD']-gaia_ref_mjd)/365.*df_gc['GAIA_pmDE']/3.6e6
            df_gc.loc[df_gc['GC_GAIA_RA'].isnull(),'GC_GAIA_RA'] = df_gc.loc[df_gc['GC_GAIA_RA'].isnull(),'GAIA_RA_ICRS']
            df_gc.loc[df_gc['GC_GAIA_DE'].isnull(),'GC_GAIA_DE'] = df_gc.loc[df_gc['GC_GAIA_DE'].isnull(),'GAIA_DE_ICRS']

            df_gc['GC_GAIA_PU'] = np.sqrt(df_gc['GAIA_e_Pos']**2+df_gc['GAIA_Plx']**2+df_gc['GAIA_e_Plx']**2 +(df_gc['GAIA_e_PM']*(df_gc['CATWISE_MJD']-gaia_ref_mjd)/365.)**2+df_gc['GAIA_epsi']**2)
            
            df_gc['GC_PU'] = sigma_factor * np.sqrt(df_gc['GC_GAIA_PU']**2+df_gc['CATWISE_e_Pos']**2)
            
            #df_gc['GC_sep'] = SkyCoord(df_gc['GC_GAIA_RA']*u.degree, df_gc['GC_GAIA_DE']*u.degree, frame='icrs').separation(SkyCoord(df_gc['CATWISE_RA_ICRS']*u.degree, df_gc['CATWISE_DE_ICRS']*u.degree, frame='icrs')).arcsec
            df_gc['GC_sep'] = SkyCoord(df_gc['GC_GAIA_RA'], df_gc['GC_GAIA_DE'], unit=u.deg, frame='icrs').separation(SkyCoord(df_gc['CATWISE_RA_ICRS'], df_gc['CATWISE_DE_ICRS'], unit=u.deg, frame='icrs')).arcsec
            
            df_gc['GC_r'] = df_gc['GC_sep']/df_gc['GC_PU']
            #print(df_gc)
            df_gc_mc = df_gc[df_gc['GC_r']<=1.]
            df_gc_mc = df_gc_mc.sort_values(by=['GAIA_DR3Name','GC_r'], ascending=True)
            df_gc_mc = df_gc_mc.drop_duplicates(subset=['GAIA_DR3Name'], keep='first').reset_index(drop=True).drop(columns=gaia_cols)
            
            #print(df_gc_mc)
             
        if len(df_a)>0:
            if verbose:  
                print('len(df_a)>0')
            # df_g_copy = pd.concat([df_g[gaia_cols2]]*len(df_a), ignore_index=True).sort_values(by=['GAIA_DR3Name']).reset_index(drop=True)
            # df_a_copy = pd.concat([df_a]*len(df_g), ignore_index=True)

            # df_ga = pd.concat([df_g_copy, df_a_copy],axis=1)

            df_ga = get_plausible_associations(df_g[gaia_cols2], df_a, 'GAIA_RA_ICRS', 'GAIA_DE_ICRS', 'ALLWISE_RA', 'ALLWISE_DEC', association_radius=5)
            
            df_ga['GA_GAIA_RA'] = df_ga['GAIA_RA_ICRS']+(55400-gaia_ref_mjd)/365.*df_ga['GAIA_pmRA']/3.6e6/(np.cos(df_ga['GAIA_DE_ICRS']*np.pi/180.))
            df_ga['GA_GAIA_DE'] = df_ga['GAIA_DE_ICRS']+(55400-gaia_ref_mjd)/365.*df_ga['GAIA_pmDE']/3.6e6
            df_ga.loc[df_ga['GA_GAIA_RA'].isnull(),'GA_GAIA_RA'] = df_ga.loc[df_ga['GA_GAIA_RA'].isnull(),'GAIA_RA_ICRS']
            df_ga.loc[df_ga['GA_GAIA_DE'].isnull(),'GA_GAIA_DE'] = df_ga.loc[df_ga['GA_GAIA_DE'].isnull(),'GAIA_DE_ICRS']

            df_ga['GA_GAIA_PU'] = np.sqrt(df_ga['GAIA_e_Pos']**2+df_ga['GAIA_Plx']**2+df_ga['GAIA_e_Plx']**2 +(df_ga['GAIA_e_PM']*(55400-gaia_ref_mjd)/365.)**2+df_ga['GAIA_epsi']**2)

            df_ga['GA_PU'] = sigma_factor * np.sqrt(df_ga['GA_GAIA_PU']**2+df_ga['ALLWISE_err0']**2)

            # df_ga['GA_sep'] = SkyCoord(df_ga['GA_GAIA_RA']*u.deg, df_ga['GA_GAIA_DE']*u.deg, frame='icrs').separation(SkyCoord(df_ga['ALLWISE_RA']*u.deg, df_ga['ALLWISE_DEC']*u.deg, frame='icrs')).arcsec
            df_ga['GA_sep'] = SkyCoord(df_ga['GA_GAIA_RA'], df_ga['GA_GAIA_DE'], unit=u.deg, frame='icrs').separation(SkyCoord(df_ga['ALLWISE_RA'], df_ga['ALLWISE_DEC'], unit=u.deg, frame='icrs')).arcsec
            
            df_ga['GA_r'] = df_ga['GA_sep']/df_ga['GA_PU']
            
            df_ga_mc = df_ga[df_ga['GA_r']<=1.]
            df_ga_mc = df_ga_mc.sort_values(by=['GAIA_DR3Name','GA_r'], ascending=True)
            df_ga_mc = df_ga_mc.drop_duplicates(subset=['GAIA_DR3Name'], keep='first').reset_index(drop=True).drop(columns=gaia_cols)
            #print(df_ga_mc)

        if len(df_t)>0:
            if verbose:
                print('len(df_t)>0')

            # df_g_copy = pd.concat([df_g[gaia_cols2]]*len(df_t), ignore_index=True).sort_values(by=['GAIA_DR3Name']).reset_index(drop=True)
            # df_t_copy = pd.concat([df_t]*len(df_g), ignore_index=True)

            # df_gt = pd.concat([df_g_copy, df_t_copy],axis=1)

            df_gt = get_plausible_associations(df_g[gaia_cols2], df_t, 'GAIA_RA_ICRS', 'GAIA_DE_ICRS', 'TMASS_RA', 'TMASS_DEC', association_radius=5)

            df_gt['GT_GAIA_RA'] = df_gt['GAIA_RA_ICRS']+(df_gt['TMASS_MJD']-gaia_ref_mjd)/365.*df_gt['GAIA_pmRA']/3.6e6/(np.cos(df_gt['GAIA_DE_ICRS']*np.pi/180.))
            df_gt['GT_GAIA_DE'] = df_gt['GAIA_DE_ICRS']+(df_gt['TMASS_MJD']-gaia_ref_mjd)/365.*df_gt['GAIA_pmDE']/3.6e6
            df_gt.loc[df_gt['GT_GAIA_RA'].isnull(),'GT_GAIA_RA'] = df_gt.loc[df_gt['GT_GAIA_RA'].isnull(),'GAIA_RA_ICRS']
            df_gt.loc[df_gt['GT_GAIA_DE'].isnull(),'GT_GAIA_DE'] = df_gt.loc[df_gt['GT_GAIA_DE'].isnull(),'GAIA_DE_ICRS']


            df_gt['GT_GAIA_PU'] = np.sqrt(df_gt['GAIA_e_Pos']**2+df_gt['GAIA_Plx']**2+df_gt['GAIA_e_Plx']**2 +(df_gt['GAIA_e_PM']*(df_gt['TMASS_MJD']-gaia_ref_mjd)/365.)**2+df_gt['GAIA_epsi']**2)
            
            df_gt['GT_PU'] = sigma_factor * np.sqrt(df_gt['GT_GAIA_PU']**2+df_gt['TMASS_err0']**2)

            # df_gt['GT_sep'] = SkyCoord(df_gt['GT_GAIA_RA']*u.deg, df_gt['GT_GAIA_DE']*u.deg, frame='icrs').separation(SkyCoord(df_gt['TMASS_RA']*u.deg, df_gt['TMASS_DEC']*u.deg, frame='icrs')).arcsec
            df_gt['GT_sep'] = SkyCoord(df_gt['GT_GAIA_RA'], df_gt['GT_GAIA_DE'], unit=u.deg, frame='icrs').separation(SkyCoord(df_gt['TMASS_RA'], df_gt['TMASS_DEC'], unit=u.deg, frame='icrs')).arcsec

            df_gt['GT_r'] = df_gt['GT_sep']/df_gt['GT_PU']
            #print(df_gt.loc[(df_gt['TMASS__2MASS'].isin(['17463763-3121514','17463827-3121426'])) & (df_gt['GT_sep']<3), ['GAIA_DR3Name','GT_sep','GT_PU','TMASS__2MASS','GT_GAIA_PU','TMASS_err0']])

            df_gt_mc = df_gt[df_gt['GT_r']<=1.]
            df_gt_mc = df_gt_mc.sort_values(by=['GAIA_DR3Name','GT_r'], ascending=True)

            df_gt_mc = df_gt_mc.drop_duplicates(subset=['GAIA_DR3Name'], keep='first').reset_index(drop=True).drop(columns=gaia_cols)
     
        #if len(df_gc_mc)>0 and len(df_ga_mc)>0:
        if len(df_c)>0 and len(df_a)>0 and len(df_t)>0:

            df_mc = pd.merge(df_gc_mc, df_ga_mc, on=['GAIA_DR3Name'], how='outer')
            df_mc = pd.merge(df_mc, df_gt_mc, on=['GAIA_DR3Name'], how='outer')
            df_c_single = df_c[~(df_c['CATWISE_Name'].isin(df_gc_mc['CATWISE_Name']))].reset_index(drop=True)
            df_a_single = df_a[~(df_a['ALLWISE_AllWISE'].isin(df_ga_mc['ALLWISE_AllWISE']))].reset_index(drop=True)
            df_t_single = df_t[~(df_t['TMASS__2MASS'].isin(df_gt_mc['TMASS__2MASS']))].reset_index(drop=True)

        elif len(df_c)>0 and len(df_a)>0: 
            df_mc = pd.merge(df_gc_mc, df_ga_mc, on=['GAIA_DR3Name'], how='outer')
            df_c_single = df_c[~(df_c['CATWISE_Name'].isin(df_gc_mc['CATWISE_Name']))].reset_index(drop=True)
            df_a_single = df_a[~(df_a['ALLWISE_AllWISE'].isin(df_ga_mc['ALLWISE_AllWISE']))].reset_index(drop=True)
            df_t_single = df_t

        elif len(df_c)>0 and len(df_t)>0:
            df_mc = pd.merge(df_gc_mc, df_gt_mc, on=['GAIA_DR3Name'], how='outer')
            df_c_single = df_c[~(df_c['CATWISE_Name'].isin(df_gc_mc['CATWISE_Name']))].reset_index(drop=True)
            df_t_single = df_t[~(df_t['TMASS__2MASS'].isin(df_gt_mc['TMASS__2MASS']))].reset_index(drop=True)
            df_a_single = df_a

        elif len(df_a)>0 and len(df_t)>0:
            df_mc = pd.merge(df_ga_mc, df_gt_mc, on=['GAIA_DR3Name'], how='outer')
            df_a_single = df_a[~(df_a['ALLWISE_AllWISE'].isin(df_ga_mc['ALLWISE_AllWISE']))].reset_index(drop=True)
            df_t_single = df_t[~(df_t['TMASS__2MASS'].isin(df_gt_mc['TMASS__2MASS']))].reset_index(drop=True)
            df_c_single = df_c

        elif len(df_c)>0:
            df_mc = df_gc_mc
            df_c_single = df_c[~(df_c['CATWISE_Name'].isin(df_gc_mc['CATWISE_Name']))].reset_index(drop=True)
            df_a_single = df_a
            df_t_single = df_t
        
        elif len(df_a)>0:
            df_mc = df_ga_mc
            df_c_single = df_c
            df_a_single = df_a[~(df_a['ALLWISE_AllWISE'].isin(df_ga_mc['ALLWISE_AllWISE']))].reset_index(drop=True)
            df_t_single = df_t

        elif len(df_t)>0:
            df_mc = df_gt_mc
            df_c_single = df_c
            df_a_single = df_a
            df_t_single = df_t[~(df_t['TMASS__2MASS'].isin(df_gt_mc['TMASS__2MASS']))].reset_index(drop=True)
 
        else:
            df_mc = df_g[['GAIA_DR3Name']]
            df_c_single = df_c
            df_a_single = df_a
            df_t_single = df_t
        
        
        #print(df_mc.columns)
        #print(df_mc)
        df_g_single = df_g[~(df_g['GAIA_DR3Name'].isin(df_mc['GAIA_DR3Name']))].reset_index(drop=True)
        if verbose:  
            print(len(df_g),len(df_g_single),len(df_g[(df_g['GAIA_DR3Name'].isin(df_mc['GAIA_DR3Name']))]))
        df_mc = pd.merge(df_mc, df_g[(df_g['GAIA_DR3Name'].isin(df_mc['GAIA_DR3Name']))].reset_index(drop=True), on='GAIA_DR3Name',how='left')
        df_match = pd.concat([df_mc, df_g_single], ignore_index=True)
        if verbose:  
            print(len(df_mc), len(df_match))

        if len(df_c_single)>0:
            if verbose:  
                print('len(df_c_single)>0')
            if len(df_a_single)>0:
                if verbose:  
                    print('len(df_a_single)>0')
                
                #print('len(df_c_single)>0 and len(df_a_single)>0')
                # df_c_copy = pd.concat([df_c_single[catwise_cols2]]*len(df_a_single), ignore_index=True).sort_values(by=['CATWISE_Name']).reset_index(drop=True)
                # df_a_copy = pd.concat([df_a_single]*len(df_c_single), ignore_index=True)
                # df_ca = pd.concat([df_c_copy, df_a_copy],axis=1)
                #print(df_gc)

                df_ca = get_plausible_associations(df_c_single[catwise_cols2], df_a_single, 'CATWISE_RA_ICRS', 'CATWISE_DE_ICRS', 'ALLWISE_RA', 'ALLWISE_DEC', association_radius=5)

                df_ca['CA_RA'] = df_ca['CATWISE_RA_ICRS']+(55400-df_ca['CATWISE_MJD'])/365.*df_ca['CATWISE_pmRA']/3.6e3/(np.cos(df_ca['CATWISE_DE_ICRS']*np.pi/180.))
                df_ca['CA_DE'] = df_ca['CATWISE_DE_ICRS']+(55400-df_ca['CATWISE_MJD'])/365.*df_ca['CATWISE_pmDE']/3.6e3
                df_ca.loc[df_ca['CA_RA'].isnull(),'CA_RA'] = df_ca.loc[df_ca['CA_RA'].isnull(),'CATWISE_RA_ICRS']
                df_ca.loc[df_ca['CA_DE'].isnull(),'CA_DE'] = df_ca.loc[df_ca['CA_DE'].isnull(),'CATWISE_DE_ICRS']

                df_ca['CA_CATWISE_PU'] = np.sqrt(df_ca['CATWISE_e_Pos']**2+(df_ca['CATWISE_e_PM']*(55400-df_ca['CATWISE_MJD'])/365.)**2)
                df_ca['CA_PU'] = sigma_factor * np.sqrt(df_ca['CA_CATWISE_PU']**2+df_ca['ALLWISE_err0']**2)
                #df_ca['CA_sep'] = SkyCoord(df_ca['CA_RA']*u.deg, df_ca['CA_DE']*u.deg, frame='icrs').separation(SkyCoord(df_ca['ALLWISE_RA']*u.deg, df_ca['ALLWISE_DEC']*u.deg, frame='icrs')).arcsec
                df_ca['CA_sep'] = SkyCoord(df_ca['CA_RA'], df_ca['CA_DE'], unit=u.deg, frame='icrs').separation(SkyCoord(df_ca['ALLWISE_RA'], df_ca['ALLWISE_DEC'], unit=u.deg, frame='icrs')).arcsec

                df_ca['CA_r'] = df_ca['CA_sep']/df_ca['CA_PU']
                df_ca_mc = df_ca[df_ca['CA_r']<=1.]
                df_ca_mc = df_ca_mc.sort_values(by=['CATWISE_Name','CA_r'], ascending=True)
                df_ca_mc = df_ca_mc.drop_duplicates(subset=['CATWISE_Name'], keep='first').reset_index(drop=True).drop(columns=catwise_cols)
                
            
            if len(df_t_single)>0:
                if verbose:  
                    print('len(df_t_single)>0')
                
                # df_c_copy = pd.concat([df_c_single[catwise_cols2]]*len(df_t_single), ignore_index=True).sort_values(by=['CATWISE_Name']).reset_index(drop=True)
                # df_t_copy = pd.concat([df_t_single]*len(df_c_single), ignore_index=True)
                # df_ct = pd.concat([df_c_copy, df_t_copy],axis=1)
                
                df_ct = get_plausible_associations(df_c_single[catwise_cols2], df_t_single, 'CATWISE_RA_ICRS', 'CATWISE_DE_ICRS', 'TMASS_RA', 'TMASS_DEC', association_radius=5)

                df_ct['CT_RA'] = df_ct['CATWISE_RA_ICRS']+(df_ct['TMASS_MJD']-df_ct['CATWISE_MJD'])/365.*df_ct['CATWISE_pmRA']/3.6e3/(np.cos(df_ct['CATWISE_DE_ICRS']*np.pi/180.))
                df_ct['CT_DE'] = df_ct['CATWISE_DE_ICRS']+(df_ct['TMASS_MJD']-df_ct['CATWISE_MJD'])/365.*df_ct['CATWISE_pmDE']/3.6e3
                df_ct.loc[df_ct['CT_RA'].isnull(),'CT_RA'] = df_ct.loc[df_ct['CT_RA'].isnull(),'CATWISE_RA_ICRS']
                df_ct.loc[df_ct['CT_DE'].isnull(),'CT_DE'] = df_ct.loc[df_ct['CT_DE'].isnull(),'CATWISE_DE_ICRS']

                df_ct['CT_CATWISE_PU'] = np.sqrt(df_ct['CATWISE_e_Pos']**2+(df_ct['CATWISE_e_PM']*(df_ct['TMASS_MJD']-df_ct['CATWISE_MJD'])/365.)**2)
                df_ct['CT_PU'] = sigma_factor * np.sqrt(df_ct['CT_CATWISE_PU']**2+df_ct['TMASS_err0']**2)
                #df_ct['CT_sep'] = SkyCoord(df_ct['CT_RA']*u.deg, df_ct['CT_DE']*u.deg, frame='icrs').separation(SkyCoord(df_ct['TMASS_RA']*u.deg, df_ct['TMASS_DEC']*u.deg, frame='icrs')).arcsec
                df_ct['CT_sep'] = SkyCoord(df_ct['CT_RA'], df_ct['CT_DE'], unit=u.deg, frame='icrs').separation(SkyCoord(df_ct['TMASS_RA'], df_ct['TMASS_DEC'], unit=u.deg, frame='icrs')).arcsec

                df_ct['CT_r'] = df_ct['CT_sep']/df_ct['CT_PU']
                df_ct_mc = df_ct[df_ct['CT_r']<=1.]
                df_ct_mc = df_ct_mc.sort_values(by=['CATWISE_Name','CT_r'], ascending=True)
                df_ct_mc = df_ct_mc.drop_duplicates(subset=['CATWISE_Name'], keep='first').reset_index(drop=True).drop(columns=catwise_cols)
     
            if len(df_a_single)>0 and len(df_t_single)>0:

                df_c_mc = pd.merge(df_ca_mc, df_ct_mc, on=['CATWISE_Name'], how='outer')
            
                df_c_mc = pd.merge(df_c_mc, df_c_single[(df_c_single['CATWISE_Name'].isin(df_c_mc['CATWISE_Name']))].reset_index(drop=True), on='CATWISE_Name',how='left')
        
                df_match = pd.concat([df_match, df_c_mc], ignore_index=True)
                df_a_single = df_a_single[~(df_a_single['ALLWISE_AllWISE'].isin(df_ca_mc['ALLWISE_AllWISE']))].reset_index(drop=True)
                df_t_single = df_t_single[~(df_t_single['TMASS__2MASS'].isin(df_ct_mc['TMASS__2MASS']))].reset_index(drop=True)


            elif len(df_a_single)>0:
                #print('len(df_c_single)>0')
                
                df_c_mc = pd.merge(df_ca_mc, df_c_single[(df_c_single['CATWISE_Name'].isin(df_ca_mc['CATWISE_Name']))].reset_index(drop=True), on='CATWISE_Name',how='left')
        
                df_match = pd.concat([df_match, df_c_mc], ignore_index=True)
                if verbose:
                    print(len(df_match))
                df_a_single = df_a_single[~(df_a_single['ALLWISE_AllWISE'].isin(df_ca_mc['ALLWISE_AllWISE']))].reset_index(drop=True)
                
            elif len(df_t_single)>0:

                df_c_mc = pd.merge(df_ct_mc, df_c_single[(df_c_single['CATWISE_Name'].isin(df_ct_mc['CATWISE_Name']))].reset_index(drop=True), on='CATWISE_Name',how='left')
        
                df_match = pd.concat([df_match, df_c_mc], ignore_index=True)
                if verbose:
                    print(len(df_match))
                df_t_single = df_t_single[~(df_t_single['TMASS__2MASS'].isin(df_ct_mc['TMASS__2MASS']))].reset_index(drop=True)

                            
            df_c_single = df_c_single[~(df_c_single['CATWISE_Name'].isin(df_match['CATWISE_Name']))].reset_index(drop=True)        
            df_match = pd.concat([df_match, df_c_single], ignore_index=True)
            if verbose:
                print(len(df_match))

            if len(df_a_single)>0 and len(df_t_single)>0:
                # df_a_copy = pd.concat([df_a_single]*len(df_t_single), ignore_index=True).sort_values(by=['ALLWISE_AllWISE']).reset_index(drop=True)
                # df_t_copy = pd.concat([df_t_single]*len(df_a_single), ignore_index=True)

                # df_at = pd.concat([df_a_copy, df_t_copy],axis=1)

                df_at = get_plausible_associations(df_a_single, df_t_single, 'ALLWISE_RA', 'ALLWISE_DEC', 'TMASS_RA', 'TMASS_DEC', association_radius=5)
        
                df_at['AT_PU'] = sigma_factor * np.sqrt(df_at['TMASS_err0']**2+df_at['ALLWISE_err0']**2)
                #df_at['AT_sep'] = SkyCoord(df_at['TMASS_RA']*u.deg, df_at['TMASS_DEC']*u.deg, frame='icrs').separation(SkyCoord(df_at['ALLWISE_RA']*u.deg, df_at['ALLWISE_DEC']*u.deg, frame='icrs')).arcsec
                df_at['AT_sep'] = SkyCoord(df_at['TMASS_RA'], df_at['TMASS_DEC'], unit=u.deg, frame='icrs').separation(SkyCoord(df_at['ALLWISE_RA'], df_at['ALLWISE_DEC'], unit=u.deg, frame='icrs')).arcsec
                
                df_at['AT_r'] = df_at['AT_sep']/df_at['AT_PU']
                #print(df_at)
                df_at_mc = df_at[df_at['AT_r']<=1.]
                df_at_mc = df_at_mc.sort_values(by=['TMASS__2MASS','AT_r'], ascending=True)
                df_at_mc = df_at_mc.drop_duplicates(subset=['TMASS__2MASS'], keep='first').reset_index(drop=True)
                #print(df_at_mc)

                df_match = pd.concat([df_match, df_at_mc], ignore_index=True)
                df_match = pd.concat([df_match, df_t_single[~(df_t_single['TMASS__2MASS'].isin(df_at_mc['TMASS__2MASS']))]], ignore_index=True)
                df_match = pd.concat([df_match, df_a_single[~(df_a_single['ALLWISE_AllWISE'].isin(df_at_mc['ALLWISE_AllWISE']))]], ignore_index=True)
             
            elif len(df_a_single)>0:
                df_match = pd.concat([df_match, df_a_single], ignore_index=True)
                if verbose:
                    print(len(df_match))
            elif len(df_t_single)>0:
                df_match = pd.concat([df_match, df_t_single], ignore_index=True)
                if verbose:
                    print(len(df_match))

        elif len(df_c_single)==0:  
            if verbose:  
                print('len(df_c_single)==0')
            if len(df_a_single)>0 and len(df_t_single)>0:
                if verbose:  
                    print('len(df_a_single)>0 and len(df_t_single)>0')

                # df_a_copy = pd.concat([df_a_single]*len(df_t_single), ignore_index=True).sort_values(by=['ALLWISE_AllWISE']).reset_index(drop=True)
                # df_t_copy = pd.concat([df_t_single]*len(df_a_single), ignore_index=True)

                # df_at = pd.concat([df_a_copy, df_t_copy],axis=1)

                df_at = get_plausible_associations(df_a_single, df_t_single, 'ALLWISE_RA', 'ALLWISE_DEC', 'TMASS_RA', 'TMASS_DEC', association_radius=5)
        
                df_at['AT_PU'] = sigma_factor * np.sqrt(df_at['TMASS_err0']**2+df_at['ALLWISE_err0']**2)
                #df_at['AT_sep'] = SkyCoord(df_at['TMASS_RA']*u.deg, df_at['TMASS_DEC']*u.deg, frame='icrs').separation(SkyCoord(df_at['ALLWISE_RA']*u.deg, df_at['ALLWISE_DEC']*u.deg, frame='icrs')).arcsec
                df_at['AT_sep'] = SkyCoord(df_at['TMASS_RA'], df_at['TMASS_DEC'], unit=u.deg, frame='icrs').separation(SkyCoord(df_at['ALLWISE_RA'], df_at['ALLWISE_DEC'], unit=u.deg, frame='icrs')).arcsec
                df_at['AT_r'] = df_at['AT_sep']/df_at['AT_PU']
                #print(df_at)
                df_at_mc = df_at[df_at['AT_r']<=1.]
                df_at_mc = df_at_mc.sort_values(by=['TMASS__2MASS','AT_r'], ascending=True)
                df_at_mc = df_at_mc.drop_duplicates(subset=['TMASS__2MASS'], keep='first').reset_index(drop=True)
                #print(df_at_mc)

                df_match = pd.concat([df_match, df_at_mc], ignore_index=True)
                df_match = pd.concat([df_match, df_t_single[~(df_t_single['TMASS__2MASS'].isin(df_at_mc['TMASS__2MASS']))]], ignore_index=True)
                df_match = pd.concat([df_match, df_a_single[~(df_a_single['ALLWISE_AllWISE'].isin(df_at_mc['ALLWISE_AllWISE']))]], ignore_index=True)
             
            elif len(df_a_single)>0:
                if verbose: 
                    print('len(df_a_single)>0') 
                df_match = pd.concat([df_match, df_a_single], ignore_index=True)
            elif len(df_t_single)>0:
                if verbose:  
                    print('len(df_t_single)>0') 
                df_match = pd.concat([df_match, df_t_single], ignore_index=True)

    else:
        if verbose: 
            print('len(df_g)==0') 

        if len(df_c)>0:
            if verbose: 
                print('len(df_c)>0') 

            if len(df_a)>0:
                if verbose: 
                    print('len(df_a)>0')  
                
                # df_c_copy = pd.concat([df_c[catwise_cols2]]*len(df_a), ignore_index=True).sort_values(by=['CATWISE_Name']).reset_index(drop=True)
                # df_a_copy = pd.concat([df_a]*len(df_c), ignore_index=True)
                # df_ca = pd.concat([df_c_copy, df_a_copy],axis=1)
                
                df_ca = get_plausible_associations(df_c[catwise_cols2], df_a, 'CATWISE_RA_ICRS', 'CATWISE_DE_ICRS', 'ALLWISE_RA', 'ALLWISE_DEC', association_radius=5)

                df_ca['CA_RA'] = df_ca['CATWISE_RA_ICRS']+(55400-df_ca['CATWISE_MJD'])/365.*df_ca['CATWISE_pmRA']/3.6e3/(np.cos(df_ca['CATWISE_DE_ICRS']*np.pi/180.))
                df_ca['CA_DE'] = df_ca['CATWISE_DE_ICRS']+(55400-df_ca['CATWISE_MJD'])/365.*df_ca['CATWISE_pmDE']/3.6e3
                df_ca.loc[df_ca['CA_RA'].isnull(),'CA_RA'] = df_ca.loc[df_ca['CA_RA'].isnull(),'CATWISE_RA_ICRS']
                df_ca.loc[df_ca['CA_DE'].isnull(),'CA_DE'] = df_ca.loc[df_ca['CA_DE'].isnull(),'CATWISE_DE_ICRS']

                df_ca['CA_CATWISE_PU'] = np.sqrt(df_ca['CATWISE_e_Pos']**2+(df_ca['CATWISE_e_PM']*(55400-df_ca['CATWISE_MJD'])/365.)**2)
                df_ca['CA_PU'] = sigma_factor * np.sqrt(df_ca['CA_CATWISE_PU']**2+df_ca['ALLWISE_err0']**2)
                # df_ca['CA_sep'] = SkyCoord(df_ca['CA_RA']*u.deg, df_ca['CA_DE']*u.deg, frame='icrs').separation(SkyCoord(df_ca['ALLWISE_RA']*u.deg, df_ca['ALLWISE_DEC']*u.deg, frame='icrs')).arcsec
                df_ca['CA_sep'] = SkyCoord(df_ca['CA_RA'], df_ca['CA_DE'], unit=u.deg, frame='icrs').separation(SkyCoord(df_ca['ALLWISE_RA'], df_ca['ALLWISE_DEC'], unit=u.deg, frame='icrs')).arcsec

                df_ca['CA_r'] = df_ca['CA_sep']/df_ca['CA_PU']
                df_ca_mc = df_ca[df_ca['CA_r']<=1.]
                df_ca_mc = df_ca_mc.sort_values(by=['CATWISE_Name','CA_r'], ascending=True)
                df_ca_mc = df_ca_mc.drop_duplicates(subset=['CATWISE_Name'], keep='first').reset_index(drop=True).drop(columns=catwise_cols)
                
            
            if len(df_t)>0:
                if verbose:
                    print('len(df_t)>0')   
                
                # df_c_copy = pd.concat([df_c[catwise_cols2]]*len(df_t), ignore_index=True).sort_values(by=['CATWISE_Name']).reset_index(drop=True)
                # df_t_copy = pd.concat([df_t]*len(df_c), ignore_index=True)
                # df_ct = pd.concat([df_c_copy, df_t_copy],axis=1)

                df_ct = get_plausible_associations(df_c[catwise_cols2], df_t, 'CATWISE_RA_ICRS', 'CATWISE_DE_ICRS', 'TMASS_RA', 'TMASS_DEC', association_radius=5)

                df_ct['CT_RA'] = df_ct['CATWISE_RA_ICRS']+(df_ct['TMASS_MJD']-df_ct['CATWISE_MJD'])/365.*df_ct['CATWISE_pmRA']/3.6e3/(np.cos(df_ct['CATWISE_DE_ICRS']*np.pi/180.))
                df_ct['CT_DE'] = df_ct['CATWISE_DE_ICRS']+(df_ct['TMASS_MJD']-df_ct['CATWISE_MJD'])/365.*df_ct['CATWISE_pmDE']/3.6e3
                df_ct.loc[df_ct['CT_RA'].isnull(),'CT_RA'] = df_ct.loc[df_ct['CT_RA'].isnull(),'CATWISE_RA_ICRS']
                df_ct.loc[df_ct['CT_DE'].isnull(),'CT_DE'] = df_ct.loc[df_ct['CT_DE'].isnull(),'CATWISE_DE_ICRS']

                df_ct['CT_CATWISE_PU'] = np.sqrt(df_ct['CATWISE_e_Pos']**2+(df_ct['CATWISE_e_PM']*(df_ct['TMASS_MJD']-df_ct['CATWISE_MJD'])/365.)**2)
                df_ct['CT_PU'] = sigma_factor * np.sqrt(df_ct['CT_CATWISE_PU']**2+df_ct['TMASS_err0']**2)
                # df_ct['CT_sep'] = SkyCoord(df_ct['CT_RA']*u.deg, df_ct['CT_DE']*u.deg, frame='icrs').separation(SkyCoord(df_ct['TMASS_RA']*u.deg, df_ct['TMASS_DEC']*u.deg, frame='icrs')).arcsec
                df_ct['CT_sep'] = SkyCoord(df_ct['CT_RA'], df_ct['CT_DE'], unit=u.deg, frame='icrs').separation(SkyCoord(df_ct['TMASS_RA'], df_ct['TMASS_DEC'], unit=u.deg, frame='icrs')).arcsec

                df_ct['CT_r'] = df_ct['CT_sep']/df_ct['CT_PU']
                df_ct_mc = df_ct[df_ct['CT_r']<=1.]
                df_ct_mc = df_ct_mc.sort_values(by=['CATWISE_Name','CT_r'], ascending=True)
                df_ct_mc = df_ct_mc.drop_duplicates(subset=['CATWISE_Name'], keep='first').reset_index(drop=True).drop(columns=catwise_cols)
     
            if len(df_a)>0 and len(df_t)>0:

                df_c_mc = pd.merge(df_ca_mc, df_ct_mc, on=['CATWISE_Name'], how='outer')
        
                df_a_single = df_a[~(df_a['ALLWISE_AllWISE'].isin(df_ca_mc['ALLWISE_AllWISE']))].reset_index(drop=True)
                df_t_single = df_t[~(df_t['TMASS__2MASS'].isin(df_ct_mc['TMASS__2MASS']))].reset_index(drop=True)

                df_c_mc = pd.merge(df_c_mc, df_c[(df_c['CATWISE_Name'].isin(df_c_mc['CATWISE_Name']))].reset_index(drop=True), on='CATWISE_Name',how='left')
                df_c_single = df_c[~(df_c['CATWISE_Name'].isin(df_c_mc['CATWISE_Name']))].reset_index(drop=True)
                df_match = pd.concat([df_c_mc, df_c_single], ignore_index=True)

            elif len(df_a)>0:
                #print('len(df_c_single)>0')
            
                df_c_mc = df_ca_mc
                
                df_a_single = df_a[~(df_a['ALLWISE_AllWISE'].isin(df_ca_mc['ALLWISE_AllWISE']))].reset_index(drop=True)
                df_t_single = df_t

                df_c_mc = pd.merge(df_c_mc, df_c[(df_c['CATWISE_Name'].isin(df_c_mc['CATWISE_Name']))].reset_index(drop=True), on='CATWISE_Name',how='left')
                df_c_single = df_c[~(df_c['CATWISE_Name'].isin(df_c_mc['CATWISE_Name']))].reset_index(drop=True)
                df_match = pd.concat([df_c_mc, df_c_single], ignore_index=True)

            elif len(df_t)>0:
                df_c_mc = df_ct_mc
                df_a_single = df_a
                df_t_single = df_t[~(df_t['TMASS__2MASS'].isin(df_ct_mc['TMASS__2MASS']))].reset_index(drop=True)

                df_c_mc = pd.merge(df_c_mc, df_c[(df_c['CATWISE_Name'].isin(df_c_mc['CATWISE_Name']))].reset_index(drop=True), on='CATWISE_Name',how='left')
                df_c_single = df_c[~(df_c['CATWISE_Name'].isin(df_c_mc['CATWISE_Name']))].reset_index(drop=True)
                df_match = pd.concat([df_c_mc, df_c_single], ignore_index=True)

            else:
                df_match = df_c
                df_a_single = df_a
                df_t_single = df_t

            #df_c_mc = pd.merge(df_c_mc, df_c[(df_c['CATWISE_Name'].isin(df_c_mc['CATWISE_Name']))].reset_index(drop=True), on='CATWISE_Name',how='left')
            #df_c_single = df_c[~(df_c['CATWISE_Name'].isin(df_c_mc['CATWISE_Name']))].reset_index(drop=True)
            #df_match = pd.concat([df_c_mc, df_c_single], ignore_index=True)

            if len(df_a_single)>0 and len(df_t_single)>0:
                # df_a_copy = pd.concat([df_a_single]*len(df_t_single), ignore_index=True).sort_values(by=['ALLWISE_AllWISE']).reset_index(drop=True)
                # df_t_copy = pd.concat([df_t_single]*len(df_a_single), ignore_index=True)

                # df_at = pd.concat([df_a_copy, df_t_copy],axis=1)

                df_at = get_plausible_associations(df_a_single, df_t_single, 'ALLWISE_RA', 'ALLWISE_DEC', 'TMASS_RA', 'TMASS_DEC', association_radius=5)
        
                df_at['AT_PU'] = sigma_factor * np.sqrt(df_at['TMASS_err0']**2+df_at['ALLWISE_err0']**2)
                # df_at['AT_sep'] = SkyCoord(df_at['TMASS_RA']*u.deg, df_at['TMASS_DEC']*u.deg, frame='icrs').separation(SkyCoord(df_at['ALLWISE_RA']*u.deg, df_at['ALLWISE_DEC']*u.deg, frame='icrs')).arcsec
                df_at['AT_sep'] = SkyCoord(df_at['TMASS_RA'], df_at['TMASS_DEC'], unit=u.deg, frame='icrs').separation(SkyCoord(df_at['ALLWISE_RA'], df_at['ALLWISE_DEC'], unit=u.deg, frame='icrs')).arcsec
                df_at['AT_r'] = df_at['AT_sep']/df_at['AT_PU']
                #print(df_at)
                df_at_mc = df_at[df_at['AT_r']<=1.]
                df_at_mc = df_at_mc.sort_values(by=['TMASS__2MASS','AT_r'], ascending=True)
                df_at_mc = df_at_mc.drop_duplicates(subset=['TMASS__2MASS'], keep='first').reset_index(drop=True)
                #print(df_at_mc)

                df_match = pd.concat([df_match, df_at_mc], ignore_index=True)
                df_match = pd.concat([df_match, df_t_single[~(df_t_single['TMASS__2MASS'].isin(df_at_mc['TMASS__2MASS']))]], ignore_index=True)
                df_match = pd.concat([df_match, df_a_single[~(df_a_single['ALLWISE_AllWISE'].isin(df_at_mc['ALLWISE_AllWISE']))]], ignore_index=True)
             
            elif len(df_a_single)>0:
                df_match = pd.concat([df_match, df_a_single], ignore_index=True)
            elif len(df_t_single)>0:
                df_match = pd.concat([df_match, df_t_single], ignore_index=True)

        elif len(df_c)==0:  
            if verbose:  
                print('len(df_c)==0')

            if len(df_a)>0 and len(df_t)>0:
                if verbose:  
                    print('len(df_a)>0 and len(df_t)>0')

                # df_a_copy = pd.concat([df_a]*len(df_t), ignore_index=True).sort_values(by=['ALLWISE_AllWISE']).reset_index(drop=True)
                # df_t_copy = pd.concat([df_t]*len(df_a), ignore_index=True)

                # df_at = pd.concat([df_a_copy, df_t_copy],axis=1)

                df_at = get_plausible_associations(df_a, df_t, 'ALLWISE_RA', 'ALLWISE_DEC', 'TMASS_RA', 'TMASS_DEC', association_radius=5)
        
                df_at['AT_PU'] = sigma_factor * np.sqrt(df_at['TMASS_err0']**2+df_at['ALLWISE_err0']**2)
                # df_at['AT_sep'] = SkyCoord(df_at['TMASS_RA']*u.deg, df_at['TMASS_DEC']*u.deg, frame='icrs').separation(SkyCoord(df_at['ALLWISE_RA']*u.deg, df_at['ALLWISE_DEC']*u.deg, frame='icrs')).arcsec
                df_at['AT_sep'] = SkyCoord(df_at['TMASS_RA'], df_at['TMASS_DEC'], unit=u.deg, frame='icrs').separation(SkyCoord(df_at['ALLWISE_RA'], df_at['ALLWISE_DEC'], unit=u.deg, frame='icrs')).arcsec
                df_at['AT_r'] = df_at['AT_sep']/df_at['AT_PU']
                #print(df_at)
                df_at_mc = df_at[df_at['AT_r']<=1.]
                df_at_mc = df_at_mc.sort_values(by=['TMASS__2MASS','AT_r'], ascending=True)
                df_at_mc = df_at_mc.drop_duplicates(subset=['TMASS__2MASS'], keep='first').reset_index(drop=True)
                #print(df_at_mc)

                #df_match = pd.concat([df_match, df_at_mc], ignore_index=True)
                df_match = pd.concat([df_at_mc, df_t[~(df_t['TMASS__2MASS'].isin(df_at_mc['TMASS__2MASS']))]], ignore_index=True)
                df_match = pd.concat([df_match, df_a[~(df_a['ALLWISE_AllWISE'].isin(df_at_mc['ALLWISE_AllWISE']))]], ignore_index=True)
             
            elif len(df_a)>0:
                if verbose:  
                    print('len(df_a)>0')
                df_match = df_a
            elif len(df_t)>0:
                if verbose:  
                    print('len(df_t)>0')
                df_match = df_t

    df_match['MW'] = ''
    for MW_iden, cat_n in zip(['GAIA_DR3Name','CATWISE_Name','ALLWISE_AllWISE','TMASS__2MASS'],['g','c','a','t']):
        if MW_iden in df_match.columns:
            
            df_match.loc[~(df_match[MW_iden].isnull()) & ~(df_match[MW_iden]==''), 'MW'] = df_match.loc[~(df_match[MW_iden].isnull()) & ~(df_match[MW_iden]==''), 'MW'] + cat_n

    df_match = df_match.reset_index(drop=True)
    #df_match['RA'], df_match['DEC'], df_match['err0'], df_match['err1'], df_match['errPA'], = np.nan, np.nan, np.nan, np.nan, np.nan

    # id column just to match between smaller fits file used by nway.py for crossmatching to Chandra sources, and full csv file with all MW properties
    df_match['id'] = df_match.index+1

    # Find separation of MW associations to ra_x, dec_x, could just be separation to cluster center
    df_match_X = pd.DataFrame()

    df_match_g = df_match[df_match['MW'].str.contains('g')].reset_index(drop=True)
    if len(df_match_g)>0:
        
        mean_mjd = ref_mjd.mean()
        delta_yr = (mean_mjd - gaia_ref_mjd)/365.
        delta_mean_mjd = max(abs((ref_mjd - mean_mjd)/365.))
        delta_max_mjd = max(abs((ref_mjd - gaia_ref_mjd)/365.))
        
        #df_match_g['RA']  = df_match_g['GAIA_RA_ICRS']
        #df_match_g['DEC'] = df_match_g['GAIA_DE_ICRS']
        df_match_g['RA']  = df_match_g['GAIA_RA_ICRS']+delta_yr*df_match_g['GAIA_pmRA']/(np.cos(df_match_g['GAIA_DE_ICRS']*np.pi/180.)*3.6e6)
        df_match_g['DEC'] = df_match_g['GAIA_DE_ICRS']+delta_yr*df_match_g['GAIA_pmDE']/3.6e6
        
        df_match_g.loc[df_match_g['RA'].isnull(),'RA']   = df_match_g.loc[df_match_g['RA'].isnull(),'GAIA_RA_ICRS']
        df_match_g.loc[df_match_g['DEC'].isnull(),'DEC'] = df_match_g.loc[df_match_g['DEC'].isnull(),'GAIA_DE_ICRS']

        df_match_g['err0'] = sigma_factor_X * np.sqrt(df_match_g['GAIA_e_Pos']**2+df_match_g['GAIA_Plx']**2+df_match_g['GAIA_e_Plx']**2+(df_match_g['GAIA_PM']*delta_mean_mjd)**2+(df_match_g['GAIA_e_PM']*delta_max_mjd)**2+df_match_g['GAIA_epsi']**2)
        df_match_g['err1'] = df_match_g['err0'] 
        df_match_g['errPA'] = 0.

        #df_q['n_srcs'] = len(df_q)
        c = SkyCoord(ra=ra_x, dec=dec_x, unit=u.deg, frame='icrs')
        # df_match_g['sep'] = SkyCoord(ra=df_match_g['RA']*u.degree, dec=df_match_g['DEC']*u.degree).separation(c).arcsec
        df_match_g['sep'] = SkyCoord(ra=df_match_g['RA'], dec=df_match_g['DEC'], unit=u.deg, frame='icrs').separation(c).arcsec
        
        df_match_X = pd.concat([df_match_X, df_match_g[['id','MW','RA','DEC','err0','err1','errPA','sep']]], ignore_index=True, sort=False)
    
    df_match_c = df_match[(df_match['MW'].str.contains('c')) &  (~df_match['MW'].str.contains('g'))].reset_index(drop=True)
    if len(df_match_c):
        
        mean_mjd = ref_mjd.mean()
        delta_mean_mjd = max(abs((ref_mjd - mean_mjd)/365.))
        df_match_c['delta_max_mjd'] = 0.
        df_match_c['delta_max_mjd'] = df_match_c.apply(lambda r: max(abs((ref_mjd - r.CATWISE_MJD)/365.)), axis=1)
        df_match_c.loc[df_match_c['delta_max_mjd'].isnull(),'delta_max_mjd'] = 0.

        
        #df_match_c['RA']  = df_match_c['CATWISE_RA_ICRS']
        #df_match_c['DEC'] = df_match_c['CATWISE_DE_ICRS']
        
                
        df_match_c['RA']  = df_match_c['CATWISE_RA_ICRS']+(mean_mjd - df_match_c['CATWISE_MJD'])/365.*df_match_c['CATWISE_pmRA']/(np.cos(df_match_c['CATWISE_DE_ICRS']*np.pi/180.)*3.6e3)
        df_match_c['DEC'] = df_match_c['CATWISE_DE_ICRS']+(mean_mjd - df_match_c['CATWISE_MJD'])/365.*df_match_c['CATWISE_pmDE']/3.6e3
        
        df_match_c.loc[df_match_c['RA'].isnull(),'RA']   = df_match_c.loc[df_match_c['RA'].isnull(),'CATWISE_RA_ICRS']
        df_match_c.loc[df_match_c['DEC'].isnull(),'DEC'] = df_match_c.loc[df_match_c['DEC'].isnull(),'CATWISE_DE_ICRS']

        df_match_c['err0'] = sigma_factor_X * np.sqrt(df_match_c['CATWISE_e_Pos']**2 + (df_match_c['CATWISE_PM']*delta_mean_mjd)**2+(df_match_c['CATWISE_e_PM']*df_match_c['delta_max_mjd'])**2)
        df_match_c['err1'] = df_match_c['err0'] 
        df_match_c['errPA'] = 0.

        c = SkyCoord(ra=ra_x, dec=dec_x, unit=u.deg, frame='icrs')
        # df_match_c['sep'] = SkyCoord(ra=df_match_c['RA']*u.degree, dec=df_match_c['DEC']*u.degree).separation(c).arcsec
        df_match_c['sep'] = SkyCoord(ra=df_match_c['RA'], dec=df_match_c['DEC'], unit=u.deg, frame='icrs').separation(c).arcsec
        

        df_match_X = pd.concat([df_match_X, df_match_c[['id','MW','RA','DEC','err0','err1','errPA','sep']]], ignore_index=True, sort=False)

    df_match_t = df_match[(df_match['MW'].str.contains('t')) &  (~df_match['MW'].str.contains('g|c'))].reset_index(drop=True) 
    if len(df_match_t)>0:
        
        df_match_t['RA'] = df_match_t['TMASS_RAJ2000']
        df_match_t['DEC'] = df_match_t['TMASS_DEJ2000']
            
        df_match_t['err0'] = df_match_t['TMASS_errMaj'] * sigma_factor_X
        df_match_t['err1'] = df_match_t['TMASS_errMin'] * sigma_factor_X
        df_match_t['errPA'] = df_match_t['TMASS_errPA']

        c = SkyCoord(ra=ra_x, dec=dec_x, unit=u.deg, frame='icrs')
        # df_match_t['sep'] = SkyCoord(ra=df_match_t['RA']*u.degree, dec=df_match_t['DEC']*u.degree).separation(c).arcsec
        df_match_t['sep'] = SkyCoord(ra=df_match_t['RA'], dec=df_match_t['DEC'], unit=u.deg, frame='icrs').separation(c).arcsec
        

        df_match_X = pd.concat([df_match_X, df_match_t[['id','MW','RA','DEC','err0','err1','errPA','sep']]], ignore_index=True, sort=False)

    df_match_a = df_match[(df_match['MW'].str.contains('a')) &  (~df_match['MW'].str.contains('g|c|t'))].reset_index(drop=True) 
    if len(df_match_a)>0:
        df_match_a['RA'] = df_match_a['ALLWISE_RA_pm']
        df_match_a['DEC'] = df_match_a['ALLWISE_DE_pm']
        
        # we don't use the proper motion measurements from allwise as they as affected by parallax and not reliable
        df_match_a['err0'] = df_match_a['ALLWISE_eeMaj'] * sigma_factor_X
        df_match_a['err1'] = df_match_a['ALLWISE_eeMin'] * sigma_factor_X
        df_match_a['errPA'] = df_match_a['ALLWISE_errPA']

        c = SkyCoord(ra=ra_x, dec=dec_x, unit=u.deg, frame='icrs')
        # df_match_a['sep'] = SkyCoord(ra=df_match_a['RA']*u.degree, dec=df_match_a['DEC']*u.degree).separation(c).arcsec
        df_match_a['sep'] = SkyCoord(ra=df_match_a['RA'], dec=df_match_a['DEC'], unit=u.deg, frame='icrs').separation(c).arcsec
        
        
        df_match_X = pd.concat([df_match_X, df_match_a[['id','MW','RA','DEC','err0','err1','errPA','sep']]], ignore_index=True, sort=False)

    
    new_t = Table.from_pandas(df_match_X[['id','MW','RA','DEC','err0','err1','errPA','sep']])
    new_t.write(f'{data_dir}/{X_name}_MW_crossmatch.fits', overwrite=True)
    df_match.to_csv(f'{data_dir}/{X_name}_MW_crossmatch.csv',index=False)#, overwrite=True)
    
    # try:
    #     new_t.write(f'{data_dir}/{X_name}_MW_crossmatch.fits', overwrite=True)
    # except:
    #     print(f'{data_dir}/{X_name}_MW_crossmatch.fits can not be produced at the first time.')
    #     if (catalog == 'gaia'):
    #         df_q.to_csv(f'{data_dir}/{X_name}_{catalog}.csv',index=False)
    #         df_q_csv = pd.read_csv(f'{data_dir}/{X_name}_{catalog}.csv')
    #         new_t = Table.from_pandas(df_q_csv[['RA','DEC','PU','sep','n_srcs','Source','RA_ICRS','DE_ICRS','_r','e_Pos','Plx','e_Plx','PM','pmRA','pmDE','e_PM','epsi','Gmag','BPmag','RPmag','e_Gmag','e_BPmag','e_RPmag','AllWISE','dAllWISE','f_AllWISE','AllWISEoid','_2MASS','d2MASS','f_2MASS','_2MASScoid']])
    #         new_t.write(f'{data_dir}/{X_name}_{catalog}.fits', overwrite=True)
    #     elif (catalog == 'allwise'):
    #         df_q.to_csv(f'{data_dir}/{X_name}_{catalog}.csv',index=False)
    #         df_q_csv = pd.read_csv(f'{data_dir}/{X_name}_{catalog}.csv')
    #         new_t = Table.from_pandas(df_q_csv[['RA','DEC','err0','err1','errPA','RAJ2000','DEJ2000','e_RA_pm','e_DE_pm','sep','n_srcs','AllWISE','_r','W1mag','W2mag','W3mag','W4mag','e_W1mag','e_W2mag','e_W3mag','e_W4mag','_2Mkey','d2M','_2M','Jmag','e_Jmag','Hmag','e_Hmag','Kmag','e_Kmag']])
    #         new_t.write(f'{data_dir}/{X_name}_{catalog}.fits', overwrite=True)
    

    area = np.pi * (search_radius/60)**2

    os.system(f'python {nway_dir}nway-write-header.py {data_dir}/{X_name}_MW_crossmatch.fits MW {area}')
    

    #print('finish.')
    return df_match, df_match_X


def nway_globular_cluster_prepare(ra, dec, cluster, search_radius, data_dir, sigma, verbose=True, astrometric_shifts=None):
    '''
    Prepare the data for nway crossmatch for a globular cluster

    Parameters
    ----------
    ra : float
        Right ascension of the globular cluster in degrees
    dec : float
        Declination of the globular cluster in degrees
    cluster : str
        Name of the globular cluster
    search_radius : float
        Search radius in arcmin, for HSC search
    data_dir : str
        Directory to save the data
    sigma : int
        Sigma for crossmatch, not used
    verbose : bool
    astrometric_shifts : tuple
        Manually determined astrometric shifts between CSC and HST coordinates in (ra, dec), defined as CSC coordinates - HST coordinates
    '''

    if sigma == 2: # 95%
        sigma_factor_X = np.sqrt(np.log(20)*2)
    elif sigma == 1: # ≈ 39.347% in 2-D  
        sigma_factor_X = 1
    elif sigma == 3: # ≈ 99.73% in 2-D   
        sigma_factor_X = np.sqrt(np.log(1./0.0027)*2)

    #sigma = 5. 
    sigma_percentage = 0.9999994267 # 5-sigma
    #sigma_percentage = 0.999937 # 4-sigma
    #sigma_percentage = 0.9972 # 3-sigma
    sigma_factor = np.sqrt(-2.*np.log(1.-sigma_percentage))

    hugs_clusters = ['ngc104', 'ngc288', 'ngc2808', 'ngc6093', 'ngc6121', 'ngc6205', 'ngc6304', 'ngc6341', 'ngc6388', 'ngc6397', 'ngc6652', 'ngc6656', 'ngc6752', 'ngc6809', 'ngc6838', 'ngc7078', 'ngc7099']

    try: 
        df_hugs = pd.read_csv(f'{data_dir}/{cluster}_hugs.csv', engine='pyarrow')

    except:
        if cluster in hugs_clusters:
            print(f'Downloading {cluster} HUGS catalog')
            cluster_url = cluster.replace(' ', '').lower()
            if cluster.lower() == 'ngc104':
                cluster_url = 'ngc0104'
            if cluster.lower() == 'ngc288':
                cluster_url = 'ngc0288'
            try:
                url = f'https://archive.stsci.edu/hlsps/hugs/{cluster_url}/hlsp_hugs_hst_wfc3-uvis-acs-wfc_{cluster_url}_multi_v1_catalog-meth3.txt'
                r = requests.get(url, allow_redirects=True)
                with open(f'{data_dir}/{cluster}_hugs.txt', 'wb') as f:
                    f.write(r.content)
            except:
                print(f'{cluster} not in HUGS')

            df_hugs = pd.read_csv(f'{data_dir}/{cluster}_hugs.txt', comment='#', sep='\s+', names=['X_pos','Y_pos','F275W','F275W_RMS','F275W_fit','F275W_sharp','F275W_exp_count','F275W_well_meas_count','F336W','F336W_RMS','F336W_fit','F336W_sharp','F336W_exp_count','F336W_well_meas_count','F438W','F438W_RMS','F438W_fit','F438W_sharp','F438W_exp_count','F438W_well_meas_count','F606W','F606W_RMS','F606W_fit','F606W_sharp','F606W_exp_count','F606W_well_meas_count','F814W','F814W_RMS','F814W_fit','F814W_sharp','F814W_exp_count','F814W_well_meas_count','Mem_Prob','RA','DEC','Star_ID','Star_iter_found'])
            df_hugs.to_csv(f'{data_dir}/{cluster}_hugs.csv', index=False)

            # delete the txt file
            os.remove(f'{data_dir}/{cluster}_hugs.txt')
        
        else:
            df_hugs = hscsearch_subfield(ra, dec, field_size=search_radius/60, subfield_size=2/60, table="summary", release='v3', magtype="magaper2", verbose=True, format='table')

            df_hugs['RA'] = df_hugs['MatchRA']
            df_hugs['DEC'] = df_hugs['MatchDec']
            df_hugs['Star_ID'] = df_hugs['MatchID']
            df_hugs['F275W'] = df_hugs['W3_F275W']
            df_hugs['F275W_RMS'] = df_hugs['W3_F275W_MAD']
            df_hugs['F336W'] = df_hugs['W3_F336W'].fillna(df_hugs['W2_F336W'])
            df_hugs['F336W_RMS'] = df_hugs['W3_F336W_MAD'].fillna(df_hugs['W2_F336W_MAD'])
            df_hugs['F438W'] = df_hugs['W3_F438W'].fillna(df_hugs['A_F435W'])
            df_hugs['F438W_RMS'] = df_hugs['W3_F438W_MAD'].fillna(df_hugs['A_F435W_MAD'])
            df_hugs['F606W'] = df_hugs['A_F606W'].fillna(df_hugs['W3_F606W']).fillna(df_hugs['W2_F606W'])
            df_hugs['F606W_RMS'] = df_hugs['A_F606W_MAD'].fillna(df_hugs['W3_F606W_MAD']).fillna(df_hugs['W2_F606W_MAD'])
            df_hugs['F814W'] = df_hugs['A_F814W'].fillna(df_hugs['W3_F814W']).fillna(df_hugs['W2_F814W'])
            df_hugs['F814W_RMS'] = df_hugs['A_F814W_MAD'].fillna(df_hugs['W3_F814W_MAD']).fillna(df_hugs['W2_F814W_MAD'])

            df_hugs.to_csv(f'{data_dir}/{cluster}_hugs.csv', index=False)

    df_hugs['err'] = 0.01 # assume 0.01 arcsec

    if astrometric_shifts is not None:
        print('Applying Astrometric Shifts: ', astrometric_shifts)
        df_hugs['RA'] = df_hugs['RA'] + astrometric_shifts[0]/3600
        df_hugs['DEC'] = df_hugs['DEC'] + astrometric_shifts[1]/3600

    table = Table.from_pandas(df_hugs[['RA', 'DEC', 'err', 'Star_ID']])

    table.write(f'{data_dir}/{cluster}_hugs.fits', overwrite=True)

    area = np.pi * (search_radius/60)**2

    os.system(f'python {nway_dir}nway-write-header.py {data_dir}/{cluster}_hugs.fits HUGS {area}')

    return df_hugs


def nway_merged_XMM_prepare(df_q,X_name,name_col='iauname',ra_col='sc_ra', dec_col='sc_dec',r0_col='sc_r0',r1_col='sc_r1',PA_col='sc_ang',data_dir='data',sigma=2):

    # if sigma == 2: # 95%
    #     sigma_factor = np.sqrt(np.log(20)*2)
    # elif sigma == 1: # ≈ 39.347% in 2-D   
    #     sigma_factor = 1
    # elif sigma == 3: # ≈ 39.347% in 2-D   
    #     sigma_factor = np.sqrt(np.log(1./0.0027)*2) 

    if sigma == 2: # 95%
        sigma_factor = 1.#np.sqrt(np.log(20)*2)
    elif sigma == 1: # ≈ 39.347% in 2-D   
        sigma_factor = 1./np.sqrt(np.log(20)*2)
    elif sigma == 3: # ≈ 39.347% in 2-D   
        sigma_factor = np.sqrt(np.log(1./0.0027)*2)/np.sqrt(np.log(20)*2)
   
    df_q['_4XMM'] = df_q[name_col]
    df_q['RA']  = df_q[ra_col]
    df_q['DEC'] = df_q[dec_col]
    
    df_q['ID'] = df_q.index + 1
    df_q['err_r0'] = df_q[r0_col]*sigma_factor
    df_q['err_r1'] = df_q[r1_col]*sigma_factor
    df_q['PA'] = df_q[PA_col]

    new_t = Table.from_pandas(df_q[['ID','RA','DEC','err_r0','err_r1','PA','_4XMM']]) # r0 is 95%, should be consistent with other PUs, 

    new_t.write(f'{data_dir}/{X_name}_XMM.fits', overwrite=True)

    area = 1328./656997
    
    os.system(f'python {nway_dir}nway-write-header.py {data_dir}/{X_name}_XMM.fits XMM {area}')
    
    return None


def nway_XMM_matching_merged_mw(args):

    (TD, i, name_col,ra_col,dec_col,r0_col,r1_col,PA_col,
        data_dir,explain,rerun,sigma, cp_prior) = args
    
    X_name, ra_X, dec_X, r0 = TD.loc[i, name_col][5:], TD.loc[i, ra_col], TD.loc[i, dec_col], TD.loc[i, r0_col]
    #try:

    r_3sigma = r0 * np.sqrt(np.log(1./0.0027)*2)/np.sqrt(np.log(20)*2) # ~1.4
    #r_search = r0
    if sigma == 2:
        r_search = r0
    elif sigma == 3:
        r_search = r_3sigma
        
    #clas = TD.loc[i, 'Class']
    

    if glob.glob(f'{data_dir}/{X_name}_MW_match.fits') == [] or rerun==True:
    
        print(i, X_name, ra_X, dec_X)

        mjds = np.array([TD.loc[i, 'mjd_first'], TD.loc[i, 'mjd_last']])
        
        if path.exists(f'{data_dir}/{X_name}_XMM.fits') == False or rerun==True:
            
            nway_merged_XMM_prepare(TD.iloc[[i]].reset_index(drop=True),X_name=X_name,name_col=name_col,ra_col=ra_col, dec_col=dec_col,r0_col=r0_col,r1_col=r1_col,PA_col=PA_col,data_dir=data_dir,sigma=sigma)

    
        if path.exists(f'{data_dir}/{X_name}_MW_crossmatch.fits') == False or rerun==True:

            nway_merged_mw_prepare(ra_X, dec_X,  X_name=X_name, ref_mjd=mjds, data_dir=data_dir,sigma=sigma, verbose=False)
            


        df_rad = Table.read(f'{data_dir}/{X_name}_MW_crossmatch.fits', format='fits').to_pandas()
        #print(df_rad)
        if len(df_rad[df_rad['sep']<=5])>0:
            r_2 = max(df_rad.loc[df_rad['sep']<=5, 'err0'])
            
        else:
            r_2 = 0
        #print(r_2, np.sqrt(r_search**2+r_2**2))
        os.system(f'python {nway_dir}nway.py {data_dir}/{X_name}_XMM.fits :err_r0:err_r1:PA {data_dir}/{X_name}_MW_crossmatch.fits :err0:err1:errPA \
            --out={data_dir}/{X_name}_MW_match.fits --radius {np.sqrt(r_search**2+r_2**2)} --prior-completeness {cp_prior}') # r0 is 2-sigma f=0.98 -> c=50

        if explain:

            os.system(f'python {nway_dir}nway-explain.py {data_dir}/{X_name}_MW_match.fits 1') 

    #except:
        #print(f'{X_name}......failed.')
    return X_name


def nway_CSC_matching_merged_mw(args):

    '''
    description: 

    cross-match a single X-ray source with a combined multiwavelength catalog using Gaia, TMASS, CatWISE2020, and AllWISE

    inputs:

    TD: dataframe of sources for crossmatching, does not necessary to be TD
    i: ith source from TD to be matched
    radius: the radius to search the CSC catalog for the per-observation MJDs, usually set to very small (0.01'~0.6"), where the coordinate is using ra_csc_col, dec_csc_col columns 
    query_dir: the query directory to save results from CSCviewsearch 
    name_col: 2CXO name column 
    ra_col,dec_col: ra and dec column names, this might be different from ra_csc_col, dec_csc_col if the coordinates have been updated from astrometry correction
    ra_csc_col,dec_csc_col: the original ra and dec column names
    PU_col: positional uncertainty column, usually the same as r0_col
    r0_col,r1_col,PA_col: semi-major, semi-minor, positional angle columns
    data_dir: directory to save the matching results 
    explain: if set to True, nway-explain.py will be run to produce visualziation plots of cross-matching 
    rerun: if set to rerun, the cross-matching will be rerun again 
    sigma: significance level for cross-matching between multiwavelength associations and Chandra, default set to 2-sigma (95%) 
    newcsc: create the X-ray fits file based on TD dataframe not from astroquery to CSC
    per_file: a self defined dataframe to provide per-observation MJD information, if set to 'txt', then will use CSCviewsearch
    self_mjd: a list of MJD for X-ray source or use CSCviewsearch if set to False
    cp_prior: prior-completeness for nway, cp_prior = caf/(1-caf), where caf (catalog association fraction) can be derived from A&A 674, A136 (2023)

    
    '''

    (TD, field_name, i, data_dir, name_col,ra_col,dec_col,r0_col,r1_col,PA_col,\
        explain,rerun,sigma,newcsc,mjds,cp_prior) = args

    csc_name, ra_X, dec_X,r0 = TD.loc[i, name_col][5:],TD.loc[i, ra_col],TD.loc[i, dec_col],TD.loc[i, r0_col]  #'r0']#err_ellipse_r0']#r0']
    #try:
    r_3sigma = r0 * np.sqrt(np.log(1./0.0027)*2)/np.sqrt(np.log(20)*2) # ~1.4
    #r_search = r0
    if sigma == 2:
        r_search = r0
    elif sigma == 3:
        r_search = r_3sigma
        
    nwaydata_dir = f'{data_dir}/nway'
    #clas = TD.loc[i, 'Class']
    try:

        if glob.glob(f'{nwaydata_dir}/{csc_name}_MW_match.fits') == [] or rerun==True:
            # print(i, csc_name, ra_X, dec_X)

            df_mjd = pd.read_csv(f'{data_dir}/{field_name}_wget.txt', comment='#', sep='\t', na_values=' '*9)
            df_mjd['name'] = df_mjd['name'].str.strip()
            df_mjd = df_mjd[df_mjd['name']==TD.loc[i, name_col]]
            df_mjd['mjd'] = np.nan
            df_mjd['mjd'] = df_mjd.apply(lambda r: Time(r['gti_obs'], format='isot', scale='utc').mjd,axis=1)
            mjds = df_mjd['mjd'].values
            
            if path.exists(f'{nwaydata_dir}/{csc_name}_CSC.fits') == False or rerun==True:
                if newcsc:
                    newcsc_prepare(TD.iloc[[i]].reset_index(drop=True),X_name=csc_name,name_col=name_col,ra_col=ra_col, dec_col=dec_col,r0_col=r0_col,r1_col=r1_col,PA_col=PA_col,data_dir=nwaydata_dir,sigma=sigma)
                else:
                    nway_mw_prepare_hierarchical_v3(ra_X, dec_X,  X_name=csc_name, ref_mjd=mjds, catalog='CSC',data_dir=nwaydata_dir,sigma=sigma)
        
            #'''
            if path.exists(f'{nwaydata_dir}/{csc_name}_MW_crossmatch.fits') == False or rerun==True:

                nway_merged_mw_prepare(ra_X, dec_X, X_name=csc_name, ref_mjd=mjds, data_dir=nwaydata_dir,sigma=sigma, verbose=False)
                

            df_rad = Table.read(f'{nwaydata_dir}/{csc_name}_MW_crossmatch.fits', format='fits').to_pandas()
            #print(df_rad)
            if len(df_rad[df_rad['sep']<=3])>0:
                r_2 = max(df_rad.loc[df_rad['sep']<=3, 'err0'])
                
            else:
                r_2 = 0
            #print(r_2, np.sqrt(r_search**2+r_2**2))
            os.system(f'python {nway_dir}nway.py {nwaydata_dir}/{csc_name}_CSC.fits :err_r0:err_r1:PA {nwaydata_dir}/{csc_name}_MW_crossmatch.fits :err0:err1:errPA \
                --out={nwaydata_dir}/{csc_name}_MW_match.fits --radius {np.sqrt(r_search**2+r_2**2)} --prior-completeness {cp_prior}') # r0 is 2-sigma f=0.93 -> c=13

            if explain:

                os.system(f'python {nway_dir}nway-explain.py {nwaydata_dir}/{csc_name}_MW_match.fits 1 None') 
        return csc_name
    except:
        print(f'{csc_name}......failed.')
        return csc_name


def nway_CSC_matching_merged_mw_cluster(args):

    '''
    description: 

    cross-match a field of X-ray sources with a combined multiwavelength catalog using Gaia, TMASS, CatWISE2020, and AllWISE

    inputs:

    TD: dataframe of sources for crossmatching, does not necessary to be TD
    ra: ra of the X-ray source
    dec: dec of the X-ray source
    radius: the radius to search the CSC catalog for sources
    query_dir: the query directory to save results from CSCviewsearch 
    field_name: the name of the CSC field
    name_col: 2CXO name column 
    ra_col,dec_col: ra and dec column names, this might be different from ra_csc_col, dec_csc_col if the coordinates have been updated from astrometry correction
    ra_csc_col,dec_csc_col: the original ra and dec column names
    PU_col: positional uncertainty column, usually the same as r0_col
    r0_col,r1_col,PA_col: semi-major, semi-minor, positional angle columns
    data_dir: directory to save the matching results 
    explain: if set to True, nway-explain.py will be run to produce visualziation plots of cross-matching 
    rerun: if set to rerun, the cross-matching will be rerun again 
    sigma: significance level for cross-matching between multiwavelength associations and Chandra, default set to 2-sigma (95%) 
    newcsc: create the X-ray fits file based on TD dataframe, not from astroquery to CSC
    per_file: a self defined dataframe to provide per-observation MJD information, if set to 'txt', then will use CSCviewsearch
    self_mjd: a list of MJD for X-ray source, or use CSCviewsearch if set to False
    cp_prior: prior-completeness for nway, cp_prior = caf/(1-caf), where caf (catalog association fraction) can be derived from A&A 674, A136 (2023)
    csc_version: the version of CSC catalog, '2.0' or 'current' for 2.1
    globular_cluster: if set to True, then will crossmatch to HUGS catalog instead of Gaia, TMASS, CatWISE2020, and AllWISE
    '''

    (ra, dec, radius, query_dir, field_name, name_col,ra_col,dec_col, ra_csc_col,dec_csc_col,PU_col,r0_col,r1_col,PA_col,\
        data_dir,explain,rerun,sigma,newcsc,per_file,self_mjd,cp_prior,csc_version,globular_cluster,astrometric_shifts) = args
        
    #clas = TD.loc[i, 'Class']
    # try:

    if glob.glob(f'{data_dir}/{field_name}_MW_match.fits') == [] or rerun==True:
        # print(csc_name, ra_X, dec_X)
        if type(per_file) == pd.DataFrame:
            df_csc = per_file.reset_index(drop=True)   
        elif type(per_file) == str and per_file == 'txt':
            # download CSC data
            if path.exists(f'{data_dir}/{field_name}_curl.txt') == False:
                CSCviewsearch(field_name, ra, dec, radius, query_dir, csc_version=csc_version)
                print('test')
            df_csc = pd.read_csv(f'{data_dir}/{field_name}_curl.txt', comment='#', sep='\t', na_values=' '*9)
        else:
            print('else')
            
        df_csc['mjd'] = df_csc.apply(lambda r: Time(r['gti_obs'], format='isot', scale='utc').mjd,axis=1)
        mjds = df_csc['mjd'].values

        df_csc = df_csc.drop_duplicates(subset=['name'], keep='first').reset_index(drop=True)
        df_csc = df_csc.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        df_csc['ID'] = df_csc.index + 1
        # df_csc = df_csc.rename(columns={'name': '_2CXO', 'ra': 'RA', 'dec': 'DEC', 'err_ellipse_r0': 'err_r0', 'err_ellipse_r1': 'err_r1', 'err_ellipse_ang': 'PA', 'separation': '_r', 'extent_flag': 'fe', 'conf_flag': 'fc'})
        df_csc = df_csc.rename(columns={'ra': 'RA', 'dec': 'DEC',})
        coords = SkyCoord(ra=df_csc['RA'], dec=df_csc['DEC'], unit=(u.hourangle, u.deg))
        df_csc['RA'] = coords.ra.deg
        df_csc['DEC'] = coords.dec.deg
        df_csc = df_csc.apply(pd.to_numeric, errors='ignore')
        table_csc = Table.from_pandas(df_csc[['name','ID','RA','DEC','err_ellipse_r0','err_ellipse_r1','err_ellipse_ang','separation','extent_flag','conf_flag']])

        table_csc.write(f'{data_dir}/{field_name}_CSC.fits', overwrite=True)

        area = 550./317000

        os.system(f'python {nway_dir}nway-write-header.py {data_dir}/{field_name}_CSC.fits CSC {area}')

        if globular_cluster:
            nway_globular_cluster_prepare(ra, dec, cluster=field_name, search_radius=radius, data_dir=data_dir, sigma=sigma, verbose=True, astrometric_shifts=astrometric_shifts)
            
            # HUGS doesn't have position errors
            os.system(f'python {nway_dir}nway.py {data_dir}/{field_name}_CSC.fits :err_ellipse_r0:err_ellipse_r1:err_ellipse_ang {data_dir}/{field_name}_hugs.fits :err:err:err \
                --out={data_dir}/{field_name}_MW_match.fits --radius 0.5 --prior-completeness {cp_prior}') # r0 is 2-sigma f=0.93 -> c=13
        
        else: 
            # download and crossmatch multiwavelength catalogs
            if path.exists(f'{data_dir}/{field_name}_MW_crossmatch.fits') == False or rerun==True:

                nway_merged_mw_prepare(ra, dec, X_name=field_name, search_radius=radius, ref_mjd=mjds, data_dir=data_dir,sigma=sigma, verbose=True)

            #print(r_2, np.sqrt(r_search**2+r_2**2))
            os.system(f'python {nway_dir}nway.py {data_dir}/{field_name}_CSC.fits :err_ellipse_r0:err_ellipse_r1:err_ellipse_ang {data_dir}/{field_name}_MW_crossmatch.fits :err0:err1:errPA \
                --out={data_dir}/{field_name}_MW_match.fits --radius 2.0 --prior-completeness {cp_prior}') # r0 is 2-sigma f=0.93 -> c=13
    else:
        df_csc = Table.read(f'{data_dir}/{field_name}_CSC.fits', format='fits').to_pandas()

    if explain:
        for i in range(len(df_csc)):
            os.system(f'python {nway_dir}nway-explain.py {data_dir}/{field_name}_MW_match.fits {i+1} CSC_name')
        
    return field_name
    # except Exception as e:
    #     print(f'Error for {field_name}: {e}')
    #     return field_name

def nway_merged_eRASS_prepare(df_q,X_name,name_col='name',ra_col='RA', dec_col='DEC',PU_col='POS_ERR',data_dir='data',sigma=3):

    # if sigma == 2: # 95%
    #     sigma_factor = np.sqrt(np.log(20)*2)
    # elif sigma == 1: # ≈ 39.347% in 2-D   
    #     sigma_factor = 1
    # elif sigma == 3: # ≈ 39.347% in 2-D   
    #     sigma_factor = np.sqrt(np.log(1./0.0027)*2) 

    if sigma == 1: # 95%
        sigma_factor = 1.#np.sqrt(np.log(20)*2)
    elif sigma == 2: # ≈ 39.347% in 2-D   
        sigma_factor = np.sqrt(np.log(20)*2)/np.sqrt(np.log(1./0.32)*2)
    elif sigma == 3: # ≈ 39.347% in 2-D   
        sigma_factor = np.sqrt(np.log(1./0.0027)*2)/np.sqrt(np.log(1./0.32)*2)
   
    df_q['_eRASS'] = df_q[name_col]
    df_q['RA']  = df_q[ra_col]
    df_q['DEC'] = df_q[dec_col]
    
    df_q['ID'] = df_q.index + 1
    df_q['PU'] = df_q[PU_col]*sigma_factor
    # df_q['err_r1'] = df_q[r1_col]*sigma_factor
    # df_q['PA'] = df_q[PA_col]

    new_t = Table.from_pandas(df_q[['ID','RA','DEC','PU','_eRASS']]) # r0 is 95%, should be consistent with other PUs, 

    new_t.write(f'{data_dir}/{X_name}_eRASS.fits', overwrite=True)

    area = 41253./(2*903521) # 1328./656997
    
    os.system(f'python {nway_dir}nway-write-header.py {data_dir}/{X_name}_eRASS.fits eRASS {area}')
    
    return None

def nway_eRASS_matching_merged_mw(args):

    (TD, i, name_col,ra_col,dec_col,PU_col,
        data_dir,explain,rerun,sigma, cp_prior) = args
    
    X_name, ra_X, dec_X, r0 = TD.loc[i, name_col], TD.loc[i, ra_col], TD.loc[i, dec_col], TD.loc[i, PU_col]
    #try:

    # r_3sigma = r0 * np.sqrt(np.log(1./0.0027)*2)/np.sqrt(np.log(20)*2) # ~1.4
    #r_search = r0
    if sigma == 2:
        r_search = r0 * np.sqrt(np.log(20.)*2)/np.sqrt(np.log(1./0.32)*2)
    elif sigma == 3:
        r_search = r0 * np.sqrt(np.log(1./0.0027)*2)/np.sqrt(np.log(1./0.32)*2)
        
    #clas = TD.loc[i, 'Class']
    

    if glob.glob(f'{data_dir}/{X_name}_MW_match.fits') == [] or rerun==True:
    
        print(i, X_name, ra_X, dec_X)

        mjds = np.array([TD.loc[i, 'MJD_MIN'], TD.loc[i, 'MJD_MAX']])
        
        if path.exists(f'{data_dir}/{X_name}_eRASS.fits') == False or rerun==True:
            
            nway_merged_eRASS_prepare(TD.iloc[[i]].reset_index(drop=True),X_name=X_name,name_col=name_col,ra_col=ra_col, dec_col=dec_col,PU_col=PU_col,data_dir=data_dir,sigma=sigma)

    
        if path.exists(f'{data_dir}/{X_name}_MW.fits') == False or rerun==True:

            nway_merged_mw_prepare(ra_X, dec_X,  X_name=X_name, ref_mjd=mjds, data_dir=data_dir,sigma=sigma, verbose=False)
            


        df_rad = Table.read(f'{data_dir}/{X_name}_MW.fits', format='fits').to_pandas()
        #print(df_rad)
        if len(df_rad[df_rad['sep']<=10])>0:
            r_2 = max(df_rad.loc[df_rad['sep']<=10, 'err0'])
            
        else:
            r_2 = 0
        #print(r_2, np.sqrt(r_search**2+r_2**2))
        os.system(f'python {nway_dir}nway.py {data_dir}/{X_name}_eRASS.fits :PU {data_dir}/{X_name}_MW.fits :err0:err1:errPA \
            --out={data_dir}/{X_name}_MW_match.fits --radius {np.sqrt(r_search**2+r_2**2)} --prior-completeness {cp_prior}') # r0 is 2-sigma f=0.98 -> c=50

        if explain:

            os.system(f'python {nway_dir}nway-explain.py {data_dir}/{X_name}_MW_match.fits 1') 

    #except:
        #print(f'{X_name}......failed.')
    return X_name
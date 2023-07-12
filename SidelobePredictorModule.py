# %%
"""
---
### This notebook contains the code to identify sidelobe streaks for a given subtile based on the uv-plane of the subtile
---
"""

# %%
# import local module containing all global variables
import GlobalVariables

# %%
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord
import numpy as np
from itertools import combinations
import astropy.units as units
import pandas as pd
import glob

# %%
def antenna_selection(configuration):
    
    ''' Given the configuration of the subtile, the function returns the 
        antennas and their locations that were used for a given configuration.
        Values taken from: https://science.nrao.edu/facilities/vla/docs/manuals/oss/ant_positions.pdf'''
        
    # select all antennas that has B in its config
    if configuration.upper() == 'B':
        antennae_list = GlobalVariables.antenna[np.where([[configuration[0].upper() in x] 
                                 for x in GlobalVariables.antenna['config']])[0]]
        
    # since the North arm is the same as A config but East and West arms are the same as B config
    # select these two separately
    else:
        # North arm
        arm_select = np.where([['N' in x] 
                               for x in GlobalVariables.antenna['arm']])[0]
        antennaN   = GlobalVariables.antenna[arm_select]
        selectN    = np.where([[configuration[2].upper() in x] 
                               for x in antennaN['config']])[0]
        
        # East and West arm
        arm_select = np.where([['N' not in x]
                               for x in GlobalVariables.antenna['arm']])[0]
        antennaEW  = GlobalVariables.antenna[arm_select]
        selectEW   = np.where([[configuration[0].upper() in x] 
                               for x in antennaEW['config']])[0]
        antennae_list = np.append(antennaEW[selectEW],antennaN[selectN])
    
    return antennae_list

# %%
def uvwCalc(L_x, L_y, L_z, obsDEC, ha):
    ''' Returns the (u, v, w) coordinates of the antennas given the (x, y, z), dec and hour angle.
        Formula in the Synthesis Imaging book.'''
    
    # wavelength of observation
    wavelength = GlobalVariables.c/GlobalVariables.frequence
    
    # rotation matrix defined by the book
    rotationMatrix = np.array([
    [                np.sin(ha),                 np.cos(ha),              0.],
    [-np.sin(obsDEC)*np.cos(ha),  np.sin(obsDEC)*np.sin(ha),  np.cos(obsDEC)],
    [ np.cos(obsDEC)*np.cos(ha), -np.cos(obsDEC)*np.sin(ha),  np.sin(obsDEC)]
    ])
    
    # every pair of antennas
    combX = list(combinations(L_x, 2))
    combY = list(combinations(L_y, 2))
    combZ = list(combinations(L_z, 2))
    
    u, v, w = [], [], []
    
    for x, y, z in zip(combX, combY, combZ):
        baseline = np.array([x[0]-x[1], y[0]-y[1], z[0]-z[1]])
        (tempU, tempV, tempW) = np.transpose(np.matmul(rotationMatrix, baseline)/wavelength)
        
        # for given (x,y,z) uv sampling will have points in the opposite quadrant too.
        # the distance btw two antennas can be measured in both ways.
        u.extend([tempU, -tempU])
        v.extend([tempV, -tempV])
        w.extend([tempW, -tempW])
    
    return zip(u, v, w)

# %%
def SidelobePredictor(subtile_identifier, system, screen=''):
    
    ''' Given the subtile_identifier, the function to predict the 
        sidelobe streaks angles based on uv-plane of the image. '''
    
    
    # output variable
    sidelobe_angles = np.zeros(3)
    
    # to ensure that the code is consistent with the one of the local system, a uniserval code is created with a flag.
    # based on the flag, the version of the code is chosen
    # here the only different btw the two code is the extraction of the image_fits_loc
    if system == 'server':
        # given subtile_identifier get the OBSRA, OBSDEC and DATE-OBS from the fits file
        image_fits_loc = GlobalVariables.image_fits_list[GlobalVariables.image_fits_list.str.contains(subtile_identifier, regex=False)].values[0]

    if system == 'local':
        # given subtile_identifier get the OBSRA, OBSDEC and DATE-OBS from the fits file
        image_fits_loc = GlobalVariables.image_fits_list[GlobalVariables.image_fits_list.str.contains(subtile_identifier[:9]+'ql.'+subtile_identifier[9:], regex=False)].values[0]
        
    if system == 'local_server':
        # given subtile_identifier get the OBSRA, OBSDEC and DATE-OBS from the fits file
        
        # get the image and rms image location (screen folders)
        signal_file_list = pd.Series(glob.glob('/Users/suhasini/Documents/ArtifactIdentification/safetyfolder/expanded_server/fits/image/screen'+str(screen)+'/*'))
        rms_file_list = pd.Series(glob.glob('/Users/suhasini/Documents/ArtifactIdentification/safetyfolder/expanded_server/fits/rms/screen'+str(screen)+'/*'))
        
        image_fits_loc = signal_file_list[signal_file_list.str.contains(subtile_identifier, regex=False)].values[0]
    
    with fits.open(image_fits_loc) as hdul:
        obsRA, obsDEC = hdul[0].header['OBSRA'], hdul[0].header['OBSDEC'] * units.deg
        obsTime = Time(hdul[0].header['DATE-OBS'])
            
    # phase center of the subtile in observation
    phaseCentre = SkyCoord(obsRA, obsDEC, frame='icrs', unit='deg')
    
    # local sidreal time
    lst = (Time(obsTime, format='isot', scale='utc',location=(GlobalVariables.VLA_loc.lon, GlobalVariables.VLA_loc.lat))).sidereal_time('apparent')
    
    # hour angle of the subtile
    ha = lst - phaseCentre.ra
    
    executionblock_csv = pd.read_csv(GlobalVariables.executionblock_csv_loc, sep=',')
    
    # get the configuration of VLA at the time this subtile was imaged
    configuration = executionblock_csv[(executionblock_csv['ObsSTART ISOT'] <= obsTime) & 
                                      (obsTime <= executionblock_csv['ObsEND ISOT'])
                                     ]['Array Config'].values[0]
        
    # since some configuration are in transition,
    # we change all B-->BNA to B
    # and BNA-->A to BNA
    if configuration == 'B->BnA':
        configuration = 'B'
    
    if configuration == 'BnA->A':
        configuration = 'BNA'
        
    # select the antennas given the configuration
    antennae = antenna_selection(configuration)
    
    # antennas in each arm are given one after the other. Choose one arm each time.
    # each arm has 9 antennas
    for i in np.arange(3):
        
        minAnt = i * 9
        maxAnt = (i + 1) * 9
        
        # choose antennas in each arm
        L_x_jj = (antennae['L_x'])[minAnt:maxAnt]
        L_y_jj = (antennae['L_y'])[minAnt:maxAnt]
        L_z_jj = (antennae['L_z'])[minAnt:maxAnt]
        
        # taken the median distance of each arm from the center
        x1, y1, z1 = np.array([1,
                     np.median((L_y_jj[:-1]-L_y_jj[-1])/
                               (L_x_jj[:-1]-L_x_jj[-1])),
                     np.median((L_z_jj[:-1]-L_z_jj[-1])/
                               (L_x_jj[:-1]-L_x_jj[-1]))
                          ])
                               
        # get the angle (in PA) of each arm from the center
        x = [x1, 0]
        y = [y1, 0]
        z = [z1, 0]
                        
        # get the u,v,w of these 3 points 
        (u, v, w) = zip(*uvwCalc(x, y, z, obsDEC, ha))
        
        # Currently returns the mathematically defined PA
        # if East is left and North is up
        sidelobe_angles[i] = np.arctan2(u[0], v[0])*180/np.pi
        
    # This adds 90 to make it astronomically defined PA and then does
    # a modulus trick that should ensure output is 0-180 degrees.
    sidelobe_angles = np.mod(sidelobe_angles + 450, 180)
    
    # round off to 3 digits after decimal points
    sidelobe_angles = [round(item, 3) for item in sidelobe_angles]
    
    return sidelobe_angles
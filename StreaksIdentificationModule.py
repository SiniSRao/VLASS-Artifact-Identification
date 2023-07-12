# %%
"""
---
### This notebook contains the code to identify streak artifacts and their width
---
"""

# %%
import pandas as pd
import numpy as np
from astropy.wcs import WCS
from astropy.io import fits
from astropy.stats import mad_std
import skimage
from skimage.transform import hough_line
from astropy.coordinates import SkyCoord
from scipy.signal import find_peaks, peak_widths, peak_prominences
from photutils.aperture import RectangularAperture, aperture_photometry, CircularAperture, ApertureStats, ApertureMask
from photutils import aperture
import matplotlib.pyplot as plt
import time
import glob

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
pd.set_option('display.max_rows', None)

# %%
# import local module containing all global variables
import GlobalVariables

# %%
def clipping(data):
    ''' Given the data, the function returns a thresholded/clipped image.
        Clipping the data to 1.5 sigma.
        We use mad_std to get sigma -- more robust than regular std'''
    
    # sigma for clipping
    clipping_sigma =  1.64   # 90th percentile
    
    # clipping/thresholding
    clipping_thresh = np.median(data) + mad_std(data)*clipping_sigma
    
    positive_space = np.where(data > clipping_thresh, 1, 0)
    negative_space = np.where(data < -clipping_thresh, 1, 0)
    
    return (positive_space, negative_space)

# %%
def getComponentsCoord(tile, subtile, subtile_bmaj, subtile_bmin, wcs):
    ''' Given the subtile's details, the function returns the physical coordinates 
        of the all the components in a given subtile.
        Components that have their bmaj and bmin < subtile's bmaj and bmin gets 
        replaced with the subtile's bmaj and bmin''' 
    
    # given the declination, find the csv file
    csv_file = pd.read_csv(GlobalVariables.selected_tile_loc + tile + '.csv', usecols=['RA', 'DEC', 'Subtile', 'Maj_img_plane', 'Min_img_plane', 'PA_img_plane'])
    
    # find all components in the csv that have the same subtile
    component_csv = csv_file[csv_file['Subtile'] == subtile]
        
    # Xposn and Yposn from the catalog has a offset that does not match the RA and DEC
    # calculate the Xposn and Yposn from the RA and DEC given
    Xposn, Yposn = [], []
    for ra, dec in np.array(component_csv[['RA', 'DEC']]):
        x, y = wcs.world_to_pixel(SkyCoord(ra, dec, unit='deg'))
        Xposn.append(float(x))
        Yposn.append(float(y))
        
    component_csv.insert(2, 'Xposn', Xposn)
    component_csv.insert(3, 'Yposn', Yposn)
        
    # replace all bmaj and bmin < subtile's bmaj and bmin with subtile's bmaj and bmin
    # if either of the two is less than subtile's values, replace both
    component_csv.loc[component_csv['Maj_img_plane'] < subtile_bmaj, 'Maj_img_plane'] = subtile_bmaj
    component_csv.loc[component_csv['Maj_img_plane'] < subtile_bmaj, 'Min_img_plane'] = subtile_bmin
    
    component_csv.loc[component_csv['Min_img_plane'] < subtile_bmin, 'Min_img_plane'] = subtile_bmin
    component_csv.loc[component_csv['Min_img_plane'] < subtile_bmin, 'Maj_img_plane'] = subtile_bmaj
    
    # reset the index of the returning df
    component_csv = component_csv.reset_index()
        
    # since some components have their bmaj regions intersecting, the component with the smaller bmaj will use the larger component's bmaj to calculate the width of the peak
    # as the annular region is small for the smaller bmaj component, the entire width is not gotten from the s=r*theta calc. therefore we will use a bigger r to get the s.
    # the corrected bmaj will only be used to calculate the width. Nowhere else.
    component_csv.insert(7, 'Overlapping_maj_flag', [0]*component_csv.shape[0])
    component_csv.insert(8, 'Overlapping_maj_corrected', [0]*component_csv.shape[0])
    
    # compare every component with every other component
    for idx1, row1 in component_csv.iterrows():
        for idx2, row2 in component_csv.iterrows():
            
            # to make sure you are not comparing a component with itself
            if idx1 != idx2:
                
                # find the distance btw the two component centers
                d = round(np.sqrt((row1['Xposn'] - row2['Xposn'])**2 + (row1['Yposn'] - row2['Yposn'])**2))
                
                # check if the two bmaj circles are intersecting
                if d <= round(row1['Maj_img_plane'] + row2['Maj_img_plane']):
                    
                    # if intersecting, find the smaller bmaj component
                    if row1['Maj_img_plane'] < row2['Maj_img_plane']:
                        to_change_idx = idx1
                        to_value = row2['Maj_img_plane']
                    else:
                        to_change_idx = idx2
                        to_value = row1['Maj_img_plane']
                
                    # change the smaller bmaj component's bmaj to the bigger one's. Also flag that component.
                    # here we are adding a corrected maj column and not actually modifying the bmaj
                    component_csv.loc[to_change_idx, 'Overlapping_maj_flag'] = 1
                    component_csv.loc[to_change_idx, 'Overlapping_maj_corrected'] = max(component_csv.iloc[to_change_idx]['Overlapping_maj_corrected'], to_value)
    
    t = component_csv[['RA', 'DEC', 'Xposn', 'Yposn', 'Maj_img_plane', 'Min_img_plane', 'PA_img_plane', 'Overlapping_maj_flag', 'Overlapping_maj_corrected']]
    
    return (component_csv[['RA', 'DEC', 'Xposn', 'Yposn', 'Maj_img_plane', 'Min_img_plane', 'PA_img_plane', 'Overlapping_maj_flag', 'Overlapping_maj_corrected']])

# %%
def componentsMasking(component_coord, shape):
    ''' Given the component_coord df containing all details of all 
        components identified by pyBDF in that subtile, the function 
        returns a mask that masks all the components. '''
    
    comps_mask = np.zeros(shape)
    
    # iterate through each component to create mask
    for x, y, comp_maj, comp_min, pa in np.array(component_coord[['Xposn', 'Yposn', 'Maj_img_plane', 'Min_img_plane', 'PA_img_plane']]):
        
        semi_maj = comp_maj/2   # comp_maj is the major axis of the component. We need semi major axis to draw ellipse
        semi_min = comp_min/2   # comp_min is the minor axis of the component. We need semi minor axis to draw ellipse

        # draw ellipse to mask each component
        rr, cc = skimage.draw.ellipse(y, x, semi_maj*2, semi_min*2, rotation=(180-pa)*np.pi/180., shape=shape)   # the x and y are swapped because its more like row and col. row --> y and col --> x
                                                                                                # here we have used twice the size of the semi major and minor axis as the radius
                                                                                                # to make sure the whole compoenent is masked
        # generate the mask
        comps_mask[rr, cc] = 1
        
    return comps_mask

# %%
def annulusMaskGenerator(component_coordinates, shape, annulus_radii_factor):
    ''' Returns a annulus mask of a given outer and inner radius.
        The radius is usually m and n times the major axis of the component (pixel coord) where m > n'''
    
    # In VLASS quicklook images, 1 pixel = 1 arcsec. Calculation -- CDELT2 = deg per pixel (2.7778e-04). Converting from deg/pixel to arcsec/pixel gives ~ 1
    # In the quicklook paper, the peak to ring ratio is taken with an inner and outer radius of 5" and 10" respectively. 
    # But since that is too small (some comps have bigger maj than that), we currently have 5x and 50x of the maj of th comp
    inner_radius_factor, outer_radius_factor = annulus_radii_factor # here it is, inner_radius and outer_radius is that many times the semi major axis of the component
    
    # source mask region
    # given in csv -- major axis (FWHM). Need, semi major axis. Hence component_coord.iloc[index][2]/2
    comp_sma = component_coordinates[2]/2
    
    # calculate radii
    inner_radius, outer_radius = comp_sma * inner_radius_factor, comp_sma * outer_radius_factor
    
    # get the indicies of the disk
    outer_rr, outer_cc = skimage.draw.disk((component_coordinates[1], component_coordinates[0]),
                                            outer_radius, shape=shape)

    inner_rr, inner_cc = skimage.draw.disk((component_coordinates[1], component_coordinates[0]),
                                            inner_radius, shape=shape)
    
    # use the indicies to generate mask
    annulus_mask = np.ones(shape)

    annulus_mask[outer_rr, outer_cc] = 0
    annulus_mask[inner_rr, inner_cc] = 1
    
    return annulus_mask

# %%
def getAnnulusHTS(posSpace_src_resp_func, negSpace_src_resp_func, bkg_resp_func, tested_angles):
    ''' Returns the hough transform of src and bkg response function and the used angle and distance'''
    
    posSpace_src_resp_HTS = hough_line(posSpace_src_resp_func, tested_angles)[0]
    negSpace_src_resp_HTS = hough_line(negSpace_src_resp_func, tested_angles)[0]
    bkg_resp_HTS = hough_line(bkg_resp_func, tested_angles)[0]
    
    return (posSpace_src_resp_HTS, negSpace_src_resp_HTS, bkg_resp_HTS)

# %%
def histogram_scaling(src_histogram, bkg_histogram):
    ''' Returns a scaled bkg_histogram to match the src_histogram.
        The mean of the noise in the bkg_subtracted_histogram is calculated using an iterative process.
        This mean is added to the bkg_histogram'''
    
    # initially, scaled_bkg_histogram is the same as the bkg_histogram since the mean has not been added has been done
    scaled_bkg_histogram = bkg_histogram
    
    # number of iteration
    ITMAX = 3
    
    for i in range(ITMAX):
    
        # condition is to keep track of the while loop. If the mean does not change btw iterations, is a NaN or runs for more than 3 times, the loop is terminated
        # old_mean is to keep track of the mean to check the variation of the means btw iterations
        # counter is to count the number of iterations
        condition, old_noise_mean, noise_mean = True, 0, 0

        # background subtraction
        scaled_bkg_subtracted_src_histogram = src_histogram - scaled_bkg_histogram

        # initally, all values are considered to calculate the rms
        values_below_rms_limit = scaled_bkg_subtracted_src_histogram

        # noise_sigma * rms will be the limit below which will be considered as signal with noise
        noise_sigma = 3
        
        # loop to calculate mean of the noise
        while condition == True:
            # rms and limit calculation
            noise_rms = 1.4826 * np.median(np.abs(values_below_rms_limit))   # converts from (rms version of) mad to std
            noise_limit = noise_rms * noise_sigma
            
            # only values which as noise
            values_below_rms_limit = scaled_bkg_subtracted_src_histogram[np.abs(scaled_bkg_subtracted_src_histogram) < noise_limit]
            
            # mean of the signal with noise
            noise_mean = np.mean(values_below_rms_limit)
            # print(noise_mean)
            
            # if the mean crosses the iteration number limit, is NaN or does not change in value btw iterations, the loop terminates
            if old_noise_mean == noise_mean or np.isnan(noise_mean) == True:
                condition = False
            else:
                # preparation for the next iteration
                old_noise_mean = noise_mean
        
        # scaling: "add" the mean of the noise from the src histogram
        scaled_bkg_histogram = scaled_bkg_histogram * (1 + old_noise_mean/np.median(scaled_bkg_histogram))

    return scaled_bkg_histogram

# %%
def findingPeaks(src_histogram, bkg_histogram, component_coordinates):
    ''' Returns the angles and angle index at which we detect peaks in the bkg subtracted src histogram.
        We smoothen the bkg subtracted src histogram to lower false positive detection.'''
        
    # standardizing to get both histograms in the same range
    # iterative method used to find the mean of the noise in the bkg, which is added to the bkg to bring it up to the src level (same scale)
    scaled_bkg_histogram = histogram_scaling(src_histogram, bkg_histogram)
    
    # bkg subtraction
    bkg_subtracted_src_histogram = src_histogram - scaled_bkg_histogram
            
    # smoothening the bkg_subtracted_src_histogram
    angle_per_pixel = 0.2
    smoothening_sigma_deg = 1 # smoothening size in degrees

    smoothening_sigma = smoothening_sigma_deg/angle_per_pixel
    
    smoothened_bkg_subtracted_src_histogram = skimage.filters.gaussian(bkg_subtracted_src_histogram, sigma=smoothening_sigma, preserve_range=True)
    
    #threshold for finding peaks in the smoothened_bkg_subtracted_src_histogram
    threshold_sigma = 5
    peak_detection_threshold = mad_std(smoothened_bkg_subtracted_src_histogram)*threshold_sigma
    
    # since find_peaks algorithm cannot detect peaks at the beginning and the end of an array,
    # the solution is to roll the array and input that to the findpeaks function -- mosts the identified peak indices by 0.2 deg (i.e., 1 pixel)
    # move back the indices by one to compensate for the earlier rolling of the array ny 1 to detect peaks at ends. Visa vera is also true
    # peak finding -- we use find_peaks from scipy --the height in the function is w.r.t 0
    peaks_index = np.concatenate((
                            find_peaks(np.roll(smoothened_bkg_subtracted_src_histogram, 1), height=peak_detection_threshold, distance=10/angle_per_pixel)[0] - 1,
                            find_peaks(np.roll(smoothened_bkg_subtracted_src_histogram, -1), height=peak_detection_threshold, distance=10/angle_per_pixel)[0] + 1
    ))
    
    # remove duplicate angle peak indices
    peaks_index = np.unique(peaks_index)
    
    # angles in radians (and then converted to deg)
    # round of the angles to 1 digit after decimal point
    peaks_angle = [round(i, 1) for i in GlobalVariables.tested_angles[peaks_index] * 180/np.pi]
    
    peaks_angle = np.array(peaks_angle)

    return (peaks_angle, peaks_index)

# %%
def findingPeaksWidth(src_HTS, angle_peaks, angle_peaks_index, HTS_d, comp_d_values, component_coordinates):
    ''' Given the source response function's HTS, peak angles (and its indices)
        dist array from HT algorithm and the dist of the component from the origin,
        this function finds the width of the streaks (only those that pass through
        the component). '''
    
    streak_width, streak_coord = [], []
    
    # for each angle (streak) identified, find the width of the streak.
    # in the HTS, at the peak angle column the line's pixels create a very noticable peak of high counts. 
    # Importantantly, the width of this peak is the the width of the streak.
    for i in range(len(angle_peaks_index)):
        
        # get the identified angle's column in the HTS
        peak_column = src_HTS[:, angle_peaks_index[i]]
        
        # since we know the position of the each component, we know where it lines in the HT (distance axis) for a given angle.
        # comp_d_values provide the xcos + ysin at each peak angle theta. Find the index of the peak in the HTS_d array
        comp_d = comp_d_values[i]
        comp_d_index = np.where(HTS_d == round(comp_d))[0][0]
        
        # sometimes there can we other streaks in the annular positive space region which might be brighter. In such cases to avoid identifying their width as the width of the streak,
        # we will only search for streaks that are within 2 arcmin of the component position
        window_width = 120                                         # 2 arcmin total width with 1 arcmin on either side of the source.
        peak_column_src_centered = peak_column[comp_d_index - round(window_width/2) : comp_d_index + round(window_width/2)]
        d_src_centered = HTS_d[comp_d_index - round(window_width/2) : comp_d_index + round(window_width/2)]
                
        # Identify only the peak that is 3 sigma above the std
        # only use nonzero values because most of the values in the column are 0. This is bias the calculated std
        column_peak_threshold = np.median(peak_column[peak_column != 0]) + mad_std(peak_column[peak_column != 0]) * 3
        
        # high counts in the column is created by the streak itself. find the peak of this column
        # sometimes multiple peaks maybe found. In that case only the peak that is above the threshold and has a width that passes through the component is considered.
        # To do that we need to make sure that the peak's width covers the r of the source i.e., center of the peak_column_src_centered
        column_highest_count_indices, column_highest_count_values = find_peaks(peak_column_src_centered, height=column_peak_threshold)

        # find the width of th streaks
        # to find the width of the streak, the rel_height at which the width needs to be calculated must be evaluated
        rel_height = 0.5    # since 0.5 is FWHM
        
        # if the identified peaks and their width don't go over the comp (center of the peak_column_src_centered) then reject it
        column_allpeak_widths, _, column_allpeak_width_leftedges, column_allpeak_width_rightedges = (peak_widths(peak_column_src_centered, column_highest_count_indices, rel_height=rel_height)[0:])
        
        # select all column peaks that contains the comp_d.
        valid_column_allpeak_widths = [ width
                                        for peak, width, left, right in zip(column_highest_count_indices, column_allpeak_widths, column_allpeak_width_leftedges, column_allpeak_width_rightedges) 
                                        if (abs(abs(comp_d - d_src_centered[peak]) * np.cos(angle_peaks[i]*np.pi/180.)) < component_coordinates[2]) & 
                                            (abs(abs(comp_d - d_src_centered[peak]) * np.sin(angle_peaks[i]*np.pi/180.)) < component_coordinates[2])
                                      ]
                
        # valid_column_allpeak_widths = [ width
        #                                 for width, left, right in zip(column_allpeak_widths, column_allpeak_width_leftedges, column_allpeak_width_rightedges) 
        #                                 if d_src_centered[np.floor(left).astype(int)] < comp_d < d_src_centered[np.ceil(right).astype(int)]]
        
        # of the valid peaks in the distance profile, select the peak with the widest peak
        column_peak_width = np.max(valid_column_allpeak_widths, initial=0.0)
        
        # get the center and index of the widest peak
        # from this get the center coordinates of the streak
        if column_peak_width != 0:
            column_peak_index = column_highest_count_indices[np.where(column_allpeak_widths == column_peak_width)[0]]
            column_peak_center = d_src_centered[column_peak_index][0]
            
            streak_x, streak_y = (component_coordinates[0] - (comp_d - column_peak_center) * np.cos(angle_peaks[i]*np.pi/180.), 
                                  component_coordinates[1] - (comp_d - column_peak_center) * np.sin(angle_peaks[i]*np.pi/180.))
        
        # if the streak does not go through comp_d then width and the streak coordinates are 0
        else:
            streak_x, streak_y = 0, 0
        
        # add the width and coordinates of the streak
        streak_width.append(column_peak_width)
        streak_coord.append([streak_x, streak_y])
    
    # convert to numpy for easy
    streak_width = np.array(streak_width)
    streak_coord = np.array(streak_coord)
        
    return streak_width, streak_coord
                

# %%
def streak_center_aperture_photometry(masked_data, streaks_coordinates, peaks_angle, peaks_angle_width, subtile_bmaj, subtile_bmin, component_coordinates):
    ''' Given the signal image, streak angles and its width, this function
        performs aperture photometry to find intensity of the streak.'''
    
    # since the VLASS data is given in Jy/bm, convert that into Jy/pix before performing aperture photometry
    # to do this divide the image by beam volume = (2 * pi * FWHM_A * FWHM_B)/(2*np.sqrt(2*np.log(2)))**2. This is the number of pixels there are in a beam. 
    # https://yuweiastro.github.io/astronomy/2020/07/21/radio-astro-tips/
    beam_volume = (np.pi * subtile_bmaj * subtile_bmin)/ (4 * np.log(2))
    masked_data_in_fluxdensity = masked_data/beam_volume
        
    # collect all streak parameters like flux density (flux), area of the region, rms and surface brightness
    allstreak_parameters = []
    
    # since the peaks_angle are P.A, we need to convert it to start at the x-axis to get the slope of the peak
    peaks_angle_0_90 = (np.array(peaks_angle)+90) % 180   # w.r.t x-axis
    
    # we consider background region as the annular region after masking the streak regions
    # therefore we mask all the streaks in the annular region
    comp_bmaj = component_coordinates[2]
    
    streaks_mask = np.zeros(masked_data_in_fluxdensity.shape)
    for streak_coords, ang, wid in zip(streaks_coordinates, peaks_angle, peaks_angle_width):
        streaks_mask = np.logical_or(streaks_mask,
                                     RectangularAperture([streak_coords[0], streak_coords[1]],                                                 # x, y for the aper
                                                         wid, comp_bmaj*GlobalVariables.annulus_radii_factor[1],                # width and length of the aper
                                                         ang*np.pi/180).to_mask(method='center').to_image(masked_data_in_fluxdensity.shape)    # angle of the aper
                                    )

    # here the streak will be divided into 4 quadrants and perform aperture photometry.
    # choose the nth peak of the identified peak and get the 4 quarters of the streak
    
    for i in range(len(peaks_angle)):
        
        # coordinates of the streak and the comp
        streak_x, streak_y = streaks_coordinates[i]
        comp_x, comp_y = component_coordinates[0], component_coordinates[1]
        
        # find the slope of each peak
        m = np.tan(peaks_angle_0_90*np.pi/180)[i]
        
        # find the inner and outer radii of the annula region to calculate the center x and y of each quadrant
        inner_radius, outer_radius = comp_bmaj/2 * GlobalVariables.annulus_radii_factor[0], comp_bmaj/2 * GlobalVariables.annulus_radii_factor[1]

        # cos and sin of the peak angles : https://math.stackexchange.com/questions/9365/endpoint-of-a-line-knowing-slope-start-and-distance
        c, s = 1/np.sqrt(1 + m**2), m/np.sqrt(1 + m**2)

        # (x1, y1) = left edge of the inner radius
        # (x2, y2) = right edge of the inner radius
        x1, y1 = streak_x - inner_radius*c, streak_y - inner_radius*s
        x2, y2 = streak_x + inner_radius*c, streak_y + inner_radius*s

        quarter_length = ((comp_bmaj * GlobalVariables.annulus_radii_factor[1]) - (comp_bmaj * GlobalVariables.annulus_radii_factor[0]))/4

        # q1, q2 = left 2 quadrants of the line(rectangle) segment
        # q3, q4 = right 2 quadrants of the line(rectangle) segment
        q1_x, q1_y = x1 - (3*quarter_length/2)*c, y1 - (3*quarter_length/2)*s
        q2_x, q2_y = x1 - (quarter_length/2)*c, y1 - (quarter_length/2)*s

        q3_x, q3_y = x2 + (quarter_length/2)*c, y2 + (quarter_length/2)*s
        q4_x, q4_y = x2 + (3*quarter_length/2)*c, y2 + (3*quarter_length/2)*s
        
        # calculate the apertures of the 4 quadrants and the background circular annular region
        q1_aper = RectangularAperture([q1_x, q1_y], peaks_angle_width[i], quarter_length, peaks_angle[i]*np.pi/180)   # here angle is PA. Not from x-axis
        q2_aper = RectangularAperture([q2_x, q2_y], peaks_angle_width[i], quarter_length, peaks_angle[i]*np.pi/180)   # here angle is PA. Not from x-axis
        q3_aper = RectangularAperture([q3_x, q3_y], peaks_angle_width[i], quarter_length, peaks_angle[i]*np.pi/180)   # here angle is PA. Not from x-axis
        q4_aper = RectangularAperture([q4_x, q4_y], peaks_angle_width[i], quarter_length, peaks_angle[i]*np.pi/180)   # here angle is PA. Not from x-axis
        cir_aper = CircularAperture([comp_x, comp_y], comp_bmaj/2 * GlobalVariables.annulus_radii_factor[1])
        
        # get aperture information
        q1_img_stats = ApertureStats(masked_data_in_fluxdensity, q1_aper)
        q2_img_stats = ApertureStats(masked_data_in_fluxdensity, q2_aper)
        q3_img_stats = ApertureStats(masked_data_in_fluxdensity, q3_aper)
        q4_img_stats = ApertureStats(masked_data_in_fluxdensity, q4_aper)
        cir_img_stats = ApertureStats(np.ma.array(masked_data_in_fluxdensity, mask=streaks_mask), cir_aper)
        
        # get the data of only the values inside the aperture selected
        cir_img_data = cir_img_stats.data_sumcutout
        q1_img_data, q2_img_data = q1_img_stats.data_sumcutout, q2_img_stats.data_sumcutout
        q3_img_data, q4_img_data = q3_img_stats.data_sumcutout, q4_img_stats.data_sumcutout
                        
        # calculate photon counts, area of the aperture and the rms of the aperture
        # corrected for masking
        q1_sum = q1_img_data.sum()
        q2_sum = q2_img_data.sum()
        q3_sum = q3_img_data.sum()
        q4_sum = q4_img_data.sum()
        cir_sum, cir_rms = cir_img_data.sum(), GlobalVariables.median_absolute_value(cir_img_data)
        
        try:
            q1_area = q1_img_stats.sum_aper_area.value
        except:
            q1_area = 0
            
        try:
            q2_area = q2_img_stats.sum_aper_area.value
        except:
            q2_area = 0
            
        try:
            q3_area = q3_img_stats.sum_aper_area.value
        except:
            q3_area = 0
        
        try:
            q4_area = q4_img_stats.sum_aper_area.value
        except:
            q4_area = 0
            
        try:
            cir_area = cir_img_stats.sum_aper_area.value
        except:
            cir_area = 0
        
        # some quadrants can fully lie outside the image. These quadrants have nan or -- as their value for sum, area and/or rms.
        # this is corrected below
        q1_sum, q1_area = np.nan_to_num(float(q1_sum)), np.nan_to_num(float(q1_area))
        q2_sum, q2_area = np.nan_to_num(float(q2_sum)), np.nan_to_num(float(q2_area))
        q3_sum, q3_area = np.nan_to_num(float(q3_sum)), np.nan_to_num(float(q3_area))
        q4_sum, q4_area = np.nan_to_num(float(q4_sum)), np.nan_to_num(float(q4_area))
        cir_sum, cir_area, cir_rms = np.nan_to_num(float(cir_sum)), np.nan_to_num(float(cir_area)), np.nan_to_num(float(cir_rms))
                
        # calculate the surface brightness of each quadrant/
        # surface brightness = total flux/solid angle of the image aka aperture area
        q1_SB, q2_SB, q3_SB, q4_SB = q1_sum/q1_area, q2_sum/q2_area, q3_sum/q3_area, q4_sum/q4_area
        cir_SB = np.nan_to_num(cir_sum/cir_area)
        
        # consolidate all streak parameters
        streak_parameters = {'q1_sum':q1_sum, 'q1_area': q1_area, 'q1_SB': q1_SB,
                             'q2_sum':q2_sum, 'q2_area': q2_area, 'q2_SB': q2_SB,
                             'q3_sum':q3_sum, 'q3_area': q3_area, 'q3_SB': q3_SB,
                             'q4_sum':q4_sum, 'q4_area': q4_area, 'q4_SB': q4_SB,
                             'cir_sum':cir_sum, 'cir_area': cir_area, 'cir_rms': cir_rms, 'cir_SB': cir_SB}
        
        allstreak_parameters.append(streak_parameters)
        
    return allstreak_parameters

# %%
def src_center_aperture_photometry(masked_data, component_coordinates, peaks_angle, peaks_angle_width, subtile_bmaj, subtile_bmin):
    ''' Given the signal image, streak angles and its width, this function
        performs aperture photometry to find intensity of the streak.'''
    
    # since the VLASS data is given in Jy/bm, convert that into Jy/pix before performing aperture photometry
    # to do this divide the image by beam volume = (2 * pi * FWHM_A * FWHM_B)/(2*np.sqrt(2*np.log(2)))**2. This is the number of pixels there are in a beam. 
    # https://yuweiastro.github.io/astronomy/2020/07/21/radio-astro-tips/
    beam_volume = (np.pi * subtile_bmaj * subtile_bmin)/ (4 * np.log(2))
    masked_data_in_fluxdensity = masked_data/beam_volume
        
    # collect all streak parameters like flux density (flux), area of the region, rms and surface brightness
    allstreak_parameters = []
    
    # since the peaks_angle are P.A, we need to convert it to start at the x-axis to get the slope of the peak
    peaks_angle_0_90 = (np.array(peaks_angle)+90) % 180   # w.r.t x-axis
    
    # we consider background region as the annular region after masking the streak regions
    # therefore we mask all the streaks in the annular region
    comp_bmaj = component_coordinates[2]
        
    streaks_mask = np.zeros(masked_data_in_fluxdensity.shape)
    for ang, wid in zip(peaks_angle, peaks_angle_width):
        streaks_mask = np.logical_or(streaks_mask,
                                     RectangularAperture([component_coordinates[0], component_coordinates[1]],                                                 # x, y for the aper
                                                         wid, comp_bmaj*GlobalVariables.annulus_radii_factor[1],                # width and length of the aper
                                                         ang*np.pi/180).to_mask(method='center').to_image(masked_data_in_fluxdensity.shape)    # angle of the aper
                                    )
    
    # here the streak will be divided into 4 quadrants and perform aperture photometry.
    # choose the nth peak of the identified peak and get the 4 quarters of the streak
    
    for i in range(len(peaks_angle)):
        
        # coordinates of the comp
        comp_x, comp_y = component_coordinates[0], component_coordinates[1]
        
        # find the slope of each peak
        m = np.tan(peaks_angle_0_90*np.pi/180)[i]
        
        # here we are taking the width of the streak as the size of the component itself. 
        # the multiplication factor is to convert from fwhm to width. fwhm is 2.355*std --> 1 side is 1.775
        # To convert from 1 side fwhm to 3*std the factor is 3/1.775
        src_size_streak_width = max(component_coordinates[2], subtile_bmaj)*3/1.1775
        
        # find the inner and outer radii of the annula region to calculate the center x and y of each quadrant
        inner_radius, outer_radius = comp_bmaj/2 * GlobalVariables.annulus_radii_factor[0], comp_bmaj/2 * GlobalVariables.annulus_radii_factor[1]

        # cos and sin of the peak angles : https://math.stackexchange.com/questions/9365/endpoint-of-a-line-knowing-slope-start-and-distance
        c, s = 1/np.sqrt(1 + m**2), m/np.sqrt(1 + m**2)

        # (x1, y1) = left edge of the inner radius
        # (x2, y2) = right edge of the inner radius
        x1, y1 = comp_x - inner_radius*c, comp_y - inner_radius*s
        x2, y2 = comp_x + inner_radius*c, comp_y + inner_radius*s

        quarter_length = ((comp_bmaj * GlobalVariables.annulus_radii_factor[1]) - (comp_bmaj * GlobalVariables.annulus_radii_factor[0]))/4

        # q1, q2 = left 2 quadrants of the line(rectangle) segment
        # q3, q4 = right 2 quadrants of the line(rectangle) segment
        q1_x, q1_y = x1 - (3*quarter_length/2)*c, y1 - (3*quarter_length/2)*s
        q2_x, q2_y = x1 - (quarter_length/2)*c, y1 - (quarter_length/2)*s

        q3_x, q3_y = x2 + (quarter_length/2)*c, y2 + (quarter_length/2)*s
        q4_x, q4_y = x2 + (3*quarter_length/2)*c, y2 + (3*quarter_length/2)*s
        
        # calculate the apertures of the 4 quadrants and the background circular annular region
        q1_aper = RectangularAperture([q1_x, q1_y], peaks_angle_width[i], quarter_length, peaks_angle[i]*np.pi/180)   # here angle is PA. Not from x-axis
        q2_aper = RectangularAperture([q2_x, q2_y], peaks_angle_width[i], quarter_length, peaks_angle[i]*np.pi/180)   # here angle is PA. Not from x-axis
        q3_aper = RectangularAperture([q3_x, q3_y], peaks_angle_width[i], quarter_length, peaks_angle[i]*np.pi/180)   # here angle is PA. Not from x-axis
        q4_aper = RectangularAperture([q4_x, q4_y], peaks_angle_width[i], quarter_length, peaks_angle[i]*np.pi/180)   # here angle is PA. Not from x-axis
        cir_aper = CircularAperture([comp_x, comp_y], comp_bmaj/2 * GlobalVariables.annulus_radii_factor[1])
        
        # get aperture information
        q1_img_stats = ApertureStats(masked_data_in_fluxdensity, q1_aper)
        q2_img_stats = ApertureStats(masked_data_in_fluxdensity, q2_aper)
        q3_img_stats = ApertureStats(masked_data_in_fluxdensity, q3_aper)
        q4_img_stats = ApertureStats(masked_data_in_fluxdensity, q4_aper)
        cir_img_stats = ApertureStats(np.ma.array(masked_data_in_fluxdensity, mask=streaks_mask), cir_aper)
        
        # get the data of only the values inside the aperture selected
        cir_img_data = cir_img_stats.data_sumcutout
        q1_img_data, q2_img_data = q1_img_stats.data_sumcutout, q2_img_stats.data_sumcutout
        q3_img_data, q4_img_data = q3_img_stats.data_sumcutout, q4_img_stats.data_sumcutout
                        
        # calculate photon counts, area of the aperture and the rms of the aperture
        # corrected for masking
        q1_sum = q1_img_data.sum()
        q2_sum = q2_img_data.sum()
        q3_sum = q3_img_data.sum()
        q4_sum = q4_img_data.sum()
        cir_sum, cir_rms = cir_img_data.sum(), GlobalVariables.median_absolute_value(cir_img_data)
        
        try:
            q1_area = q1_img_stats.sum_aper_area.value
        except:
            q1_area = 0
            
        try:
            q2_area = q2_img_stats.sum_aper_area.value
        except:
            q2_area = 0
            
        try:
            q3_area = q3_img_stats.sum_aper_area.value
        except:
            q3_area = 0
        
        try:
            q4_area = q4_img_stats.sum_aper_area.value
        except:
            q4_area = 0
            
        try:
            cir_area = cir_img_stats.sum_aper_area.value
        except:
            cir_area = 0
        
        # some quadrants can fully lie outside the image. These quadrants have nan or -- as their value for sum, area and/or rms.
        # this is corrected below
        q1_sum, q1_area = np.nan_to_num(float(q1_sum)), np.nan_to_num(float(q1_area))
        q2_sum, q2_area = np.nan_to_num(float(q2_sum)), np.nan_to_num(float(q2_area))
        q3_sum, q3_area = np.nan_to_num(float(q3_sum)), np.nan_to_num(float(q3_area))
        q4_sum, q4_area = np.nan_to_num(float(q4_sum)), np.nan_to_num(float(q4_area))
        cir_sum, cir_area, cir_rms = np.nan_to_num(float(cir_sum)), np.nan_to_num(float(cir_area)), np.nan_to_num(float(cir_rms))
                
        # calculate the surface brightness of each quadrant/
        # surface brightness = total flux/solid angle of the image aka aperture area
        q1_SB, q2_SB, q3_SB, q4_SB = q1_sum/q1_area, q2_sum/q2_area, q3_sum/q3_area, q4_sum/q4_area
        cir_SB = np.nan_to_num(cir_sum/cir_area)
        
        # consolidate all streak parameters
        streak_parameters = {'q1_sum':q1_sum, 'q1_area': q1_area, 'q1_SB': q1_SB,
                             'q2_sum':q2_sum, 'q2_area': q2_area, 'q2_SB': q2_SB,
                             'q3_sum':q3_sum, 'q3_area': q3_area, 'q3_SB': q3_SB,
                             'q4_sum':q4_sum, 'q4_area': q4_area, 'q4_SB': q4_SB,
                             'cir_sum':cir_sum, 'cir_area': cir_area, 'cir_rms': cir_rms, 'cir_SB': cir_SB}
        
        allstreak_parameters.append(streak_parameters)
        
    return allstreak_parameters

# %%
def srccenter_srcwidth_aperture_photometry(masked_data, component_coordinates, peaks_angle, peaks_angle_width, subtile_bmaj, subtile_bmin):
    ''' Given the signal image, streak angles, this function performs aperture photometry to 
        find intensity of the streak with the width set to the component size.'''
    
    # since the VLASS data is given in Jy/bm, convert that into Jy/pix before performing aperture photometry
    # to do this divide the image by beam volume = (2 * pi * FWHM_A * FWHM_B)/(2*np.sqrt(2*np.log(2)))**2. This is the number of pixels there are in a beam. 
    # https://yuweiastro.github.io/astronomy/2020/07/21/radio-astro-tips/
    beam_volume = (np.pi * subtile_bmaj * subtile_bmin)/ (4 * np.log(2))
    masked_data_in_fluxdensity = masked_data/beam_volume
        
    # collect all streak parameters like flux density (flux), area of the region, rms and surface brightness
    allstreak_parameters = []
    
    # since the peaks_angle are P.A, we need to convert it to start at the x-axis to get the slope of the peak
    peaks_angle_0_90 = (np.array(peaks_angle)+90) % 180   # w.r.t x-axis
    
    # we consider background region as the annular region after masking the streak regions
    # therefore we mask all the streaks in the annular region
    comp_bmaj = component_coordinates[2]
        
    streaks_mask = np.zeros(masked_data_in_fluxdensity.shape)
    for ang, wid in zip(peaks_angle, peaks_angle_width):
        streaks_mask = np.logical_or(streaks_mask,
                                     RectangularAperture([component_coordinates[0], component_coordinates[1]],                                                 # x, y for the aper
                                                         wid, comp_bmaj*GlobalVariables.annulus_radii_factor[1],                # width and length of the aper
                                                         ang*np.pi/180).to_mask(method='center').to_image(masked_data_in_fluxdensity.shape)    # angle of the aper
                                    )
    
    # here the streak will be divided into 4 quadrants and perform aperture photometry.
    # choose the nth peak of the identified peak and get the 4 quarters of the streak
    
    for i in range(len(peaks_angle)):
        
        # coordinates of the comp
        comp_x, comp_y = component_coordinates[0], component_coordinates[1]
        
        # find the slope of each peak
        m = np.tan(peaks_angle_0_90*np.pi/180)[i]
        
        # here we are taking the width of the streak as the size of the component itself. 
        # the multiplication factor is to convert from fwhm to width. fwhm is 2.355*std --> 1 side is 1.775
        # To convert from 1 side fwhm to 3*std the factor is 3/1.775
        src_size_streak_width = max(component_coordinates[2], subtile_bmaj)*3/1.1775
        
        # find the inner and outer radii of the annula region to calculate the center x and y of each quadrant
        inner_radius, outer_radius = comp_bmaj/2 * GlobalVariables.annulus_radii_factor[0], comp_bmaj/2 * GlobalVariables.annulus_radii_factor[1]

        # cos and sin of the peak angles : https://math.stackexchange.com/questions/9365/endpoint-of-a-line-knowing-slope-start-and-distance
        c, s = 1/np.sqrt(1 + m**2), m/np.sqrt(1 + m**2)

        # (x1, y1) = left edge of the inner radius
        # (x2, y2) = right edge of the inner radius
        x1, y1 = comp_x - inner_radius*c, comp_y - inner_radius*s
        x2, y2 = comp_x + inner_radius*c, comp_y + inner_radius*s

        quarter_length = ((comp_bmaj * GlobalVariables.annulus_radii_factor[1]) - (comp_bmaj * GlobalVariables.annulus_radii_factor[0]))/4

        # q1, q2 = left 2 quadrants of the line(rectangle) segment
        # q3, q4 = right 2 quadrants of the line(rectangle) segment
        q1_x, q1_y = x1 - (3*quarter_length/2)*c, y1 - (3*quarter_length/2)*s
        q2_x, q2_y = x1 - (quarter_length/2)*c, y1 - (quarter_length/2)*s

        q3_x, q3_y = x2 + (quarter_length/2)*c, y2 + (quarter_length/2)*s
        q4_x, q4_y = x2 + (3*quarter_length/2)*c, y2 + (3*quarter_length/2)*s
        
        # calculate the apertures of the 4 quadrants and the background circular annular region
        q1_aper = RectangularAperture([q1_x, q1_y], src_size_streak_width, quarter_length, peaks_angle[i]*np.pi/180)   # here angle is PA. Not from x-axis
        q2_aper = RectangularAperture([q2_x, q2_y], src_size_streak_width, quarter_length, peaks_angle[i]*np.pi/180)   # here angle is PA. Not from x-axis
        q3_aper = RectangularAperture([q3_x, q3_y], src_size_streak_width, quarter_length, peaks_angle[i]*np.pi/180)   # here angle is PA. Not from x-axis
        q4_aper = RectangularAperture([q4_x, q4_y], src_size_streak_width, quarter_length, peaks_angle[i]*np.pi/180)   # here angle is PA. Not from x-axis
        cir_aper = CircularAperture([comp_x, comp_y], comp_bmaj/2 * GlobalVariables.annulus_radii_factor[1])
        
        # get aperture information
        q1_img_stats = ApertureStats(masked_data_in_fluxdensity, q1_aper)
        q2_img_stats = ApertureStats(masked_data_in_fluxdensity, q2_aper)
        q3_img_stats = ApertureStats(masked_data_in_fluxdensity, q3_aper)
        q4_img_stats = ApertureStats(masked_data_in_fluxdensity, q4_aper)
        cir_img_stats = ApertureStats(np.ma.array(masked_data_in_fluxdensity, mask=streaks_mask), cir_aper)
        
        # get the data of only the values inside the aperture selected
        cir_img_data = cir_img_stats.data_sumcutout
        q1_img_data, q2_img_data = q1_img_stats.data_sumcutout, q2_img_stats.data_sumcutout
        q3_img_data, q4_img_data = q3_img_stats.data_sumcutout, q4_img_stats.data_sumcutout
                        
        # calculate photon counts, area of the aperture and the rms of the aperture
        # corrected for masking
        q1_sum = q1_img_data.sum()
        q2_sum = q2_img_data.sum()
        q3_sum = q3_img_data.sum()
        q4_sum = q4_img_data.sum()
        cir_sum, cir_rms = cir_img_data.sum(), GlobalVariables.median_absolute_value(cir_img_data)
        
        try:
            q1_area = q1_img_stats.sum_aper_area.value
        except:
            q1_area = 0
            
        try:
            q2_area = q2_img_stats.sum_aper_area.value
        except:
            q2_area = 0
            
        try:
            q3_area = q3_img_stats.sum_aper_area.value
        except:
            q3_area = 0
        
        try:
            q4_area = q4_img_stats.sum_aper_area.value
        except:
            q4_area = 0
            
        try:
            cir_area = cir_img_stats.sum_aper_area.value
        except:
            cir_area = 0
        
        # some quadrants can fully lie outside the image. These quadrants have nan or -- as their value for sum, area and/or rms.
        # this is corrected below
        q1_sum, q1_area = np.nan_to_num(float(q1_sum)), np.nan_to_num(float(q1_area))
        q2_sum, q2_area = np.nan_to_num(float(q2_sum)), np.nan_to_num(float(q2_area))
        q3_sum, q3_area = np.nan_to_num(float(q3_sum)), np.nan_to_num(float(q3_area))
        q4_sum, q4_area = np.nan_to_num(float(q4_sum)), np.nan_to_num(float(q4_area))
        cir_sum, cir_area, cir_rms = np.nan_to_num(float(cir_sum)), np.nan_to_num(float(cir_area)), np.nan_to_num(float(cir_rms))
                
        # calculate the surface brightness of each quadrant/
        # surface brightness = total flux/solid angle of the image aka aperture area
        q1_SB, q2_SB, q3_SB, q4_SB = q1_sum/q1_area, q2_sum/q2_area, q3_sum/q3_area, q4_sum/q4_area
        cir_SB = np.nan_to_num(cir_sum/cir_area)
        
        # consolidate all streak parameters
        streak_parameters = {'q1_sum':q1_sum, 'q1_area': q1_area, 'q1_SB': q1_SB,
                             'q2_sum':q2_sum, 'q2_area': q2_area, 'q2_SB': q2_SB,
                             'q3_sum':q3_sum, 'q3_area': q3_area, 'q3_SB': q3_SB,
                             'q4_sum':q4_sum, 'q4_area': q4_area, 'q4_SB': q4_SB,
                             'cir_sum':cir_sum, 'cir_area': cir_area, 'cir_rms': cir_rms, 'cir_SB': cir_SB}
        
        allstreak_parameters.append(streak_parameters)
        
    return allstreak_parameters

# %%
def StreaksIdentifier(subtile_identifier, system, screen=''):
    ''' Given the subtile_identifier, this function identifies the angle(PA)
        and width all the streaks going through a given component. '''
                
    # get tile and subtile names
    tile, subtile = subtile_identifier[9:].split('.')

    # to ensure that the code is consistent with the one of the local system, a uniserval code is created with a flag.
    # based on the flag, the version of the code is chosen
    # here the only different btw the two code is the extraction of the signal_file and rms_file
    if system == 'server':
        # get the image and rms image location
        signal_file = GlobalVariables.image_fits_list[GlobalVariables.image_fits_list.str.contains(subtile_identifier, regex=False)].values[0]
        rms_file = GlobalVariables.rms_fits_list[GlobalVariables.rms_fits_list.str.contains(subtile_identifier, regex=False)].values[0]
    
    if system == 'local':
        # get the image and rms image location
        signal_file = GlobalVariables.image_fits_list[GlobalVariables.image_fits_list.str.contains(subtile_identifier[:9]+'ql.'+subtile_identifier[9:], regex=False)].values[0]
        rms_file = GlobalVariables.rms_fits_list[GlobalVariables.rms_fits_list.str.contains(subtile_identifier[:9]+'ql.'+subtile_identifier[9:], regex=False)].values[0]
    
    if system == 'local_server':
        # get the image and rms image location (screen folders)
        signal_file_list = pd.Series(glob.glob('/Users/suhasini/Documents/ArtifactIdentification/safetyfolder/expanded_server/fits/image/screen'+str(screen)+'/*'))
        rms_file_list = pd.Series(glob.glob('/Users/suhasini/Documents/ArtifactIdentification/safetyfolder/expanded_server/fits/rms/screen'+str(screen)+'/*'))
        
        signal_file = signal_file_list[signal_file_list.str.contains(subtile_identifier, regex=False)].values[0]
        rms_file = rms_file_list[rms_file_list.str.contains(subtile_identifier, regex=False)].values[0]
    
    # given signal data, divide rms from the image to get to the SNR.
    # This will be used as "data" to identify streaks
    # access signal and rms headers
    signal_hdr = fits.open(signal_file)
    rms_hdr = fits.open(rms_file)
        
    # access signal image
    signal_data = signal_hdr[0].data
    signal_data = signal_data.reshape(signal_data.shape[2], signal_data.shape[3])

    # access rms image
    rms_data = rms_hdr[0].data
    rms_data = rms_data.reshape(rms_data.shape[2], rms_data.shape[3])

    # signal/noise image
    data = signal_data/rms_data
    
    # some subtiles may have NaN values. 
    data[np.isnan(data)] = 0   # change NaN values to 0
    
    # close files
    signal_hdr.close()
    rms_hdr.close()
        
    # get subtile bmaj, bmin and wcs
    subtile_bmaj = signal_hdr[0].header['BMAJ']/np.abs(signal_hdr[0].header['CDELT1'])   # in pixels
    subtile_bmin = signal_hdr[0].header['BMIN']/np.abs(signal_hdr[0].header['CDELT1'])   # in pixels
    wcs = WCS(signal_hdr[0].header, naxis=2)
    
    # clip/threshold the image to 1.5 sigma.
    # Any value abovw 1.5 sigma is 1, rest is 0. This represents the positive space in the rest of the code    
    positive_space, negative_space = clipping(data)
    bkg_space = np.ones(data.shape)
        
    # there can be NaN values in data. Those pixels should be masked in both positive_space, negative_space and bkg_space
    NaN_mask = np.zeros(data.shape)
    NaN_mask[np.isnan(data)] = 1
    
    positive_space, negative_space = np.ma.array(positive_space, mask=NaN_mask), np.ma.array(negative_space, mask=NaN_mask)
    bkg_space = np.ma.array(bkg_space, mask=NaN_mask)
    
    # get x, y, bmaj, bmin and pa of all the components in the given subtile
    component_coord = getComponentsCoord(tile, subtile, subtile_bmaj, subtile_bmin, wcs)
    
    # get a mask which masks all the components in the given subtile.
    # this is just a mask, not the masked image
    # the subtile's bmaj and bmin is given so that if a component's bmaj and bmin respectively is less than subtile's bmaj and bmin, it is replaces with the subtile's bmaj and bmin
    comps_mask = componentsMasking(component_coord, shape=data.shape)
        
    # masking all compoenents from positive_space, negative space and bkg space
    positive_space = np.ma.array(positive_space, mask=comps_mask)
    negative_space = np.ma.array(negative_space, mask=comps_mask)
    bkg_space = np.ma.array(bkg_space, mask=comps_mask)
    
    # perform hough transform on the whole image i.e., both positive and negative space
    posSpace_HTS, t, d = hough_line(positive_space, GlobalVariables.tested_angles)    
    negSpace_HTS = hough_line(negative_space, GlobalVariables.tested_angles)[0]
        
    # for each coordinate, find the streaks going through the coordinate and add it to the output dataframe
    for idx, coord_row in component_coord.iterrows():
        
        # print(idx)
        
        # if the user has given the component/source's maj (FWHM) and it is greater than the subtile_bmaj, then use that
        # if not, use the subtile_bmaj
        if float(coord_row['Maj_img_plane']) <= float(subtile_bmaj):
            src_info = [coord_row['Xposn'], coord_row['Yposn'], subtile_bmaj, coord_row['Overlapping_maj_flag'], coord_row['Overlapping_maj_corrected']]
        else:
            src_info = [coord_row['Xposn'], coord_row['Yposn'], coord_row['Maj_img_plane'], coord_row['Overlapping_maj_flag'], coord_row['Overlapping_maj_corrected']]
        
        # Generate a mask to get only an annulus around a given source
        # this mask will be used to generate src and bkg response function -- masked inputs to the HT function
        annulus_mask = annulusMaskGenerator(src_info,                  # x, y, bmaj and bmin to draw annulii
                                            shape=data.shape,                             # to generate a mask image the same size as that of the data
                                            annulus_radii_factor=GlobalVariables.annulus_radii_factor)    # if the bmaj and bmin of the comp < subtile's bmaj and bmin, this will be used
                
        # mask the postive space and negative space to get src response function in both spaces. mask all 1s image to get bkg response function
        posSpace_src_resp_func = np.ma.array(positive_space, mask=annulus_mask)
        negSpace_src_resp_func = np.ma.array(negative_space, mask=annulus_mask)
        bkg_resp_func = np.ma.array(bkg_space, mask=annulus_mask)

        # perform hough transform on the positive and negative space annulii i.e., src response function and bkg response function
        posSpace_src_resp_HTS, negSpace_src_resp_HTS, bkg_resp_HTS = getAnnulusHTS(posSpace_src_resp_func, negSpace_src_resp_func, bkg_resp_func, GlobalVariables.tested_angles)
                
        # generate src histogram and bkg histogram w.r.t angles -- this is a weighted histogram which highlights only the region in the annulus
        posSpace_src_histogram = (posSpace_src_resp_HTS * posSpace_HTS).sum(axis=0) / posSpace_src_resp_HTS.sum(axis=0)
        posSpace_bkg_histogram = (bkg_resp_HTS * posSpace_HTS).sum(axis=0) / bkg_resp_HTS.sum(axis=0)

        negSpace_src_histogram = (negSpace_src_resp_HTS * negSpace_HTS).sum(axis=0) / negSpace_src_resp_HTS.sum(axis=0)
        negSpace_bkg_histogram = (bkg_resp_HTS * negSpace_HTS).sum(axis=0) / bkg_resp_HTS.sum(axis=0)

        # find the peaks in angles in the bkg subtracted src histogram
        # this indicates the angles at which there is a potential streak going through the "source" in question
        posSpace_peaks_angle, posSpace_peaks_angle_index  = findingPeaks(posSpace_src_histogram, posSpace_bkg_histogram, src_info)
        negSpace_peaks_angle, negSpace_peaks_angle_index = findingPeaks(negSpace_src_histogram, negSpace_bkg_histogram, src_info)
                
        # In the next few steps the streak angles that are not strong are removed. 
        # To make sure we know which angles are detected but did not make the cut, we store them here
        prelimPosAngle = posSpace_peaks_angle
        prelimNegAngle = negSpace_peaks_angle
                
        # for each peak identified, find the width of each streak
        # position of the component at each identified peak is given as r. This will help find the width of the peak of the streak going through the component instead
        # of finding the width of a peak at the same angle but different location from the comp but is a brighter streak.
        comp_d_values = [i for i in 
               src_info[0]*np.cos(posSpace_peaks_angle * np.pi/180) + src_info[1]*np.sin(posSpace_peaks_angle * np.pi/180)]
        posSpace_peaks_angle_width, posSpace_streak_coord = findingPeaksWidth(posSpace_src_resp_HTS, posSpace_peaks_angle, posSpace_peaks_angle_index, d, comp_d_values, src_info)
        
        comp_d_values = [i for i in 
               src_info[0]*np.cos(negSpace_peaks_angle * np.pi/180) + src_info[1]*np.sin(negSpace_peaks_angle * np.pi/180)]
        negSpace_peaks_angle_width, negSpace_streak_coord = findingPeaksWidth(negSpace_src_resp_HTS, negSpace_peaks_angle, negSpace_peaks_angle_index, d, comp_d_values, src_info)
        
        # where ever the width is 0, it indicates that it was a false positive streak detection
        # remove all those peaks and width
        zero_width_false_positive_index = np.where(posSpace_peaks_angle_width == 0)[0]
        posSpace_peaks_angle = np.delete(posSpace_peaks_angle, zero_width_false_positive_index).tolist()
        posSpace_peaks_angle_width = np.delete(posSpace_peaks_angle_width, zero_width_false_positive_index).tolist()
        posSpace_streak_coord = np.delete(posSpace_streak_coord, zero_width_false_positive_index, axis=0).tolist()
        
        zero_width_false_positive_index = np.where(negSpace_peaks_angle_width == 0.)[0]
        negSpace_peaks_angle = np.delete(negSpace_peaks_angle, zero_width_false_positive_index).tolist()
        negSpace_peaks_angle_width = np.delete(negSpace_peaks_angle_width, zero_width_false_positive_index).tolist()
        negSpace_streak_coord = np.delete(negSpace_streak_coord, zero_width_false_positive_index, axis=0).tolist()
        
        # perform aperture photometry on the identified streaks
        # divide the streak (given the angle and its width) into 4 quadrants and perform aperture photometry on these sections
        # for aperture photmetry, you need to mask the signal image, rms image (to get the rms of the quadrants)
        masked_signal_data = np.ma.array(signal_data, mask=NaN_mask)
        masked_signal_data = np.ma.array(masked_signal_data, mask=comps_mask)
        masked_signal_data = np.ma.array(masked_signal_data, mask=annulus_mask)
                
        # perform aperture photometry
        # here send the streak coordinates instead of the comp_coordinates to get better accuracy in aperture photometry
        if len(posSpace_peaks_angle) != 0:
            streak_center_posStreak_parameters = streak_center_aperture_photometry(masked_signal_data, posSpace_streak_coord, posSpace_peaks_angle, posSpace_peaks_angle_width, subtile_bmaj, subtile_bmin, src_info)
        else:
            streak_center_posStreak_parameters = []
            
        if len(negSpace_peaks_angle) != 0:
            streak_center_negStreak_parameters = streak_center_aperture_photometry(masked_signal_data, negSpace_streak_coord, negSpace_peaks_angle, negSpace_peaks_angle_width, subtile_bmaj, subtile_bmin, src_info)
        else:
            streak_center_negStreak_parameters = []
        
        # but also just in case, get the aperture photometry with src/comp as the center
        if len(posSpace_peaks_angle) != 0:
            src_center_posStreak_parameters = src_center_aperture_photometry(masked_signal_data, src_info, posSpace_peaks_angle, posSpace_peaks_angle_width, subtile_bmaj, subtile_bmin)
        else:
            src_center_posStreak_parameters = []
            
        if len(negSpace_peaks_angle) != 0:
            src_center_negStreak_parameters = src_center_aperture_photometry(masked_signal_data, src_info, negSpace_peaks_angle, negSpace_peaks_angle_width, subtile_bmaj, subtile_bmin)
        else:
            src_center_negStreak_parameters = []
            
        # and also just in case, get the aperture photometry with src/comp as the center 
        # with width as the size of the component instead of the streak width calculated
        if len(posSpace_peaks_angle) != 0:
            srccenter_srcwidth_posStreak_parameters = srccenter_srcwidth_aperture_photometry(masked_signal_data, src_info, posSpace_peaks_angle, posSpace_peaks_angle_width, subtile_bmaj, subtile_bmin)
        else:
            srccenter_srcwidth_posStreak_parameters = []
            
        if len(negSpace_peaks_angle) != 0:
            srccenter_srcwidth_negStreak_parameters = srccenter_srcwidth_aperture_photometry(masked_signal_data, src_info, negSpace_peaks_angle, negSpace_peaks_angle_width, subtile_bmaj, subtile_bmin)
        else:
            srccenter_srcwidth_negStreak_parameters = []
                
        # add values to output dataframe
        row = pd.Series([coord_row['RA'], coord_row['DEC'], coord_row['Xposn'], coord_row['Yposn'],
                         subtile_identifier, '', 
                         prelimPosAngle, prelimNegAngle,
                         posSpace_peaks_angle, negSpace_peaks_angle, 
                         posSpace_peaks_angle_width, negSpace_peaks_angle_width,
                         posSpace_streak_coord, negSpace_streak_coord,
                         streak_center_posStreak_parameters, streak_center_negStreak_parameters,
                         src_center_posStreak_parameters, src_center_negStreak_parameters,
                         srccenter_srcwidth_posStreak_parameters, srccenter_srcwidth_negStreak_parameters], index=GlobalVariables.output_df.columns)
        
        GlobalVariables.output_df = GlobalVariables.output_df.append(row, ignore_index=True)
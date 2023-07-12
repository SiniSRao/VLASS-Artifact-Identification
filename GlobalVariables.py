# %%
"""
---
### This notebook contains all the global variables needed for all the modules
---
"""

# %%
import pandas as pd
import numpy as np
import astropy.units as u
from astropy.coordinates import EarthLocation
import glob

# %%
input_csv_loc, input_csv = '', ''
selected_tile_loc, selected_tile_with_peaks_loc = '', ''
executionblock_csv_loc = ''
image_fits_list, rms_fits_list = '', ''
output_csv_loc = ''

# %%
def declare(sys):
    ''' Assigned variables based type of system the code is being run on.'''
    
    global input_csv_loc, input_csv, selected_tile_loc, selected_tile_with_peaks_loc, executionblock_csv_loc, image_fits_list, rms_fits_list, output_csv_loc
    
    if sys == 'server':
        # input file
        input_csv_loc = '~/ArtifactIdentification/csv/unique_subtile_identifiers.csv'

        # selected tile's location
        selected_tile_loc = '/mnt/bigdata/suhasin1/VLASS_data/VLASS1_components_tiles/'
        selected_tile_with_peaks_loc = '~/ArtifactIdentification/csv/VLASS1_components_tiles_with_peaks/'

        # to get the start and end time of the observation
        # this will be used to calculate the predicted angles
        executionblock_csv_loc = '~/ArtifactIdentification/csv/sorted_execution_block.csv'

        # location of all fits files and rms files
        image_fits_list = pd.Series(glob.glob('/mnt/bigdata/suhasin1/VLASS_data/image/*'))
        rms_fits_list = pd.Series(glob.glob('/mnt/bigdata/suhasin1/VLASS_data/rms/*'))

        # output file loc
        output_csv_loc = '/home/suhasin1/ArtifactIdentification/output/'
        
    if sys == 'local':
        # input file
        input_csv_loc = '/Users/suhasini/Documents/ArtifactIdentification/csv/unique_subtile_identifiers.csv'

        # selected tile's location
        selected_tile_loc = '/Users/suhasini/Documents/ArtifactIdentification/csv/VLASS1_components_tiles/'
        selected_tile_with_peaks_loc = '/Users/suhasini/Documents/ArtifactIdentification/csv/VLASS1_components_tiles_with_peaks/'

        # to get the start and end time of the observation
        # this will be used to calculate the predicted angles
        executionblock_csv_loc = '/Users/suhasini/Documents/ArtifactIdentification/csv/sorted_execution_block.csv'

        # location of all fits files and rms files
        image_fits_list = pd.Series(glob.glob('/Users/suhasini/Documents/transient/fits/signal/*'))
        rms_fits_list = pd.Series(glob.glob('/Users/suhasini/Documents/transient/fits/noise/*'))

        # output file loc
        output_csv_loc = '/Users/suhasini/Documents/ArtifactIdentification/output/'
        
    if sys == 'local_server':
        # input file
        input_csv_loc = '/Users/suhasini/Documents/ArtifactIdentification/csv/unique_subtile_identifiers.csv'

        # selected tile's location
        selected_tile_loc = '/Users/suhasini/Documents/ArtifactIdentification/csv/VLASS1_components_tiles/'
        selected_tile_with_peaks_loc = '/Users/suhasini/Documents/ArtifactIdentification/csv/VLASS1_components_tiles_with_peaks/'

        # to get the start and end time of the observation
        # this will be used to calculate the predicted angles
        executionblock_csv_loc = '/Users/suhasini/Documents/ArtifactIdentification/csv/sorted_execution_block.csv'

        # location of all fits files and rms files
        server_image_list_csv = pd.read_csv('/Users/suhasini/Documents/ArtifactIdentification/csv/unique_subtile_identifiers_withloc.csv', index_col=0)
        image_fits_list = server_image_list_csv['image_loc']
        rms_fits_list = server_image_list_csv['rms_loc']

        # output file loc
        output_csv_loc = '/Users/suhasini/Documents/ArtifactIdentification/safetyfolder/expanded_server/csv/'
        
    input_csv = pd.read_csv(input_csv_loc, index_col=0)

# %%
c = 2.99792458e8   # speed of light. (x,y,z) of VLA antennas is given in ns. This is needed to convert to m
frequence = 3e9    # VLA observation freq
antNum = 27        # number of antennas in VLA

VLA_loc = EarthLocation.of_site('VLA')   # coordinates of VLA

# %%
antenna = np.array([
        ('W1'  ,'W', 'D'   ,     76.69,     11.67,   -108.36),
        ('W2'  ,'W', 'CD'  ,     49.29,   -123.87,    -67.42),
        ('W3'  ,'W', 'D'   ,     96.46,   -248.46,   -136.94),
        ('W4'  ,'W', 'BCD' ,    156.49,   -407.06,   -225.51),
        ('W5'  ,'W', 'D'   ,    228.83,   -597.84,   -331.98),
        ('W6'  ,'W', 'CD'  ,    311.96,   -817.22,   -454.39),
        ('W7'  ,'W', 'D'   ,    405.70,  -1064.49,   -592.36),
        ('W8'  ,'W', 'ABCD',    509.53,  -1338.54,   -745.23),
        ('W9'  ,'W', 'D'   ,    623.12,  -1638.19,   -912.51),
        ('W10' ,'W', 'C'   ,    747.12,  -1962.88,  -1093.09),
        ('W12' ,'W', 'BC'  ,   1021.28,  -2683.76,  -1494.63),
        ('W14' ,'W', 'C'   ,   1328.35,  -3496.20,  -1948.61),
        ('W16' ,'W', 'ABC' ,   1667.27,  -4396.35,  -2452.41),
        ('W18' ,'W', 'C'   ,   2040.62,  -5381.34,  -3002.15),
        ('W20' ,'W', 'B'   ,   2446.13,  -6447.72,  -3596.19),
        ('W24' ,'W', 'AB'  ,   3353.71,  -8816.08,  -4910.74),
        ('W28' ,'W', 'B'   ,   4391.16, -11485.64,  -6382.93),
        ('W32' ,'W', 'AB'  ,   5470.50, -14443.13,  -8061.25),
        ('W36' ,'W', 'B'   ,   6671.47, -17678.17,  -9883.19),
        ('W40' ,'W', 'A'   ,   7988.65, -21181.36, -11844.80),
        ('W48' ,'W', 'A'   ,  10925.70, -28961.66, -16194.06),
        ('W56' ,'W', 'A'   ,  14206.44, -37731.09, -21114.62),
        ('W64' ,'W', 'A'   ,  17842.85, -47447.29, -26566.65),
        ('W72' ,'W', 'A'   ,  21802.57, -58074.16, -32540.96),
        ('E1'  ,'E', 'D'   ,    151.26,     23.33,   -218.44),
        ('E2'  ,'E', 'CD'  ,     37.71,    135.65,    -50.59),
        ('E3'  ,'E', 'D'   ,     73.37,    271.95,   -103.23),
        ('E4'  ,'E', 'BCD' ,    118.76,    445.77,   -170.46),
        ('E5'  ,'E', 'D'   ,    173.02,    653.27,   -250.51),
        ('E6'  ,'E', 'CD'  ,    235.66,    893.16,   -343.18),
        ('E7'  ,'E', 'D'   ,    305.29,   1163.76,   -448.46),
        ('E8'  ,'E', 'ABCD',    381.68,   1463.33,   -565.35),
        ('E9'  ,'E', 'D'   ,    465.79,   1790.89,   -692.95),
        ('E10' ,'E', 'C'   ,    558.29,   2145.87,   -830.16),
        ('E12' ,'E', 'BC'  ,    765.39,   2933.01,  -1133.62),
        ('E14' ,'E', 'C'   ,    999.66,   3822.35,  -1475.27),
        ('E16' ,'E', 'ABC' ,   1257.45,   4806.65,  -1855.04),
        ('E18' ,'E', 'C'   ,   1548.02,   5883.17,  -2264.55),
        ('E20' ,'E', 'B'   ,   1868.27,   7049.02,  -2704.15),
        ('E24' ,'E', 'AB'  ,   2552.45,   9638.20,  -3698.88),
        ('E28' ,'E', 'B'   ,   3331.17,  12556.45,  -4814.90),
        ('E32' ,'E', 'AB'  ,   4180.34,  15789.68,  -6060.60),
        ('E36' ,'E', 'B'   ,   5118.75,  19326.37,  -7416.80),
        ('E40' ,'E', 'A'   ,   6127.38,  23156.24,  -8890.33),
        ('E48' ,'E', 'A'   ,   8324.92,  31661.66, -12190.73),
        ('E56' ,'E', 'A'   ,  10813.96,  41248.78, -15902.56),
        ('E64' ,'E', 'A'   ,  13620.16,  51870.75, -19982.12),
        ('E72' ,'E', 'A'   ,  16204.22,  63678.31, -24269.82),
        ('N1'  ,'N', 'D'   ,      2.24,      0.05,      1.71),
        ('N2'  ,'N', 'CD'  ,   -100.24,    -15.93,    152.45),
        ('N3'  ,'N', 'D'   ,   -174.91,    -27.56,    262.39),
        ('N4'  ,'N', 'BCD' ,   -249.59,    -39.15,    372.31),
        ('N5'  ,'N', 'D'   ,   -361.68,    -56.66,    537.09),
        ('N6'  ,'N', 'CD'  ,   -495.22,    -77.43,    733.79),
        ('N7'  ,'N', 'D'   ,   -645.82,   -100.90,    955.52),
        ('N8'  ,'N', 'ABCD',   -812.58,   -126.88,   1200.98),
        ('N9'  ,'N', 'D'   ,   -995.39,   -155.53,   1469.71),
        ('N10' ,'N', 'C'   ,  -1193.00,   -186.06,   1760.66),
        ('N12' ,'N', 'BC'  ,  -1632.09,   -254.47,   2406.72),
        ('N14' ,'N', 'C'   ,  -2126.47,   -331.52,   3135.28),
        ('N16' ,'N', 'ABC' ,  -2673.19,   -416.88,   3943.10),
        ('N18' ,'N', 'C'   ,  -3271.25,   -510.29,   4826.72),
        ('N20' ,'N', 'B'   ,  -3917.13,   -611.39,   5784.82),
        ('N24' ,'N', 'AB'  ,  -5538.93,   -865.16,   8187.02),
        ('N28' ,'N', 'B'   ,  -6976.44,  -1089.30,  10305.16),
        ('N32' ,'N', 'AB'  ,  -8769.76,  -1369.81,  12961.02),
        ('N36' ,'N', 'B'   , -10732.66,  -1677.65,  15864.79),
        ('N40' ,'N', 'A'   , -12857.81,  -2009.03,  19009.59),
        ('N48' ,'N', 'A'   , -17583.09,  -2747.09,  25991.26),
        ('N56' ,'N', 'A'   , -22918.88,  -3578.90,  33852.66),
        ('N64' ,'N', 'A'   , -28827.03,  -4500.61,  42565.13),
        ('N72' ,'N', 'A'   , -35282.55,  -5508.69,  52098.85)],
        dtype=[('pad','U3'),('arm','U1'),('config','U4'),
              ('L_x','<f4'),('L_y','<f4'),('L_z','<f4')])

# %%
# angles (in radians) at which the HT is computed
tested_angles = np.linspace(0, np.pi, 180*5+1)

# %%
# annulus side of the source and background region we are considering
# while finding the linear streaks

# this value will be multiplied by the bmaj (or cmaj, if given) to get the annular region's radii
annulus_radii_factor=[5, 50]

# %%
# output of the artifact identification
output_df = pd.DataFrame({
    'RA': [],
    'DEC': [],
    'Xposn': [],
    'Yposn': [],
    'Identifier': [],
    'predictedAngle': [],
    'prelimPosAngle': [],
    'prelimNegAngle': [],
    'posAngle': [],
    'negAngle': [],
    'posAngleWidth': [],
    'negAngleWidth': [],
    'posStreakCoord': [],
    'negStreakCoord': [],
    'streak_center_posStreak_parameters': [],
    'streak_center_negStreak_parameters': [],
    'src_center_posStreak_parameters': [],
    'src_center_negStreak_parameters': [],
    'srccenter_srcwidth_posStreak_parameters': [],
    'srccenter_srcwidth_negStreak_parameters': []
})

output_df['predictedAngle'] = output_df['predictedAngle'].astype(object)

# %%
def median_absolute_value(data):
    ''' Given an array, it returns the median absolute value.
        This is the rms equivalent to RMS. '''
    return np.nanmedian(np.abs(data.data))
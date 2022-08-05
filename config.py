# --- EMULATING DATA variables ---
f_sampling = 16000           # sampling frequency in Hz
t_start = 0                 # start time of simulation 
t_end = 1                   # end time of simulation

# Source variables
away_distance = 700         # distance between the array and sources
# source1
f_start1 = 300              # lowest frequency that the source emitts
f_end1 = 350                # highest frequency that the source emitts
f_res1 = 20                 # resolution of frequency
theta_deg1 = 20             # theta angel of source placement, relative to origin of array
phi_deg1 = 40              # phi angel of source placement, relative to origin of array
t_start1 = 0                # start time of emission
t_end1 = 1.5                # end time of emission

# source1
f_start2 = 300              # lowest frequency that the source emitts
f_end2 = 350                # highest frequency that the source emitts
f_res2 = 20                 # resolution of frequency
theta_deg2 = -20             # theta angel of source placement, relative to origin of array
phi_deg2 = 40              # phi angel of source placement, relative to origin of array
t_start2 = 0                # start time of emission
t_end2 = 1.5                # end time of emission

# --- ANTENNA ARRAY setup variables ---
r_a1 = [-0.08, 0, 0]        # coordinate position of origin of array1
r_a2 = [0.08, 0, 0]         # coordinate position of origin of array2
r_a3 = [-0.24, 0, 0]        # coordinate position of origin of array3
r_a4 = [0.24, 0, 0]         # coordinate position of origin of array4
rows = 8                    # number of rows
columns = 8                 # number of columns
elements = rows*columns     # number of elements
distance = 20 * 10**(-3)    # distance between elements (m)

# --- FILTER variables ---
filter_order = 199          # filter order
scale_factor = 10000        # scale factor, adjusting filter width
f_bands_N = 45                                  # number of frequency bands
bandwidth = [100, f_sampling/2-f_sampling/100]  # bandwidth of incoming audio signal

# --- OTHER variables ---
# Modes for adaptive weights
modes = 7

# Beamforming resolution
x_res = 5                          # resolution in x
y_res = 5                          # resolution in y

# Listening values
theta_listen = 0
phi_listen = 45

# Initial values (startup values of microphone signals) to ignore
audio_signals = 'recorded'
initial_values = 40000

verbose = True

# Backend
backend = "cpu"  # Or gpu

# Index of which GPU device to use
gpu_device = 0
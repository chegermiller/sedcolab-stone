"""
Read output from SWASH.
Calculate short wave group envelope, incoming and outgoing IG waves, and phase lags between incoming IG and short wave envelope.

"""

import numpy as np
from scipy.signal import butter, filtfilt, lfilter, freqz, hilbert
import matplotlib.pyplot as plt
#import matplotlib.dates as mdates
#from matplotlib import ticker, cm, colors, colormaps

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Specify paths
genpath = '/Users/christiehegermiller/Projects/sedcolab-swash/'
runfolder = genpath + 'slope_1on35cont_swl_1p3/'
filename = 'grid_output3.tbl'
figfolder = genpath + 'figures/'

# Read the file into a NumPy array
try:
    data = np.loadtxt(runfolder + filename)
    print("2D Array from file")
    # print(data)
except FileNotFoundError:
    print(f"File '{filename}' not found.")
except ValueError as e:
    print(f"Error reading the file: {e}")

# Specify frequencies of bichromatic wave forcing
f1 = 0.4
f2 = 0.5
fp = (f1+f2)/2
fg = (np.abs(f1-f2))/2
# cutoff frequency for low-pass and high-pass filters
cutoff = fp/2

def butter_filter(data, cutoff, fs, btype='low', order=5):
    b, a = butter(order, cutoff, fs=fs, btype=btype, analog=False)
    #y = lfilter(b, a, data)
    y = filtfilt(b, a, data) # forwards and backwards to reduce phase shift
    return y

# Get the filter coefficients so we can understand its frequency response.
# b, a = butter(5, cutoff, fs=fs, btype='low', analog=False)

# Plot the frequency response.
# w, h = freqz(b, a, fs=fs, worN=8000)

# plt.subplot(2, 1, 1)
# plt.plot(w, np.abs(h), 'b')
# plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
# plt.axvline(cutoff, color='k')
# plt.xlim(0, 0.5*fs)
# plt.title("Lowpass Filter Frequency Response")
# plt.xlabel('Frequency [Hz]')
# plt.grid()
# plt.show()

# Find unique cross-shore locations in the array
unique_values, unique_indices = np.unique(data[:, 1], return_index=True)

bot_lev = data[unique_indices, 3]

phi_lag = []
r_lag = []
r0 = []
Hs_ig = []
Hs_ss = []

# xc_loc = unique_values[160] # at cross-shore position 40 m
for xc_loc in unique_values:
    # Find the indices of each unique value
    indices = np.where(data[:, 1] == xc_loc)[0]

    data_temp = data[indices, :]

    eta = data_temp[:, 2]
    depth = data_temp[:, 4]
    u = data_temp[:, 5]
    fs = 1/(data_temp[2, 0]-data_temp[1, 0]) # sampling frequency

    # Filter the data
    eta_lf = butter_filter(eta, cutoff, fs, btype='low', order=5)
    eta_hf = butter_filter(eta, cutoff, fs, btype='high', order=5)
    u_lf = butter_filter(u, cutoff, fs, btype='low', order=5)
    u_hf = butter_filter(u, cutoff, fs, btype='high', order=5)

    # Calculate Hs real quick
    Hs_ig.append(4 * np.nanstd(eta_lf))
    Hs_ss.append(4 * np.nanstd(eta_hf))

    # Find the short wave group envelope, amplitude and velocity
    A = np.abs(eta_hf + np.imag(hilbert(eta_hf)))
    A = butter_filter(A, cutoff, fs, btype='low', order=5)
    U = np.abs(u_hf + np.imag(hilbert(u_hf)))
    U = butter_filter(U, cutoff, fs, btype='low', order=5)

    # Identify the shoreward propagating IG
    g = 9.81
    eta_lf_onshore = (eta_lf + u_lf*np.sqrt(depth/g))/2
    eta_lf_offshore = (eta_lf - u_lf*np.sqrt(depth/g))/2

    if xc_loc == 0:
        fig, ax = plt.subplots(1)
        ax.plot(data_temp[:, 0], eta, '-', color='k', linewidth=1, label='$\eta$')
        ax.plot(data_temp[:, 0], eta_lf, '-', color='g', linewidth=1, label='$\eta_{IG}$')
        ax.plot(data_temp[:, 0], eta_lf_onshore, 'g--', linewidth=1, label='$\eta_{IG} onshore$')
        ax.plot(data_temp[:, 0], eta_hf, 'r-', linewidth=1, label='$\eta_{SS}$')
        ax.plot(data_temp[:, 0], A, 'b--', linewidth=1, label='$SS envelope$')
        ax.set_xlabel(r'\textit{Time} (sec)')
        ax.set_ylabel(r'\textit{[-]} (m)')
        ax.tick_params(direction="in")
        ax.grid()
        ax.legend()
        fig.set_figwidth(8)
        # fig.show()
        fig.savefig('1on20_SSIG_envelope_timeseries_offshore', dpi=350)

    # calculate lagged correlations and find phase lag
    # subsample to remove filtering edge impacts
    A_sub = A[20:100]
    eta_lf_onshore_sub = eta_lf_onshore[20:100]

    r = []
    tau = np.arange(0, 20, 0.5) # seconds
    for t in tau:
        A2 = A_sub**2
        A2_tau = np.ones_like(A2)*np.nan
        A2_tau[0:(len(A2_tau)-int(t*2))] = A2[int(t*2):] # lagged timeseries
        std_eta_lf_onshore = np.nanstd(eta_lf_onshore_sub)
        std_A2 = np.nanstd(A2_tau)
        r.append(np.nanmean(eta_lf_onshore_sub * A2_tau)/(std_eta_lf_onshore*std_A2)) # correlation
        #r.append(np.corrcoef(eta_lf_onshore_sub, A2_tau)[1,0])
    r = np.asarray(r)
    tau_maxr = tau[np.argmin(r)]

    omega_g = 2*np.pi/(1/fg) # IG angular frequency
    phi_lag.append(omega_g*tau_maxr)
    r_lag.append(np.abs(np.min(r)))

    # calculate correlation between IG and SS envelope velocities
    # subsample to remove filtering edge impacts
    U_sub = U[20:100]
    u_lf_sub = u_lf[20:100]

    std_u_lf = np.nanstd(u_lf_sub)
    std_U = np.nanstd(U_sub)
    r0.append(np.corrcoef(u_lf_sub, U_sub)[1,0])
    #r0.append(np.nanmean(u_lf_sub * U_sub) / (std_u_lf * std_U))  # correlation

# organize the data by time now and find breakpoint
unique_times = np.unique(data[:, 0])

bkpt = []

for t in unique_times:
    # Find the indices of each unique value
    indices = np.where(data[:, 0] == t)[0]

    bk = data[indices, 7]
    ind_bk = np.where(bk == 1)[0]
    if len(ind_bk)>0:
        ind_bk = ind_bk[0]
    elif len(ind_bk)==0:
        ind_bk = np.nan
    bkpt.append(ind_bk)

phi_lag = np.array(phi_lag)
r_lag = np.array(r_lag)
r0 = np.array(r0)
Hs_ig = np.array(Hs_ig)
Hs_ss = np.array(Hs_ss)
bkpt = int(np.nanmin(np.array(bkpt)))

fig, ax = plt.subplots(2, 1)
ax[0].plot(unique_values, -bot_lev, 'k-', lw=1)
ax[0].plot(unique_values, np.zeros_like(unique_values), '-.', color='steelblue', lw=1)
ax[0].plot(unique_values, Hs_ig, label='Hs IG')
ax[0].plot(unique_values, Hs_ss, label='Hs SS')
ax[0].fill_between(unique_values, np.ones_like(unique_values)*-1.3, -bot_lev, facecolor='gainsboro', edgecolor='None')
# define corner points
ft2m = 1./.3048
glassx = [115.41/ft2m, (115.41+48)/ft2m, (115.41+48)/ft2m, 115.41/ft2m]
glassy = [-1.3, -1.3, 0.3, 0.3]
ax[0].fill(glassx,glassy, edgecolor='darkgray', facecolor='None', alpha=.5)
ax[0].plot([unique_values[bkpt], unique_values[bkpt]], [-1.3, 0.3], '--', c='black', lw=1)
ax[0].set_xlim([-0.5, 59.5])
ax[0].set_ylim([-1.3, 0.3])
ax[0].tick_params(direction="in")
ax[0].set_ylabel(r'\textit{[-]} (m)')
ax[0].legend()

#ax[1].plot(unique_values, Hs_ig, label='Hs IG')
#ax[1].plot(unique_values, Hs_ss, label='Hs SS')
#ax[1].plot(unique_values, Hs_ig/Hs_ss, label='Hs_{IG}/Hs_{SS}')
ax[1].plot(unique_values, np.abs(r0), label='r0')
ax[1].plot([unique_values[bkpt], unique_values[bkpt]], [0, 1.5], '--', c='black', lw=1)
glassy = [0, 0, 1.5, 1.5]
ax[1].fill(glassx,glassy, edgecolor='darkgray', facecolor='None', alpha=.5)
ax[1].tick_params(direction="in")
ax[1].set_xlim([-0.5, 59.5])
ax[1].set_ylim([0, 1.5])
ax[1].set_ylabel(r'\textit{r0} (-)')
ax[1].set_xlabel(r'\textit{along-flume distance} (m)')
ax[1].legend(loc=3)
ax[1].grid()

ax2 = ax[1].twinx()
ax2.plot(unique_values, phi_lag, 'k.', markersize=4, label=r'$\phi_{lag}$')
ax2.set_yticks([0, np.pi, 2*np.pi], [r'\textit{0}', r'\textit{$\pi$}', r'\textit{2$\pi$}'])
ax2.tick_params(direction="in")
ax2.set_xlim([-0.5, 59.5])
ax2.set_ylim([0, 2*np.pi])
ax2.set_ylabel(r'\textit{$\phi_{lag}$} (rad)')
ax2.legend()
fig.set_figwidth(10)
#fig.show()
fig.savefig('1on20_SSIG_correlation', dpi=350)
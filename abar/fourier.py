"""
Fourier coefficient extraction of periodic Angstrom Bar data
11 October 2024
Nathan Ngqwebo
"""
import matplotlib.pyplot as plt
import ipywidgets as widgets
import scipy.stats as stats
import seaborn as sns
import numpy as np
import os
from uncertainties.umath import atan2, sqrt, log
from uncertainties import unumpy
from uncertainties import ufloat
noise_path = os.path.join(os.getcwd(), 'cold_noise.csv')
data_path = os.path.join(os.getcwd(), 'data.csv')
plt.rcParams.update({'font.size': 15})


def read_angstrom_data(filename:str) -> np.array:
    data = np.loadtxt(filename, skiprows=4, delimiter=',').astype(float)
    return data


def square_wave(t: np.ndarray, T_on: float, T_off: float) -> np.ndarray:
    """generate square wave given array of time steps"""
    period = T_on + T_off
    t_mod = t % period  # Find where each t falls within the current period
    return np.where(t_mod < T_on, 1, 0)


def square_analytic_decomposition(r:float, M: int = 10) -> tuple[list, list]:
    """Returns a_m, b_m Fourier coefficients of a square wave signal (non-symmetric).

    Parameters
    ----------
    signal : np.array
        Input signal array (assumed to contain 0s and 1s).
    M : int
        Number of modes (harmonics) to compute up to M.

    Returns
    -------
    tuple : (np.array, np.array)
        Arrays of a_m (sine coefficients) and b_m (cosine coefficients).
    """
    # Define a_m and b_m for the Fourier series
    def a_m(m:int):
        return np.sin(2*np.pi * m * r) / (m*np.pi)
    
    def b_m(m:int):
        return (1 - np.cos(2*np.pi * m * r)) / (m*np.pi) 
    
    return np.array([a_m(m) for m in range(1,M+1)]), np.array([b_m(m) for m in range(1,M+1)])


# DEPRECATED
# def numerical_decomposition(signal: np.array, T: float, M: int) -> tuple[list, list]:
#     """Returns numerically determined a_m, b_m Fourier coefficients of a periodic signal

#     Parameters
#     ----------
#     signal : np.array
#         Input signal array.
#     T : float
#         signal period.
#     N : int
#         Number of samples (dt = T / N).
#     M : int
#         Number of modes (harmonics) to compute up to M.
#     """
#     # signal = signal_full[np.where(signal_full[:,0] < T),:]
#     N = len(signal[:,0])
#     dt = T / N
#     print(dt)
#     # signal[:-1, 1] --- all but last element of amplitude values in signal
#     def a_m(m: int) -> float:
#         j = signal[:, 0] 
#         cosine_values = np.cos(2 * np.pi * m * j)  # Calculate the sine values for all j
#         result = (2 / N) * np.sum(signal[:, 1] * cosine_values)  # Perform the vectorized sum TODO debug signal[:-1,1]
#         return result
    
#     def b_m(m: int) -> float:
#         j = signal[:, 0] 
#         sine_values = np.sin(2 * np.pi * m * j)  # Calculate the sine values for all j
#         result = (2 / N) * np.sum(signal[:, 1] * sine_values)  # Perform the vectorized sum TODO debug signal[:-1,1]
#         return result

#     # a_m = (2 / N) * np.sum(signal[:-1, 1] * np.sin(2 * np.pi * m * ))
#     return np.array([a_m(m) for m in range(1,M+1)]), np.array([b_m(m) for m in range(1,M+1)])


def numerical_decomposition(signal_raw: np.array, T: float, M: int, delta_f:float=0.2) -> tuple[np.array, np.array]:
    """Returns numerically determined a_m, b_m Fourier coefficients of a periodic signal.

    Parameters
    ----------
    signal : np.array
        Input signal array with shape (N, 2) where column 0 is time and column 1 is amplitude.
    T : float
        Signal period.
    N : int
        Number of samples.
    M : int
        Number of modes (harmonics) to compute up to M.
    delta_f : float
        Standard uncertainty of  thermocouple readings
    """

    # signal = signal_full[(signal_full[:, 0] >= 0) & (signal_full[:, 0] < T)]
    N = len(signal_raw[:,0])
    dt = T / N
    
    # Precompute the time values
    t = signal_raw[:, 0]

    # give signal standard uncertainty: delta_f
    signal = np.column_stack((t, unumpy.uarray(signal_raw[:,1], np.full_like(signal_raw[:,1], delta_f))))

    # Initialize coefficients arrays
    # a_coefficients = np.zeros((M, 2))
    # b_coefficients = np.zeros((M, 2))
    a_coefficients = unumpy.uarray(np.zeros(M), np.zeros(M))
    b_coefficients = unumpy.uarray(np.zeros(M), np.zeros(M))


    for m in range(1, M):
        # Compute a_m
        a_coefficients[m - 1] = (2 / N) * np.sum(signal[:, 1] * np.cos(2 * np.pi * m * t / T))
        # Compute b_m
        b_coefficients[m - 1] = (2 / N) * np.sum(signal[:, 1] * np.sin(2 * np.pi * m * t / T))
    return a_coefficients, b_coefficients
    

def reconstruct_signal(signal: np.array, T: float, a_m: np.array, b_m: np.array):
    t = signal[:,0]
    a_0 = np.mean(signal[:,1])
    recon = np.full_like(signal[:, 1], a_0)  # Initialize with signal average value
    
    for m, (a, b) in enumerate(zip(a_m, b_m), start=1):
        # Add each harmonic contribution using a contracted expression
        d = np.sqrt(np.power(a,2) + np.power(b,2))
        phi = np.arctan2(a, b)
        recon += d * np.sin(2 * np.pi * m * t / T + phi)
    
    return np.column_stack((t, recon))


def reconstruct_square_signal(signal: np.array, T: float, r: float, a_m: np.array, b_m: np.array):
    """Reconstructs a square wave signal given its Fourier coefficients and period.

    Parameters
    ----------
    signal : np.array
        Array where the first column is time values, and the second column contains signal values.
    T : float
        Signal period.
    r : float
        Ratio of on-time to period: T_on / T. (Duty Cycle)
    a_m : np.array
        Fourier sine coefficients (for higher harmonics).
    b_m : np.array
        Fourier cosine coefficients (for higher harmonics).

    Returns
    -------
    np.array
        Reconstructed signal values corresponding to the input times.
    """
    t = signal[:, 0]  # Extract time values
    recon = np.full_like(signal[:, 1], r)  # Initialize with DC component (average)

    # Loop over the harmonic terms
    # for m, (a, b) in enumerate(zip(a_m, b_m), start=1):
    #     # Add each harmonic contribution
    #     recon += a * np.cos(2 * np.pi * m * t / T) + b * np.sin(2 * np.pi * m * t / T)

    for m, (a, b) in enumerate(zip(a_m, b_m), start=1):
        # Add each harmonic contribution using a contracted expression
        d = np.sqrt(np.power(a,2) + np.power(b,2))
        phi = np.arctan2(a, b)
        recon += d * np.sin(2 * np.pi * m * t / T + phi)

    # for m in range(1, len(a_m)+1):
    #     print(m)
    #     recon += (2 * np.sin(np.pi * m * r) * np.cos(2 * np.pi * m * t / T)) / (m * np.pi)
    return np.column_stack((t, recon))


def plot_coefficients(a_m:np.array, b_m:np.array) -> None:
    """produces stem plot of coefficients"""
    fig, ax = plt.subplots(1, figsize=(8, 5))
    m = np.arange(1, a_m.shape[0]+1, 1)
    ax.stem(m, a_m, linefmt='r', label='$a_m$')
    ax.stem(m, b_m, linefmt='b', label='$b_m$')
    ax.set(xlabel='$m$', ylabel='Coefficent Amplitude')
    ax.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_signal(signal:np.array, recon:np.array=None) -> None:
    fig, ax = plt.subplots(1, figsize=(8, 5))
    
    if recon is not None:
        ax.plot(recon[:,0], recon[:,1], '-r', linewidth=3, label='reconstruction')

    ax.step(signal[:,0], signal[:,1], '-k', linewidth=3, where='post', label='signal')
    ax.set(xlabel='time [s]', ylabel='Amplitude')
    ax.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def square_wave_example()->None:
    T_on = 500
    T_off = 400
    T = T_on + T_off
    time =  np.linspace(0, 2*T, 1000)
    signal = np.column_stack((time, square_wave(time, T_on, T_off)))

    # Plotting Signal   
    # plot_signal(signal)

    # plotting coefficients
    coefficients = square_analytic_decomposition(r=T_on/T, M=50)
    n_coefficients = numerical_decomposition(signal, T, 50)
    plot_coefficients(*coefficients)
    plot_coefficients(unumpy.nominal_values(n_coefficients[0]), unumpy.nominal_values(n_coefficients[1]))

    # plotting reconstrunction
    recon = reconstruct_square_signal(signal, T, T_on/T, *coefficients)
    n_recon = reconstruct_signal(signal, T, *n_coefficients)
    # plot_signal(recon, True)
    
    fig, ax = plt.subplots(1, figsize=(8, 5))
    
    ax.plot(recon[:,0], recon[:,1], '-r', linewidth=3, label='analytic reconstruction')
    ax.plot(n_recon[:,0], n_recon[:,1], '-b', linewidth=3, label='numerical reconstruction')
    ax.step(signal[:,0], signal[:,1], '-k', linewidth=3, where='post', label='Square Wave Signal')
    ax.set(xlabel='time [s]', ylabel='Amplitude')
    ax.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def get_thermal_uncertainty(noise:np.array) -> float:
    # noise = noise_signal[noise_signal[:,1] < 1]
    # print(noise)
    # fig, ax = plt.subplots(1)
    # ax.hist(noise[:,1], bins=50, density=True, label='dark signal')
    # # ax.scatter(x=noise[:,0], y=noise[:,1], label='noise')
    # plt.legend()
    # plt.show()


    # Sample data (replace with your dark data)
    # data = np.random.normal(loc=0, scale=1, size=100)  # Example normal data
    data = noise[:,1]
    std_dev = np.std(data)

    # 1. Visual Inspection
    # Histogram
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(data, bins=6, kde=True)
    plt.title('Histogram')

    # Q-Q Plot
    plt.subplot(1, 2, 2)
    stats.probplot(data, dist="norm", plot=plt)
    plt.title('Q-Q Plot')
    plt.tight_layout()
    plt.show()

    # 2. Statistical Tests
    shapiro_test = stats.shapiro(data)
    print("Shapiro-Wilk Test: statistic =", shapiro_test.statistic, ", p-value =", shapiro_test.pvalue)

    anderson_test = stats.anderson(data)
    print("Anderson-Darling Test: statistic =", anderson_test.statistic, ", critical values =", anderson_test.critical_values)

    ks_test = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
    print("Kolmogorov-Smirnov Test: statistic =", ks_test.statistic, ", p-value =", ks_test.pvalue)
    # var = np.histogram(noise[:,1])

    return std_dev


# def get_diffusivity(P_coeff: np.array, Q_coeff: np.array, T: float, L: float):
#     # Print coefficients for debugging
#     # print("P_coeff:", P_coeff)
#     # print("Q_coeff:", Q_coeff)

#     # Extract nominal values and uncertainties
#     P_a_m = unumpy.nominal_values(P_coeff[0])
#     P_b_m = unumpy.nominal_values(P_coeff[1])
#     Q_a_m = unumpy.nominal_values(Q_coeff[0])
#     Q_b_m = unumpy.nominal_values(Q_coeff[1])

#     P_a_std = unumpy.std_devs(P_coeff[0])
#     P_b_std = unumpy.std_devs(P_coeff[1])
#     Q_a_std = unumpy.std_devs(Q_coeff[0])
#     Q_b_std = unumpy.std_devs(Q_coeff[1])

#     m = np.arange(1, P_a_m.shape[0] + 1, 1)  # Modes from 1 to M for D_m

#     # Calculate arctan2 for P and Q coefficients
#     Tan = np.arctan2(P_a_m, P_b_m) - np.arctan2(Q_a_m, Q_b_m)

#     # Uncertainty propagation for arctan2
#     # Partial derivatives
#     dTan_dP_a = P_b_m / (P_a_m**2 + P_b_m**2)
#     dTan_dP_b = -P_a_m / (P_a_m**2 + P_b_m**2)
#     dTan_dQ_a = -Q_b_m / (Q_a_m**2 + Q_b_m**2)
#     dTan_dQ_b = Q_a_m / (Q_a_m**2 + Q_b_m**2)

#     # Combined uncertainty for Tan
#     Tan_uncertainty = np.sqrt(
#         (dTan_dP_a * P_a_std)**2 +
#         (dTan_dP_b * P_b_std)**2 +
#         (dTan_dQ_a * Q_a_std)**2 +
#         (dTan_dQ_b * Q_b_std)**2
#     )

#     # Calculate Ln
#     Ln = np.log(np.sqrt((np.power(P_a_m, 2) + np.power(P_b_m, 2)) / (np.power(Q_a_m, 2) + np.power(Q_b_m, 2))))

#     # Uncertainty propagation for Ln
#     # Partial derivatives
#     dLn_dP_a = (P_a_m / (P_a_m**2 + P_b_m**2)) / (2 * Ln)  # dLn/dP_a
#     dLn_dP_b = (P_b_m / (P_a_m**2 + P_b_m**2)) / (2 * Ln)  # dLn/dP_b
#     dLn_dQ_a = (-Q_a_m / (Q_a_m**2 + Q_b_m**2)) / (2 * Ln)  # dLn/dQ_a
#     dLn_dQ_b = (-Q_b_m / (Q_a_m**2 + Q_b_m**2)) / (2 * Ln)  # dLn/dQ_b

#     # Combined uncertainty for Ln
#     Ln_uncertainty = np.sqrt(
#         (dLn_dP_a * P_a_std)**2 +
#         (dLn_dP_b * P_b_std)**2 +
#         (dLn_dQ_a * Q_a_std)**2 +
#         (dLn_dQ_b * Q_b_std)**2
#     )

#     # Calculate diffusivity
#     diffusivity_m = ((np.power(L, 2) * np.pi * m) / T) * np.power(Ln * Tan , -1)  # Adjust for Tan uncertainty
#     diffusivity_m_uncertainty = diffusivity_m * np.sqrt((Tan_uncertainty / Tan)**2 + (Ln_uncertainty / Ln)**2)
#     print(diffusivity_m)
#     print(diffusivity_m_uncertainty)
#     # Combine results as ufloat for diffusivity
#     diffusivity_m_results = unumpy.uarray(unumpy.nominal_values(diffusivity_m), unumpy.std_devs(diffusivity_m_uncertainty))
#     # diffusivity_results = [ufloat(d, u) for d, u in zip(unumpy.nominal_values(diffusivity_m), diffusivity_m_uncertainty)]

#     return diffusivity_m_results


def get_diffusivity(P_coeff: np.array, Q_coeff: np.array, T: float, L: float):
    # print("P_coeff:", P_coeff)
    # print("Q_coeff:", Q_coeff)

    P_a_m = P_coeff[0]
    P_b_m = P_coeff[1]
    
    Q_a_m = Q_coeff[0]
    Q_b_m = Q_coeff[1]

    # Print to check for potential zero values
    # print("Q values for denominator check:", np.power(Q_a_m, 2) + np.power(Q_b_m, 2))

    # Calculate modes
    m = np.arange(1, P_a_m.shape[0] + 1, 1)  # Available modes from 1 to M for D_m

    # Safe calculation for Tan using a custom atan2 that handles zero cases
    Tan = np.array([
        atan2(P_a, P_b) - atan2(Q_a, Q_b) 
        if (Q_a != 0 or Q_b != 0) else 0  # Assign a default value if both Q_a and Q_b are zero
        for P_a, P_b, Q_a, Q_b in zip(P_a_m, P_b_m, Q_a_m, Q_b_m)
    ])

    # Prevent division by zero in Ln calculation
    denominator = np.power(Q_a_m, 2) + np.power(Q_b_m, 2)
    numerator = np.power(P_a_m, 2) + np.power(P_b_m, 2)
    args = np.array([sqrt((num) / np.where(den == 0, np.nan, den)) for num, den in zip(numerator, denominator)])
    Ln = np.array([log(arg) for arg in args])

    # Calculate diffusivity
    diffusivity_m = ((np.power(L, 2) * np.pi * m) / T) * np.power(Tan * Ln, -1)


    return diffusivity_m, conductivity



def analysis() -> None:
    # square_wave_example()
    noise = read_angstrom_data(noise_path)
    data = read_angstrom_data(data_path)
    # data = data[data[:, 0] > 2200, :]
    # print(len(data))
    T_on = 500
    T = 800
    noise_signal = noise[:, [0, 1]]
    heating_signal = data[:, [0, 1]]
    P_signal = data[:, [0, 2]]
    Q_signal = data[:, [0, 3]]
    # plot_signal(heating_signal)
    # plot_signal(P_signal)
    # plot_signal(Q_signal)
    
    SIGNAL = P_signal

    # plotting coefficients 
    n_coefficients = numerical_decomposition(SIGNAL, 800, 20)
    plot_coefficients(a_m=unumpy.nominal_values(n_coefficients[0]), b_m=unumpy.nominal_values(n_coefficients[1])) # TODO fix for uarrays

    # plotting reconstrunction
    n_recon = reconstruct_signal(SIGNAL, T, a_m=unumpy.nominal_values(n_coefficients[0]), b_m=unumpy.nominal_values(n_coefficients[1]))
    plot_signal(SIGNAL, n_recon)

    r = len(np.where(data[:,1] > 0)) / len(data[:,1])
    # print(f"r = {r} vs {500/800}")


def square_wave_example():
    T_on = 500
    T_off = 400
    T = T_on + T_off
    time = np.linspace(0, 2*T, 1000)
    signal = np.column_stack((time, square_wave(time, T_on, T_off)))

    # Plotting coefficients
    coefficients = square_analytic_decomposition(r=r, M=50)
    n_coefficients = numerical_decomposition(signal, T, 50)
    plot_coefficients(*coefficients)
    plot_coefficients(*n_coefficients)

    # Plotting reconstruction
    recon = reconstruct_square_signal(signal, T, r, *coefficients)
    n_recon = reconstruct_signal(signal, T, *n_coefficients)

    fig, ax = plt.subplots(1, figsize=(8, 5))

    ax.plot(recon[:, 0], recon[:, 1], '-r', linewidth=3, label='Analytic Reconstruction')
    ax.plot(n_recon[:, 0], n_recon[:, 1], '-b', linewidth=3, label='Numerical Reconstruction')
    ax.step(signal[:, 0], signal[:, 1], '-k', linewidth=3, where='post', label='Square Wave Signal')
    ax.set(xlabel='Time [s]', ylabel='Amplitude')
    ax.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    # TODO: maybe choose different r value to get steady state less heating more cooling

    # Noise Characterisation
    noise = read_angstrom_data(noise_path)
    noise = noise[noise[:,1] < 1]
    thermal_uncertainty = get_thermal_uncertainty(noise[:,[0, 2]])
    print(thermal_uncertainty)
    # RESULT: thermal uncertainty = FWHM of histogram = 0.2 degrees C

    # Diffusivity
    # square_wave_example()
    analysis() 
    data = read_angstrom_data(data_path)
    # data = data[data[:, 0] > 2200, :]
    # print(len(data))
    T_on = 500
    T = 800
    M = 20 # number of coefficients
    # delta_f = get_thermal_uncertainty(noise[:,[0, 2]]) # thermocouple standard uncertainty
    delta_f = 0.2
    noise_signal = noise[:, [0, 1]]
    heating_signal = data[:, [0, 1]]
    P_signal = data[:, [0, 2]]
    Q_signal = data[:, [0, 3]]

    P_coef = numerical_decomposition(P_signal, T, M, delta_f)
    Q_coef = numerical_decomposition(Q_signal, T, M, delta_f)

    L = ufloat(6, 0.1) / 100
    T = ufloat(800, 1)

    D_m = get_diffusivity(P_coef, Q_coef, T, L)
    # Prepare the data for saving
    data_to_save = np.array([[d.nominal_value, d.std_dev] for d in D_m])

    # Save to a text file
    np.savetxt('Diff_data.txt', data_to_save, fmt=['%.8e', '%.8e'], delimiter=',', header='Diffusivity, Uncertainty', comments='')
    
    print(D_m)
    D_m = np.array([d.nominal_value for d in D_m])
    
    fig, ax = plt.subplots(1, figsize=(8, 5))
    m = np.arange(1, D_m.shape[0] + 1, 1)
    ax.stem(m, D_m, linefmt='r', label='$D_m$')
    ax.set(xlabel='modes', ylabel='Diffusivity [$m^2/s$]')
    ax.legend()
    # ax.set(ylim=(-1e-2, 1e-2))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()


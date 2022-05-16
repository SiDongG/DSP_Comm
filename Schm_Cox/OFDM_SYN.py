import scipy.io
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class OFDM:
    pass
ofdm=OFDM()
ofdm.K = 1024                      # Number of OFDM subcarriers
ofdm.Kon = 600                     # Number of switched-on subcarriers
ofdm.CP = 128                      # CP length
ofdm.ofdmSymbolsPerFrame = 5       # N, number of payload symbols in each frame
ofdm.L = ofdm.K//2                 # Parameter L, denotes the length of one repeated part of the preamble

def random_qam(ofdm):
    qam = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    return np.random.choice(qam, size=(ofdm.Kon), replace=True)

def ofdm_modulate(ofdm, qam):
    assert (len(qam) == ofdm.Kon)
    fd_data = np.zeros(ofdm.K, dtype=complex)
    off = (ofdm.K - ofdm.Kon)//2
    fd_data[off:(off+len(qam))] = qam 
    fd_data = np.fft.fftshift(fd_data)
    symbol = np.fft.ifft(fd_data) * np.sqrt(ofdm.K)
    return np.hstack([symbol[-ofdm.CP:], symbol])

qam_preamble = np.sqrt(2)*random_qam(ofdm)
qam_preamble[::2] = 0   # delete every second data to make the preamble 2-periodic

preamble = ofdm_modulate(ofdm, qam_preamble)

plt.subplot(121)
plt.plot(abs(preamble))

plt.subplot(122)
f = np.linspace(-ofdm.K/2, ofdm.K/2, 4*len(preamble), endpoint=False)
plt.plot(f, 20*np.log10(abs(np.fft.fftshift(np.fft.fft(preamble, 4*len(preamble))/np.sqrt(len(preamble))))))

plt.show()

def createFrame(ofdm, qam_preamble=None):
    if qam_preamble is None:
        qam_preamble = np.sqrt(2)*random_qam(ofdm)
        qam_preamble[::2] = 0
    preamble = ofdm_modulate(ofdm, qam_preamble)
    
    payload = np.hstack([ofdm_modulate(ofdm, random_qam(ofdm)) for _ in range(ofdm.ofdmSymbolsPerFrame)])
    return np.hstack([preamble, payload])
frame = createFrame(ofdm, qam_preamble=qam_preamble*2)
plt.plot(abs(frame))
plt.show()

def addCFO(signal, cfo):  # Add carrier frequency offset (unused in this notebook)
    return signal * np.exp(2j*np.pi*cfo*np.arange(len(signal)))

def addSTO(signal, sto):  # add some time offset
    return np.hstack([np.zeros(sto), signal])

def addNoise(signal, sigma2):  # add AWGN
    noise = np.sqrt(sigma2/2) * (np.random.randn(len(signal)) + 1j*np.random.randn(len(signal)))
    return signal + noise

def addChannel(signal, h):       # add some multipath impulse response (unused in this notebook)
    return scipy.signal.lfilter(h, (1,), signal)

x = createFrame(ofdm, qam_preamble=qam_preamble*2)
sto = ofdm.K//2     #This is adding to Time domain symbols
r = addNoise(addSTO(x, sto), 0.5)

#Calculate P(d)--Direct Calculation
d_set = np.arange(0, sto + ofdm.K)
P1 = np.zeros(len(d_set), dtype=complex)
for i, d in enumerate(d_set):
    P1[i] = sum(r[d+m].conj()*r[d+m+ofdm.L] for m in range(ofdm.L))
plt.plot(d_set, abs(P1))
plt.show()

#Calculate P(d)--Recursive method given by Schmidl&Cox
P0 = sum(r[m].conj()*r[m+ofdm.L] for m in range(ofdm.L))  # initialize P[0] with the default method
def calcP_method2(r, d_set, P0):
    P2 = np.zeros(len(d_set), dtype=complex)
    P2[0] = P0
    for d in d_set[:-1]:
        P2[d+1] = P2[d] + r[d+ofdm.L].conj()*r[d+2*ofdm.L] - r[d].conj()*r[d+ofdm.L]
    return P2

#Calculate R(d)--Direct and recursive approach (Energy)
def calcR_method1(r, d_set):
    # calculation based on the iterative method
    R = np.zeros(len(d_set))
    for i, d in enumerate(d_set):
        R[i] = sum(abs(r[d+m+ofdm.L])**2 for m in range(ofdm.L))
    return R

def calcR_method2(r, d_set, R0):
    # calculation based on non-causal recursive expression
    R = np.zeros(len(d_set))
    R[0] = R0
    for d in d_set[:-1]:
        R[d+1] = R[d] + abs(r[d+2*ofdm.L])**2-abs(r[d+ofdm.L])**2
    return R
R1 = calcR_method1(r, d_set)
R2 = calcR_method2(r, d_set, R1[0])
plt.plot(d_set, abs(R1))
plt.show()

#Calculate Metric M(d)
M = abs(P1)**2/R1**2
plt.plot(d_set,M)
plt.show()
plt.plot(abs(r), label='$r[n]$', color='cyan')
plt.plot(M, label='$M(d)$')
plt.show()

def createFrame_withSyncInfo(ofdm):
    qam_preamble = np.sqrt(2)*random_qam(ofdm)
    qam_preamble[::2] = 0
    preamble = ofdm_modulate(ofdm, qam_preamble)

    payload = np.hstack([ofdm_modulate(ofdm, random_qam(ofdm)) for _ in range(ofdm.ofdmSymbolsPerFrame)])
    
    frame = np.hstack([preamble, payload])
    preamble_valid = np.zeros(len(frame))
    payload_valid = np.zeros(len(frame))
    preamble_valid[ofdm.CP:ofdm.CP+ofdm.K] = 1
    for f in range(ofdm.ofdmSymbolsPerFrame):
        payload_valid[(ofdm.CP+ofdm.K)*(f+1) + ofdm.CP + np.arange(ofdm.K)] = 1
    
    return frame, preamble_valid, payload_valid
frame, preamble_valid, payload_valid = createFrame_withSyncInfo(ofdm)
plt.plot(abs(frame), label='Signal');
plt.plot(preamble_valid, label='Preamble valid')
plt.plot(payload_valid, label='Payload valid')

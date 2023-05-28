from __future__ import division
import numpy as np
import scipy.interpolate 
import math
import os
import torch
import torch.nn as nn
import torch.optim as optim 

K = 64
CP = K//4
P = 64 # number of pilot carriers per OFDM block
#pilotValue = 1+1j
allCarriers = np.arange(K)  # indices of all subcarriers ([0, 1, ... K-1])
pilotCarriers = allCarriers[::K//P] # Pilots is every (K/P)th carrier.
#pilotCarriers = np.hstack([pilotCarriers, np.array([allCarriers[-1]])])
#P = P+1
dataCarriers = np.delete(allCarriers, pilotCarriers)
mu = 2
payloadBits_per_OFDM = len(dataCarriers)*mu  # number of payload bits per OFDM symbol

payloadBits_per_OFDM = K*mu

SNRdb = 20  # signal to noise-ratio in dB at the receiver 

mapping_table = {
    (0,0) : -1-1j,
    (0,1) : -1+1j,
    (1,0) : 1-1j,
    (1,1) : 1+1j,
}

demapping_table = {v : k for k, v in mapping_table.items()}

def Modulation(bits):                                        
    bit_r = bits.reshape((int(len(bits)/mu), mu))                  
    return (2*bit_r[:,0]-1)+1j*(2*bit_r[:,1]-1)                                    # This is just for QAM modulation

def OFDM_symbol(Data, pilot_flag):
    symbol = np.zeros(K, dtype=complex) # the overall K subcarriers
    #symbol = np.zeros(K) 
    symbol[pilotCarriers] = pilotValue  # allocate the pilot subcarriers 
    symbol[dataCarriers] = Data  # allocate the pilot subcarriers
    return symbol

def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data)

def addCP(OFDM_time):
    cp = OFDM_time[-CP:]               # take the last CP samples ...
    return np.hstack([cp, OFDM_time])  # ... and add them to the beginning

def channel(signal,channelResponse,SNRdb):
    convolved = np.convolve(signal, channelResponse)
    signal_power = np.mean(abs(convolved**2))
    sigma2 = signal_power * 10**(-SNRdb/10)  
    noise = np.sqrt(sigma2/2) * (np.random.randn(*convolved.shape)+1j*np.random.randn(*convolved.shape))
    return convolved + noise

def removeCP(signal):
    return signal[CP:(CP+K)]

def DFT(OFDM_RX):
    return np.fft.fft(OFDM_RX)


def equalize(OFDM_demod, Hest):
    return OFDM_demod / Hest

def get_payload(equalized):
    return equalized[dataCarriers]

def PS(bits):
    return bits.reshape((-1,))

def ofdm_simulate(codeword, channelResponse,SNRdb):       
    OFDM_data = np.zeros(K, dtype=complex)
    OFDM_data[allCarriers] = pilotValue
    OFDM_time = IDFT(OFDM_data)
    OFDM_withCP = addCP(OFDM_time)
    OFDM_TX = OFDM_withCP
    OFDM_RX = channel(OFDM_TX, channelResponse,SNRdb)
    OFDM_RX_noCP = removeCP(OFDM_RX)

    # ----- target inputs ---
    symbol = np.zeros(K, dtype=complex)
    codeword_qam = Modulation(codeword)
    symbol[np.arange(K)] = codeword_qam
    OFDM_data_codeword = symbol
    OFDM_time_codeword = np.fft.ifft(OFDM_data_codeword)
    OFDM_withCP_cordword = addCP(OFDM_time_codeword)
    OFDM_RX_codeword = channel(OFDM_withCP_cordword, channelResponse,SNRdb)
    OFDM_RX_noCP_codeword = removeCP(OFDM_RX_codeword)
    return np.concatenate((np.concatenate((np.real(OFDM_RX_noCP),np.imag(OFDM_RX_noCP))), np.concatenate((np.real(OFDM_RX_noCP_codeword),np.imag(OFDM_RX_noCP_codeword))))), abs(channelResponse) 


Pilot_file_name = 'Pilot_'+str(P)
if os.path.isfile(Pilot_file_name):
    print ('Load Training Pilots txt')
    # load file
    bits = np.loadtxt(Pilot_file_name, delimiter=',')
else:
    # write file
    bits = np.random.binomial(n=1, p=0.5, size=(K*mu, ))
    np.savetxt(Pilot_file_name, bits, delimiter=',')


pilotValue = Modulation(bits)

import torch
import torch.nn as nn

class DNN(nn.Module):
    def __init__(self, n_input, n_output, n_hidden_1, n_hidden_2, n_hidden_3):
        super(DNN, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(n_input, n_hidden_1),
            nn.ReLU(),
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.ReLU(),
            nn.Linear(n_hidden_2, n_hidden_3),
            nn.ReLU(),
            nn.Linear(n_hidden_3, n_output),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded


    
def training():
    training_epochs = 20
    batch_size = 256
    display_step = 5
    test_step = 1000
    examples_to_show = 10

    n_hidden_1 = 500
    n_hidden_2 = 250
    n_hidden_3 = 120
    n_input = 256
    n_output = 16 


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DNN(n_input, n_output, n_hidden_1, n_hidden_2, n_hidden_3).to(device)
    criterion = nn.MSELoss()
    learning_rate = 0.001
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)


    H_folder_train = 'C:/Users/sguo93/OFDM_DNN/DNN_Detection/H_dataset/'
    H_folder_test = 'C:/Users/sguo93/OFDM_DNN/DNN_Detection/H_dataset/'
    train_idx_low = 1
    train_idx_high = 301
    test_idx_low = 301
    test_idx_high = 401
    # Saving Channel conditions to a large matrix
    channel_response_set_train = []
    for train_idx in range(train_idx_low,train_idx_high):
        print("Processing the ", train_idx, "th document")
        H_file = H_folder_train + str(train_idx) + '.txt'
        with open(H_file) as f:
            for line in f:
                numbers_str = line.split()
                numbers_float = [float(x) for x in numbers_str]
                h_response = np.asarray(numbers_float[0:int(len(numbers_float)/2)])+1j*np.asarray(numbers_float[int(len(numbers_float)/2):len(numbers_float)])
                channel_response_set_train.append(h_response)

    channel_response_set_test = []
    for test_idx in range(test_idx_low,test_idx_high):
        print("Processing the ", test_idx, "th document")
        H_file = H_folder_test + str(test_idx) + '.txt'
        with open(H_file) as f:
            for line in f:
                numbers_str = line.split()
                numbers_float = [float(x) for x in numbers_str]
                h_response = np.asarray(numbers_float[0:int(len(numbers_float)/2)])+1j*np.asarray(numbers_float[int(len(numbers_float)/2):len(numbers_float)])
                channel_response_set_test.append(h_response)
            
    training_epochs = 2000
    print('Start')
    for epoch in range(training_epochs):
        print(epoch)
        if epoch > 0 and epoch%2000 == 0:
            learning_rate = learning_rate/5
        avg_cost = 0
        total_batch = 50

        for index_m in range(total_batch):
            input_samples = []
            input_labels = []
            for index_k in range(0, 1000):
                bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))
                channel_response = channel_response_set_train[np.random.randint(0,len(channel_response_set_train))]
                signal_output, para = ofdm_simulate(bits,channel_response,SNRdb)   
                input_labels.append(bits[16:32])
                input_samples.append(signal_output)
            batch_x = np.asarray(input_samples)
            batch_y = np.asarray(input_labels)

            batch_x = torch.from_numpy(batch_x).float()
            batch_y = torch.from_numpy(batch_y).float()

            outputs = model(batch_x)

            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_cost += loss/total_batch
        
        if epoch % display_step == 0:
            print("Epoch:",'%04d' % (epoch+1), "cost=", \
                        "{:.9f}".format(avg_cost))
            input_samples_test = []
            input_labels_test = []
            test_number = 1000
            # set test channel response for this epoch                    
            if epoch % test_step == 0:
                print ("Big Test Set ")
                test_number = 10000
            for i in range(0, test_number):
                bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))                        
                channel_response= channel_response_set_test[np.random.randint(0,len(channel_response_set_test))]
                signal_output, para = ofdm_simulate(bits,channel_response,SNRdb)
                input_labels_test.append(bits[16:32])
                input_samples_test.append(signal_output)
            batch_x = np.asarray(input_samples_test)
            batch_y = np.asarray(input_labels_test)

            batch_x = torch.from_numpy(batch_x).float()
            batch_y = torch.from_numpy(batch_y).float()

            output = model(batch_x)
            mean_error = torch.mean(abs(output - batch_y))
            #mean_error_rate = 1 - torch.mean(torch.mean(torch.tensor(torch.equal(torch.sign(output-0.5), torch.tensor(batch_y-0.5).float())), dim=1))
            print("OFDM Detection QAM output number is", n_output,",SNR = ", SNRdb, ",Num Pilot = ", P,", prediction and the mean error on test set are:", mean_error)

            batch_x = np.asarray(input_samples)
            batch_y = np.asarray(input_labels)

            batch_x = torch.from_numpy(batch_x).float()
            batch_y = torch.from_numpy(batch_y).float()

            output = model(batch_x)
            mean_error = torch.mean(abs(output - batch_y))
            #mean_error_rate = 1 - torch.mean(torch.mean(torch.tensor(torch.equal(torch.sign(output-0.5), torch.tensor(batch_y-0.5).float())), dim=1))
            print("Prediction and the mean error on train set are:", mean_error)

    print("finished")

training()
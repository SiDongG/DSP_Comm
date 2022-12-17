import numpy as np
import matplotlib.pyplot as plt
import math
tscale, x = np.load('groundtruth.npy')
tscale_measurement, y = np.load('measurements2.npy')
print(np.size(tscale))

lambda_p = 0.1
lambda_m = 0.3
delta_t = 0.002
g = 9.8
Q = lambda_p**2*np.array([[delta_t**3/3,delta_t**2/2],[delta_t**2/2,delta_t]])
G = np.array([0,delta_t])
R = lambda_m**2
A = G@Q@np.transpose(G)

s=np.arcsin(y[0])+np.arcsin(y[1])+np.arcsin(y[2])+np.arcsin(y[3])+np.arcsin(y[5])+np.arcsin(y[6])+np.arcsin(y[8])+np.arcsin(y[9])+np.arcsin(y[10])+np.arcsin(y[11])

print(s/10)

Num=len(tscale_measurement)
Kalman_Estimation=np.zeros((2,1,Num))
print(np.size(Kalman_Estimation))
Kalman_Prediction=np.zeros((2,1,Num))
Kalman_Estimation[0,:,0]=-0.96
Kalman_P=np.zeros((2,2,Num))
Kalman_Pprediction=np.zeros((2,2,Num))
Kalman_K=np.zeros((2,1,Num))



for t in np.arange(0,Num-1):
    # Prediction
    Kalman_Prediction[0,0,t] = Kalman_Estimation[0,0,t]+Kalman_Estimation[1,0,t]*delta_t
    Kalman_Prediction[1,0,t] = Kalman_Estimation[1,0,t]-g*delta_t*math.sin(Kalman_Estimation[0,0,t])
    # Construct F Matrix
    F = np.array([[1,delta_t],[-g*delta_t*math.cos(Kalman_Estimation[0,0,t]),1]])
    Kalman_Pprediction[:,:,t] = F@Kalman_P[:,:,t]@np.transpose(F)+A
    # Construc H Matrix 
    H = np.array([math.cos(Kalman_Prediction[0,0,t]),0])
    Kalman_K[:,0,t] = Kalman_Pprediction[:,0,t]@np.transpose(H)*(1/(H@Kalman_Pprediction[:,:,t]@np.transpose(H)+R))
    # Correction
    Kalman_Estimation[:,:,t+1] = Kalman_Prediction[:,:,t]+Kalman_K[:,:,t]*(y[t+1]-math.sin(Kalman_Prediction[0,0,t]))
    Kalman_P[:,:,t+1] = (np.identity(2)-Kalman_K[:,0,t]@H)@Kalman_Pprediction[:,:,t]

plt.figure(figsize=(8,6))
plt.scatter(tscale,x,label='Ground_Measurements')
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(tscale_measurement,Kalman_Estimation[0,0,:],label='Estimated_Measurements')
plt.legend(fontsize=10)
plt.xlabel('Time');plt.ylabel('Estimated Angle')
plt.grid(True)
plt.show()

print(len(x))
print(len(Kalman_Estimation))
#Calculate rms
rms = 0
for t in np.arange(0,Num):
    rms = rms + (Kalman_Estimation[0,0,t]-x[2*t])**2

print(rms)
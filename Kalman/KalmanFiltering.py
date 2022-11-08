import numpy as np 
from scipy import signal
import matplotlib.pyplot as plt
import math

np.random.seed(1992)
NumSteps = 201
TimeScale = np.linspace(0,10,NumSteps)
DeltaSim = np.diff(TimeScale)[0]
SigmaInput = 1
SigmaNoise = 0.5
F = np.array([[1,0,DeltaSim,0],[0,1,0,DeltaSim],[0,0,1,0],[0,0,0,1]])
Q = SigmaInput**2 * np.array([[DeltaSim**3/3,0,DeltaSim**2/2,0],
[0,DeltaSim**3/3,0,DeltaSim**2/2],
[DeltaSim**2/2,0,DeltaSim,0],
[0,DeltaSim**2/2,0,DeltaSim]])
H = np.array([[1,0,0,0],[0,1,0,0]])
R = SigmaNoise**2 * np.identity(2)
State = np.zeros((4,NumSteps))
NoisyMeasurements = np.zeros((2,NumSteps))
for t in np.arange(1,NumSteps):
    ProcessNoise = np.squeeze(np.matmul(np.linalg.cholesky(Q),np.random.randn(4,1)))
    State[:,t] = np.matmul(F,State[:,t-1]) + ProcessNoise
    MeasurementNoise = SigmaNoise * np.squeeze(np.random.randn(2))
    NoisyMeasurements[:,t] = np.matmul(H,State[:,t]) + MeasurementNoise

StateX1 = State[0,:]
StateX2 = State[1,:]
DownSampling = 2
NoisyMeasurements2 = NoisyMeasurements[:,::DownSampling]
MeasurementY1 = NoisyMeasurements[0,:]
MeasurementY2 = NoisyMeasurements[1,:]

## Construct Kalman Filter 
# Initialization
KalmanMeasurements = np.zeros((4,NumSteps))
KalmanP = np.zeros((4,4,NumSteps))
Innovation = np.zeros((2,NumSteps))
KalmanF = np.zeros((4,2,NumSteps))
ConditionalX = np.zeros((4,NumSteps))
ConditionalP = np.zeros((4,4,NumSteps))

print(ConditionalP.shape)
ConditionalP[:,:,0]=np.identity(4)

for t in np.arange(0,NumSteps-1):
    # Compute Innvoation
    Innovation[:,t] = NoisyMeasurements[:,t]-H@ConditionalX[:,t]
    KalmanF[:,:,t] = ConditionalP[:,:,t]@np.transpose(H)@np.linalg.inv(H@ConditionalP[:,:,t]@np.transpose(H)+R)
    # Correction on X and P
    KalmanMeasurements[:,t] = ConditionalX[:,t]+KalmanF[:,:,t]@Innovation[:,t]
    KalmanP[:,:,t] = ConditionalP[:,:,t]-KalmanF[:,:,t]@H@ConditionalP[:,:,t]
    # Prediction on Conditional X and P
    ConditionalX[:,t+1]=F@KalmanMeasurements[:,t]
    ConditionalP[:,:,t+1]=F@KalmanP[:,:,t]@np.transpose(F)+Q

Kalman1=KalmanMeasurements[0,0:-1]
Kalman2=KalmanMeasurements[1,0:-1]

plt.figure(figsize=(8,6))
plt.scatter(MeasurementY1,MeasurementY2,label='Measurements')
plt.scatter(StateX1[0],StateX2[0],label='start point',color='red')
plt.scatter(StateX1[len(StateX1)-1],StateX2[len(StateX2)-1],label='stop point',color='green')
plt.plot(StateX1,StateX2,label='Trajectory',color='black')
plt.plot(Kalman1,Kalman2,label='KalmanTrajectory',color='brown')
plt.legend(fontsize=10)
plt.xlabel('x1 coordinates');plt.ylabel('x2 coordinates')
plt.grid(True)
plt.show()

## Calculate Measurement rms
rms = 0
rmsK = 0
for t in np.arange(0,NumSteps-2):
    rms = rms + (StateX1[t]-MeasurementY1[t])**2+(StateX2[t]-MeasurementY2[t])**2
    rmsK = rmsK + (StateX1[t]-Kalman1[t])**2+(StateX2[t]-Kalman2[t])**2

rms = math.sqrt(rms)
rmsK = math.sqrt(rmsK)

print(rms)
print(rmsK)
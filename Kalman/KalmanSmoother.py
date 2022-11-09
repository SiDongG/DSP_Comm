import numpy as np 
from scipy import signal
import matplotlib.pyplot as plt
import math

np.random.seed(1992)
NumSteps = 201
TimeScale = np.linspace(0,10,NumSteps)
DeltaSim = np.diff(TimeScale)[0]
#DeltaSim = 0.1
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
NoisyMeasurements = NoisyMeasurements[:,::DownSampling]
MeasurementY1 = NoisyMeasurements[0,:]
MeasurementY2 = NoisyMeasurements[1,:]

Delta = 0.1
Fk = np.array([[1,0,Delta,0],[0,1,0,Delta],[0,0,1,0],[0,0,0,1]])
Qk = SigmaInput**2 * np.array([[Delta**3/3,0,Delta**2/2,0],
[0,Delta**3/3,0,Delta**2/2],
[Delta**2/2,0,Delta,0],
[0,Delta**2/2,0,Delta]])
NumDown = 101
## Construct Kalman Filter 
# Initialization
KalmanMeasurements = np.zeros((4,NumDown))
KalmanP = np.zeros((4,4,NumDown))
Innovation = np.zeros((2,NumDown))
KalmanF = np.zeros((4,2,NumDown))
ConditionalX = np.zeros((4,NumDown))
ConditionalP = np.zeros((4,4,NumDown))

ConditionalP[:,:,0]=np.identity(4)

for t in np.arange(0,NumDown):
    # Compute Innvoation
    Innovation[:,t] = NoisyMeasurements[:,t]-H@ConditionalX[:,t]
    KalmanF[:,:,t] = ConditionalP[:,:,t]@np.transpose(H)@np.linalg.inv(H@ConditionalP[:,:,t]@np.transpose(H)+R)
    # Correction on X and P
    KalmanMeasurements[:,t] = ConditionalX[:,t]+KalmanF[:,:,t]@Innovation[:,t]
    KalmanP[:,:,t] = ConditionalP[:,:,t]-KalmanF[:,:,t]@H@ConditionalP[:,:,t]
    # Prediction on Conditional X and P
    if t != NumDown-1:
        ConditionalX[:,t+1]=Fk@KalmanMeasurements[:,t]
        ConditionalP[:,:,t+1]=Fk@KalmanP[:,:,t]@np.transpose(Fk)+Qk

Kalman1=KalmanMeasurements[0,0:NumDown]
Kalman2=KalmanMeasurements[1,0:NumDown]

KalmanSM = np.zeros((4,NumDown))
KalmanKI = np.zeros((4,4,NumDown))
KalmanSM[:,NumDown-1]=KalmanMeasurements[:,NumDown-1]

## Construct Kalman Smoother 
for t in np.arange(0,NumDown-1):
    KalmanKI[:,:,NumDown-1-t] = KalmanP[:,:,NumDown-2-t]@np.transpose(Fk)@np.linalg.inv(Fk@KalmanP[:,:,NumDown-2-t]@np.transpose(Fk)+Qk)
    KalmanSM[:,NumDown-2-t] = KalmanMeasurements[:,NumDown-2-t]+KalmanKI[:,:,NumDown-1-t]@(KalmanSM[:,NumDown-1-t]-Fk@KalmanMeasurements[:,NumDown-2-t])

KalmanSM[:,0]=KalmanMeasurements[:,0]


KalmanS1 = KalmanSM[0,0:NumDown]
KalmanS2 = KalmanSM[1,0:NumDown]

print(KalmanS1.shape)
print(Kalman1.shape)

plt.figure(figsize=(8,6))
plt.scatter(MeasurementY1,MeasurementY2,label='Measurements')
plt.scatter(StateX1[0],StateX2[0],label='start point',color='red')
plt.scatter(StateX1[len(StateX1)-1],StateX2[len(StateX2)-1],label='stop point',color='green')
plt.plot(StateX1,StateX2,label='Trajectory',color='black')
plt.plot(Kalman1,Kalman2,label='KalmanTrajectory',color='brown')
plt.plot(KalmanS1,KalmanS2,label='SmoothedTrajectory',color='orange')
plt.legend(fontsize=10)
plt.xlabel('x1 coordinates');plt.ylabel('x2 coordinates')
plt.grid(True)
plt.show()

rms = 0
rmsK = 0
rmsS = 0
for t in np.arange(0,NumDown-1):
    rms = rms + (StateX1[2*t]-MeasurementY1[t])**2+(StateX2[2*t]-MeasurementY2[t])**2
    rmsK = rmsK + (StateX1[2*t]-Kalman1[t])**2+(StateX2[2*t]-Kalman2[t])**2
    rmsS = rmsS + (StateX1[2*t]-KalmanS1[t])**2+(StateX2[2*t]-KalmanS2[t])**2

rms = math.sqrt(rms)
rmsK = math.sqrt(rmsK)
rmsS = math.sqrt(rmsS)

print(rms)
print(rmsK)
print(rmsS)
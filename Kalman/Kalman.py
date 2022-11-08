import numpy as np 
from scipy import signal
import matplotlib.pyplot as plt

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

print(State)
StateX1 = State[0,:]
StateX2 = State[1,:]
DownSampling = 2
NoisyMeasurements = NoisyMeasurements[:,::DownSampling]
MeasurementY1 = NoisyMeasurements[0,:]
MeasurementY2 = NoisyMeasurements[1,:]


plt.figure(figsize=(8,6))
plt.scatter(MeasurementY1,MeasurementY2,label='Measurements')
plt.scatter(StateX1[0],StateX2[0],label='start point',color='red')
plt.scatter(StateX1[len(StateX1)-1],StateX2[len(StateX2)-1],label='stop point',color='green')
plt.plot(StateX1,StateX2,label='Trajectory',color='black')
plt.legend(fontsize=10)
plt.xlabel('x1 coordinates');plt.ylabel('x2 coordinates')
plt.grid(True)
plt.show()

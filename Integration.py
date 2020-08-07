import numpy as np
from scipy import interpolate
from scipy.interpolate import interp1d

def integrate(accel, dt = 1e-2, init_motion = None):
    if init_motion is None:
        init_motion = np.zeros((3,3))
    accel = np.array(accel)
    velocity = np.zeros_like(accel)
    pose = np.zeros_like(accel)
    N = accel.shape[1]
    dts = 1e-4
    motion = np.array(init_motion)
    for i in range(N):
        for k in range(int(dt/dts)):
            new_accel = accel[:,i]
            motion[0] = new_accel
            for j in range(2):
                motion[j+1] += motion[j] * dts
        velocity[:,i] = np.array(motion[1])
        pose[:,i] = np.array(motion[2])
    return accel, velocity, pose
    
def integrate_smooth(accel, dt = 1e-2, init_motion = None):
    if init_motion is None:
        init_motion = np.zeros((3,3))
    time = (np.arange(accel.shape[1],dtype=np.float64))*dt
    vel = np.zeros_like(accel)
    pose = np.zeros_like(accel)
    for i in range(3):
        tck = interpolate.splrep(time, accel[i], s=0,k=1)
        for j in range(accel[i].shape[0]):
            vel[i,j] = interpolate.splint(0, time[j], tck)
        vel[i]+=init_motion[1,i]
        tck = interpolate.splrep(time, vel[i], s=0,k=1)
        for j in range(vel[i].shape[0]):
            pose[i,j] = interpolate.splint(0, time[j], tck)
        pose[i]+=init_motion[2,i]
    return accel, vel, pose
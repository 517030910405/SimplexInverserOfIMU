import numpy as np
from scipy import interpolate
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
def function1():
    GT = np.loadtxt("pose_left.txt", delimiter=" ")


    x = np.linspace(0, 10, num=11, endpoint=True)
    y = np.cos(-x**2/9.0)
    f = interp1d(x, y)
    f2 = interp1d(x, y, kind='cubic')

    xnew = np.linspace(0, 10, num=41, endpoint=True)
    plt.plot(x, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--')
    plt.legend(['data', 'linear', 'cubic'], loc='best')
    plt.savefig("tmp.png")
def function2():
    x = np.arange(0, 2*np.pi+np.pi/4, 2*np.pi/8)
    y = np.sin(x)
    tck = interpolate.splrep(x, y, s=0,k=4)
    print(tck)
    xnew = np.arange(0, 2*np.pi, np.pi/50)
    ynew = interpolate.splev(xnew, tck, der=0)

    plt.figure()
    plt.plot(x, y, 'x', xnew, ynew,'-', xnew, np.sin(xnew),'--', x, y, 'b')
    plt.legend(['Linear', 'Cubic Spline', 'True'])
    plt.axis([-0.05, 6.33, -1.05, 1.05])
    plt.title('Cubic-spline interpolation')
    plt.show()
    plt.savefig("tmp2.png")
def fit(GT):
    t = np.arange(GT.shape[0],dtype=np.float64)*.1
    t_new = np.arange((GT.shape[0]-1)*10+1,dtype=np.float64)*.01
    # print(t)
    # print(t_new)
    pose = []
    vel = []
    accel = []
    for i in range(3):
        x = GT[:,i]
        tck = interpolate.splrep(t, x, s = 0, k = 5)
        x_new = interpolate.splev(t_new, tck, der=0)
        vel_new = interpolate.splev(t_new, tck, der = 1)
        accel_new = interpolate.splev(t_new, tck, der = 2)
        pose.append(x_new)
        # print("x",x)
        # print("x new",x_new)
        vel.append(vel_new)
        accel.append(accel_new)
    accel = np.array(accel)
    np.linalg.norm(accel,axis=0)
    vel = np.array(vel)
    print(accel)
    print(vel)

    vel_simplex = np.loadtxt("result_vel.csv")
    accel_simplex = np.loadtxt("result_accel.csv")
    t_simplex = np.arange(accel_simplex.shape[1])*1e-2
    # plt.figure()
    # plt.plot(pose[0], pose[1],'-', GT[:,0], GT[:,1],'--')
    # plt.legend(['Interpolation', 'True'])
    # plt.title('interpolation')
    # plt.savefig("interpolation_k=5,s=0.png")

    # plt.figure()
    # plt.plot(t_new, np.linalg.norm(vel,axis=0),'-')
    # plt.legend(['Interpolation', 'Simplex'])
    # plt.title('velocity')
    # plt.savefig("vel_interpolation_k=5,s=0.png")
    
    # plt.figure()
    # plt.plot(t_new, np.linalg.norm(accel,axis=0),'-')
    # plt.legend(['Interpolation', 'Simplex'])
    # plt.title('accel')
    # plt.savefig("accel_interpolation_k=5,s=0.png")

    # plt.figure()
    # plt.plot(t_new, np.linalg.norm(vel,axis=0),'--', t_simplex, np.linalg.norm(vel_simplex,axis=0),'--')
    # plt.legend(['Interpolation', 'Simplex'])
    # plt.title('velocity')
    # plt.savefig("vel_compare_k=5,s=0.png")
    
    plt.figure()
    plt.plot(t_new, np.linalg.norm(accel,axis=0),'--', t_simplex, np.linalg.norm(accel_simplex,axis=0),'--')
    plt.legend(['Interpolation', 'Simplex'])
    plt.title('accel')
    plt.savefig("accel_compare_.png")
    
    pass
def main():
    GT = np.loadtxt("pose_left.txt",delimiter=" ")
    fit(GT)
# function2()
main()
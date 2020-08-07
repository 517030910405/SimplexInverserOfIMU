import numpy as np
from scipy import interpolate
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from Integration import integrate,integrate_smooth
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
    pose = np.array(pose)
    return accel, vel, pose

def demo_interpolation(GT, accel, vel, pose):
    t = np.arange(GT.shape[0],dtype=np.float64)*.1
    t_new = np.arange((GT.shape[0]-1)*10+1,dtype=np.float64)*.01
    print(accel)
    print(vel)

    vel_simplex = np.loadtxt("result_vel.csv")
    accel_simplex = np.loadtxt("result_accel.csv")
    t_simplex = np.arange(accel_simplex.shape[1])*1e-2

    # print(np.arange(GT.shape[0],dtype=np.int32)*10)
    print(np.linalg.norm((np.array(pose).T)[np.arange(GT.shape[0],dtype=np.int64)*10]-GT[:,:3],axis=1).mean())

    init_motion_interpolation = np.array(
        [accel[:,0],
        vel[:,0],
        pose[:,0],]
    )

    init_motion_simplex = np.array(
        [[0,0,0],
        [0,0,0],
        GT[0,:3]]
    )


    accel_sth, vel_sth, pose_sth = integrate_smooth(accel,init_motion=init_motion_interpolation,dt=1e-2)
    accel_, vel_, pose_ = integrate(accel,init_motion=init_motion_interpolation,dt=1e-2)


    plt.figure()
    plt.plot(pose[0], pose[1],'-', GT[:,0], GT[:,1],'--')
    plt.legend(['Interpolation', 'True'])
    plt.title('interpolation')
    plt.savefig("interpolation_direct.png")

    plt.figure()
    plt.plot(pose_[0], pose_[1],'-',pose_sth[0], pose_sth[1],'-', GT[:,0], GT[:,1],'--')
    plt.legend(['Interpolation-reintegral-simple integration','Interpolation-reintegral-spline during integration', 'True'])
    plt.title('interpolation')
    plt.savefig("interpolation_reintegral.png")

    plt.figure()
    plt.plot(t_new, np.linalg.norm(vel,axis=0),'-')
    plt.legend(['Interpolation', 'Simplex'])
    plt.title('velocity')
    plt.savefig("vel_interpolation_direct.png")
    
    plt.figure()
    plt.plot(t_new, np.linalg.norm(accel,axis=0),'-')
    plt.legend(['Interpolation', 'Simplex'])
    plt.title('accel')
    plt.savefig("accel_interpolation_direct.png")

    plt.figure()
    plt.plot(t_new, np.linalg.norm(vel,axis=0),'--', t_simplex, np.linalg.norm(vel_simplex,axis=0),'--')
    plt.legend(['Interpolation', 'Simplex'])
    plt.title('velocity')
    plt.savefig("vel_compare_direct.png")

    plt.figure()
    plt.plot(t_new, np.linalg.norm(accel,axis=0),'--', t_simplex, np.linalg.norm(accel_simplex,axis=0),'--')
    plt.legend(['Interpolation', 'Simplex'])
    plt.title('accel')
    plt.savefig("accel_compare_.png")
    

def demo(GT, accel, vel, pose, rt = 'img2/'):
    t = np.arange(GT.shape[0],dtype=np.float64)*.1
    t_new = np.arange((GT.shape[0]-1)*10+1,dtype=np.float64)*.01
    print(accel)
    print(vel)

    vel_simplex = np.loadtxt("result_vel.csv")
    accel_simplex = np.loadtxt("result_accel.csv")
    t_simplex = np.arange(accel_simplex.shape[1])*1e-2

    # print(np.arange(GT.shape[0],dtype=np.int32)*10)
    print(np.linalg.norm((np.array(pose).T)[np.arange(GT.shape[0],dtype=np.int64)*10]-GT[:,:3],axis=1).mean())

    init_motion_interpolation = np.array(
        [accel[:,0],
        vel[:,0],
        pose[:,0],]
    )

    init_motion_simplex = np.array(
        [[0,0,0],
        [0,0,0],
        GT[0,:3]]
    )


    accel_sth, vel_sth, pose_sth = integrate_smooth(accel,init_motion=init_motion_interpolation,dt=1e-2)
    accel_, vel_, pose_ = integrate(accel,init_motion=init_motion_interpolation,dt=1e-2)

    accel_sth_resim, vel_sth_resim, pose_sth_resim = integrate_smooth(accel_simplex,init_motion=init_motion_simplex,dt=1e-2)
    accel_resim, vel_resim, pose_resim = integrate(accel_simplex,init_motion=init_motion_simplex,dt=1e-2)



    plt.figure()
    plt.plot(pose[0], pose[1],'-', GT[:,0], GT[:,1],'--')
    plt.legend(['Interpolation', 'True'])
    plt.title('interpolation')
    plt.savefig(rt+"interpolation_direct.png")

    plt.figure()
    plt.plot(pose_[0], pose_[1],'-',pose_sth[0], pose_sth[1],'-', GT[:,0], GT[:,1],'--')
    plt.legend(['Interpolation-reintegral-simple integration','Interpolation-reintegral-spline during integration', 'True'])
    plt.title('interpolation')
    plt.savefig(rt+"interpolation_reintegral.png")


    plt.figure()
    plt.plot(pose_resim[0], pose_resim[1],'-',pose_sth_resim[0], pose_sth_resim[1],'-', GT[:,0], GT[:,1],'--')
    plt.legend(['Simplex-reintegral-simple integration','Simplex-reintegral-spline during integration', 'True'])
    plt.title('Simplex')
    plt.savefig(rt+"Simplex_reintegral.png")

    plt.figure()
    plt.plot(pose[0][2500:], pose[1][2500:],'-', GT[250:,0], GT[250:,1],'--')
    plt.legend(['Interpolation', 'True'])
    plt.title('interpolation')
    plt.savefig(rt+"interpolation_ZoomIn.png")

    plt.figure()
    plt.plot(t_new, np.linalg.norm(vel,axis=0),'-')
    plt.legend(['Interpolation', 'Simplex'])
    plt.title('velocity')
    plt.savefig(rt+"interpolation_vel_direct.png")
    
    plt.figure()
    plt.plot(t_new, np.linalg.norm(accel,axis=0),'-')
    plt.legend(['Interpolation', 'Simplex'])
    plt.title('accel')
    plt.savefig(rt+"interpolation_accel_direct.png")

    plt.figure()
    plt.plot(t_new, np.linalg.norm(vel,axis=0),'--', t_simplex, np.linalg.norm(vel_simplex,axis=0),'--')
    plt.legend(['Interpolation', 'Simplex'])
    plt.title('velocity')
    plt.savefig(rt+"vel_compare_direct.png")

    plt.figure()
    plt.plot(
        t_new, np.linalg.norm(vel_,axis=0),'--', 
        t_new, np.linalg.norm(vel_sth,axis=0),'--',
        t_simplex, np.linalg.norm(vel_resim,axis=0),'--',
        t_simplex, np.linalg.norm(vel_sth_resim,axis=0),'--',
    )
    plt.legend([
        'Interpolation - Simple Integration','Interpolation - Spline Integration',
        'Simplex - Simple Integration','Simplex - Spline Integration',
    ])
    plt.title('velocity after re-integration')
    plt.savefig(rt+"vel_compare_re-integration.png")


    plt.figure()
    plt.plot(t_new, np.linalg.norm(accel,axis=0),'--', t_simplex, np.linalg.norm(accel_simplex,axis=0),'--')
    plt.legend(['Interpolation', 'Simplex'])
    plt.title('accel')
    plt.savefig(rt+"accel_compare_direct.png")
    
    pass
def main():
    GT = np.loadtxt("pose_left.txt",delimiter=" ")
    accel,vel,pose = fit(GT)
    demo(GT,accel,vel,pose)
# function2()
main()
## 
# Using Simplex Methods 
##

import numpy as np 
from scipy import optimize
import matplotlib.pyplot as plt

def upperTriangle(N):
    return np.where(
        np.arange(N).reshape((-1,1))>np.arange(N).reshape((1,-1)),
        np.ones((N,N)),
        np.zeros((N,N))
    )+ np.eye(N)
    pass

def IntegralBuilder(N, M, dT, InitMotion = None):
    """
    - Input
        - N: Time Steps
        - M: Integral Order
        - dT: delta T
        - InitMotion: Shape[M]
            - InitMotion[M-1] is the Init Position
            - InitMotion[M-2] is the Init Vel
    """
    if InitMotion is None:
        InitMotion = np.zeros((M,))
    A = np.eye(N)
    b = np.zeros((N,1))
    for i in range(M):
        Ap = upperTriangle(N) * dT
        A = np.matmul(Ap,A)
        b = np.matmul(Ap,b) + InitMotion[i]
    return [
        A,
        b,
    ]

def Integral(X, N, M, dT, InitMotion = None, early_split = 0):
    if InitMotion is None:
        InitMotion = np.zeros((M,))
    AnsMotion = np.zeros((M,))
    X = np.array(X)
    accel = [X]
    for i in range(M):
        Ap = upperTriangle(N) * dT
        X = np.matmul(Ap,X) + InitMotion[i]
        AnsMotion[i] = X[X.shape[0]-1-early_split]
        accel.append(X)
    return X, AnsMotion, accel 
def SimplexBuilder(A, b, GT, GTI):
    """
    - Input
        - A: shape[N,N]
        - b: shape[N,1]
        - GT: ground Truth shape[Length] or list
        - GTI: ground Truth Index shape[Length] or list
    """
    N = A.shape[0]
    GT = np.array(GT)
    GTI = np.array(GTI)
    K = GTI.shape[0]
    # print(K,N)
    a_eq = np.zeros((K,N+1))
    a_eq[:,:N] = np.array(
        A[GTI]
    )
    b_eq = GT-b.reshape((-1,))[GTI]
    z = np.zeros((N+1,))
    z[N] = 1
    a_ub = np.zeros((N*2,N+1))
    b_ub = np.zeros((N*2,))

    a_ub[:N,:N] = np.eye(N)
    a_ub[:,N:N+1] -=1
    a_ub[N:,:N] = -np.eye(N)
    # a_ub[:,N:N+1] -=1
    
    # print(a_eq)
    # print(b_eq)
    # print(a_ub)
    # print(b_ub)
    res = optimize.linprog(
        z,A_eq=a_eq,b_eq=b_eq,A_ub=a_ub,b_ub=b_ub,method="revised simplex",
        bounds=[(None,None)for i in range(N+1)],
    )
    # print(res)
    xE = np.copy(res.get("x"))
    return xE, res

def demo():
    A,b = IntegralBuilder(100,2,1e-2)
    X, res = SimplexBuilder(A,b,[5.],[99])
    X = X[:100]
    print(A)
    print(np.matmul(A,X))

def run_part(GT = None,init = None,order=3,early_split = 0):
    # init = np.array([1.,1.,1.])
    GT = np.array(GT)
    PoseFs = 10.
    RealFs = 100.
    MAXSIZE = GT.shape[0]*10
    FsFs = int(RealFs/PoseFs)
    A,b = IntegralBuilder(MAXSIZE,order,1/RealFs,init)
    SGT = list(GT)
    # print(SGT)
    X,res = SimplexBuilder(A,b,SGT,[(i+1)*FsFs-1 for i in range(0,len(SGT))])
    X = X[:MAXSIZE]
    # print("POSE",(np.matmul(A,X)+b.reshape((-1,))))
    Pose = Integral(X, MAXSIZE, order, 1/RealFs,init,early_split=early_split)
    # print(Pose[0],Pose[0].shape)
    return Pose[0], Pose[1], Pose[2]

def run():
    GT = np.loadtxt("pose_left.txt",delimiter=" ")
    GT[:,:3] -= GT[0:1,:3]
    # print(GT)
    # return
    PoseFs = 10.
    RealFs = 100.
    FsFs = int(RealFs/PoseFs)
    A,b = IntegralBuilder(100,3,1/RealFs,None)
    SGT = list(GT[:10,0])
    X,res = SimplexBuilder(A,b,SGT,[i*FsFs for i in range(0,len(SGT))])
    # print(res)
    X = X[:100]
    # print(X)
    # print(A.shape)
    # print(X.shape)
    print("POSE",(np.matmul(A,X)+b.reshape((-1,))))
    pass

def SampleSimplex():
    z = np.array([2, 3, 1])

    a = np.array([[1, 4, 2], [3, 2, 0]])

    b = np.array([8, 6])

    x1_bound = x2_bound = x3_bound =(0, None)

    from scipy import optimize

    res = optimize.linprog(z, A_ub=-a, b_ub=-b,bounds=(x1_bound, x2_bound, x3_bound),)
    print(res)
    print(res.get("x"))

def GetAccel(GT , higher = 1, init = None, batch = 20, MoreGT = 5, early_split = 50):
    GT = np.array(GT)
    order = 2+higher
    if init is None:
        init = np.zeros((order,))
        init[order-1] = GT[0]
    ACCEL = []
    VEL = []
    for i in range(GT.shape[0]//batch):
        Pose,init,accel = run_part(GT[i*batch:(i+1)*batch+MoreGT],init,order,early_split=early_split)
        print(init)
        ACCEL+=list(accel[len(accel)-2-1])[:batch*10]
        VEL +=list(accel[len(accel)-1-1])[:batch*10]
    init = np.zeros((2,))
    init[2-1] = GT[0]
    ANS = Integral(ACCEL,len(ACCEL),2,1e-2,init)
    # print(ANS[0])
    # print(ANS[1])
    # print(GT[:GT.shape[0]//batch*batch])
    print("OK")
    return ACCEL, VEL,  ANS[0]


def GetAllAccel(GT, higher = 0, init = [None,None,None] , batch = 30, MoreGT = 5, early_split = 50):
    allaccel = []
    allpose  = []
    allvel = []
    for i in range(3):
        accel, vel, pose = GetAccel(GT[:,i],higher,init[i],batch, MoreGT, early_split)
        allaccel.append(accel)
        allpose.append(pose)
        allvel.append(vel)
    allpose = np.array(allpose)
    allaccel = np.array(allaccel)
    allvel = np.array(allvel)
    return allpose, allvel, allaccel
    # print(allpose)

if (__name__=="__main__"):
    # demo()
    if False:
        ## This Part is the higher order estimation for lower variance in ACCEL
        GT = np.loadtxt("pose_left.txt",delimiter=" ")
        pose, vel, accel = GetAllAccel(GT[:],higher=1,batch=30)
        plt.plot(pose[0],pose[1])
        plt.savefig("simplex11.png")
        plt.plot(GT[:,0],GT[:,1])
        plt.savefig("simplex12.png")
    if True:
        ## This Part is the lower order estimation for lower vibration in POSE
        GT = np.loadtxt("pose_left.txt",delimiter=" ")
        pose, vel, accel = GetAllAccel(GT[:],higher=0,batch=40,MoreGT=5,early_split=50)
        plt.figure()
        plt.plot(pose[0],pose[1],'-',GT[:,0],GT[:,1],'--')
        plt.legend(['Simplex', 'True'])
        plt.title("Simplex")
        plt.savefig("simplexRt.png")
        np.savetxt("result_x.csv",pose)
        np.savetxt("result0.csv",accel)

        np.savetxt("result_vel.csv",vel)
        np.savetxt("result_accel.csv",accel)
    if False:
        GT = np.loadtxt("pose_left.txt",delimiter=" ")
        pose = np.loadtxt("result_x.csv")
        plt.figure()
        plt.plot(pose[0][2500:],pose[1][2500:],'-',GT[250:,0],GT[250:,1],'--')
        plt.legend(['Simplex', 'True'])
        plt.title("Simplex")
        plt.savefig("simplexZoomIn.png")


    if False:
        vel = np.loadtxt("result_vel.csv")
        accel = np.loadtxt("result_accel.csv")
        plt.figure()
        plt.plot(np.arange(accel.shape[1])*1e-2, np.linalg.norm(accel,axis=0),'-')
        plt.legend(['Simplex', 'True'])
        plt.title('accel')
        plt.savefig("sim_accel_k=5,s=0.png")
    if False:
        vel = np.loadtxt("result_vel.csv")
        accel = np.loadtxt("result_accel.csv")
        plt.figure()
        plt.plot(np.arange(vel.shape[1])*1e-2, np.linalg.norm(vel,axis=0),'-')
        plt.legend(['Simplex', 'True'])
        plt.title('velocity')
        plt.savefig("sim_vel_k=5,s=0.png")

        pass
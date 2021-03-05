from __future__ import print_function
from agimus_vision import py_agimus_vision as av
import numpy as np, pinocchio, sys, os

def toURDFTag(M):
    x,y,z = M.translation
    ori = '<origin xyz="{} {} {}" '.format(x,y,z)
    r,p,y = pinocchio.rpy.matrixToRpy(M.rotation)
    ori += 'rpy="{} {} {}"/>'.format(r,p,y)
    return ori

def vpToSE3(M):
    return pinocchio.SE3(np.array(M))
def se3toVP(M):
    return av.HomogeneousMatrix(M.homogeneous)

def cam_proj(x):
    return x[:2] / x[2]
def Jcam_proj(x):
    d = 1/x[2]
    return np.array([ [ d, 0, -x[0]*d**2],
                      [ 0, d, -x[1]*d**2] ])

def J_M_times_P(M,P):
    return np.hstack((M.rotation, -np.dot(M.rotation, pinocchio.skew(P))))

def image_point_residual(pij, cMi, oPij):
    return pij - cam_proj(cMi * oPij)
def Jimage_point_residual(pij, cMi, oPij):
    return - np.dot(Jcam_proj(cMi * oPij), J_M_times_P(cMi, oPij))

def image_point_residuals(pijss, cMis, oPijss):
    return np.hstack([ pij - cam_proj(cMi * oPij) for pijs, cMi, oPijs in zip(pijss, cMis, oPijss) for pij, oPij in zip(pijs, oPijs) ])
def Jimage_point_residuals(pijss, cMis, oPijss):
    ncols = len(cMis)*6
    nrows = 0
    col = 0
    JcMi = []
    for pijs, cMi, oPijs in zip(pijss, cMis, oPijss):
        for pij, oPij in zip(pijs, oPijs):
            J = np.zeros((2,ncols))
            J[:,col:col+6] = Jimage_point_residual(pij, cMi, oPij)
            JcMi.append(J)
            nrows += 2
        col += 6
    Jpi = np.zeros((nrows, 4))
    return np.hstack((Jpi, np.vstack(JcMi)))

def plane_residual(pi, cMis, oPijss):
    value = np.array([ (np.dot(pi[:3], cMi * oPij) + pi[3] ) for cMi, oPijs in zip(cMis, oPijss) for oPij in oPijs ])
    assert value.shape[0] == 5*4
    value[-2*4:] -= 0.004
    return value
    #TODO should be this but we add the 4 mm that we measured on the real model
    #return np.array([ (np.dot(pi[:3], cMi * oPij) + pi[3] ) for cMi, oPijs in zip(cMis, oPijss) for oPij in oPijs ])
def Jplane_residual(pi, cMis, oPijss):
    # derivative wrt to pi
    Jpi = [ (cMi*oPij).tolist() + [1,] for cMi, oPijs in zip(cMis, oPijss) for oPij in oPijs ]
    # derivative wrt to cMis
    JcMis = np.zeros((len(Jpi), len(cMis)*6))
    row = 0
    col = 0
    for cMi, oPijs in zip(cMis, oPijss) :
        for oPij in oPijs:
            JcMis[row,col:col+6] = np.dot(pi[:3], J_M_times_P(cMi, oPij))
            row += 1
        col += 6
    return np.hstack(( Jpi, JcMis ))

def plane_param_constraint(pi):
    return np.sum(pi[:3]**2) - 1
def Jplane_param_constraint(pi, ncols):
    return np.hstack((2*pi[:3], np.zeros(ncols-3)))

class System:
    def __init__(self, pijss, oPijss):
        self.pijss = pijss
        self.oPijss = oPijss

        from pinocchio import liegroups
        self.space = liegroups.Rn(4)
        for pijs in pijss:
            self.space *= liegroups.SE3()

    def image_point_residuals(self, cMis):
        return image_point_residuals(self.pijss, cMis, self.oPijss)
    def Jimage_point_residuals(self, cMis):
        return Jimage_point_residuals(self.pijss, cMis, self.oPijss)

    def plane_residual(self, pi, cMis):
        return plane_residual(pi, cMis, self.oPijss)
    def Jplane_residual(self, pi, cMis):
        return Jplane_residual(pi, cMis, self.oPijss)

    def plane_param_constraint(self, pi):
        return plane_param_constraint(pi)
    def Jplane_param_constraint(self, pi):
        return Jplane_param_constraint(pi, self.space.nv)

# minimize image_point_residuals
# pi, cMi
#  s.t.    plane_residual = 0

def gauss_newton(value, jacobian, x0, space,
        iter=100,
        ethr=0.001,
        Jthr=0.001,
        mthr=1e-4,
        ):
    X = x0

    def norm2(a): return np.sum(a**2)
    from numpy.linalg import norm

    while iter > 0:
        err = value(X)
        J = jacobian(X)
        els = norm2(err)
        if norm(err) < ethr:
            print("Error is very small")
            break
        if norm(J) < Jthr:
            print("Jacobian is very small")
            break
        d,res,rank,s = np.linalg.lstsq(J, -err)
        # do line search on els = norm2(err), Jls = 2 * err^T * J
        # els(u) = norm2(err(q + u*d)) ~ els(0) + u * Jls * d
        Jls = 2 * np.dot(err,J)
        m = np.dot(Jls, d)
        if abs(m) < mthr:
            print("m is very small.", m)
            break
        assert m < 0, str(m) + " should be negative"
        alpha = 1.
        c = 0.1 # factor for the linear part
        rate = 0.5
        alphaDefault = None
        while True:
            X2 = space.integrate(X, alpha*d)
            err2 = value(X2)
            els2 = norm2(err2)
            if els2 < els + c * alpha * m:
                break
            if alphaDefault is None and els2 < els:
                alphaDefault = alpha
            alpha *= rate
            if alpha < 1e-5:
                if alphaDefault is None:
                    print("failed to find a alpha that makes the error decrease. m =", m)
                    return toSE3dict(X)
                print("failed to find correct alpha")
                alpha = alphaDefault
                X2 = space.integrate(X, alpha*d)
                break
        if iter%10 == 0:
            print("{:4} {:^8} {:^8} {:^8} {:^8}".format("iter", "err", "J", "d","alpha"))
        print("{:4} {:8.5} {:8.5} {:8.5} {:8.5}".format(iter, np.sqrt(els2), norm(J), norm(d), alpha))
        X = X2
        iter -= 1
    return X

def penalty(system, pi0, cMis0,
        mu0 = 1e-5,
        maxIter=100,
        cthr = 1e-6,
        innerKwArgs={}):

    def xToArgs(X):
        return X[:4], [ pinocchio.XYZQUATToSE3(X[r:r+7]) for r in range(4, len(X), 7) ]
    def argsToX(pi, cMis):
        return np.hstack([pi,]+[ pinocchio.SE3ToXYZQUAT(cMi) for cMi in cMis ])

    class AL:
        def __init__(self, system):
            self.s = system
            self.mu = 0

        def residuals(self, x):
            pi, cMis = xToArgs(x)
            return np.hstack((self.s.image_point_residuals(cMis),
                              self.mu * self.s.plane_residual(pi, cMis),
                              self.mu * self.s.plane_param_constraint(pi)))
        def Jresiduals(self, x):
            pi, cMis = xToArgs(x)
            return np.vstack((self.s.Jimage_point_residuals(cMis),
                              self.mu * self.s.Jplane_residual(pi, cMis),
                              self.mu * self.s.Jplane_param_constraint(pi),
                              ))


    al = AL(system)
    al.mu = mu0
    X = argsToX(pi0, cMis0)

    for iter in range(maxIter):
        X = gauss_newton(al.residuals, al.Jresiduals, X, system.space, **innerKwArgs)
        pi, cMis = xToArgs(X)
        cstrv = np.linalg.norm(system.plane_residual(pi, cMis))
        cost = np.linalg.norm(system.image_point_residuals(cMis))
        print("Penalty: {:4} {:8.5} {:8.5} {:8.5}".format(iter, al.mu, cstrv, cost))
        if cstrv < cthr:
            break
        al.mu *= 5

    return pi, cMis

datadir="/home/jmirabel/devel/tiago/catkin_ws/src/tiago_calibration/part/"

I = av.Image()
I.read(datadir+"hole_01/frame0000.jpg")
#I.initDisplay()
I.display()

cam = av.makeTiagoCameraParameters()

pi0 = [1, 0, 0, 0]
cMis0 = []
pijss = []
oPijss = []

tags = ( (15, 0.0845), (6, 0.1615), (1, 0.0845), (100, 0.041), (101, 0.041), )

for id, size in tags:
    aprilTag = av.makeAprilTag()
    aprilTag.cameraParameters(cam)
    aprilTag.addTag(id, size, av.HomogeneousMatrix())
    if aprilTag.detect(I):
        aprilTag.drawDebug(I)
        oMt = np.array(aprilTag.getPose())
        cMis0.append(vpToSE3(oMt))
        pijss.append([ np.array(v) for v in aprilTag.getPoints(cam, id) ])
        oPijss.append([ np.array(v) for v in av.aprilTagPoints(size) ])
    else:
        raise ValueError("Could not detect tag " + str(id))
    del aprilTag

I.flush()
I.getClick()

# First find the pose in the camera frame
system = System(pijss, oPijss)
cpi, cMis = penalty(system, pi0, cMis0,
        #innerKwArgs={"mthr": 1e-8}
        )

# Now find the pose of the object in camera frame
#bMt15 = pinocchio.XYZQUATToSE3([ -0.1536, -0.03, 0.4985, 0.729386, -0.00577831, -0.00480235, 0.684062 ])
bMt6 = pinocchio.XYZQUATToSE3([ -0.045, 0, 0.005, 0, 0.0249974, 0, 0.999688 ]).inverse() \
        * pinocchio.SE3(pinocchio.rpy.rpyToMatrix(2.6653495081, -1.49863665688, -1.10843302167),
                np.array((0.255441487911, -0.0185894364178, 0.225616029867)))

cMb = cMis[1] * bMt6.inverse()
bMc = cMb.inverse()
bMis = [ bMc * cMi for cMi in cMis ]
# Compute pi in object frame
def changePlaneFrame(api, bMa):
    bpi = api.copy()
    bpi[:3] = np.dot(bMa.rotation, api[:3])
    bpi[3] = api[3] - np.dot(bMa.translation, bpi[:3])
    return bpi
bpi = changePlaneFrame(cpi, bMc)

aprilTagPart = av.makeAprilTag()
aprilTagPart.cameraParameters(cam)
for bMi, (id, size) in zip(bMis, tags):
    aprilTagPart.addTag(id, size, se3toVP(bMi))
    #aprilTagPart.addTag(id, size, av.HomogeneousMatrix(bMi.homogeneous[:3,:].tolist()))

# Now compute the position of the holes. They are assumed to be onto the computed plane.
aprilTagHole = av.makeAprilTag()
aprilTagHole.cameraParameters(cam)
aprilTagHole.addTag(230, 0.041, av.HomogeneousMatrix())
detector = aprilTagHole.detector()
J = av.Image()
initDisplay = True
bCs = []
for i in range(1, 19):
    holedatadir = datadir + "hole_{:02}/".format(i)
    tagCenters = []
    for fname in os.listdir(holedatadir):
        img = holedatadir + fname
        print("reading " + img)
        J.read(img)
        if not initDisplay:
            initDisplay = True
            J.initDisplay()
        J.display()
        detector.imageReady = False

        partDetected = aprilTagPart.detect(J)
        holeDetected = aprilTagHole.detect(J)
        if partDetected and holeDetected:
            cMb = vpToSE3(aprilTagPart.getPose())

            aprilTagPart.drawDebug(J)
            aprilTagHole.drawDebug(J)
            ps = [ np.array(v+[1,]) for v in aprilTagHole.getPoints(cam, 230) ]
            # Compute pi in camera frame
            _cpi = changePlaneFrame(bpi, cMb)
            # Project p onto plane pi, then compute centroid.
            # find t such that t * pi[:3]^T * p + pi[3] = 0
            ts = [ -_cpi[3] / np.dot(_cpi[:3], p) for p in ps ]
            cPs = [ t * p for t, p in zip(ts, ps) ]
            #print(ts)
            #print(cPs)
            #print([ np.dot(_cpi[:3], cP) + _cpi[3] for cP in cPs ])
            cC = np.mean(cPs, axis=0)
            bC = cMb.inverse() * cC
            tagCenters.append(bC)
            #p.project
            #oP = [ np.array(v) for v in av.aprilTagPoints(0.041) ]
        else:
            print("Could not detect part or hole. Detected tags {}".format(detector.getTagsId()))

        J.flush()
        J.getClick()
    bCs.append(np.mean(tagCenters, axis=0))
    print("hole", i, bC, np.var(tagCenters, axis=0))

# display result
def displayResult():
    for i in range(1, 19):
        holedatadir = datadir + "hole_{:02}/".format(i)
        img = holedatadir + "frame0000.jpg"
        print("reading " + img)
        J.read(img)
        J.initDisplay()
        J.display()
        detector.imageReady = False

        partDetected = aprilTagPart.detect(J)
        if partDetected:
            cMb = vpToSE3(aprilTagPart.getPose())
            for bC in bCs:
                cC = cMb * bC
                c = cam_proj(cC)
                u, v = cam.convertMeterPixel(c[0], c[1])
                J.displayPoint(u, v, 4)
        J.flush()
        J.getClick()

#print("Plane coeffs: {}".format(bpi))
#for (id, size), bMi in zip(tags, bMis):
#    print("Tag {}:\n{}".format(id, toURDFTag(bMi)))

def tagsInUrdfFormat(tags, bMis):
    tag_fmt="""
      <link name="tag36_11_{id:05}">
        <visual>
          <geometry>
            <mesh filename="package://gerard_bauzil/meshes/apriltag_36h11/tag36_11_{id:05}.dae" scale="{size} {size} 1."/>
          </geometry>
        </visual>
      </link>
      <joint name="to_tag_{id:05}" type="fixed">
        <parent link="base_link"/>
        <child link="tag36_11_{id:05}"/>
        {origin}
      </joint>"""
    urdf_str = ""
    for (id, size), bMi in zip(tags, bMis):
        urdf_str += tag_fmt.format(id=id, size=size, origin=toURDFTag(bMi))
    return urdf_str

def holesInUrdfFormat(bpi, bCs):
    # Check that plane normal points inward
    if bpi[1] < 0:
        R = pinocchio.Quaternion(np.array([1,0,0]), -bpi[:3])
    else:
        R = pinocchio.Quaternion(np.array([1,0,0]), bpi[:3])

    hole_fmt="""
      <link name="hole_{id:02}_link">
        <visual>
          <geometry>
            <sphere radius="0.005" />
          </geometry>
        </visual>
      </link>
      <joint name="to_hole_{id:02}" type="fixed">
        <parent link="base_link"/>
        <child link="hole_{id:02}_link"/>
        {origin}
      </joint>"""
    hole_str = ""
    for i, bC in zip(range(1, 19), bCs):
        hole_str += hole_fmt.format(id=i, origin = toURDFTag(pinocchio.SE3(R, bCs[i-1])))
    return hole_str

def generate_urdf(tags, bMis, bCs, bpi):
    return tagsInUrdfFormat(tags, bMis) + "\n" + holesInUrdfFormat(bpi, bCs)

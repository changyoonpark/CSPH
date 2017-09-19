import numpy as np

class Particle:
    def __init__(self, idx, x, f):
        self.idx = idx
        self.pos = x
        self.f = f
        self.lapl_f = None
        self.grad_f = None
        self.vol = None
        self.B = None
        self.L = None
        self.A = None
        self.M = None
        self.N0 = None
        self.N1 = None
        self.N2 = None
        self.P = None
        self.L = None
        self.neigh = []

    def __str__(self):

        s = "---------------------------------------\n"
        s+= "Particle : {}\n".format(self.idx)
        s+= "vol : " + str(self.vol) + "\n"
        s+= "pos : " + self.pos.__str__() + "\n"
        s+= "f : " + self.f.__str__() + "\n"
        s+= "grad_f : " + self.grad_f.__str__() + "\n"
        s+= "lapl_f : " + self.lapl_f.__str__() + "\n"
        s+= "P : " + self.P.__str__() + "\n"
        s+= "---------------------------------------"
        return s

def W(dist, smoothingLength) :
    H = smoothingLength * 0.5
    q = (dist / H)
    if (q > 2.0) :
 	    return 0.0;
    else : 
        foo = (1.0 - 0.5 * q);
        bar = foo * foo;
        res = (0.55704230082163367519109317180380 / (H * H)) * bar * bar * (2.0 * q + 1.0)
        return res;

def gW(dist, smoothingLength, direction):
    H = smoothingLength * 0.5;
    q = (dist / H);
    if (q > 2.0) :
        return np.array([0,0])
    else :
        foo = 1.0 - 0.5 * q;
        c = ((0.557042300821633675191)/(H*H*H))*(-5.0 * q)*foo*foo*foo;
        return c * direction;

    

import numpy as np
import random
import math
from helper import W, gW, Particle

EPS_SMALL = 1.0E-10

l = 1.0
n = 10

# dx = l/float(n)
# h = 2.5 * dx
dx = 0.1
h = 0.25

IDX2D = [(0, 0), (1, 1), (0, 1)]
NEG_DELTA_MN = np.array([-1.0, -1.0, 0.0])

particles = []

idx = 0
per = 0
for i in range(0,n+1):
    for j in range(0,n+1):
        pos = np.array([float(i) * dx, float(j) * dx])
        pos = pos + per * np.array([random.uniform(-1,1),random.uniform(-1,1)])
        f = pos[0] *pos[0]
        # f = math.sin(pos[0] * 2.0 * 3.141592) * math.sin(pos[1] * 2.0 * 3.141592)
        p = Particle(idx,pos,f)
        particles.append(p)
        idx += 1

# Find Volume
for i in range(0, len(particles)):
    pi = particles[i]
    kernelSum = 0.0
    for j in range(0, len(particles)):
        pj = particles[j]

        dist_ij = np.linalg.norm((pi.pos - pj.pos), 2)
        if dist_ij > h :
            continue        
        W_ij = W(dist_ij, h)
        pi.nneigh = pi.nneigh + 1 

        kernelSum += W_ij        
    pi.vol = 1.0 / kernelSum

# Find Renorm Matrix (1st derivative)
for i in range(0, len(particles)):
    pi = particles[i]
    vol_i = pi.vol
    f_i = pi.f
    B_i = np.matrix([[0.0,0.0],[0.0,0.0]])
    
    grad_f_i = np.array([0.0,0.0])
    for j in range(0, len(particles)):
        pj = particles[j]

        dist_ij = np.linalg.norm((pi.pos - pj.pos), 2)
        if ( dist_ij > h or pi == pj):
            continue        
        vol_j = pj.vol
        f_j = pj.f

        r_ij = (pi.pos - pj.pos)
        e_ij = (pi.pos - pj.pos) / dist_ij
        W_ij = W(dist_ij, h)
        gW_ij = gW(dist_ij,h,e_ij)
        
        B_i = B_i - vol_j * np.einsum('m,n->mn',r_ij, gW_ij)
        grad_f_i = grad_f_i + (vol_j * (f_j - f_i) * gW_ij)
    
    B_i = np.linalg.inv(B_i)
    pi.B = B_i
    foo = B_i.dot(grad_f_i.transpose())
    pi.grad_f = np.array([foo[0,0],foo[0,1]])


# Find Renorm Matrix (2nd derivative)
def method1():
    for i in range(0, len(particles)):
        pi = particles[i]
        vol_i = pi.vol
        f_i = pi.f
        grad_f_i = pi.grad_f
        B_i = pi.B

        A_i = np.einsum('m,n,q->mnq',np.array([0.0,0.0]),np.array([0.0,0.0]),np.array([0.0,0.0]))
        for j in range(0, len(particles)):
            pj = particles[j]

            dist_ij = np.linalg.norm((pi.pos - pj.pos), 2)
            if ( dist_ij > h or pi == pj):
                continue        
            vol_j = pj.vol
            f_j = pj.f

            r_ij = (pi.pos - pj.pos)
            e_ij = (pi.pos - pj.pos) / dist_ij
            W_ij = W(dist_ij, h)
            gW_ij = gW(dist_ij,h,e_ij)

            G_i_ok_gW_k = np.einsum('ok,k->o',B_i,gW_ij)
            A_i = A_i + vol_j * np.einsum('m,n,o->mno',r_ij,r_ij,G_i_ok_gW_k)
            # M_i_mnq = M_i_mnq + vol_j * np.einsum('m,n,q->mnq',r_ij,r_ij,gW_ij)

        # pi.A = np.einsum('kq,mnq->kmn',B_i,M_i_mnq)    
        pi.A = A_i


    for i in range(0, len(particles)):
        pi = particles[i]

        vol_i = pi.vol
        f_i = pi.f
        grad_f_i = pi.grad_f
        B_i = pi.B
        A_i = pi.A
        
        _N_i = np.einsum('i,j->ij',np.array([0.0,0.0,0.0]),np.array([0.0,0.0,0.0]))
        # print(pi)
        for j in range(0, len(particles)):
            pj = particles[j]

            dist_ij = np.linalg.norm((pi.pos - pj.pos), 2)
            if ( dist_ij > h or pi == pj):
                continue        
            vol_j = pj.vol
            f_j = pj.f

            r_ij = (pi.pos - pj.pos)
            e_ij = (pi.pos - pj.pos) / dist_ij
            W_ij = W(dist_ij, h)
            gW_ij = gW(dist_ij,h,e_ij)

            A_i_kmn_e_ij_k = np.einsum('kmn,k->mn',A_i,e_ij)
            r_ij_e_ij_mn = np.einsum('m,n->mn',r_ij,e_ij)             
            e_ij_gW_ij_op = np.einsum('o,p->op',e_ij,gW_ij)   
            
            # (0,0)(0,1)(1,1)
            for I in range(0,len(IDX2D)):
                m = IDX2D[I][0]
                n = IDX2D[I][1]

                for J in range(0,len(IDX2D)):
                    o = IDX2D[J][0]
                    p = IDX2D[J][1]

                    if o == p :
                        _N_i[I,J] = _N_i[I,J] + vol_j * ( A_i_kmn_e_ij_k[m,n] + r_ij_e_ij_mn[m,n] ) * (e_ij_gW_ij_op[o,p])
                    else :
                        _N_i[I,J] = _N_i[I,J] + vol_j * ( A_i_kmn_e_ij_k[m,n] + r_ij_e_ij_mn[m,n] ) * (e_ij_gW_ij_op[o,p] + e_ij_gW_ij_op[p,o])

        # print(_N_i)
        if np.linalg.cond(_N_i) < 100.0 :
            print("good : " + str(np.linalg.cond(_N_i)))         
            N_i = np.linalg.inv(_N_i).dot(NEG_DELTA_MN)
            pi.L = np.array( [ [ N_i[0], N_i[2] ],
                            [ N_i[2], N_i[1] ] ] )
        else:
            print(np.linalg.cond(_N_i))
            pi.L = np.array( [ [ float('nan'), float('nan') ],
                            [ float('nan'), float('nan') ] ] )
            # pi.L = np.array( [ [ 1.0, 0.0 ],
            #                    [ 0.0, 1.0 ] ] )



def method2():
    # Find Renorm Matrix (2nd derivative)
    for i in range(0, len(particles)):
        pi = particles[i]
        vol_i = pi.vol
        f_i = pi.f
        grad_f_i = pi.grad_f
        B_i = pi.B
        
        # \hat{B}_i^{op} : [R_i + S_i * B_i * P_i] = -\delta_{mn}

        R_i = np.einsum('m,n,o,p->mnop',np.array([0.0,0.0]),np.array([0.0,0.0]),np.array([0.0,0.0]),np.array([0.0,0.0]))
        S_i = np.einsum('m,n,k->mnk',np.array([0.0,0.0]),np.array([0.0,0.0]),np.array([0.0,0.0]))
        P_i = np.einsum('l,o,p->lop',np.array([0.0,0.0]),np.array([0.0,0.0]),np.array([0.0,0.0]))
        # B_i -> 'kl'
        for j in range(0, len(particles)):
            pj = particles[j]

            dist_ij = np.linalg.norm((pi.pos - pj.pos), 2)
            if ( dist_ij > h or pi == pj):
                continue        
            vol_j = pj.vol
            f_j = pj.f

            r_ij = (pi.pos - pj.pos)
            e_ij = (pi.pos - pj.pos) / dist_ij
            W_ij = W(dist_ij, h)
            gW_ij = gW(dist_ij,h,e_ij)
            
            R_i = R_i + vol_j * np.einsum('m,n,o,p->mnop',r_ij,e_ij,e_ij,gW_ij)
            S_i = S_i + vol_j * np.einsum('m,n,k->mnk',e_ij,e_ij,gW_ij)
            P_i = P_i + vol_j * np.einsum('l,o,p->lop',r_ij,r_ij,gW_ij)
        
        Q_i = R_i + np.einsum('mnk,kl,lop->mnop',S_i,B_i,P_i)
        
        _Q_i = np.array([[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]])
        
        # Flatten down to a matrix (2nd order tensor)
        for I in range(0,len(IDX2D)):
            m = IDX2D[I][0]
            n = IDX2D[I][1]
            for J in range(0,len(IDX2D)):
                o = IDX2D[J][0]
                p = IDX2D[J][1]
                _Q_i[I,J] = Q_i[m,n,o,p]
                

        pi.cond = np.linalg.cond(_Q_i)
        if (pi.cond > 10):
            pi.L = np.array([[float('nan'),float('nan')],[float('nan'),float('nan')]])
        else:
            _L_i = np.linalg.inv(_Q_i).dot(NEG_DELTA_MN)
            pi.L = np.array([[_L_i[0],_L_i[2]],[_L_i[2],_L_i[1]]])









method2()







for i in range(0, len(particles)):
    pi = particles[i]
    vol_i = pi.vol
    f_i = pi.f
    grad_f_i = pi.grad_f
    lapl_f_i = 0.0

    B_i = pi.B
    L_i = pi.L
    
    foo = 0.0

    for j in range(0, len(particles)):
        pj = particles[j]

        dist_ij = np.linalg.norm((pi.pos - pj.pos), 2)
        if ( dist_ij > h or pi == pj):
            continue        
        vol_j = pj.vol
        f_j = pj.f

        r_ij = (pi.pos - pj.pos)
        e_ij = (pi.pos - pj.pos) / dist_ij
        W_ij = W(dist_ij, h)
        gW_ij = gW(dist_ij,h,e_ij)

        if pi.cond > 10 or pi.nneigh < 15:
            foo = foo + vol_j * np.einsum('i,i->',(pj.grad_f - pi.grad_f),gW_ij)
            # foo = foo + vol_j * np.einsum('i,i->',(pj.grad_f),gW_ij)
            # foo = foo + 2.0 * vol_j * (pi.f - pj.f) / dist_ij * np.einsum('i,i->',e_ij,gW_ij)
        else:
            foo = foo + 2.0 * vol_j * ( np.einsum('mn,mn->',L_i,np.einsum('m,n->mn',e_ij,gW_ij)) * ((f_i - f_j) / (dist_ij) - np.einsum('i,i->',e_ij,grad_f_i) ))

    pi.lapl_f = foo;

        # print(pi)
    
print('"x","y","f","grad_x","grad_y","lapl","cond","nneigh","kernelsum","vol"')
for p in particles:
    print("{},{},{},{},{},{},{},{},{},{}".format(p.pos[0],p.pos[1],p.f,p.grad_f[0],p.grad_f[1],p.lapl_f,p.cond,p.nneigh,p.kernelSum,p.vol))



import numpy as np
import random
from helper import W, gW, Particle

EPS_SMALL = 1.0E-10

l = 1.0
n = 10

dx = l/float(n)
h = 3.0 * dx

# IDX2D = [(0,0),(1,1),(0,1),(1,0)]
IDX2D = [(0, 0), (1, 1), (0, 1), (1, 0)]
# NEG_DELTA_MN = np.array([-1.0,-1.0,0.0,0.0])
NEG_DELTA_MN = np.array([-1.0, -1.0, 0.0, 0.0])

particles = []

idx = 0
for i in range(0,n+1):
    for j in range(0,n+1):
        pos = np.array([float(i) * dx, float(j) * dx])
        pos = pos + 0.01 * np.array([random.uniform(-1,1),random.uniform(-1,1)])
        f = pos[0]
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

        # e_ij = (pi.pos - pj.pos) / dist_ij
        W_ij = W(dist_ij, h)
        # gW_ij = gW(dist,h,e_ij)

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

    pi.B = B_i = np.linalg.inv(B_i)
    pi.grad_f = B_i.dot(grad_f_i.transpose())



# Find Renorm Matrix (2nd derivative)
for i in range(0, len(particles)):
    pi = particles[i]
    vol_i = pi.vol
    f_i = pi.f
    grad_f_i = pi.grad_f
    B_i = pi.B
    
    # \hat{B}_i^{op} : [M_i^{mnop} + N_i^{mnop}] = -\delta_{mn}
    # N_i^{mnop} = (R_i^{mnk} . S_i^{opl} . B_i^{kl})

    M_i = np.einsum('m,n,o,p->mnop',np.array([0.0,0.0]),np.array([0.0,0.0]),np.array([0.0,0.0]),np.array([0.0,0.0]))
    N_i = np.einsum('m,n,o,p->mnop',np.array([0.0,0.0]),np.array([0.0,0.0]),np.array([0.0,0.0]),np.array([0.0,0.0]))
    R_i = np.einsum('m,n,k->mnk',np.array([0.0,0.0]),np.array([0.0,0.0]),np.array([0.0,0.0]))
    S_i = np.einsum('o,p,l->opl',np.array([0.0,0.0]),np.array([0.0,0.0]),np.array([0.0,0.0]))

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
        M_i = M_i + vol_j * np.einsum('m,n,o,p->mnop',r_ij,e_ij,e_ij,gW_ij)
        R_i = R_i + vol_j * np.einsum('m,n,k->mnk',e_ij,e_ij,gW_ij)
        S_i = S_i + vol_j * np.einsum('o,p,l->opl',r_ij,r_ij,gW_ij)
    
    N_i = np.einsum('mnk,opl,kl->mnop',R_i,S_i,B_i)

    P_i = M_i + N_i
    
    _P_i = np.array([[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]])
    
    for I in range(0,len(IDX2D)):
        m = IDX2D[I][0]
        n = IDX2D[I][1]
        for J in range(0,len(IDX2D)):
            o = IDX2D[J][0]
            p = IDX2D[J][1]
            _P_i[I,J] = P_i[m,n,o,p]
            

    try:
        _L_i = np.linalg.inv(_P_i).dot(NEG_DELTA_MN)
        pi.L = np.array([[_L_i[0],_L_i[2]],[_L_i[3],_L_i[1]]])
    except:
        pi.L = np.array([[1.0,0.0],[0.0,1.0]])






for i in range(0, len(particles)):
    pi = particles[i]
    vol_i = pi.vol
    f_i = pi.f
    grad_f_i = np.array([pi.grad_f[0,0],pi.grad_f[0,1]])
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

        foo = foo + 2.0 * vol_j * ( np.einsum('mn,mn->',L_i,np.einsum('m,n->mn',e_ij,gW_ij)) * ((f_i - f_j) / (dist_ij) - np.einsum('i,i->',e_ij,grad_f_i) ))
        # print(grad_f_i)
        # print(np.array(r_ij))
        # print(e_ij)
        # np.einsum('i,i->',e_ij,grad_f_i)
    pi.lapl_f = foo;

        # print(pi)
    
print('"x","y","f","grad_x","grad_y","lapl"')
for p in particles:
    print("{},{},{},{},{},{}".format(p.pos[0],p.pos[1],p.f,p.grad_f[0,0],p.grad_f[0,1],p.lapl_f))






























# # Find Renorm Matrix (2nd derivative)
# for i in range(0, len(particles)):
#     pi = particles[i]
#     vol_i = pi.vol
#     f_i = pi.f
#     grad_f_i = pi.grad_f
#     B_i = pi.B

#     A_i = np.einsum('m,n,q->mnq',np.array([0,0]),np.array([0,0]),np.array([0,0]))
#     for j in range(0, len(particles)):
#         pj = particles[j]

#         dist_ij = np.linalg.norm((pi.pos - pj.pos), 2)
#         if ( dist_ij > h or pi == pj):
#             continue        
#         vol_j = pj.vol
#         f_j = pj.f

#         r_ij = (pi.pos - pj.pos)
#         e_ij = (pi.pos - pj.pos) / dist_ij
#         W_ij = W(dist_ij, h)
#         gW_ij = gW(dist_ij,h,e_ij)

#         G_i_ok_gW_k = np.einsum('ok,k->o',B_i,gW_ij)
#         A_i = A_i + vol_j * np.einsum('m,n,o->mno',r_ij,r_ij,G_i_ok_gW_k)
#         # M_i_mnq = M_i_mnq + vol_j * np.einsum('m,n,q->mnq',r_ij,r_ij,gW_ij)

#     # pi.A = np.einsum('kq,mnq->kmn',B_i,M_i_mnq)    
#     pi.A = A_i



# # Find Renorm Matrix (2nd derivative)
# # for i in range(0, len(particles)):
# #     pi = particles[i]
# #     vol_i = pi.vol
# #     f_i = pi.f
# #     grad_f_i = pi.grad_f
# #     B_i = pi.B

# #     N0_i = np.einsum('m,n->mn',np.array([0,0]),np.array([0,0]))
# #     N1_i = np.einsum('m,n->mn',np.array([0,0]),np.array([0,0]))
# #     N2_i = np.einsum('m,n->mn',np.array([0,0]),np.array([0,0]))

# #     for j in range(0, len(particles)):
# #         pj = particles[j]

# #         dist_ij = np.linalg.norm((pi.pos - pj.pos), 2)
# #         if ( dist_ij > h or pi == pj):
# #             continue        
# #         vol_j = pj.vol
# #         f_j = pj.f

# #         r_ij = (pi.pos - pj.pos)
# #         e_ij = (pi.pos - pj.pos) / dist_ij
# #         W_ij = W(dist_ij, h)
# #         gW_ij = gW(dist_ij,h,e_ij)

# #         N0_i = N1_i + vol_j * np.einsum('m,n->mn',r_ij,gW_ij)
# #         N1_i = N1_i + vol_j * np.einsum('m,n->mn',e_ij,gW_ij)
# #         N2_i = N2_i + vol_j * np.einsum('m,n->mn',r_ij,r_ij * np.norm(gW_ij, 2))

# #     pi.A = np.einsum('kq,mnq->kmn',B_i,M_i_mnq)    
# #     pi.N0 = N0_i
# #     pi.N1 = N1_i
# #     pi.N2 = N2_i


# # for i in range(0, len(particles)):
# #     pi = particles[i]
# #     vol_i = pi.vol
# #     f_i = pi.f
# #     grad_f_i = pi.grad_f
# #     B_i = pi.B

# #     M_i = np.einsum('m,n,o->mno',np.array([0,0]),np.array([0,0]),np.array([0,0]))
# #     for j in range(0, len(particles)):
# #         pj = particles[j]

# #         dist_ij = np.linalg.norm((pi.pos - pj.pos), 2)
# #         if ( dist_ij > h or pi == pj):
# #             continue        
# #         vol_j = pj.vol
# #         f_j = pj.f

# #         r_ij = (pi.pos - pj.pos)
# #         e_ij = (pi.pos - pj.pos) / dist_ij
# #         W_ij = W(dist_ij, h)
# #         gW_ij = gW(dist_ij,h,e_ij)
        
# #         M_i = M_i + vol_j * np.einsum('m,n->mn',r_ij,gW_ij) + 


# # Find Renorm Matrix (continued)

# # for i in range(0, len(particles)):
# #     pi = particles[i]

# #     vol_i = pi.vol
# #     f_i = pi.f
# #     grad_f_i = pi.grad_f
# #     B_i = pi.B
# #     A_i = pi.A
    
# #     # N_i = np.einsum('m,n,o,p->mnop',np.array([0,0]),np.array([0,0]),np.array([0,0]),np.array([0,0]))
# #     nneigh = 0
# #     for j in range(0, len(particles)):
# #         pj = particles[j]

# #         dist_ij = np.linalg.norm((pi.pos - pj.pos), 2)
# #         if ( dist_ij > h or pi == pj):
# #             continue        
# #         nneigh += 1
# #         vol_j = pj.vol
# #         f_j = pj.f

# #         r_ij = (pi.pos - pj.pos)
# #         e_ij = (pi.pos - pj.pos) / dist_ij
# #         W_ij = W(dist_ij, h)
# #         gW_ij = gW(dist_ij,h,e_ij)

# #         A_i_kmn_e_ij_k = np.einsum('kmn,k->mn',A_i,e_ij)
# #         r_ij_e_ij_mn = np.einsum('m,n->mn',r_ij,e_ij)             
# #         e_ij_gW_ij_op = np.einsum('o,p->op',e_ij,gW_ij)   
# #         _N_i = _N_i + vol_j * np.einsum('mn,op->mnop',( A_i_kmn_e_ij_k + r_ij_e_ij_mn ), e_ij_gW_ij_op)

# #     N_i = np.einsum('I,J->IJ',np.array([0,0,0,0]),np.array([0,0,0,0]))
# #     for I in range(len(IDX2D)):
# #         m = IDX2D[I][0]
# #         n = IDX2D[I][1]

# #         for J in range(len(IDX2D)):
# #             o = IDX2D[J][0]
# #             p = IDX2D[J][1]

# #             N_i[I,J] = _N_i[m,n,o,p]
# #     print(nneigh)
#     # print(np.linalg.inv(N_i).dot(NEG_DELTA_MN))
    




# for i in range(0, len(particles)):
#     pi = particles[i]

#     vol_i = pi.vol
#     f_i = pi.f
#     grad_f_i = pi.grad_f
#     B_i = pi.B
#     A_i = pi.A
    
#     N_i = np.einsum('i,j->ij',np.array([0,0,0]),np.array([0,0,0]))
#     print(pi)
#     for j in range(0, len(particles)):
#         pj = particles[j]

#         dist_ij = np.linalg.norm((pi.pos - pj.pos), 2)
#         if ( dist_ij > h or pi == pj):
#             continue        
#         vol_j = pj.vol
#         f_j = pj.f

#         r_ij = (pi.pos - pj.pos)
#         e_ij = (pi.pos - pj.pos) / dist_ij
#         W_ij = W(dist_ij, h)
#         gW_ij = gW(dist_ij,h,e_ij)

#         A_i_kmn_e_ij_k = np.einsum('kmn,k->mn',A_i,e_ij)
#         r_ij_e_ij_mn = np.einsum('m,n->mn',r_ij,e_ij)             
#         e_ij_gW_ij_op = np.einsum('o,p->op',e_ij,gW_ij)   
        
#         # (0,0)(0,1)(1,1)
#         for I in range(0,len(IDX2D)):
#             m = IDX2D[I][0]
#             n = IDX2D[I][1]

#             for J in range(0,len(IDX2D)):
#                 o = IDX2D[J][0]
#                 p = IDX2D[J][1]

#                 if o == p :
#                     N_i[I,J] = N_i[I,J] + vol_j * ( A_i_kmn_e_ij_k[m,n] + r_ij_e_ij_mn[m,n] ) * (e_ij_gW_ij_op[o,p])
#                 else :
#                     N_i[I,J] = N_i[I,J] + vol_j * ( A_i_kmn_e_ij_k[m,n] + r_ij_e_ij_mn[m,n] ) * (e_ij_gW_ij_op[o,p] + e_ij_gW_ij_op[p,o])

#     print(N_i)
#     # print(np.linalg.inv(N_i).dot(NEG_DELTA_MN))

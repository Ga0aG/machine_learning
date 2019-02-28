#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 10:47:22 2019

@author: ga
"""

import numpy as np
''' 
  Q ,所有可能的状态的集合; 
  V ,所有可能的观测的集合； 
  A ,转移矩阵；
  B ,观测概率矩阵 emission probability;
  pi ,初始概率向量
  I ,长度为T的状态序列
  seq(O) ,对应的观测序列
'''
#A,B should be numpy array
class hmm():
    def __init__(self,
                 Q = None,
                 V = None,
                 A = None,
                 B = None,
                 pi = None):
        
        self.Q = [0,1,2] if Q is None else Q
        self.V = [0,1] if V is None else V
        self.N = len(self.Q) 
        self.M = len(self.V)
        # initial parameter ()
        self.A = A if A is not None else np.ones((self.N,self.N))/self.N
        self.B = B if B is not None else np.ones((self.N,self.M))/self.M
        self.pi = pi if pi is not None else np.ones(self.N)/self.N
        
        self.em_flag = False
        
    
    def forward(self, seq, pre=None, step=0):
        if step == 0:
            lst = [self.pi[i]*self.B[i,seq[0]] for i in range(self.N)]
            return self.forward(seq,lst,1)
        else:  
            if step == len(seq):
                if self.em_flag:
                    return pre
                else:
                    return sum(pre)
            else:
                lst = [0]*self.N
                for i in range(self.N):
                    for index,value in enumerate(pre):
                        lst[i] += value*self.A[index,i]
                    lst[i]*=self.B[i,seq[step]]
                return self.forward(seq,lst,step+1)
 
    def backward(self, seq, post=None, step=None):
        if step == None:
            post = [1]*self.N
            return self.backward(seq,post,len(seq)-1)
        else:
            if step == 0:
                if self.em_flag:
                    return post
                else:
                    return sum([self.pi[i]*self.B[i,seq[0]]*post[i] for i in range(self.N)])
            else:
                lst = [0]*self.N
                for i in range(self.N):
                    for index,value in enumerate(post):
                        lst[i] += value*self.A[i,index]*self.B[index,seq[step]]
                return self.backward(seq,lst,step-1)

    def chaseBack(self,preprob,preroute,seq,current):
        route=[[],[],[]]
        potemp = np.multiply(self.A,preprob)
        prob = []
        for i in range(len(preprob)):
            temp = np.argmax(potemp[:,i])
            route[i] = preroute[temp].copy()  
            route[i].append(i)
            prob.append(np.max(potemp[:,i])*self.B[i,seq[current]])
        prob = np.array(prob).reshape((self.N,1))
        if current==len(seq)-1:
            return route[ np.argmax(prob) ]
        else:
            return self.chaseBack(prob,route,seq,current+1)
    # seq is number
    def viterbi(self,seq):
        route = [[i] for i in range(self.N)]
        pro = np.multiply(self.pi,self.B[:,seq[0]]).reshape((self.N,1))
        return self.chaseBack(pro,route,seq,1)
    
    def baum_welch(self,seq,tol=0.01,max_iter=10):
        iters = 0
        while True:
            P_seq_old = self.forward(seq) #P(O|lamda)
            self.em_flag = True 
            gamma = [[]]*len(seq)
            for t in range(len(seq)):
                gamma[t] = [self.forward(seq[:t+1])[i]* self.backward(seq[t:])[i]/P_seq_old for i in range(self.N)]
            gamma = np.array(gamma)
            self.gamma=gamma
            
            epsilon = [np.empty_like(self.A)] * len(seq)
            for t in range(len(seq)-1):
                for i in range(self.N):
                    for j in range(self.N):
                        epsilon[t][i,j] = self.forward(seq[:t+1])[i]*self.A[i,j]*self.B[j,seq[t+1]]*self.backward(seq[t+1:])[j]
                epsilon[t]/=P_seq_old
            epsilon = np.array(epsilon)
            
            A = np.empty_like(self.A)
            for i in range(self.N):
                for j in range(self.N):
                    A[i,j] = np.sum(epsilon[:len(seq)-1,i,j])/np.sum(gamma[:len(seq)-1,i])
            B = np.empty_like(self.B)
            for j in range(self.N):
                for k in range(self.M):
                    B[j,k] = np.sum(gamma[np.where(np.array(seq)==k)][:,j])/np.sum(gamma[:,j])
            pi = gamma[0]
            iters+=1
            if iters>=max_iter:
                print('max_iter')
                break
            def reduce(A,B,pi):
                return np.concatenate((A.reshape(-1,1),B.reshape(-1,1),pi.reshape(-1,1)))
            if np.linalg.norm(reduce(A,B,pi)-reduce(self.A,self.B,self.pi))<tol:
                self.em_flag = False 
                break
            else:
                self.A,self.B,self.pi = A,B,pi
                self.em_flag = False 
        return A, B, pi
            
        


model = hmm(
        A = np.array([[0.6,0.2,0.2],[0.3,0.5,0.2],[0.5,0.2,0.3]]),
        B = np.array([[0.3,0.7],[0.6,0.4],[0.5,0.5]]),
        pi = np.array([0.3,0.4,0.3]) )

##Os = [[u,v,w]for u in [0,1] for v in [0,1] for w in [0,1]]
##results = []
##for O in Os:
##     prob = 0
##     for i1 in [0,1,2]:
##          for i2 in [0,1,2]:
##               for i3 in [0,1,2]:
##                    prob += model.pi[i1]*model.A[i1,i2]*model.A[i2,i3]*model.B[i1,O[0]]*model.B[i2,O[1]]*model.B[i3,O[2]]
##     results.append((O,prob))
#
print(model.forward([1,1,0])) #0.124502
model.em_flag = True
print(sum([model.forward([1,1,0])[i]* model.backward([0])[i] for i in range(model.N)]))#0.124502
print(sum([model.forward([1])[i]* model.backward([1,1,0])[i] for i in range(model.N)]))#0.124502
model.em_flag = False

model2 = hmm()
seq = [1,1,0]
print(model2.baum_welch(seq))

     

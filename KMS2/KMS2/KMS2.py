import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from os.path import join
from numba import njit

# p(k), <x>(t), eps(t), N(t)

#z polem
#tabela omega/n: K:1,5,10

data_dir_psi='out_psi.txt'
data_dir_pom='out_pom.txt'
data_dir_pk='out_pk.dat'
N=100
n=1
d_tau=.0001
d_x=1/N
xk=np.array([k*d_x for k in range(N+1)])
epochs=500000
K=0
t=0
omega=0*.5*(np.pi**2)
s_psi=250
s_NXEpk=250




def main():
	global data_dir_psi, data_dir_pom,data_dir_pk,t
	if len(sys.argv) == 5:
		zaladuj_parametry()
		data_dir_psi=sys.argv[2]
		data_dir_pom=sys.argv[3]
		data_dir_pk=sys.argv[4]
	#print(N,n,d_tau,d_x,xk,epochs,K,omega,s_psi,s_NXEpk)

	Psi_r=np.array([np.sqrt(2)*np.sin(np.pi*n*k*d_x) if np.sin(np.pi*n*k*d_x) != np.sin(np.pi) else 0 for k in range(N+1)])
	Psi_i=np.zeros(N+1)
	H_r=hamiltonian(Psi_r,t)
	H_i=hamiltonian(Psi_i,t)

	zapisz_psi_H(Psi_r,Psi_i,H_r,H_i,'w')

	rw_mode='w'
	i = 1
	j = 1
	for e in range(epochs):
		Psi_r,Psi_i,H_r,H_i=calkowanie(Psi_r,Psi_i,t)
		
		if e%s_psi == 0:
			zapisz_psi_H(Psi_r,Psi_i,H_r,H_i,'a')
			i+=1

		if e%s_NXEpk == 0:
			NN,XX,EPS,pk=pomocnicze(Psi_r,Psi_i,H_r,H_i)
			zapisz_NXE(NN,XX,EPS,rw_mode)
			zapisz_pk(pk,rw_mode)
			if rw_mode == 'w':
				rw_mode = 'a'
		t+=d_tau
#		print(t)

	#for ii in range(1,i):
	#	with open(join('output',data_dir_psi + f'{ii}.txt'),'r',encoding='utf8') as f:
	#		pass
	#for jj in range(1,j):
	#	with open(join('output',data_dir_pk + f'{jj}.txt'),'r',encoding='utf8') as f:
	#		x=[]
	#		for line in f:
	#			x.append(float(line.split('	')[1]))
	#
	#		pass

def zaladuj_parametry():
	global N,n,d_tau,d_x,xk,epochs,K,omega,s_psi,s_NXEpk

	with open(sys.argv[1], mode='r', encoding='utf8') as file:
		N=int(file.readline().split('#')[0])
		n=int(file.readline().split('#')[0])
		d_tau=float(file.readline().split('#')[0])
		epochs=int(file.readline().split('#')[0])
		K=float(file.readline().split('#')[0])
		omega=.5*(np.pi**2)*int(file.readline().split('#')[0])
		s_psi=int(file.readline().split('#')[0])
		s_NXEpk=int(file.readline().split('#')[0])

	d_x=1/N
	xk=np.array([k*d_x for k in range(N+1)])

def zapisz_psi_H(psi_r,psi_i,h_r,h_i,rw):
	data_dir=join('output',data_dir_psi)
	with open(data_dir, mode=rw, encoding='utf8') as f:
		if rw == 'w':
			f.write('x	Psi_r	Psi_i	H_r	H_i\n')
		else:
			f.write('\n')
		for k in range(N+1):
			f.write(f'\n{k/N}	{psi_r[k]}	{psi_i[k]}	{h_r[k]}	{h_i[k]}')

			
def zapisz_NXE(N,X,Eps,rw):
	global t
	data_dir=join('output',data_dir_pom)
	with open(data_dir, mode=rw, encoding='utf8') as f:
		if rw == 'w':
			f.write('t	N	x	epsilon\n')
		f.write(f'{t}	{N}	{X}	{Eps}\n')
		
def zapisz_pk(pk,rw):
	data_dir=join('output',data_dir_pk)
	with open(data_dir, mode=rw, encoding='utf8') as f:
		if rw == 'a':
			f.write('\n\n')
		for k in range(N+1):
			f.write(f'{k}	{pk[k]}\n')

@njit
def calkowanie(psi_r,psi_i,t):
	h_i = hamiltonian(psi_i,t)
	psi_r = psi_r + .5*h_i*d_tau
	h_r = hamiltonian(psi_r,t+.5*d_tau)
	psi_i = psi_i - h_r*d_tau
	h_i = hamiltonian(psi_i,t + d_tau)
	psi_r = psi_r + .5*h_i*d_tau

	return psi_r, psi_i, h_r, h_i

@njit
def hamiltonian(arr,tau):
	return np.array(
		[0] +
		[K*(d_x*k-.5)*arr[k]*np.sin(omega*tau) - .5*(arr[k+1] + arr[k-1] - 2*arr[k])/(d_x**2) for k in range(1,N)]
		+ [0])


@njit
def pomocnicze(psi_r,psi_i,h_r,h_i):
	pk=((psi_r**2)+(psi_i**2))

	NN=d_x*np.sum(pk)
	x=d_x*np.sum(xk*pk)
	eps=d_x*np.sum(psi_r*h_r+psi_i*h_i)

	return NN,x,eps,pk


if __name__ == '__main__':
	main()

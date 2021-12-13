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
omega=0
s_psi=250
s_NXEpk=250




def main():
	Psi_r=np.array([np.sqrt(2)*np.sin(np.pi*n*k/N) if np.sin(np.pi*n*k/N) != np.sin(np.pi) else 0 for k in range(N+1)])
	Psi_i=np.zeros(N+1)
	H_r=hamiltonian(Psi_r)
	H_i=hamiltonian(Psi_i)

	zapisz_psi_H(Psi_r,Psi_i,H_r,H_i,'w')

	rw_mode='w'
	i = 1
	j = 1
	for e in range(epochs):
		global t

		Psi_r,Psi_i,H_r,H_i=calkowanie(Psi_r,Psi_i)
		
		if e%s_psi == 0:
			zapisz_psi_H(Psi_r,Psi_i,H_r,H_i,'a')
			i+=1

		if e%s_NXEpk == 0:
			NN,XX,EPS,pk=pomocnicze(Psi_r,Psi_i)
			zapisz_NXE(NN,XX,EPS,rw_mode)
			zapisz_pk(pk,rw_mode)
			if rw_mode == 'w':
				rw_mode = 'a'
		t+=d_tau

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

#def zaladuj_parametry(data_dir):
#	global n,N
#	with open(data_dir, mode='r', encoding='utf8') as file:
#		n=int(file.readline().split('#')[0])
#		N=int(file.readline().split('#')[0])
#		d_tau=float(file.readline().split('#')[0])

def zapisz_psi_H(psi_r,psi_i,h_r,h_i,rw):
	data_dir=join('output',data_dir_psi+'.txt')
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
			f.write('t	N	x	epsilon\n\n')
		f.write(f'{t}	{N}	{X}	{Eps}\n')
		
def zapisz_pk(pk,rw):
	data_dir=join('output',data_dir_pk)
	with open(data_dir, mode=rw, encoding='utf8') as f:
		if rw == 'a':
			f.write('\n\n')
		for k in range(N+1):
			f.write(f'{k}	{pk[k]}\n')

@njit
def calkowanie(psi_r,psi_i):
	h_i=hamiltonian(psi_i)
	psi_r+=(h_i*d_tau*.5)
	h_r=hamiltonian(psi_r)
	psi_i-=(h_r*d_tau)
	h_i=hamiltonian(psi_i)
	psi_r+=(h_i*d_tau*.5)

	return psi_r, psi_i, h_r, h_i

@njit
def hamiltonian(arr):
	return np.array(
		[0] +
		[K*arr[k]*np.sin(omega*t)*(d_x*k-.5)-(arr[k+1] + arr[k-1] - 2*arr[k])/(2*(d_x**2)) for k in range(1,N)]
		+ [0])


@njit
def pomocnicze(psi_r,psi_i):
	pk=((psi_r**2)+(psi_i**2))

	NN=d_x*np.sum(pk)
	x=d_x*np.sum(xk*pk)
	eps=d_x*np.sum(psi_r*hamiltonian(psi_r)+psi_i*hamiltonian(psi_i))

	return NN,x,eps,pk


if __name__ == '__main__':
	main()

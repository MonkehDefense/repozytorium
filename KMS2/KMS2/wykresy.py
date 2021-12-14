import numpy as np
from matplotlib import pyplot as plt, animation
from os.path import join

N=100
d_x=1/N
x=np.array([k*d_x for k in range(N+1)])

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
fig, ax = plt.subplots()
ax.set(xlim=(-0.1,1.1), ylim=(-3, 3))
t = np.sin(x*2*np.pi)

X2, T2 = np.meshgrid(x, t)
sinT2 = np.sin(2 * np.pi * T2 / T2.max())
F = 0.9 * sinT2 * np.sinc(X2 * (1 + sinT2))
line, = ax.plot(x, F[0, :], color='k', lw=2)
def animate(i):
	y=[]
	with open(join('output',f'out_pk_{i}.txt'),'r',encoding='utf8') as f:
		for l in f:
			y.append(float(l.split('	')[1]))
	line.set_ydata(np.array(y))
anim = animation.FuncAnimation(fig, animate, interval=100, frames=range(1,1000))
anim.save('503.gif')
plt.show()

#import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib.animation import FuncAnimation, PillowWriter
#from os import listdir
#from os.path import join
#
#data_dir = 'output'
#
#fig, ax = plt.subplots()
#
#x = np.arange(0, 2*np.pi, 0.01)
#wykr, = ax.plot(x, np.sin(x))
#
#
#def animate(i):
#    wykr.set_ydata(np.sin(x + i / 50))  # update the data.
#    return wykr
#
#
#
#hl,=plt.plot([],[])
#plt.show()
#
#for file in listdir(data_dir):
#	x=[]
#	y=[]
#	if file[4:6] == 'pk':
#		with open(join(data_dir,file),'r',encoding='utf8') as f:
#			for line in f:
#				if len(line) > 0:
#					x.append(float(line.split('	')[0]))
#					y.append(float(line.split('	')[1]))
#		hl.set_xdata(np.array(x))
#		hl.set_ydata(np.array(y))
#		plt.draw()
		#print(f'odczytany plik {file}')

from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np


def EA_fitn_dev(fitness_res, run_s):
    fitness_s = fitness_res.copy()
    fitness_s.reset_index()

    fitness_s = fitness_s[fitness_s['run']==run_s]
    fitness_s = fitness_s.drop('run', axis = 1)
    fitness_s = fitness_s.set_index('generation')

    fitness_s.loc[:, fitness_s.columns.difference(['fitness_std'])].plot()
    fitness_s.plot(y='fitness_std')


def plt_land(f, domain, steps, a=15, b=-80):
    (x_min, x_max, y_min, y_max) = domain
    
    # Create a 3D array
    # meshgrid produces all combinations of given x and y
    x=np.linspace(x_min,x_max,steps)
    y=np.linspace(y_min,y_max,steps)
    X,Y=np.meshgrid(x,y)  # combine all x with all y
    
    # Applying the function
    Z = f(X,Y)

    norm = plt.Normalize(Z.min(), Z.max())
    colors = cm.ocean(norm(Z))
    rcount, ccount, _ = colors.shape

    fig = plt.figure(figsize=(9, 3))
    # plotting the surface
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.view_init(a,b)
    surf = ax.plot_surface(X, Y, Z, rcount=rcount, ccount=ccount, facecolors=colors, shade=False)
    surf.set_facecolor((0,0,0,0))
    ax.set_aspect('auto')
    ax.autoscale_view()
    ax.scatter( -1, -1, 0, color='r',s=10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    #plotting level curves
    levels = 15
    ay = fig.add_subplot(1,2,2)
    CS = ay.contour(X, Y, Z, levels, cmap=cm.ocean)
    ay.scatter(-1, -1, color='r')
    ay.clabel(CS, inline=True, fontsize=8)
    ay.set_aspect('auto')
  
    #adjusting
    plt.tight_layout()
    plt.show()
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


def EA_plt_land(f, domain, steps, a=15, b=-80):
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

    fig = plt.figure(figsize=(7, 3))
    # plotting the surface
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.view_init(a,b)
    surf = ax.plot_surface(X, Y, Z, rcount=rcount, ccount=ccount, facecolors=colors, shade=False)
    surf.set_facecolor((0,0,0,0))
    ax.scatter( -1, -1, 0, color='r', s=15)
    ax.set_aspect('auto')
    ax.autoscale_view()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    #plotting level curves
    levels = 15
    ay = fig.add_subplot(1,2,2)
    CS = ay.contour(X, Y, Z, levels, cmap=cm.ocean)
    ay.scatter(-1, -1, color='r', s=15)
    ay.clabel(CS, inline=True, fontsize=8)
    ay.set_aspect('auto')
    ay.set_xlabel('x')
    ay.set_ylabel('y')

    #adjusting
    plt.tight_layout()
    plt.show()


def EA_plt_pop(f, domain, steps, genera_res, run_s, gen_s, a=15, b=-80):
    query = (genera_res['function']=='population') & (genera_res['generation']==gen_s) & (genera_res['run']==run_s)
    population_s = genera_res[query]
    xp = population_s['gen_x'].values
    yp = population_s['gen_y'].values
    zp = population_s['fitness'].values

    (x_min, x_max, y_min, y_max) = domain

    # Create a 3D array
    # meshgrid produces all combinations of given x and y
    x=np.linspace(x_min,x_max,steps)
    y=np.linspace(y_min,y_max,steps)
    X,Y=np.meshgrid(x,y)  # combine all x with all y
    # Applying the function
    Z = f(X,Y)

    fig = plt.figure(figsize=(7, 3))
    # plotting the surface
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.view_init(a,b)
    surf = ax.plot_surface(X, Y, Z, cmap=cm.ocean, alpha=0.25)
    ax.scatter(xp, yp, zp, color='g', s=15)
    ax.set_aspect('auto')
    ax.autoscale_view()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    #plotting level curves
    levels = 15
    ay = fig.add_subplot(1,2,2)
    ay.contour(X, Y, Z, levels, cmap=cm.ocean)
    ay.scatter(xp, yp, color='g',  s=15)
    ay.set_aspect('auto')
    ay.set_xlabel('x')
    ay.set_ylabel('y')

    #adjusting
    plt.tight_layout()
    plt.show()
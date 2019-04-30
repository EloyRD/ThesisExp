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


def EA_plt_land(f, domain, point, steps, a=15, b=-80, imgsize=(15,10)):
    (x_min, x_max, y_min, y_max) = domain
    (x_plot, y_plot) = point

    # Create a 3D array
    # meshgrid produces all combinations of given x and y
    x=np.linspace(x_min,x_max,steps)
    y=np.linspace(y_min,y_max,steps)
    X,Y=np.meshgrid(x,y)  # combine all x with all y

    # Applying the function
    Z = f(X,Y)

    norm = plt.Normalize(Z.min(), Z.max())
    colors = cm.viridis(norm(Z))
    rcount, ccount, _ = colors.shape

    fig = plt.figure(figsize=imgsize)
    # plotting the surface
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter( x_plot, y_plot, f(x_plot, y_plot), color='r', s=20)
    ax.view_init(a,b)
    surf = ax.plot_surface(X, Y, Z, rcount=rcount, ccount=ccount, facecolors=colors, shade=False)
    surf.set_facecolor((0,0,0,0))
    ax.set_aspect('auto')
    ax.autoscale_view()
    ax.set_xlabel('gen_x')
    ax.set_ylabel('gen_y')
    ax.set_zlabel('fitness')

    #plotting level curves
    levels = 15
    ay = fig.add_subplot(1,2,2)
    ay.scatter(x_plot, y_plot, color='r', s=20)
    CS = ay.contour(X, Y, Z, levels, cmap='viridis', linewidths=1)
    ay.clabel(CS, inline=True, fontsize=8)
    ay.set_aspect('auto')
    ay.set_xlabel('gen_x')
    ay.set_ylabel('gen_y')

    #adjusting
    plt.tight_layout()
    plt.show()


def EA_plt_pop(f, domain, steps, genera_res, run_s, gen_s, a=15, b=-80, imgsize=(15,10)):
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

    fig = plt.figure(figsize=imgsize)
    # plotting the surface
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.view_init(a,b)
    ax.scatter(xp, yp, zp, color='r', s=20)
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.25, linewidth=0)
    ax.set_aspect('auto')
    ax.autoscale_view()
    ax.set_xlabel('gen_x')
    ax.set_ylabel('gen_y')
    ax.set_zlabel('fitness')

    #plotting level curves
    levels = 15
    ay = fig.add_subplot(1,2,2)
    ay.scatter(xp, yp, color='r',  s=20)
    ay.contour(X, Y, Z, levels, cmap='viridis', linewidths=.75)
    ay.set_aspect('auto')
    ay.set_xlabel('gen_x')
    ay.set_ylabel('gen_y')

    #adjusting
    plt.tight_layout()
    plt.show()


def EA_plt_gen(f, domain, steps, genera_res, run_s, gen_s, a=15, b=-80, imgsize=(15,10)):
    query = (genera_res['function']=='population') & (genera_res['generation']==gen_s) & (genera_res['run']==run_s)
    population_s = genera_res[query]
    xp = population_s['gen_x'].values
    yp = population_s['gen_y'].values
    zp = population_s['fitness'].values

    query = (genera_res['function']=='progeny') & (genera_res['generation']==gen_s) & (genera_res['run']==run_s)
    progeny_s = genera_res[query]
    xg = progeny_s['gen_x'].values
    yg = progeny_s['gen_y'].values
    zg = progeny_s['fitness'].values

    (x_min, x_max, y_min, y_max) = domain

    # Create a 3D array
    # meshgrid produces all combinations of given x and y
    x=np.linspace(x_min,x_max,steps)
    y=np.linspace(y_min,y_max,steps)
    X,Y=np.meshgrid(x,y)  # combine all x with all y
    # Applying the function
    Z = f(X,Y)

    fig = plt.figure(figsize=imgsize)
    # plotting the surface
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.view_init(a,b)
    ax.scatter(xp, yp, zp, color='r', s=20)
    ax.scatter(xg, yg, zg, color='g', s=17.5)
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.3, linewidth=0)
    ax.set_aspect('auto')
    ax.autoscale_view()
    ax.set_xlabel('gen_x')
    ax.set_ylabel('gen_y')
    ax.set_zlabel('fitness')

    #plotting level curves
    levels = 15
    ay = fig.add_subplot(1,2,2)
    ay.scatter(xp, yp, color='r',  s=20)
    ay.scatter(xg, yg, color='g', s=17.5)
    ay.contour(X, Y, Z, levels, cmap='viridis', linewidths=.75)
    ay.set_aspect('auto')
    ay.set_xlabel('gen_x')
    ay.set_ylabel('gen_y')

    #adjusting
    plt.tight_layout()
    plt.show()
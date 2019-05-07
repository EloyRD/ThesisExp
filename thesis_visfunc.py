from matplotlib import cm
from matplotlib import gridspec
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np


def EA_fitn_dev(fitness_res, run_s, min_f=0):
    fitness_s = fitness_res.copy()
    fitness_s.reset_index()

    fitness_s = fitness_s[fitness_s['run']==run_s]
    fitness_s = fitness_s.drop('run', axis=1)
    fitness_s = fitness_s.set_index('generation')

    fitness_s.loc[:, fitness_s.columns.difference(['fitness_std'])].plot()
    plt.xlim(0, None)
    plt.ylim(min_f, None)
    fitness_s.plot(y='fitness_std')
    plt.xlim(0, None)


def EA_plt_land(f, domain, point, steps, a=30, b=-60, imgsize=(15, 10), min_f='None', ratio_w=1.5, ln=1):
    (x_min, x_max, y_min, y_max) = domain
    (x_plot, y_plot) = point

    # Create arrays
    # # meshgrid produces all combinations of given x and y
    x = np.linspace(x_min, x_max,steps)
    y = np.linspace(y_min, y_max,steps)
    X, Y = np.meshgrid(x, y)  # combine all x with all y
    ## Applying the function
    Z = f(X,Y)

    # Set up the axes with gridspec
    fig = plt.figure(figsize=imgsize)
    gs = gridspec.GridSpec(1, 2, width_ratios=[ratio_w,1])
    ax = fig.add_subplot(gs[0], projection='3d')
    ay = fig.add_subplot(gs[1])

    # Plotting the surface
    ## Some values for the surface plot
    norm = plt.Normalize(Z.min(), Z.max())
    colors = cm.viridis(norm(Z))
    rcount, ccount, _ = colors.shape
    ax.view_init(a,b)  # Visualization angles
    ## Plotting points
    ax.scatter( x_plot, y_plot, f(x_plot, y_plot), color='r', s=20)
    ## Plotting surface
    surf = ax.plot_surface(X, Y, Z, rcount=rcount, ccount=ccount, facecolors=colors, shade=False)
    surf.set_facecolor((0,0,0,0))
    if min_f != 'None':
        ax.set_zlim(bottom=min_f)
    ax.set_xlabel('gen_x')
    ax.set_ylabel('gen_y')
    ax.set_zlabel('fitness')
    #ax.set_aspect('auto')
    ax.autoscale_view(True,True,True,True)

    # Plotting level curves
    ## Plotting points
    ay.scatter(x_plot, y_plot, color='r', s=20, label='Minima')
    ## Plotting contour
    levels = 15
    CS = ay.contour(X, Y, Z, levels, cmap='viridis', linewidths=ln)
    ay.clabel(CS, inline=True, fontsize=8)
    ay.set_xlabel('gen_x')
    ay.set_ylabel('gen_y')
    #ay.set_aspect('auto')
    ay.autoscale_view(True,True,True)
    ay.legend()

    #adjusting
    plt.tight_layout()
    plt.show()


def EA_plt_pop(f, domain, steps, genera_res, run_s, gen_s, a=30, b=-60, imgsize=(15, 10), min_f='None', ratio_w=1.5, ln=1):
    query = (genera_res['function']=='population') & (genera_res['generation']==gen_s) & (genera_res['run']==run_s)
    population_s = genera_res[query]
    xp = population_s['gen_x'].values
    yp = population_s['gen_y'].values
    zp = population_s['fitness'].values

    (x_min, x_max, y_min, y_max) = domain

    # Create arrays
    # meshgrid produces all combinations of given x and y
    x = np.linspace(x_min, x_max, steps)
    y = np.linspace(y_min, y_max, steps)
    X, Y = np.meshgrid(x, y)  # combine all x with all y
    # Applying the function
    Z = f(X, Y)

    # Set up the axes with gridspec
    fig = plt.figure(figsize=imgsize)
    gs = gridspec.GridSpec(1, 2, width_ratios=[ratio_w, 1])
    ax = fig.add_subplot(gs[0], projection='3d')
    ay = fig.add_subplot(gs[1])

    # Plotting the surface
    ax.view_init(a, b)
    ## Plotting points
    ax.scatter(xp, yp, zp, color='r', s=20)
    ## Plotting surface
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.3, linewidth=0)
    if min_f != 'None':
        ax.set_zlim(bottom=min_f)
    ax.set_xlabel('gen_x')
    ax.set_ylabel('gen_y')
    ax.set_zlabel('fitness')
    # ax.set_aspect('auto')
    ax.autoscale_view(True, True, True, True)

    # Plotting level curves
    # # Plotting points
    ay.scatter(xp, yp, color='r', s=20, label='population')
    # # Plotting contour
    levels = 15
    ay.contour(X, Y, Z, levels, cmap='viridis', linewidths=ln)
    ay.set_xlabel('gen_x')
    ay.set_ylabel('gen_y')
    # ay.set_aspect('auto')
    ay.autoscale_view(True,True,True)
    ay.legend()

    # adjusting
    plt.tight_layout()
    plt.show()


def EA_plt_gen(f, domain, steps, genera_res, run_s, gen_s, a=30, b=-60, imgsize=(15,10), min_f='None', ratio_w=1.5, ln=1):
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

    # Create arrays
    # meshgrid produces all combinations of given x and y
    x=np.linspace(x_min,x_max,steps)
    y=np.linspace(y_min,y_max,steps)
    X, Y=np.meshgrid(x,y)  # combine all x with all y
    # Applying the function
    Z = f(X,Y)

    # Set up the axes with gridspec
    fig = plt.figure(figsize=imgsize)
    gs = gridspec.GridSpec(1, 2, width_ratios=[ratio_w,1])
    ax = fig.add_subplot(gs[0], projection='3d')
    ay = fig.add_subplot(gs[1])

    # Plotting the surface
    ax.view_init(a, b)
    # # Plotting points
    ax.scatter(xp, yp, zp, color='r', s=20)
    ax.scatter(xg, yg, zg, color='g', s=17.5)
    # # Plotting surface
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.3, linewidth=0)
    if min_f != 'None':
        ax.set_zlim(bottom=min_f)
    ax.set_xlabel('gen_x')
    ax.set_ylabel('gen_y')
    ax.set_zlabel('fitness')
    # ax.set_aspect('auto')
    ax.autoscale_view(True, True, True, True)

    # Plotting level curves
    # # Plotting points
    ay.scatter(xp, yp, color='r',  s=20, label='population')
    ay.scatter(xg, yg, color='g', s=17.5, label='progeny')
    # # Plotting contour
    levels = 15
    ay.contour(X, Y, Z, levels, cmap='viridis', linewidths=ln)
    ay.set_xlabel('gen_x')
    ay.set_ylabel('gen_y')
    # ay.set_aspect('auto')
    ay.autoscale_view(True, True, True)
    ay.legend()

    # adjusting
    plt.tight_layout()
    plt.show()

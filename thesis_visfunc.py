def EA_fitn_dev(fitness_res, run_s):
    fitness_s = fitness_res.copy()
    fitness_s.reset_index()

    fitness_s = fitness_s[fitness_s['run']==run_s]
    fitness_s = fitness_s.drop('run', axis = 1)
    fitness_s = fitness_s.set_index('generation')

    fitness_s.loc[:, fitness_s.columns.difference(['fitness_std'])].plot()
    fitness_s.plot(y='fitness_std')
import numpy as np
import pandas as pd


def EA_start(pop_s, domain, f, birthcounter):
    gen_n = 0

    (x_min, x_max, y_min, y_max) = domain

    # Creating initial population
    initial = np.ones((pop_s, 6))

    # #Gen y  in 6th column
    initial[:, 5] = np.random.uniform(y_min, y_max, (pop_s))

    # #Gen x  in 5th column
    initial[:, 4] = np.random.uniform(x_min, x_max, (pop_s))

    # #Fitness in 4th column
    initial[:, 3] = f(initial[:, 4], initial[:, 5])

    # #Function in 3rd column. "111" and "222" are alias for parent and progeny
    initial[:, 2] = np.ones(pop_s)*111

    # #Generation number in 2nd column
    initial[:, 1] = np.ones(pop_s)*int(gen_n)

    # #Birthnumber in first column
    initial[:, 0] = np.arange(pop_s)

    birthcounter = birthcounter + (len(initial)-1)

    # Creating data storage
    cols = ['birthdate', 'generation', 'function', 'fitness', 'gen_x', 'gen_y']

    generations = pd.DataFrame(initial, columns=cols)

    # #Renaming in the function column
    query = generations['function']==111
    generations.loc[query, 'function'] = 'population'

    generations = generations.astype({'birthdate': int, 'generation': int})

    # We set the parents population as the
    population = np.copy(initial)

    return population, generations, birthcounter, gen_n


def EA_prog(population, par_s, prog_s, birthcounter, gen_n, mut_p, mut_s, domain, par_selection='Ranking', crossover='None', mutation='random_co_dis'):
    parents = EA_par_selection(population, par_s, par_selection)
    progeny = EA_prog_CrosUMut(parents, prog_s, birthcounter, mut_p, mut_s, domain, crossover, mutation)

    # #Fitness in 3rd column
    progeny[:, 3] = f(progeny[:, 4], progeny[:, 5])

    # #Function in 2nd column. "111" and "222" are alias for population and progeny
    progeny[:, 2] = np.ones(par_s)*222

    # #Generation number in 1st column
    progeny[:, 1] = np.ones(par_s)*int(gen_n)

    # #Birthnumber in first column
    progeny[:, 0] = np.arange(len(progeny)) + (birthcounter + 1)

    birthcounter = progeny[-1, 0]

    return birthcounter, progeny


def EA_par_selection(population, par_s, par_selection='Ranking'):
    if par_selection == 'Ranking':
        parents = np.copy(population)
        parents = parents[parents[:, 3].argsort()]  # Sorting by fitness
        parents = np.delete(parents, list(range(par_s, len(parents))), axis=0)

    return parents


def EA_prog_CrosUMut(parents, prog_s, birthcounter, mut_p, mut_s, domain, crossover='None', mutation='random_co_dis'):
    if crossover == 'None':
        progeny = np.copy(parents)
    if mutation == 'random_co_dis':
        # We unpack the landscape domain
        (x_min, x_max, y_min, y_max) = domain

        # We modify the x and y values of the progeny
        a = list([(i, j) for i in range(len(progeny)) for j in range(4, 6)])

        for (i, j) in a:
            r = (np.random.random() < mut_p)
            if r:
                progeny[i,j] = progeny[i,j] + (2 * (np.random.random() - 1)) * mut_s
                if j == 4:
                    if progeny[i, j] > x_max:
                        progeny[i, j] = x_max
                    if progeny[i, j] < x_min:
                        progeny[i, j] = x_min
                if j == 5:
                    if progeny[i, j] > y_max:
                        progeny[i, j] = y_max
                    if progeny[i, j] < y_min:
                        progeny[i, j] = y_min

    return progeny


def EA_prog_to_df(generations, progeny):
    # Creating data storage
    cols = ['birthdate', 'generation', 'function', 'fitness', 'gen_x', 'gen_y']
    prog = pd.DataFrame(progeny, columns=cols)

    generations = generations.append(prog, ignore_index = True)

    query = generations['function']==222
    generations.loc[query, "function"] = "progeny"

    generations = generations.astype({'birthdate': int, 'generation': int})

    return generations


def EA_new_population(population, progeny, gen_n, pop_s, method='Ranking'):
    gen_n += 1

    population = np.append(population, progeny, axis=0)

    if method=='Ranking':
        population = population[population[:,3].argsort()]
        population = np.delete(population, list(range(pop_s,len(population))), axis=0)

    # #Fitness in 4th column
    population[:, 3] = f(population[:, 4],population[:, 4])

    # #Function in 3rd column. "111" and "222" are alias for parent and progeny
    population[:, 2] = np.ones(pop_s)*111

    # #Generation number in 2nd column
    population[:, 1] = np.ones(pop_s)*int(gen_n)

    # #Resetting progeny array
    progeny = np.zeros((1, 6))

    return gen_n, population, progeny


def EA_pop_to_df(generations, population):
    # Creating data storage
    cols = ['birthdate', 'generation', 'function', 'fitness', 'gen_x', 'gen_y']
    pop = pd.DataFrame(population, columns=cols)

    generations = generations.append(pop, ignore_index = True)

    query = generations['function']==111
    generations.loc[query, "function"] = 'population'

    generations = generations.astype({'birthdate': int, 'generation': int})

    return generations


def EA_fitn_summary(generations):
    fitness = generations[generations['function']=='parent'].groupby('generation').agg({'fitness': ['min', 'max', 'mean']})
    fitness.columns = ["_".join(x) for x in fitness.columns.ravel()]

    return fitness


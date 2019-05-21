import numpy as np
import pandas as pd


def EA_exp(exp_n, gen_f, f, domain, pop_s, par_s, prog_s, mut_p, mut_s, par_selection='truncation', crossover='none', mutation='random_co_dis', population_new='truncation'):
    fitn_res_cols = ['run', 'generation', 'fitness_min',
                     'fitness_max', 'fitness_mean', 'fitness_std']
    gene_res_cols = ['run', 'birthdate', 'generation',
                     'function', 'fitness', 'gen_x', 'gen_y']

    fitness_res = pd.DataFrame(columns=fitn_res_cols)
    genera_res = pd.DataFrame(columns=gene_res_cols)

    for j in range(exp_n):
        run_n = j
        birthcounter = 0

        population, generations, birthcounter, gen_n = EA_start(
            pop_s, domain, f, birthcounter)

        for i in range(gen_f):
            birthcounter, progeny = EA_prog(population, par_s, prog_s, birthcounter,
                                            gen_n, mut_p, mut_s, domain, f, par_selection, crossover, mutation)
            generations = EA_prog_to_df(generations, progeny)
            gen_n, population, progeny = EA_new_population(
                population, progeny, gen_n, pop_s, f, population_new)
            generations = EA_pop_to_df(generations, population)

        fitness = EA_fitn_summary(generations)
        fitness = fitness.reset_index()
        fitness.insert(0, 'run', run_n)
        fitness_res = fitness_res.append(fitness, ignore_index=True)

        generations = generations.reset_index()
        generations.insert(0, 'run', run_n)
        genera_res = genera_res.append(generations, ignore_index=True)

    fitness_res = fitness_res[fitn_res_cols]
    fitness_res = fitness_res.sort_values(by=['run', 'generation'])
    genera_res = genera_res[['run', 'generation',
                             'birthdate', 'function', 'fitness', 'gen_x', 'gen_y']]
    genera_res = genera_res.sort_values(by=['run', 'generation'])

    fitness_res = fitness_res.infer_objects()
    genera_res = genera_res.infer_objects()

    return genera_res, fitness_res


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

    initial = initial[initial[:, 3].argsort()]

    # Creating data storage
    cols = ['birthdate', 'generation', 'function', 'fitness', 'gen_x', 'gen_y']

    generations = pd.DataFrame(initial, columns=cols)

    # #Renaming in the function column
    query = generations['function'] == 111
    generations.loc[query, 'function'] = 'population'

    generations = generations.infer_objects()

    # We set the parents population as the
    population = np.copy(initial)

    return population, generations, birthcounter, gen_n


def EA_prog(population, par_s, prog_s, birthcounter, gen_n, mut_p, mut_s, domain, f, par_selection='truncation', crossover='simple', mutation='random_co_dis'):
    parents = EA_par_selection(population, par_s, par_selection)
    progeny = EA_prog_cross_u_mut(
        parents, prog_s, birthcounter, mut_p, mut_s, domain, crossover, mutation)

    # #Fitness in 3rd column
    progeny[:, 3] = f(progeny[:, 4], progeny[:, 5])
    # #Function in 2nd column. "111" and "222" are alias for population and progeny
    progeny[:, 2] = np.ones(par_s)*222
    # #Generation number in 1st column
    progeny[:, 1] = np.ones(par_s)*int(gen_n)
    # #Birthnumber in first column
    progeny[:, 0] = np.arange(len(progeny)) + (birthcounter + 1)

    birthcounter = progeny[-1, 0]

    progeny = progeny[progeny[:, 3].argsort()]

    # Elitist reduction of progeny
    if prog_s < par_s:
        progeny = np.delete(progeny, list(range(prog_s, len(progeny))), axis=0)

    return birthcounter, progeny


def EA_par_selection(population, par_s, par_selection='truncation'):
    if par_selection == 'truncation':
        parents = np.copy(population)
        parents = parents[parents[:, 3].argsort()]  # Sorting by fitness
        parents = np.delete(parents, list(range(par_s, len(parents))), axis=0)
    elif par_selection == 'fitness_proportional_selection':
        parents = select_fitness_proportional(population, par_s)
    elif par_selection == 'tournment_k3':
        parents = select_tournment_k(population, par_s, 3)
    return parents


def EA_prog_cross_u_mut(parents, prog_s, birthcounter, mut_p, mut_s, domain, crossover='simple', mutation='random_co_dis'):
    if crossover == 'none':
        progeny = np.copy(parents)
    elif crossover == 'simple':
        progeny = np.copy(parents)
        # Gene sets
        gn1 = progeny[:, -2:]
        gn2 = gn1.copy()
        # Random shuffle of second sets
        np.random.shuffle(gn2)
        # For the progeny , we randomly pick from the shuffled and unshuffled sets
        sieve = np.random.randint(2, size=(len(gn1), 2))
        not_sieve = sieve ^ 1
        progeny[:, -2:] = sieve*gn1 + not_sieve*gn2

    if mutation == 'random_co_dis':
        # We unpack the landscape domain
        (x_min, x_max, y_min, y_max) = domain

        # We modify the x and y values of the progeny
        a = list([(i, j) for i in range(len(progeny)) for j in range(4, 6)])

        for (i, j) in a:
            r = (np.random.random() < mut_p)
            if r:
                progeny[i, j] = progeny[i, j] + \
                    (2 * np.random.random() - 1) * mut_s
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


def EA_new_population(population, progeny, gen_n, pop_s, f, population_new='truncation'):
    gen_n += 1

    population = np.append(population, progeny, axis=0)  # Overlapping generations

    if population_new == 'truncation':
        population = population[population[:, 3].argsort()]
        population = np.delete(population, list(
            range(pop_s, len(population))), axis=0)
    elif population_new == 'tournment_k3':
        population = select_tournment_k(population, pop_s, 3)
    elif population_new == 'fitness_proportional_selection':
        population = select_fitness_proportional(population, pop_s)

    # #Fitness in 4th column
    population[:, 3] = f(population[:, 4], population[:, 5])
    # #Function in 3rd column. "111" and "222" are alias for parent and progeny
    population[:, 2] = np.ones(pop_s)*111
    # #Generation number in 2nd column
    population[:, 1] = np.ones(pop_s)*int(gen_n)

    # #Resetting progeny array
    progeny = np.zeros((1, 6))

    population = population[population[:, 3].argsort()]

    return gen_n, population, progeny


def select_fitness_proportional(individuals, selection_size):
    total_fitness = np.sum(individuals[:, 3])
    select_probs = individuals[:, 3]/total_fitness
    selected = individuals[np.random.choice(
        len(individuals), selection_size, p=select_probs)]
    return selected


def select_tournment_k(individuals, selection_size, tournment_size):
    selected = []
    for t in range(selection_size):
        tournment = individuals[np.random.choice(
            len(individuals), tournment_size)]
        winner = tournment[individuals[:, 3].argmax()]
        selected.append(winner)
    return np.array(selected)


def EA_prog_to_df(generations, progeny):
    # Creating data storage
    cols = ['birthdate', 'generation', 'function', 'fitness', 'gen_x', 'gen_y']
    prog = pd.DataFrame(progeny, columns=cols)

    generations = generations.append(prog, ignore_index=True)

    query = generations['function'] == 222
    generations.loc[query, 'function'] = 'progeny'

    generations = generations.infer_objects()

    return generations


def EA_pop_to_df(generations, population):
    # Creating data storage
    cols = ['birthdate', 'generation', 'function', 'fitness', 'gen_x', 'gen_y']
    pop = pd.DataFrame(population, columns=cols)

    generations = generations.append(pop, ignore_index=True)

    query = generations['function'] == 111
    generations.loc[query, "function"] = 'population'

    generations = generations.infer_objects()

    return generations


def EA_fitn_summary(generations):
    fitness = generations[generations['function'] == 'population'].groupby(
        'generation').agg({'fitness': ['min', 'max', 'mean', 'std']})
    fitness.columns = ["_".join(x) for x in fitness.columns.ravel()]

    return fitness


def shifter(df, col_to_shift, pos_to_move):
    arr = df.columns.values
    idx = df.columns.get_loc(col_to_shift)
    if idx == pos_to_move:
        pass
    elif idx > pos_to_move:
        arr[pos_to_move+1: idx+1] = arr[pos_to_move: idx]
    else:
        arr[idx: pos_to_move] = arr[idx+1: pos_to_move+1]
    arr[pos_to_move] = col_to_shift
    df.columns = arr
    return df

import thesis_EAfunc, thesis_visfunc

#starting seed
np.random.seed(654321)

#Problem domain
x_min = -2.5
x_max = 2.5
y_min = -2.5
y_max = 2.5

#Problem definition
def f(x, y):
    D = 2
    alpha = 1/8
    a = np.abs(x ** 2 + y ** 2 - D) ** (alpha * D)
    b = ( 0.5 * (x ** 2 + y ** 2) + (x + y) ) / D
        
    return (a + b + 0.5)

# Population size
pop_s = 50



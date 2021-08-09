def cost_plus(costs,stock=0):
    total_costs = costs
    stock = stock
    price = total_costs+(0.4*total_costs)
    #sp = [tc / (100 - % of margin)]*100
    p_0 = (total_costs/(100-50))*100


    for x in range(1,stock+1):
        sales = (price * x)
        s_0 = (p_0*x)
        pro_0 = s_0 - (total_costs*x)
        profit = sales -(total_costs*x)

        #print('Sale: %d  Sales: %d Profit: %d' %(x,sales,profit))
        #print('Sale: %d  Sales: %d Profit: %d' %(x,s_0,pro_0))
    return price


def price(x, a=200, b=10, d=10, t=np.linspace(1,10,10)):
    """ Returns the price given a demand x and time t
    See equation 4 above"""

    return (a - b * x) * d / (d + t)

def price_values(p):

    p_vals = []
    for x in range(1, 10, 1):
        p = p - (0.033*p)
        p_vals.append(p)

    return p_vals


def demand(p, a=200, b=10, d=10, t=np.linspace(1,10,10)):
    """ Return demand given an array of prices p for times t
    (see equation 5 above)"""

    return  1.0 / b  * ( a - p * ( d + t ) / d )

def objective(x_t, a=512, b=10, d=10, t=np.linspace(1,10,10)):
    return -1.0 * np.sum( x_t * price(x_t, a=a, b=b, d=d, t=t) )

def constraint_1(x_t, s_0=150):
    return s_0 - np.sum(x_t)

def constraint_2(x_t):
    return x_t

def constraint_3(x_t, a=200, b=10):
    return (a / b) - x_t

 import random
 def dynamic_pricing(time=0,stock=0)
    # Model parameters :
    s_0 = stock
    a = 1650#random.randint(0,1000000)
    b = 10.0
    d = 10.0
    t = time

    # Starting values :
    x_start = 3.0 * np.ones(len(t))

    # bounds on the values :
    bounds = tuple((0,20.0) for x in x_start)

    # Constraints :
    constraints = ({'type': 'ineq', 'fun':  lambda x, s_0=s_0:  constraint_1(x,s_0=s_0)},
            {'type': 'ineq', 'fun':  lambda x: constraint_2(x)},
            {'type': 'ineq', 'fun': lambda x, a=a, b=b: constraint_3(x, a=a, b=b)})

    import scipy.optimize as optimize

    opt_results = optimize.minimize(objective, x_start, args=(a, b, d, t),
                                method='SLSQP', bounds=bounds,  constraints=constraints)

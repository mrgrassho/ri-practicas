from scipy.optimize import curve_fit

func = lambda x, a, b: a * (x**b)

n_terms = l
n_tokens = list(range(1, len(l)+1))

popt, pcov = curve_fit(func, n_tokens, n_terms) #calcula los coeficientes de la función
adata = func(n_tokens, *popt)

import pandas as pd
import numpy as np
import math
from sympy import diff
import sympy as sp
from sympy import re
from sympy import plot_implicit, Eq
import matplotlib.pyplot as plt
x_var = sp.symbols('x')


def newton_coefs(x, y):
    n = len(x)
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            y[i] = (y[i] - y[i - 1]) / (x[i] - x[i - j])
    return y

def value_at_point(point, coefs, x):
    n = len(coefs)
    prev = coefs[n - 1]
    for i in range(n - 2, -1, -1):
        prev = prev * (point - x[i]) + coefs[i]
    return prev


def nearest_points(x, y, point, num):
    # массив с расстояниями
    distances = np.abs(np.array(x) - point) 
    # сортируем точки по расстоянию
    pairs = sorted(zip(distances, range(len(x))))
    # индексы ближайших
    indexes = list(map(lambda x: x[1], pairs))[:num]
    x = [x[i] for i in indexes]
    y = [y[i] for i in indexes]
    return x, y


# поиск корня методом Ньютона
def newton_root(f, x0, epsilon, max_iter):
    Df = diff(f)
    xn = x0
    for n in range(0, max_iter):
        fxn = f.subs({x_var : xn})
        if abs(fxn) < epsilon:
            print('решение нашлось после',n,'итераций.')
            return xn
        Dfxn = Df.subs({x_var:xn})
        if Dfxn == 0:
            print('нулевая производная.')
            return None
        xn = xn - fxn / Dfxn
        
    print('решение не найдено.')
    return None


# вся таблица
def finite_differences2(y, n):
    finite_diffs = [y]
    for curn in range(n):
        cur_diffs = []
        for i in range(1, len(y) - curn):
            cur_diffs.append(finite_diffs[-1][i] - finite_diffs[-1][i-1])
        finite_diffs.append(cur_diffs)
    return finite_diffs


# только первая строка
def finite_differences(y, n):
    finite_diffs = [y]
    for curn in range(n):
        cur_diffs = []
        for i in range(1, len(y) - curn):
            cur_diffs.append(finite_diffs[-1][i] - finite_diffs[-1][i-1])
        finite_diffs.append(cur_diffs)
    return [finite_diffs[i][0] for i in range(len(finite_diffs))]


# Nk
def get_Nk(t, k):
    if k == 0:
        return 1
    else:
        return get_Nk(t, k - 1) * (t - k + 1) / k

    
# Pk
def get_Pk(deltas_y0, t, maxk):
    Nks = []
    for k in range(maxk + 1):
        Nks.append(get_Nk(t, k))
    s = 0
    for i, j in zip(deltas_y0, Nks):
        s += j * i
    return s


# многочлен Ньютона по коэф-там в sympy
def sympy_poly(x, y, x_var, coefs):
    polynom = coefs[0]
    prod = (x_var - x[0])
    for i in range(1, len(coefs)):
        polynom += prod * coefs[i]
        prod *= x_var - x[i]
    return polynom


# рр, первая строка
def divided_differences(nodes, values, max_order, derivatives=None):
    diffs = [values]
    for order in range(max_order):
        current_diffs = []
        for i in range(1, len(nodes) - order):
            try:
                current_diffs.append((diffs[-1][i] - diffs[-1][i-1]) / (nodes[i + order] - nodes[i-1]))
            except ZeroDivisionError:
                current_diffs.append(derivatives[order + 1][i] / math.factorial(order + 1))
        diffs.append(current_diffs)
    return [diffs[i][0] for i in range(len(diffs)) if diffs[i]]


# рр, вся таблица
def divided_differences2(nodes, values, max_order, derivatives=None):
    diffs = [values]
    for order in range(max_order):
        current_diffs = []
        for i in range(1, len(nodes) - order):
            try:
                current_diffs.append((diffs[-1][i] - diffs[-1][i-1]) / (nodes[i + order] - nodes[i-1]))
            except ZeroDivisionError:
                current_diffs.append(derivatives[order + 1][i] / math.factorial(order + 1))
        diffs.append(current_diffs)
    return diffs


def first_order_deriv_1(x, y, h):
    return np.array([(y[i] - y[i - 1]) / h for i in range(1, len(x))] + [0])


def first_order_deriv_2(x, y, h):
    return np.array([0] + [(y[i] - y[i - 2]) / (2*h) for i in range(2, len(x))] + [0])


def second_order_deriv(x, y, h):
    return np.array([0] + [(y[i+1] - 2 * y[i] + y[i - 1]) / (h ** 2)
                           for i in range(1, len(x) - 1)] + [0])


def true_first_order_deriv(x):
    return np.array([2 * np.exp(2 * p) for p in x])

def true_second_order_deriv(x):
    return np.array([4 * np.exp(2 * p) for p in x])

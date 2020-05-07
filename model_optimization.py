import numpy as np
import pandas as pd
import math
import altair as alt
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy import optimize

# from data.objects.Ricci import Ricci

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def simple_threshold_data(number_a0, number_a1,
                          a0_threshold, a1_threshold,
                          a0_label, a1_label):

    a0s = np.full(number_a0, a0_label)
    a1s = np.full(number_a1, a1_label)

    a0_scores = np.linspace(0, 1, number_a0)
    a1_scores = np.linspace(0, 1, number_a1)
    score_feature = np.concatenate((a0_scores, a1_scores), axis=None)

    A = np.concatenate((a0s, a1s), axis=None)

    Y_a0 = [x > a0_threshold for x in a0_scores]
    Y_a1 = [x > a1_threshold for x in a1_scores]

    Y = np.concatenate((Y_a0, Y_a1), axis=None)

    X = pd.DataFrame({"credit_score_feature": score_feature,
                      "example_sensitive_feature": A})
    return X, Y, A

def plot_data(Xs, Ys):
    labels = np.unique(Xs["example_sensitive_feature"])

    for l in labels:
        label_string = str(l.item())
        mask = Xs["example_sensitive_feature"] == l
        plt.scatter(Xs[mask].credit_score_feature, Ys[mask], label=str("Label="+label_string))
        plt.xlabel("Credit Score")
        plt.ylabel("Got Loan")

    plt.legend()
    plt.show()


num_samples_a0 = 31
num_samples_a1 = 21

a0_threshold= 0.2
a1_threshold = 0.7

a0_label = 2
a1_label = 3

pos_label = 2

X, Y, A = simple_threshold_data(num_samples_a0, num_samples_a1, a0_threshold, a1_threshold, a0_label, a1_label)

um = LogisticRegression(solver='liblinear', fit_intercept=True)
um.fit(X, Y, sample_weight=np.random.rand(len(Y)))

weight_on_attr = pd.DataFrame(columns=["Delta a","Weight on Protected Attribute"])
losses = pd.DataFrame(columns=["Delta a","Disperate Impact"])

Y_predict_unfair = um.predict(X)

# plot_data(X, Y_predict_unfair)

def calc_pos_neg_for_attrb(atrb_val, Y, Y_hat, A):
    tn = 0 #true negative
    fn = 0 #false negative
    fp = 0 #false positive
    tp = 0 #true positive

    for y, y_hat, a in zip(Y, Y_hat, A):
        if a == atrb_val:
            if y == 0 and y_hat == 0:
                tn += 1
            if y == 1 and y_hat == 0:
                fn += 1
            if y == 0 and y_hat == 1:
                fp += 1
            if y == 1 and y_hat == 1:
                fp += 1

    return [tn, fn, fp, tp]

def calc_cost_0(y):
    if y != 0:
        return 1
    return 0

def calc_cost_1(y, lamb, prob, lambdas_sum):
    addend = 0
    lam_ai = 0
    if y != 1:
        addend = 1

    return addend + (lamb/prob) - lambdas_sum

def calc_prob(A, a):
    sum = 0

    # print(A)

    for sample in A:
        if sample == a:
            sum += 1

    return sum/len(A)

def calc_lam_a(pa, pa_prime, dela, dela_prime):


    numerator = dela_prime + dela*((1/pa_prime)-1)
    denom = (((((1/pa)-1)*((1/pa_prime)-1))+1e-15) - 1)

    # print(pa, pa_prime, ((1/pa)-1)*((1/pa_prime)-1))

    return numerator/denom

def calc_lam_a_prime(lam_a, pa, da):
    return (lam_a/pa)-lam_a-da

def assess_fairness(da):
    d_a_prime = 0.0

    # get predictions
    Y_predict = um.predict(X)

    # collect protected attribute values where h(x) = 1
    # false_predicts = []
    # for i, val in enumerate(Y_predict_unfair):
    #     # if val != Y[i]:
    #     false_predicts.append(A[i])


    # calculate probablities relative to both a and a' for false predictions
    pa = calc_prob(A, a0_label)
    pa_prime = calc_prob(A, a1_label)


    # print(pa, pa_prime, da, flush=True)
    d_a_prime = ((-pa*da)/pa_prime)
    lam_a = calc_lam_a(pa, pa_prime, da, d_a_prime)
    lam_a_prime = calc_lam_a_prime(lam_a, pa, da)

    lambdas = []
    probs = []
    for a in A:
        if a is pos_label:
            lambdas.append(lam_a)
            probs.append(pa)
        else:
            lambdas.append(lam_a_prime)
            probs.append(pa_prime)

    costs_0 = []
    costs_1 = []

    # store lambdas
    for i,y in enumerate(Y):
        costs_0.append(calc_cost_0(y))
        costs_1.append(calc_cost_1(y, lambdas[i], probs[i], lam_a+lam_a_prime))



    weights = []
    New_Y = Y
    weight = 0

    for i, (c_0, c_1) in enumerate(zip(costs_0, costs_1)):
        if A[i] == pos_label:
            weight = c_0 - c_1
        weights.append(abs(c_0 - c_1))
        # if c_0 >= c_1:
        #     New_Y[i] = 1
        # else:
        #     New_Y[i] = 0

    # print(costs_0,"\n", costs_1)

    um.fit(X, New_Y, sample_weight=weights)
    Y_hat = um.predict(X)
    # print(weights)
    # print(Y_hat)
    # print(Y_predict_unfair)

    # calc values for both attribute values
    confus_m_a0 = calc_pos_neg_for_attrb(a0_label, Y, Y_hat, A)
    confus_m_a1 = calc_pos_neg_for_attrb(a1_label, Y, Y_hat, A)

    # print(confus_m_a0, confus_m_a1)

    l = ((confus_m_a0[2]+confus_m_a0[3])/sum(confus_m_a0)) - ((confus_m_a1[2]+confus_m_a1[3])/sum(confus_m_a1))

    global weight_on_attr
    global losses

    weight_on_attr = weight_on_attr.append({"Delta a": da, "Weight on Protected Attribute": weight}, ignore_index=True)
    losses = losses.append({"Delta a": da, "Disperate Impact": l}, ignore_index=True)

    # print("Da:{} Loss:{} Weight: {}".format(da, l, weight))

    return weight

#do predictions

# sol = optimize.root_scalar(f, bracket=[-2,2], method='brentq'
# print(sol.root, sol.iterations, sol.function_calls)

# calculate da_prime for many different values of d_a

# sol = optimize.root_scalar(assess_fairness, bracket=[-2,2], method='bisect')
# print(sol.root)

range = sys.argv[1]

d_a=-float(range)

while(d_a <= float(range)):
    assess_fairness(d_a)
    d_a += .25

# c = alt.Chart(losses).mark_line().encode(
#     y="Disperate Impact",
#     x="Delta a"
# ).interactive()
#
# c.save("DisperateImpact_line{}.html".format(d_a))

c = alt.Chart(losses).mark_point().encode(
    y="Disperate Impact",
    x="Delta a"
).interactive()

c.save("DisperateImpact_point_{}.html".format(d_a))

# c = alt.Chart(weight_on_attr).mark_line().encode(
#     y="Weight on Protected Attribute",
#     x="Delta a"
# ).interactive()
#
# c.save("Weights_line_{}.html".format(d_a))

c = alt.Chart(weight_on_attr).mark_point().encode(
    y="Weight on Protected Attribute",
    x="Delta a"
).interactive()

c.save("Weights_point_{}.html".format(d_a))

# Y_hat = um.predict(X)
# plot_data(X, Y_hat)


# d_a=-2
#
# while(d_a <= 2):
#     assess_fairness(d_a)
#     d_a += .25

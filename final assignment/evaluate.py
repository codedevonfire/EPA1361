# load library

import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import time
import statistics
import pickle
import functools


from ema_workbench import (Model, RealParameter, ScalarOutcome, MultiprocessingEvaluator, 
                           ema_logging, Constant, Scenario, Policy, Constraint)
from ema_workbench import load_results
from ema_workbench.analysis import prim, dimensional_stacking, cart
from ema_workbench.util import ema_logging, utilities

from ema_workbench.em_framework.optimization import (HyperVolume,
                                                     EpsilonProgress)
from ema_workbench.em_framework import sample_uncertainties, MonteCarloSampler
from ema_workbench.em_framework.evaluators import BaseEvaluator

ema_logging.log_to_stderr(ema_logging.INFO)

from dike_model_function import DikeNetwork  # @UnresolvedImport
from problem_formulation import get_model_for_problem_formulation

results = utilities.load_results('results/base_case_75.csv')

experiments, outcomes = results
outcomes = pd.DataFrame(outcomes)
experiments = pd.DataFrame(experiments)

outcome_total = pd.read_csv('results/75policies_with_total_values.csv')


results = experiments.join(outcome_total)
results = results.drop(columns="model")

# minimise the worst outcome and minimise the standard deviation
#defined as the median value multiplied by the interquartile distance plus one

def robustness(result):
    mean = np.median(result)
    iqr = sp.stats.iqr(result) + 1 #to aviod the return being 0
    robust = mean * iqr
    return robust

def total_robustness(*result):

    total_robust = robustness(sum(result))

    return total_robust


robustness_score = pd.DataFrame(columns = ["policy", "damage", "death", "dike_invest", "rfr_cost", "evacuation"])

for p in results["policy"].unique():

    index = results[results["policy"] == p].index
    
    damage = robustness(results["Total Expected Annual Damage"][index])
    death = robustness(results["Total Expected Number of Deaths"][index])
    dike_invest = robustness(results["Total Dike Investment Costs"][index])
    rfr_cost = robustness(results["Total RfR Total Costs"][index])
    evacuation = robustness(results["Total Expected Evacuation Costs"][index])
    
        
    policy_line = [p, damage, death, dike_invest, rfr_cost, evacuation]
    
    robustness_score = robustness_score.append(pd.Series(policy_line, index=["policy", "damage", "death", "dike_invest", "rfr_cost", "evacuation"]), ignore_index=True) 



dike_lists = ['A.1', 'A.2', 'A.3', 'A.4', 'A.5']

damage_column=[]
death_column=[]
investment_column=[]
rfr_column = []
evacuation_column = []

for c in outcomes.columns:
    if "Expected Annual Damage" in c:
        damage_column.append(c)

for c in outcomes.columns:
    if "Expected Number of Deaths" in c:
        death_column.append(c)
        
for c in outcomes.columns:
    if "Dike Investment Costs" in c:
        investment_column.append(c)
        
for c in outcomes.columns:
    if "RfR Total Costs" in c:
        rfr_column.append(c)
        
for c in outcomes.columns:
    if "Expected Evacuation Costs" in c:
        evacuation_column.append(c)
        
dike_model, planning_steps = get_model_for_problem_formulation(5)

robust_metrics = [
    ScalarOutcome('Damage', variable_name=damage_column,
                  function=total_robustness, kind=ScalarOutcome.MINIMIZE, expected_range=(0, 2e16)),
    ScalarOutcome('Deaths', variable_name=death_column,
                  function=total_robustness, kind=ScalarOutcome.MINIMIZE, expected_range=(0, 1)),
    ScalarOutcome('Dike Invest', variable_name=investment_column,
                  function=total_robustness, kind=ScalarOutcome.MINIMIZE, expected_range=(1e8, 1e9)),
    ScalarOutcome('RfR Cost', variable_name=rfr_column,
                  function=total_robustness, kind=ScalarOutcome.MINIMIZE, expected_range=(3e8, 2e9)),
    ScalarOutcome('Evacuation', variable_name=evacuation_column,
                  function=total_robustness, kind=ScalarOutcome.MINIMIZE, expected_range=(0, 5e7)),
]


results, convergence = pickle.load(
    open('results/MORO_50_nfe10000.pkl', 'rb'))

results.iloc[:, 0:-5].T.to_dict()

policy = []

for i in range(len(results)):
    pol = Policy(name="Policy " + str(i), **results.iloc[i, :-5].to_dict())
    policy.append(pol)

tic = time.time()

with MultiprocessingEvaluator(dike_model) as evaluator:
    results = evaluator.perform_experiments(scenarios=200, policies=policy)

toc = time.time()
print('Total run time:{} min'.format((toc - tic)/60))

# Write the results so this step can be skipped when doing multiple analyzes
with open('results/MORO_evaluate.pkl', 'wb') as file_pi:
    pickle.dump(results, file_pi)

import functools
import time
import numpy as np

from ema_workbench import (Model, CategoricalParameter, ScalarOutcome, IntegerParameter, RealParameter, 
                           Constraint, MultiprocessingEvaluator, Policy, Scenario, ema_logging,
                           perform_experiments, SequentialEvaluator)
from ema_workbench.em_framework.optimization import (HyperVolume, 
                                                     EpsilonProgress)
from ema_workbench.em_framework.evaluators import BaseEvaluator
from ema_workbench.em_framework import sample_uncertainties
from ema_workbench.util import utilities

ema_logging.log_to_stderr(ema_logging.INFO)

from dike_model_function import DikeNetwork
from problem_formulation import get_model_for_problem_formulation

dike_model, planning_steps = get_model_for_problem_formulation(3)

# Define Robustness
def robustness(direction, threshold, data):
    if direction == SMALLER:
        return np.sum(data<=threshold)/data.shape[0]
    else:
        return np.sum(data>=threshold)/data.shape[0]

SMALLER = 'SMALLER'


# Set functions
A1_Expected_Number_of_Deaths = functools.partial(robustness, SMALLER, 0.05)
A1_Expected_Annual_Damage = functools.partial(robustness, SMALLER, 400000) 
A1_Total_Costs = functools.partial(robustness, SMALLER, 150000000)

A2_Total_Costs = functools.partial(robustness, SMALLER, 150000000)
A2_Expected_Number_of_Deaths = functools.partial(robustness, SMALLER, 0.5) 
A2_Expected_Annual_Damage = functools.partial(robustness, SMALLER, 400000) 

A3_Total_Costs = functools.partial(robustness, SMALLER, 150000000)
A3_Expected_Number_of_Deaths = functools.partial(robustness, SMALLER, 0.5)
A3_Expected_Annual_Damage = functools.partial(robustness, SMALLER, 400000)

A4_Total_Costs = functools.partial(robustness, SMALLER, 150000000)
A4_Expected_Number_of_Deaths = functools.partial(robustness, SMALLER, 0.5) 
A4_Expected_Annual_Damage = functools.partial(robustness, SMALLER, 400000)

A5_Total_Costs = functools.partial(robustness, SMALLER, 150000000)
A5_Expected_Number_of_Deaths = functools.partial(robustness, SMALLER, 0.5)
A5_Expected_Annual_Damage = functools.partial(robustness, SMALLER, 400000)

RfR_Total_Costs = functools.partial(robustness, SMALLER, 100000)
Expected_Evacuation_Costs = functools.partial(robustness, SMALLER, 100000)
                                              

n_scenarios = 5 #50
scenarios = sample_uncertainties(dike_model, n_scenarios)

MAXIMIZE = ScalarOutcome.MAXIMIZE
MINIMIZE = ScalarOutcome.MINIMIZE
                                              
#Formulate Robustness functions                                              
robustness_functions = [ScalarOutcome('A1 deaths', kind=MINIMIZE, 
                             variable_name='A.1_Expected Number of Deaths', function=A1_Expected_Number_of_Deaths),
                       ScalarOutcome('A1 costs', kind=MINIMIZE, 
                             variable_name='A.1 Total Costs', function=A1_Expected_Annual_Damage),
                       ScalarOutcome('A1 damage', kind=MINIMIZE, 
                             variable_name='A.1_Expected Annual Damage', function=A1_Total_Costs),
                        ScalarOutcome('A2 deaths', kind=MINIMIZE, 
                             variable_name='A.2_Expected Number of Deaths', function=A2_Expected_Number_of_Deaths),
                       ScalarOutcome('A2 costs', kind=MINIMIZE, 
                             variable_name='A.2 Total Costs', function=A2_Expected_Annual_Damage),
                       ScalarOutcome('A2 damage', kind=MINIMIZE, 
                             variable_name='A.2_Expected Annual Damage', function=A2_Total_Costs),
                        ScalarOutcome('A3 deaths', kind=MINIMIZE, 
                             variable_name='A.3_Expected Number of Deaths', function=A3_Expected_Number_of_Deaths),
                       ScalarOutcome('A3 costs', kind=MINIMIZE, 
                             variable_name='A.3 Total Costs', function=A3_Expected_Annual_Damage),
                       ScalarOutcome('A3 damage', kind=MINIMIZE, 
                             variable_name='A.3_Expected Annual Damage', function=A3_Total_Costs),
                        ScalarOutcome('A4 deaths', kind=MINIMIZE, 
                             variable_name='A.4_Expected Number of Deaths', function=A4_Expected_Number_of_Deaths),
                       ScalarOutcome('A4 costs', kind=MINIMIZE, 
                             variable_name='A.4 Total Costs', function=A4_Expected_Annual_Damage),
                       ScalarOutcome('A4 damage', kind=MINIMIZE, 
                             variable_name='A.4_Expected Annual Damage', function=A4_Total_Costs),
                        ScalarOutcome('A5 deaths', kind=MINIMIZE, 
                             variable_name='A.5_Expected Number of Deaths', function=A5_Expected_Number_of_Deaths),
                       ScalarOutcome('A5 costs', kind=MINIMIZE, 
                             variable_name='A.5 Total Costs', function=A5_Expected_Annual_Damage),
                       ScalarOutcome('A5 damage', kind=MINIMIZE, 
                             variable_name='A.5_Expected Annual Damage', function=A5_Total_Costs),
                        ScalarOutcome('RfR total costs', kind=MINIMIZE, variable_name='RfR Total Costs', function=RfR_Total_Costs),
                        ScalarOutcome('Evacuation costs', kind=MINIMIZE, variable_name='Expected Evacuation Costs', function=Expected_Evacuation_Costs)
                       ]

constraints = [Constraint("ConstrA.1_Expected Annual Damage", outcome_names="A.1_Expected Annual Damage",
                           function=lambda x:max(0, x-40000000)),
               Constraint("ConstrA.1_Expected Number of Deaths", outcome_names="A.1_Expected Number of Deaths",
                          function=lambda x:max(0, x-1))
               ]
                                              
# Record the run time
start = time.time()

# We assume we needed to put 17x 0 and 17x 1 for convergence, because we have 17 outcomes in Robustness_functions?
# In assignment 10, there were 4 outcomes, and also 4x 0 and 4x 1 here in the convergence part
convergence = [HyperVolume(minimum=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], maximum=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]),
               EpsilonProgress()]
nfe = 200 # int(1e4)

# Run it                                              
if __name__ == '__main__':                                            
    with MultiprocessingEvaluator(dike_model, n_processes=7) as evaluator:
        results = evaluator.robust_optimize(robustness_functions, scenarios, 
                                                   nfe=nfe, convergence=convergence, constraints=constraints,
                                                   epsilons=[0.05,]*len(robustness_functions))

    utilities.save_results(results, 'Outcomes/MOROrijkswaterstaatConstraints.tar.gz')

end = time.time()
print('Total run time:{} min'.format((end - start)/60))
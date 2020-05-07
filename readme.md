# Installation

To install the requirements for this software run

```
pip install requirements.txt
```

# Running

To run this software use the following command:

```
python3 model_optimization.py 2
```

This software takes one command line argument which can be an integer or float. This argument, "2" in the above example, specifies the negative to positive range for a grid search of models with increasing values of delta_a. In this case, our program will begin by training a model with input value delta_a = -2 and will train models with delta_a's increasing at a rate of .25 until it reaches +2.

# Outputs

This program will output several interactive Altair charts which can be opened in your web browser. One chart is the disperate impact loss calculated for each model produced by a given delta_a while the other plots the weight applied to our protected attribute on the model. In this case it is the weight placed on individuals with the attribute value "2". These second charts were generated to compare against the example "Grid Search for Binary Classification" which was provided as part of the Fairlearn package.

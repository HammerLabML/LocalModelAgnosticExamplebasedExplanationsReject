# "I do not Know! But Why?" – Local Model-Agnostic Example-based Explanations of Reject

This repository contains the implementation of the methods proposed in the paper ["I do not Know! But Why?" – Local Model-Agnostic Example-based Explanations of Reject]() by André Artelt, Roel Visser and Barbara Hammer.

The experiments, as described in the paper, are implemented in the folder [Implementation/](Implementation/).

## Abstract

The application of machine learning based decision making systems in safety critical areas requires reliable high certainty predictions in order to avoid mistakes that induce serious (e.g. life-threatening) consequences. A common strategy is to reject samples where the model can not provided a high certainty prediction. Reject options are a methodology that allows the system to reject samples where it is not certain enough about its prediction -- i.e. the system refuses to make a prediction because "it is afraid of providing a wrong prediction".
While being able to reject samples for avoiding mistakes is important and useful, it is also of importance to be able to explain why a particular sample was rejected. Explaining and understanding rejections is not only important for the trustworthiness of the system but also allows us to understand and debug & improve the overall system of model and reject option. However, how to explain general reject options is still an open research question on which only very little work exists.

In this work we propose a model-agnostic methodology for using example-based explanations, such as counterfactual and semifactual explanations, to locally understand why a particular sample got rejected. In addition to an abstract methodology, we also provide a widely applicable implementation which we evaluate empirically.

## Details

### Data 

The preprocess data sets, if not part of scikit-learn, are stored in [Implementation/data/](Implementation/data/).

### Implementation of the proposed method and the experiments

Our proposed method of using a local surrogate for computing explanations of reject is implemented in [rejectexplanation.py](Implementation/rejectexplanation.py).
The experiments themself are implemented in [experiments.py](Implementation/experiments.py) -- you can also check out this file for an example how to use the implemented explanation methods.

## Requirements

- Python3.6
- Packages as listed in [Implementation/REQUIREMENTS.txt](Implementation/REQUIREMENTS.txt).

## License

MIT license -- See [LICENSE](LICENSE).

## How to cite

TBA

import numpy
import theano.tensor as T
from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin
from quadratic_kappa import quadratic_kappa_cost

from pylearn2.models.model import Model
from pylearn2.space import VectorSpace
from pylearn2.utils import sharedX

class QuadraticKappaCost(DefaultDataSpecsMixin, Cost):
    supervised = True

    def expr(self, model, data, **kwargs):
        space, source = self.get_data_specs(model)
        space.validate(data)
        inputs, targets = data
        y_hat = model.fprop(inputs)
        loss = quadratic_kappa_cost(y_hat, targets)
        return loss

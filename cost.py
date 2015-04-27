import theano.tensor as T
from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin
from quadratic_kappa import quadratic_kappa_cost

class QuadraticKappaCost(DefaultDataSpecsMixin, Cost):
    supervised = True

    def expr(self, model, data, **kwargs):
        space, source = self.get_data_specs(model)
        space.validate(data)

        inputs, targets = data
        outputs = model.logistic_regression(inputs)
        loss = quadratic_kappa_cost(outputs, targets)
        return loss.mean()

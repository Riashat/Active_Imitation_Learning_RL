from npo import NPO
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer as ConjugateGradientOptimizer_Gating_Function


class TRPO(NPO):
    """
    Trust Region Policy Optimization
    """

    def __init__(
            self,
            optimizer=None,
            gate_optimizer=None,
            optimizer_args=None,
            **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = ConjugateGradientOptimizer(**optimizer_args)


        ## separate optimizer required for the gate
        if gate_optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            gate_optimizer = ConjugateGradientOptimizer_Gating_Function(**optimizer_args)



        super(TRPO, self).__init__(optimizer=optimizer, gate_optimizer=gate_optimizer, **kwargs)

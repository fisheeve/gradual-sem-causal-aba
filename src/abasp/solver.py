from aspforaba.src.aspforaba import ABASolver

class Solver(ABASolver):
    def get_stable_models(self):
        return self.enumerate_extensions('ST')

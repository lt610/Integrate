
class EarlyStopping(object):
    def __init__(self, patience):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.is_stop = False
        self.metrics = None

    def __call__(self, score, metrics):
        if self.best_score is None:
            self.best_score = score
            self.metrics = metrics
        elif score <= self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.is_stop = True
        else:
            self.best_score = score
            self.metrics = metrics
            self.counter = 0

    def reset(self):
        self.counter = 0
        self.best_score = None
        self.is_stop = False
        self.metrics = None

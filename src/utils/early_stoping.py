import copy


class EarlyStopping:
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_model_state = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss - val_loss > self.delta:
            self.best_loss = val_loss
            self.best_model_state = self.best_model_state = copy.deepcopy(
                model.state_dict()
            )
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def load_best_model(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)

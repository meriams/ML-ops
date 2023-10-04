from torch.optim import lr_scheduler 

class LRScheduler: 


    def __init__(self, optimizer, patience=5, min_lr=1e-6, factor=0.5) -> None:
        self.optimizer = optimizer 
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor 
        self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', 
                                                           patience=self.patience,
                                                           factor=self.factor,
                                                           min_lr=self.min_lr,
                                                           verbose=True)
        
    def __call__(self, validation_loss):
        self.lr_scheduler.step(validation_loss)



class EarlyStopping: 

    def __init__(self, patience=10, min_delta=0):
        self.early_stop_enabled = False
        self.min_delta = min_delta
        self.patience = patience 
        self.best_loss = None
        self.counter = 0
    
    def __call__(self, validation_loss):
        if self.best_loss is None:
            self.best_loss = validation_loss
        elif (self.best_loss - validation_loss) < self.min_delta:
            self.counter +=1 
            print(f'[INFO] Early stopping: {self.counter}/{self.patience}... \n\n')

            if self.counter >= self.patience:
                self.early_stop_enabled = True 
                print(f'[INFO] Early stopping enabled')
        elif (self.best_loss - validation_loss) > self.min_delta:
            self.best_loss = validation_loss
            self.counter = 0
            
import math
from torch.optim.lr_scheduler import LambdaLR

# adapted from (https://github.com/sooftware/KoSpeech)
class LearningRateScheduler(object):
    def __init__(self, optimizer, init_lr):
        self.optimizer = optimizer
        self.init_lr = init_lr

    def step(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def set_lr(optimizer, lr):
        for g in optimizer.param_groups:
            g['lr'] = lr

    def get_lr(self):
        for g in self.optimizer.param_groups:
            return g['lr']

class TransformerLRScheduler(LearningRateScheduler):
    def __init__(self, optimizer, peak_lr, final_lr, final_lr_scale, warmup_steps, decay_steps):
        assert isinstance(warmup_steps, int), "warmup_steps should be inteager type"
        assert isinstance(decay_steps, int), "total_steps should be inteager type"

        super(TransformerLRScheduler, self).__init__(optimizer, 0.0)
        self.final_lr = final_lr
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps

        self.warmup_rate = self.peak_lr / self.warmup_steps
        self.decay_factor = -math.log(final_lr_scale) / self.decay_steps

        self.lr = self.init_lr
        self.update_step = 0

    def _decide_stage(self):
        if self.update_step < self.warmup_steps:
            return 0, self.update_step

        if self.warmup_steps <= self.update_step < self.warmup_steps + self.decay_steps:
            return 1, self.update_step - self.warmup_steps
        
        return 2, None

    def step(self):
        self.update_step += 1
        stage, steps_in_stage = self._decide_stage()

        if stage == 0:
            self.lr = self.update_step * self.warmup_rate
        elif stage == 1:
            self.lr = self.peak_lr * math.exp(-self.decay_factor * steps_in_stage)
        elif stage == 2:
            self.lr = self.final_lr
        else:
            raise ValueError("Undefined stage")

        self.set_lr(self.optimizer, self.lr) 
        
        return self.lr
    
# adapted from (https://huggingface.co/transformers/main_classes/optimizer_schedules.html)
def get_cosine_schedule_with_warmup(optimizer, 
                                    num_warmup_steps, 
                                    num_training_steps, 
                                    num_cycles = 0.5, 
                                    last_epoch = -1
                                   ):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))     
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, 
                                                       num_warmup_steps, 
                                                       num_training_steps, 
                                                       num_cycles=1.0, 
                                                       last_epoch=-1
                                                      ):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        if progress >= 1.0:
            return 0.0
        
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)
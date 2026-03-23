class TransformerOptim:
    def __init__(self, optimizer, d_model: int, warmup_steps: int):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0  # 这里的 step_num 在训练过程中动态更新

    def step(self):
        "执行一次参数更新，并同步更新学习率"
        self.step_num += 1
        lr = self._get_lr()
        for p in self.optimizer.param_groups:
            p['lr'] = lr
        self.optimizer.step()

    def zero_grad(self):
        "清空梯度"
        self.optimizer.zero_grad()

    def _get_lr(self):
        "计算当前 step_num 对应的学习率"
        return (self.d_model ** -0.5) * \
               min(self.step_num ** -0.5, self.step_num * (self.warmup_steps ** -1.5))
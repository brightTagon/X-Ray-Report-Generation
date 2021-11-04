from utils.assignment import *
from utils.utils import *

class MSEGCRLatentLoss:
    def __init__(self, safe_coef: float = 0, w_loss_br: float = 1, w_loss_rb: float = 0.1):
        self.safe_coef = safe_coef
        self.w_loss_br = w_loss_br
        self.w_loss_rb = w_loss_rb

    def forward(self, B, len_B, R, len_R):
        func = batch_hungarian_gcr(safe_coef=self.safe_coef)
        R_pi, B, R, R_i = func(B, len_B, R, len_R)

        loss_fn = F.mse_loss

        # R => B must not push gradient to B
        loss_RB = loss_fn(R[R_i], B.detach(), reduction='none')
        loss_BR = loss_fn(B, R_pi.detach(), reduction='none')

        loss_RB = mean_equal_by_instance(loss_RB, len_B)
        loss_BR = mean_equal_by_instance(loss_BR, len_B)

        loss = self.w_loss_rb * loss_RB + self.w_loss_br * loss_BR
        return R_pi, R_i, loss
"""
@Time：2022.12.2
@desc：重写torch.distributions.categorical中的sample方法
动作网络输出的三维特征logtis会通过Categorical2(logits).sample()得到动作
可选确定性策略或者随机策略
原始用法:Categorical(logits).sample()
"""
import torch
from torch.distributions.categorical import Categorical
class Categorical2(Categorical):

    def __init__(self, probs=None, logits=None, validate_args=None):
        super(Categorical2, self).__init__(probs, logits, validate_args)





    def sample(self, sample_shape=torch.Size(), deterministic=False):
        if deterministic == True:
            return torch.argmax(self.probs, dim=1)
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        probs_2d = self.probs.reshape(-1, self._num_events)
        samples_2d = torch.multinomial(probs_2d, sample_shape.numel(), True).T
        return samples_2d.reshape(self._extended_shape(sample_shape))













if __name__ == "__main__":

    probs = torch.FloatTensor([[0.2, 0.3, 0.3], [0.4, 0.3, 0.3]])
    print("重写方法,随机策略")
    for _ in range(3):
        print(Categorical2(probs).sample(deterministic=False))
    print("重写方法,确定策略")
    for _ in range(3):
        print(Categorical2(probs).sample(deterministic=True))
    print("原始Categorical函数")
    for _ in range(3):
        print(Categorical(probs).sample())








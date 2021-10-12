from typing import Tuple
from torch import cuda
import torch.nn as nn
import torch as th


class ResNet(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs

class ActorCriticNetwork(nn.Module):
    def __init__(self, num_input_channels: int, num_filters: int = 32):
        super(ActorCriticNetwork, self).__init__()
        self.features_extractor = nn.Sequential(
            nn.Conv2d(num_input_channels, num_filters, kernel_size=3, stride=1, padding= 'same'),
            nn.ReLU(),
            nn.BatchNorm2d(num_filters),
            ResNet(
                nn.Sequential(
                    nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding= 'same'),
                    nn.ReLU(),
                    nn.BatchNorm2d(num_filters),
                    nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding= 'same'),
                    nn.ReLU(),
                    nn.BatchNorm2d(num_filters),
                )
            ),
            ResNet(
                nn.Sequential(
                    nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding= 'same'),
                    nn.ReLU(),
                    nn.BatchNorm2d(num_filters),
                    nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding= 'same'),
                    nn.ReLU(),
                    nn.BatchNorm2d(num_filters),
                )
            ),
            ResNet(
                nn.Sequential(
                    nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding= 'same'),
                    nn.ReLU(),
                    nn.BatchNorm2d(num_filters),
                    nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding= 'same'),
                    nn.ReLU(),
                    nn.BatchNorm2d(num_filters),
                )
            ),
        )
        self.policy_net = nn.Sequential(
            nn.Conv2d(num_filters, 2, kernel_size=1, stride=1, padding= 'same'),
            nn.ReLU(),
            nn.BatchNorm2d(2),
            nn.Conv2d(2, 1, kernel_size=1, stride=1, padding= 'same'),
            nn.Flatten(),
        )
        self.value_net = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=1, stride=1, padding= 'same'),
            nn.ReLU(),
            nn.BatchNorm2d(num_filters),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_filters, 1),
            nn.ReLU(),
        )

    def masked(self, x, mask):
        x_masked = x.clone()
        # print(x.device, x_masked.device)
        x_masked[mask == 0] = -float("inf")
        return x_masked

    def forward(self, observations: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        # observations = th.tensor(observations, dtype=th.float).cuda()
        # observations = observations.to("cuda")
        # observations = th.as_tensor(observations, device=th.device('cuda'), dtype =th.float)
        # print("leg shape = ", observations[:,1,:,:].shape)
        legal_action = th.reshape(observations[:,1,:,:], (-1, observations.shape[2] * observations.shape[3]))
        # print("leg act shape = ", legal_action.shape)
        feature  = self.features_extractor(observations)
        pi = self.policy_net(feature)
        pi_masked = self.masked(self.policy_net(feature), legal_action)
        # pi = nn.functional.softmax(pi, dim = 1)
        pi_masked = nn.functional.softmax(pi_masked, dim = 1)
        value = self.value_net(feature)
        # print(observations.device, legal_action.device, feature.device, pi.device, value.device)
        return pi, pi_masked, value

if __name__ == '__main__':
    # import torch.multiprocessing as mp
    # net = ActorCriticNetwork(num_input_channels=4).share_memory()
    # print("test = ", next(net.parameters()).is_cuda)
    # input = th.randn(1, 4, 16, 16).cuda()
    # pi, value = net(input)
    # print(pi.size(), value[0][0].data)
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    model= ActorCriticNetwork(num_input_channels=4).cuda('cuda:1')
    # model =  nn.DataParallel(model)
    # model.cuda()
    # print(model.device)
    print("test = ", next(model.parameters()).is_cuda)
    # a = th.randn(1, 4, 16, 16).cuda('cuda:0')
    b = th.randn(1, 4, 16, 16).cuda('cuda:1')
    # print(a, b)
    # pi, value = model(a)
    for _ in range(100000):
        pi, value = model(b)

    
    

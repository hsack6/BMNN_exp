import torch.nn as nn

class FNN(nn.Module):

    def __init__(self, opt, hidden_state):
        super(FNN, self).__init__()

        self.batchSize = opt.batchSize
        self.state_dim = opt.state_dim
        self.L = opt.L
        self.n_node = opt.n_node
        self.hidden_dim = hidden_state

        self.fnn = nn.Linear(self.L*self.state_dim, self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, opt.output_dim)

        self._initialization()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, prop_state):
        """"
        prop_state  :(batch_size, n_node, L, state_dim)
        input_state :(batch_size * n_node, L, state_dim)
        h_t         :(batch_size * n_node, L, hidden_dim) 最後の層の各tにおける隠れ状態
        h_n         :(num_layers * num_directions, batch_size * n_node, hidden_dim) 時系列の最後の隠れ状態
        c_n         :(num_layers * num_directions, batch_size * n_node, hidden_dim) 時系列の最後のセル状態
        num_layersはLSTMの層数、スタック数。
        num_directionsはデフォルト1、双方向で2。
        """
        input_state = prop_state.view(self.batchSize, self.n_node, self.L * self.state_dim)
        output = self.fnn(input_state)
        output = self.out(output)
        return output
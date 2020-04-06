import torch 
import torch.nn as nn 


act_func = {
    'relu': torch.relu,
    'sigmoid': torch.sigmoid
}


class GCN(nn.Module):
    def __init__(self, input_dim, output_dim, num_support, dropout=0., act='relu', bias=False, batch_norm=False, init='xav_uniform'):
        super(GCN, self).__init__()

        assert init in ('xav_uniform', 'kaiming_uniform', 'xav_normal', 'kaiming_normal')

        self.weights = nn.ParameterList()
        self.bias = None 
        self.batch_norm = None

        if bias:
            self.bias = nn.Parameter(torch.empty(output_dim))

        for _ in range(num_support):
            self.weights.append(nn.Parameter(torch.empty(input_dim, output_dim)))

        self.dropout = nn.Dropout(p=dropout)
        self.act = act_func[act]
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)

        # initialize layers
        for weight in self.weights:
            # initialize weight
            if init == 'xav_uniform':
                torch.nn.init.xavier_uniform_(weight.data, gain=nn.init.calculate_gain(act))
            elif init == 'kaiming_uniform':
                torch.nn.init.kaiming_uniform_(weight.data, nonlinearity=act)
            elif init == 'xav_normal':
                torch.nn.init.xavier_normal_(weight.data, gain=nn.init.calculate_gain(act))
            else:
                torch.nn.init.kaiming_normal_(weight.data, nonlinearity=act)
            
        # initialize bias
        if bias:
            torch.nn.init.zeros_(self.bias.data)


    def forward(self, input, supports):
        x = self.dropout(input).float()
        
        supports_out = []

        for i in range(len(supports)):
            w = self.weights[i]

            tmp = torch.matmul(x, w)  # X * Theta
            adj_s = supports[i]
            tmp = torch.sparse.mm(adj_s, tmp)  # As * X * Theta
            supports_out.append(tmp)
        
        z = sum(supports_out)

        if self.bias is not None:
            z = z + self.bias 
        
        out = self.act(z)

        if self.batch_norm is not None:
            out = self.batch_norm(out)
        
        return out 


class MLPDecoder(nn.Module):
    """
    MLP based decoder model for edge prediction.
    """
    def __init__(self, input_dim, num_classes, dropout=0., bias=False, init='xav_uniform'):
        super(MLPDecoder, self).__init__()

        assert init in ('xav_uniform', 'kaiming_uniform', 'xav_normal', 'kaiming_normal')

        self.weight = nn.Parameter(torch.empty(input_dim, num_classes if num_classes > 2 else 1))
        self.bias = None 

        if bias:
            self.bias = nn.Parameter(torch.empty(num_classes if num_classes > 2 else 1))
        
        self.dropout = nn.Dropout(p=dropout)

        # initialize layers
        if init == 'xav_uniform':
            torch.nn.init.xavier_uniform_(self.weight.data)
        elif init == 'kaiming_uniform':
            torch.nn.init.kaiming_uniform_(self.weight.data)
        elif init == 'xav_normal':
            torch.nn.init.xavier_normal_(self.weight.data)
        else:
            torch.nn.init.kaiming_normal_(self.weight.data)

        # initialize bias
        if bias:
            torch.nn.init.zeros_(self.bias.data)
    
    def forward(self, input, r_indices, c_indices):
        x = self.dropout(input).float()

        # r corrsponds to start nodes of edges and c corresponds to end nodes of edges
        start_inputs = x[r_indices]
        end_inputs = x[c_indices]

        diff = torch.abs(start_inputs - end_inputs)  # |Hi - Hj|
        out = torch.matmul(diff, self.weight)  # |Hi-Hj| * W 

        if self.bias is not None:
            out = out + self.bias 
        
        return out

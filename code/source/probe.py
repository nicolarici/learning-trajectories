import torch.nn as nn


class Probe(nn.Module):

    def __init__(self):
        super(Probe, self).__init__()


class FixedDimProbe(Probe):

    def __init__(self, args, layer):
        super(FixedDimProbe, self).__init__()

        # Defining dimensions.
        model_dim = args['model']['hidden_dim']
        intermediate_size = args['probe']['hidden_dim']
        label_space_size = args['dataset']['label_space_size']

        self.args = args
        self.layer = layer

        # Defining layers.
        self.initial_linear = nn.Linear(model_dim, intermediate_size)
        
        self.intermediate_linears = nn.ModuleList()
        for _ in range(args['probe']['probe_hidden_layers']):
            self.intermediate_linears.append(nn.Linear(intermediate_size, intermediate_size))

        self.last_linear = nn.Linear(intermediate_size, label_space_size)

        self.dropout = nn.Dropout(p=args['probe']['dropout'])
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        self.to(args['device'])


    def forward(self, batch):

        batch = self.dropout(batch)
        intermediate = self.relu(self.initial_linear(batch))

        for linear in self.intermediate_linears:
            intermediate = self.relu(linear(intermediate))
            intermediate = self.dropout(intermediate)

        return self.softmax(self.last_linear(intermediate))


class DynamicDimProbe(Probe):

    def __init__(self, args, layer):
        super(DynamicDimProbe, self).__init__()

        # Defining dimensions.
        model_dim = args['model']['hidden_dim']
        label_space_size = args['dataset']['label_space_size']
        
        net_sizes = [model_dim]

        for n in range(args['probe']['probe_hidden_layers']):
            if args['probe'][f"{n}_hidden_dim"]:
                net_sizes.append(args['probe'][f"{n}_hidden_dim"])
        
        net_sizes.append(label_space_size)

        self.args = args
        self.layer = layer

        # Defining layers.
        
        self.intermediate_linears = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(len(net_sizes)-1):
            self.intermediate_linears.append(nn.Linear(net_sizes[i], net_sizes[i+1]))
            self.dropouts.append(nn.Dropout(p=args['probe']['dropout']))

        self.first_dropout = nn.Dropout(p=args['probe']['dropout'])
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        self.to(args['device'])


    def forward(self, x):

        for linear, dropout in zip(self.intermediate_linears, self.dropouts):
            x = self.relu(linear(x))
            x = dropout(x)

        return self.softmax(x)

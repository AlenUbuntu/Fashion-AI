import torch 
import torch.nn as nn
import torch.optim as optim
from .layers import GCN, MLPDecoder
from sklearn.metrics import confusion_matrix


class CompatibilityGAE(nn.Module):
    def __init__(self, input_dim, hidden, num_classes, settings, init='xav_uniform'):
        super(CompatibilityGAE, self).__init__()

        layers = []
        self.settings = settings

        # stack GCN layers as encoder
        for i in range(len(hidden)):
            layers.append(
                GCN(
                    input_dim=input_dim,
                    output_dim=hidden[i],
                    num_support=settings['num_support'],  # each support is an adjacency matrix measuring s-hop connectivity (s=0,1,2,3,...)
                    dropout=settings['dropout'],
                    act='relu',
                    bias=not settings['batch_norm'],
                    batch_norm=settings['batch_norm'],
                    init=init
                )
            )
            input_dim = hidden[i]
        
        self.encoder = nn.Sequential(*layers)

        # create a decoder
        self.decoder = MLPDecoder(
            input_dim=input_dim, 
            num_classes=num_classes,
            dropout=0.,
            bias=True
        )

        self.optimizer = optim.Adam(
            [
                {'params': self.encoder.parameters()},
                {'params': self.decoder.parameters()}
            ],
            lr=settings['learning_rate'],
            betas=[0.9, 0.999],
            eps=1.e-8,
            weight_decay=settings['wd']
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.confusion_matrix = None 
    
    def forward(self, inputs, supports, r_indices, c_indices):
        """
        @params inputs: node features
        @params supports: support message passing adjacency matrix of a graph
        @params r_indices: start nodes of edges to predict
        @params c_indices: end nodes of edges to predict
        """
        node_embedding = inputs

        # forward encoder
        for layer in self.encoder:
            node_embedding = layer(node_embedding, supports)

        # forward decoder
        out = self.decoder(node_embedding, r_indices, c_indices)

        return out.squeeze()
    
    def compute_loss(self, logits, labels):
        # classification loss
        loss = self.criterion(logits, labels)
        return loss 
    
    def accuracy(self, pred, labels):
        """
        Accuracy for binary class labels.
        @params pred: predicted labels
        @params labels: gt labels
        """
        # if prediction probability >= 0.5, prediction = 1 else 0
        pred = pred.int()
        labels = labels.int()
        accuracy = sum(pred==labels).float()/len(pred)
        return accuracy
    
    def update_confusion_matrix(self, pred, labels):
        pred = pred.int()
        labels = labels.int()
        pred = pred.cpu().numpy()
        labels = labels.cpu().numpy()
        conf_mat = confusion_matrix(labels, pred)
        if self.confusion_matrix is None:
            self.confusion_matrix = conf_mat
        else:
            self.confusion_matrix += conf_mat
    
    def predict(self, inputs, supports, r_indices, c_indices):
        logits = self.forward(inputs, supports, r_indices, c_indices)
        pred = (torch.sigmoid(logits) >= 0.5).int()
        return pred

    def train_epoch(self, inputs, supports, r_indices, c_indices, labels):
        logit = self.forward(inputs, supports, r_indices, c_indices)
        loss = self.compute_loss(logit, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            pred = (torch.sigmoid(logit.detach()) >= 0.5).int()
            self.update_confusion_matrix(pred, labels)
            acc = self.accuracy(pred, labels)
        return loss.item(), acc
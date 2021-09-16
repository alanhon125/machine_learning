import torch
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, to_dense_adj
from torch_scatter import scatter_add
import numpy as np
import matplotlib.pyplot as plt
import time

# Convert adjacency matrix into edge index matrix with size [2,E]
def adj_to_edge_index(A):
    coordinates = [[], []]
    for i in range(len(A)):
        for j in range(len(A[i])):
            if A[i][j] > 0:
                coordinates[0].append(i)
                coordinates[1].append(j)
    edge = torch.tensor(np.array(coordinates), dtype=torch.long)
    return edge

# Convert and normalize mnist dataset into graph data object
def load_process_mnist(load_train=True):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)), ])
    if load_train:
        loader = DataLoader(
            datasets.MNIST('./datasets/train', train=True, download=True, transform=transform), shuffle=False)
    else:
        loader = DataLoader(
            datasets.MNIST('./datasets/test', train=False, download=True, transform=transform), shuffle=False)
    # Compute Adjacency Matrix(A) of MNIST images (Identical for all images)
    img_size = 28  # MNIST image width and height
    sq_img_size = img_size * img_size
    A = []
    for i in range(sq_img_size):
        A.append([0 for _ in range(sq_img_size)])
    for j in range(sq_img_size):
        if j + 1 < sq_img_size:
            A[j][j + 1] = 1
            A[j + 1][j] = 1
        if j + img_size < sq_img_size:
            A[j][j + img_size] = 1
            A[j + img_size][j] = 1
        if j - 1 > 0:
            A[j][j - 1] = 1
            A[j - 1][j] = 1

    edge = adj_to_edge_index(A)
    data_set = []
    for _, (data, target) in enumerate(loader):
        data = data.reshape([784, 1])
        d = Data(x=data, y=int(target), edge_index=edge.contiguous())
        data_set.append(d)
    return data_set

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)
        # Step 3: Compute normalization.
        edge_weight = torch.ones((edge_index.size(1),),
                                 dtype=x.dtype,
                                 device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=x.size(0))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)
        return out

    def message(self, x_j, norm):
        # x_j: after linear x and expand edge, it has shape [E, out_channels]
        # Step 4: Normalize node features.
        # return: each row is norm(embedding) vector for each edge_index pair
        return norm.view(-1, 1) * x_j

class DiffPool(torch.nn.Module):
    def __init__(self, in_channels, next_num_nodes, out_channels):
        super(DiffPool, self).__init__()
        self.embed = GCNConv(in_channels, out_channels)
        self.assign_mat = GCNConv(in_channels, next_num_nodes)

    def forward(self, x, edge_index):
        Z = self.embed(x, edge_index)
        S = F.softmax(self.assign_mat(x, edge_index), dim=-1)
        A = to_dense_adj(edge_index)
        next_X = torch.matmul(S.transpose(0, 1), Z)
        next_A = (S.transpose(0, 1)).matmul(A).matmul(S)
        next_A = torch.reshape(next_A,(next_A.size(1), next_A.size(2)))
        next_edge_index = adj_to_edge_index(next_A)
        return next_X, next_edge_index

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(1, 256)
        self.diffpool1= DiffPool(256,256,128)
        self.conv2 = GCNConv(128, 128)
        self.diffpool2 = DiffPool(128, 64, 128)
        self.conv3 = GCNConv(128, 128)
        self.diffpool3 = DiffPool(128, 1, 128)
        self.linear = torch.nn.Linear(128, 10)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x, edge_index = self.diffpool1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x, edge_index = self.diffpool2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x, edge_index = self.diffpool3(x, edge_index)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)

def train():
    model.train()
    loss_all = 0
    correct = 0
    for i, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        label = data.y.to(device)
        loss = F.nll_loss(output, label)
        _, pred = torch.max(output,1)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        correct += (pred == data.y).sum().cpu().item()
        optimizer.step()
        if i % 10 == 9:
            progress_bar = '[' + ('=' * ((i + 1) // 10)) + (' ' * ((len(train_set) // 100 - (i + 1)) // 10)) + ']'
            print('\repoch: {:d} image {:d}/{:d} loss: {:.3f} accuracy: {:.3f}% {}'
                  .format(epoch + 1, i, len(train_set), loss.cpu().item(), correct / i *100 ,progress_bar), end="  ")
    torch.save(model.state_dict(), './')
    return loss_all / len(train_set)

def evaluate(loader):
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data).detach().cpu().numpy()
            label = data.y.detach().cpu().numpy()
            predictions.append(pred)
            labels.append(label)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ## 1. prepare dataset
    training_size = 50000
    train_set = load_process_mnist(load_train=True)[:training_size]
    print("Train set size: ",len(train_set))
    test_set = load_process_mnist(load_train=False)
    print("Test set size: ", len(test_set))
    # plot_dataset(dataset)
    train_loader = DataLoader(train_set, batch_size=1 ,shuffle = True)
    test_loader = DataLoader(test_set, batch_size=1 , shuffle = True)
    ## 2. prepare model
    model = Net().to(device)
    ## 3. prepare loss function
    optimizer = torch.optim.Adam(model.parameters())
    # criterion = torch.nn.CrossEntropyLoss()
    ## 4. training
    plot = True
    load_model = False
    for epoch in range(10):
        start_time = time.time()
        loss = train()
        print("Time to train epoch # %d: --- %s seconds ---" % (epoch,time.time() - start_time))
        if load_model:
            model.load_state_dict(torch.load('./'))
        train_acc = evaluate(train_loader)
        test_acc = evaluate(test_loader)
        print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f}'.
              format(epoch, loss, train_acc, test_acc))
        if plot:
            plt.plot(train_acc, label='Train accuracy')
            plt.plot(test_acc, label='Validation accuracy')
            plt.xlabel("# Epoch")
            plt.ylabel("Accuracy")
            plt.legend(loc='upper right')
            plt.show()
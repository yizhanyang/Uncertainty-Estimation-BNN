import torch
import torch.nn as nn
import torch.optim as optim

skip_training = False

class MCDropout(nn.Module):
    def __init__(self):
        super(MCDropout,self).__init__()
        self.net = nn.Sequential(
            nn.Dropout(0.05),
            nn.Linear(8, 100),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(100,100),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(100,20),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(20, 1))
        
    def forward(self, x):
        return self.net(x)
    
class EarlyStopping:
    def __init__(self, tolerance, patience):
        
        self.tolerance = tolerance
        self.patience = patience
    
    def stop_criterion(self, val_errors):
        if len(val_errors) < self.patience + 1:
            return False
        else:
            current_best = min(val_errors[:-self.patience])
            current_stop = True
            for i in range(self.patience):
                current_stop = current_stop and (val_errors[-i-1] - current_best > self.tolerance)
            return current_stop

def training_MC(mlp, x, y, x_test, y_test, learning_rate = 0.0001, batch_size = 50, num_epoch=1000, tolerance=0.002, patience = 20):
    
    parameters = set(mlp.parameters())
    optimizer = optim.Adam(parameters, lr = learning_rate)
    early_stop = EarlyStopping(tolerance, patience)
    criterion = nn.MSELoss()

    train_errors = []
    val_errors = []

    num_data, num_dim = x.shape
    y = y.view(-1, 1)
    data = torch.cat((x, y), 1)
        
    for epoch in range(num_epoch):
        # permuate the data
        if skip_training :
            break
        data_perm = data[torch.randperm(len(data))]
        x = data_perm[:, 0:-1]
        y = data_perm[:, -1]

        for index in range(int(num_data/batch_size)):
            # data comes in
            inputs = x[index*batch_size : (index+1)*batch_size]
            labels = y[index*batch_size : (index+1)*batch_size].view(-1,1)
            #print(inputs)
            # initialize the gradient of optimizer
            optimizer.zero_grad()
            mlp.train()
            # calculate the loss function

            outputs = mlp(inputs)          
            loss = criterion(outputs, labels)

            # backpropogate the gradient     
            loss.backward()
            # optimize with SGD
            optimizer.step()

        # train and validation loss
        mlp.eval()
        train_errors.append(criterion(mlp.forward(x), y.view(-1,1)))
        val_errors.append(criterion(mlp.forward(x_test), y_test.view(-1,1)))

        # determine if early stop
        if early_stop.stop_criterion(val_errors):
            print(val_errors[epoch])
            print('Stop after %d epochs' % epoch)
            break

        if (epoch % 10) == 0:
            print('EPOACH %d: TRAIN LOSS: %.4f; VAL LOSS IS: %.5f.'% (epoch+1, train_errors[epoch], val_errors[epoch]))
        #save the model
        torch.save(mlp.state_dict(), 'MC_mlp_01.pth')



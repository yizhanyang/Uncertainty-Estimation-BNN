from sklearn import preprocessing
import torch
def data_loader(traindf, testdf):
    # training set
    trainx = traindf.drop('median_house_value', axis=1).values
    trainy = traindf['median_house_value'].values
    # test set
    testx = testdf.drop('median_house_value', axis=1).values
    testy = testdf['median_house_value'].values
   
    scalerx = preprocessing.MinMaxScaler().fit(trainx)
    trainx = scalerx.transform(trainx) + 0.5
    testx = scalerx.transform(testx) + 0.5

    scalery = preprocessing.MinMaxScaler().fit(trainy.reshape(-1,1))
    trainy = scalery.transform(trainy.reshape(-1,1)).reshape(1, -1)
    testy = scalery.transform(testy.reshape(-1,1)).reshape(1, -1)
      
    # numpy to torch tensor
    x = torch.tensor(trainx, dtype = torch.float)
    y = torch.tensor(trainy[0], dtype = torch.float)
    x_test = torch.tensor(testx, dtype = torch.float)
    y_test = torch.tensor(testy[0], dtype = torch.float)


    return x, y, x_test, y_test

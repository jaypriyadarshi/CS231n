import numpy as np 
from cs231n.classifiers.cnn import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.solver import Solver

data = get_CIFAR10_data()
for k, v in data.iteritems():
  print '%s: ' % k, v.shape

model = SevenLayerConvNetNorm(use_batchnorm=True, dropout=0.5, weight_scale=0.001, hidden_dim=1000, reg=0.001)
#model = SevenLayerConvNet(weight_scale=0.001, hidden_dim=500, reg=0.001)

solver = Solver(model, data,
                num_epochs=20, batch_size=200,
                update_rule='adam',
                optim_config={
                  'learning_rate': 1e-3,
                },
                verbose=True, print_every=20)
solver.train()

f = open('model.txt','w')
f.write(str(model))
f.close()

y_test_pred = np.argmax(model.loss(data['X_test']), axis=1)
y_val_pred = np.argmax(model.loss(data['X_val']), axis=1)
print 'Validation set accuracy: ', (y_val_pred == data['y_val']).mean()
print 'Test set accuracy: ', (y_test_pred == data['y_test']).mean()
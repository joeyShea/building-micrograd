from micrograd import MLP
from value import Value

# create model
n = MLP(3, [4, 4, 1]) # 3 inputs, 4 neurons in first layer, 4 neurons in second layer, 1 output

# create input data
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]
ys = [1.0, -1.0, -1.0, 1.0] # expected outputs

epochs = -1
while epochs < 0:
    try:
        epochs = int(input("Enter number of epochs: "))
    except ValueError:
        print("Invalid input. Please enter a positive integer.")

a = -1
while a < 0:
    try:
        a = float(input("Enter learning rate: "))
    except ValueError:
        print("Invalid input. Please enter a positive value.")

# train the model
ypred = []
loss = 0.0
for k in range(epochs):
    # forward pass
    ypred = [n(x) for x in xs] # make predictions
    loss = sum([(yout - ygt)**2 for ygt, yout in zip(ys, ypred)]) # MSE loss

    # backward pass
    for p in n.parameters():
        p.grad = 0.0
    loss.backward()

    # update
    for p in n.parameters():
        p.data += -0.1 * p.grad

    print(k, loss.data)
    
print("\nLoss (MSE): " + str(loss.data))
print("Predictions: " + str(ypred))
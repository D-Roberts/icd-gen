"""

Just in case I make some custom optimization routines, code up a
somewhat custom trainer poc.

Suppose I had two models, simplest case. Other much more intricate
and optimized interplays could be done.
"""

import torch
import torch.nn as nn


def train_loop(
    batch_num, X_train, y_train, model1, model2, epochs, loss_fn, optimizer, lr
):
    # have a dataloader instead
    for epoch in range(epochs):
        preds1 = model1.forward(X_train)
        preds2 = model2.forward(X_train)

        loss1 = loss_fn(preds1, y_train)
        loss2 = loss_fn(preds2, y_train)

        # get this for each batch
        loss = loss1 + loss2

        # backward pass
        model1.zero_grad()
        model2.zero_grad()

        trainable_weights1 = [
            param for name, param in model1.named_parameters() if param.requires_grad
        ]
        tw2 = [
            param for name, param in model2.named_parameters() if param.requires_grad
        ]

        if batch_num % 2:
            loss.backward()  # this is wrt to weights

            gradients1 = [param.grad for param in trainable_weights1]
            print(f"gradients1 {gradients1}")

            gradients2 = [param.grad for param in tw2]
            print(f"gradients2 {gradients2}")

            with torch.no_grad():
                for param in model1.parameters():
                    param.data -= lr * param.grad.data
                for param in model2.parameters():
                    param.data -= lr * param.grad.data

            print(f"loss {loss}")

        else:  # even batch numbers
            # dummy tensor loss for a second backward
            X = torch.tensor(2.0, requires_grad=True)
            fx = X**2

            loss2 = fx.sum()
            loss2.backward()

            print("The X gradient ", X.grad)


# Ad hoc test


class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


criterion = nn.MSELoss()
X_train = torch.randn(100, 10)  # 100 samples, 10 features
y_train = torch.randn(100, 1)  # 100 samples, 1 output

model1 = SimpleModel(input_dim=10, output_dim=1)
model2 = SimpleModel(input_dim=10, output_dim=1)

train_loop(0, X_train, y_train, model1, model2, 10, criterion, torch.optim.Adam, 0.01)
train_loop(1, X_train, y_train, model1, model2, 10, criterion, torch.optim.Adam, 0.01)

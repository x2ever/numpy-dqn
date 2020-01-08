from .Policy import Policy
import numpy as np

class FCNPolicy(Policy):
    #For weight initialization, used He normal init.
    #For bias initialization, used Zero init.
    def __init__(self, input_n, output_n, hidden_n=16, hidden_layer_n=1):
        self.layers = list()
        input_w = np.random.normal(scale=np.sqrt(2 / input_n), size=(input_n, hidden_n))
        input_b = np.zeros(hidden_n)

        self.layers.append((input_w, input_b))

        for i in range(hidden_layer_n):
            hidden_w = np.random.normal(scale=np.sqrt(2 / hidden_n), size=(hidden_n, hidden_n))
            hidden_b = np.zeros(hidden_n)

            self.layers.append((hidden_w, hidden_b))

        output_w = np.random.normal(scale=np.sqrt(2 / hidden_n), size=(hidden_n, output_n))
        output_b = np.zeros(output_n)
        
        self.layers.append((output_w, output_b))

    def train(self, x, y, learning_rate=0.00003):
        predict, update_helper = self.predict(x, update_mode=True)
        update_layers = list()
        d = predict - y
        cost = np.mean(np.square(d))
        d = d * 2
        reversed_layers = list(reversed(self.layers))

        for i, info in enumerate(update_helper):
            #C is cost, x is node
            prev_layer, mid_layer, layer, activate_d_func = info
            # dC/df'(wx+b)
            dl = d
            if activate_d_func is None:
                dl_a = dl
            else:
                #Calculate f'(wx + b) * dC/df'(wx+b)
                dl_a = activate_d_func(mid_layer) * dl

            #dC/db = 1 * f'(wx + b) * dC/df'(wx+b)
            db = np.mean(dl_a, axis=0)
            #dC/dw = x * f'(wx + b) * dC/df'(wx+b)
            #y.shape[0] for mean of total gradient
            dw = prev_layer.T @ dl_a / y.shape[0]
            
            w, b = reversed_layers[i]

            #For backpropagation, we need tensor shape of (batchsize, output) which has each node's gradient for each batch.
            #dC/dx = df(wx+b)/dx * dC/df(wx+b) = w * f'(wx + b) * dC/df'(wx+b) = w * dl_a
            d = (w @ dl_a.T).T
            update_layers.append((dw, db))
        
        update_layers.reverse()

        for i, l in enumerate(zip(update_layers, self.layers)):
            update_layer, layer = l
            dw, db = update_layer
            w, b = layer
            #use for clipping gradient
            #dw = np.clip(dw, -0.5, 0.5)
            #db = np.clip(db, -0.5, 0.5)
            w -= learning_rate * dw
            b -= learning_rate * db
            self.layers[i] = (w, b)

        return cost
            
    def predict(self, x, update_mode=False):
        update_helper = list()
        prev_x = x
        for param in self.layers[:-1]:
            w, b = param
            mid_x = x @ w + b
            x = self.ReLU(mid_x)
            if update_mode:
                update_helper.append((prev_x, mid_x, x, self.d_ReLU))
                prev_x = np.copy(x)

        w, b = self.layers[-1]
        if update_mode:
            mid_x = x @ w + b
            update_helper.append((x, mid_x, mid_x, None))
            return x @ w + b, list(reversed(update_helper))
        else:
            return x @ w + b

if __name__ == "__main__":
    p = FCNPolicy(2, 1)
    x = np.array([[0, 0],
                  [1, 0],
                  [0, 1],
                  [1, 1]])
    y = np.array([[100],
                  [0],
                  [0],
                  [1]])

    for i in range(10000):

        cost = p.train(x, y)

        print(cost)
    
    print(p.predict([[1, 0]]))
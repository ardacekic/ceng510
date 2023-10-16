import numpy as np

def f(x,w,b): # do not change this line!
    y1  = np.dot(x,w.T) + b
    y_hat  = np.where(y1 <= 0, 0, np.where(y1 > 1, 0, y1))
    df_dw = w
    df_db = 1
    return y_hat,df_dw,df_db

def l1loss(x,y,w,b): # do not change this line!
    #forward pass to see current loss

    y_hat,two,three = f(x,w,b)
    error = y - y_hat
    l1loss_error =  np.mean(np.abs(error))
    
    # Derivative of alpha function w.r.t. y1
    dalpha = np.where((y_hat <= 0) | (y_hat > 1), 0, 1)

    ######
    #forward pass to see current loss
    y_hat,two,three = f(x,w,b)
    error = y - y_hat
    l1loss_error =  np.mean(np.abs(error))
    
    # Derivative of updated alpha function w.r.t. y_hat
    # Derivative of y_hat with respect to w
    # Gradients for w and b
    dw = -(1 / len(x)) * np.dot(x.T, np.sign(y - y_hat)*dalpha) 

    #dw = -np.dot(de , x) / len(x)
    db = -(1 / len(x)) * np.sum(np.sign(y - y_hat)*dalpha)
    #db = -np.sum(de ) / len(x)
    
    dl_dw = dw
    dl_db = db
    
    return l1loss_error,dl_dw,dl_db

def minimize_l1loss(x,y,w,b, num_iters=10000, eta=0.0001): # do not change this line!
    iteration_loss = []
    for _ in range(num_iters):
        loss,dl_dw,dl_db = l1loss(x,y,w,b)
        # Update w and b using gradient descent
        iteration_loss.append(loss)
        w -= eta * dl_dw
        b -= eta * dl_db
    
    return w, b , iteration_loss

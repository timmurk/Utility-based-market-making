import numpy as np
import time
from util import Logger 
from IPython.display import clear_output

class AS():
    def __init__(self, logger, gamma):
        self.A = 135.
        self.k = 1.5
        self.gamma = gamma# risk aversion 
        self.logger = logger
        
    def optimize(self, volume, markets_mask):
        self.alpha = 1.5
        self.A = np.sum(volume) / np.len(np.where(markets_mask == True))
        
    def get_pnl(self, N, T):
        r, r_ask, r_bid, pnl, x, q = np.zeros(N + 2), np.zeros(N + 2), np.zeros(N + 2), np.zeros(N + 2), np.zeros(N + 2), np.zeros(N + 2)
        
        dt = T / N
        
        for i in range(N):
            r_spread = 2. / self.gamma * np.log(1 + self.gamma / self.k)    
        
            r_ask[i] = r[i] + r_spread / 2.
            r_bid[i] = r[i] - r_spread / 2.

            #Determine deltas
            delta_ask = r_ask[i] - data[i]
            delta_bid = data[i] - r_bid[i]

            #Determine lambdas
            lambda_ask = self.A * np.exp(-k * delta_ask)
            lambda_bid = self.A * np.exp(-k * delta_bid)

            #Check t + dt 

            add_ask, add_bid = 0, 0

            if np.random.rand() < lambda_ask * dt:
                add_ask = 1
            if np.random.rand() < lambda_bid * dt:
                add_bid = 1

            q[i + 1] = q[i] - add_ask + add_bid
            x[i + 1] = x[i] + r_ask[i] * add_ask - r_bid[i] * add_bid
            pnl[i + 1] = x[i + 1] + q[i + 1] * data[i]
        
        return pnl, x, q, r_ask, r_bid, r
    
    def get_pnl_online(self, stream, demonstrate=False, process=None, asset=None, flag_type=False, video=False, path=None):
        r, r_ask, r_bid, pnl, x, q = [], [], [], [], [0.], [0.]
        
        stream.start(process, flag_type, asset)
        
        dt = stream.dt
    
        asset_data = []
        
        while True:
            if demonstrate:
                try:
                    r_spread = 2. / self.gamma * np.log(1 + self.gamma / self.k) 

                    r_ask.append(r_spread / 2.)
                    r_bid.append(- r_spread / 2.)

                    #Determine deltas
                    delta_ask = r_ask[-1] - stream.value
                    delta_bid = stream.value - r_bid[-1]

                    #Determine lambdas
                    lambda_ask = self.A * np.exp(-self.k * delta_ask)
                    lambda_bid = self.A * np.exp(-self.k * delta_bid)

                    #print(lambda_ask, lambda_bid)

                    #Check t + dt 

                    add_ask, add_bid = 0, 0

                    if np.random.rand() < lambda_ask * dt:
                        add_ask = 1
                    if np.random.rand() < lambda_bid * dt:
                        add_bid = 1

                    #print(lambda_ask * dt, lambda_ask * dt)

                    q.append(q[-1] - add_ask + add_bid)
                    x.append(x[-1] + r_ask[-1] * add_ask - r_bid[-1] * add_bid)
                    pnl.append(x[-1] + q[-1] * stream.value)

                    asset_data.append(stream.value)

                    self.logger.send([(q, 'Quantity.'), (asset_data, 'Asset.'), (pnl, 'PnL')])

                    time.sleep(stream.dt)

                    stream.step(asset)

                    if demonstrate:
                        self.logger.plot(video, path)
                        
                except KeyboardInterrupt:
                    clear_output(True)
                    print("Demonstration is done.")
                    break
                

                
                
                
                
                
                
                
                
                
                
                
                
        
# Utility-based-market-making
Utility based approach in market-making problem.

The given code considered to be the NES and YSDA project, however the framework can be useful for practical purposes.

Object of interest is market-making problem in Crypto-currency market. The typical points should be considered to:
 
 
    1. Maximization of the profit to the end of the trading day.
    2. Minimization of the liquidity risk, i.e. minimizing the inventory by the end of the day.
          
The RL ideas are implemented: Double DQN + Prioritized Exp.Replay for simplification the education process. 

The current results presented here. 

For the mid-price: 
![Mid price](https://github.com/timmurk/Utility-based-market-making/blob/main/mid_price_pnl.jpg?raw=true)

For the weight-price:
![Weight price](https://github.com/timmurk/Utility-based-market-making/blob/main/wp_price_pnl.jpg?raw=true)

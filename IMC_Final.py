from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
import math
import numpy as np
from numpy import exp, sqrt, pi
import statistics

SUBMISSION = "SUBMISSION"
AMETHYSTS = "AMETHYSTS"
STARFRUIT = "STARFRUIT"
ORCHIDS = "ORCHIDS"
CHOCOLATE = "CHOCOLATE"
STRAWBERRIES = "STRAWBERRIES"
ROSES = "ROSES"
GIFT_BASKET = "GIFT_BASKET"
COCONUT_COUPON = "COCONUT_COUPON"
COCONUT = "COCONUT"


PRODUCTS = [
    AMETHYSTS,
    STARFRUIT,
    ORCHIDS,
    CHOCOLATE,
    STRAWBERRIES,
    ROSES,
    GIFT_BASKET,
    COCONUT,
    COCONUT_COUPON
]

DEFAULT_PRICES = {
    AMETHYSTS: 10_000,
    STARFRUIT: 5_000,
    ORCHIDS: 1000,
    ROSES: 14000,
    CHOCOLATE: 8000,
    STRAWBERRIES: 4000,
    GIFT_BASKET: 70000,
    COCONUT: 10000,
    COCONUT_COUPON: 10000
}

class Trader:
    def __init__(self) -> None:
        print("Initializing Trader...")
        self.position_limit = {
            AMETHYSTS: 20,
            STARFRUIT: 20,
            ORCHIDS: 100,
            CHOCOLATE: 250,
            STRAWBERRIES: 350,
            ROSES: 60,
            GIFT_BASKET: 60,
            COCONUT: 300,
            COCONUT_COUPON: 600
        }
        self.round = 0
        self.cash = 0

        self.starfruit_cache = []
        self.starfruit_dim = 7

        self.chocolate_cache = []
        self.chocolate_dim = 7

        self.strawberries_cache = []
        self.strawberries_dim = 7

        self.roses_cache = []
        self.roses_dim = 7

        self.gift_basket_cache = []
        self.gift_basket_dim = 7
        
        self.coconut_coupon_cache = []
        self.coconut_coupon_dim = 100
        



        self.ema_prices = {product: DEFAULT_PRICES[product] for product in PRODUCTS}
        self.amethysts_spread_up = 2
        self.amethysts_spread_down = 2
        self.past_sunlight = None
        self.past_humidity = None



    """
    def get_bs_price_numpy(self, S, timestamp, day, K, r, sigma):

        def erf_approx(x):
            a1 =  0.254829592
            a2 = -0.284496736
            a3 =  1.421413741
            a4 = -1.453152027
            a5 =  1.061405429
            p  =  0.3275911

            # Save the sign of x
            sign = np.sign(x)
            x = np.abs(x)

            # A&S formula 7.1.26
            t = 1.0 / (1.0 + p*x)
            y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x)

            return sign*y
        
        T = (250 - day + (1000000 - timestamp) / 1000000) / 365
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        # Calculating the cumulative distribution function values for d1 and d2 using numpy
        cdf_d1 = 0.5 * (1 + erf_approx(d1 / np.sqrt(2)))
        cdf_d2 = 0.5 * (1 + erf_approx(d2 / np.sqrt(2)))
        
        C = S * cdf_d1 - K * np.exp(-r * T) * cdf_d2
        return C
    
    """



    def get_bs_price_numpy(self, S, T, K, r, sigma):
            N = statistics.NormalDist(mu=0, sigma=1)
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            cdf_d1 = N.cdf(d1)
            cdf_d2 = N.cdf(d2)
            C = S * cdf_d1 - K * np.exp(-r * T) * cdf_d2
            return C

    






    def get_position(self, product, state: TradingState):
        return state.position.get(product, 0)    

    def get_mid_price(self, product, state: TradingState):
        default_price = self.ema_prices[product]
        if product not in state.order_depths:
            return default_price
        market_bids = state.order_depths[product].buy_orders
        if len(market_bids) == 0:
            return default_price
        market_asks = state.order_depths[product].sell_orders
        if len(market_asks) == 0:
            return default_price
        best_bid = max(market_bids)
        best_ask = min(market_asks)
        return (best_bid + best_ask) / 2

    def calc_next_price_starfruit(self):
        coef = [0.06119917724497761, 0.07612804987876423, 0.09726994388253776, 0.10943461598131106, 0.14763276238717365, 0.21174658952830833, 0.2962220188954584]
        intercept = 1.8530096469830823
        nxt_price = intercept
        for i, val in enumerate(self.starfruit_cache):
            nxt_price += val * coef[i]

        return int(round(nxt_price))


    def calc_next_price_gift_basket(self):
        coef_chocolate = [0.4804, 0.0073, -0.0199, -0.0103, -0.0463, 0.0685, 3.3640]
        intercept = 161.6499
        coef_strawberries = [1.1825, 0.1758, 0.0403, 0.0428, 0.2579, 0.6173, 3.8543]
        coef_roses = [-0.0069, -0.0293, -0.0196, 0.0083, 0.0009, 0.0240, 1.0754]
        
        
        nxt_price = intercept
 
        for i, val in enumerate(self.chocolate_cache):
            nxt_price += val * coef_chocolate[i]

        for i, val in enumerate(self.strawberries_cache):
            nxt_price += val * coef_strawberries[i]
            
        for i, val in enumerate(self.roses_cache):
            nxt_price += val * coef_roses[i]




        return int(round(nxt_price))










    def calc_next_price_chocolate(self):
        
           
        
        intercept = 0.6788    
        coef_chocolate = [0.98369464, 0.01859974, 0.00432448, -0.00408152, -0.00239515, -0.00947092, 0.00933048]
        coef_chocolate = coef_chocolate[::-1]
        
        coef_strawberries = [0.00272472, 0.01727635, -0.02607548, 0.00372565, 0.02470762, -0.02138593, -0.00126366]
        coef_strawberries = coef_strawberries[::-1]
#        coef_roses = [-0.00169739581, 0.000194764141, 0.00347008460, -0.00128169870, -0.000512740795]
        
#        coef_gift_basket = [0.00116798437, -0.000164682809, -0.00144073911, -0.00252132884, 0.00287272363]
        
        
        nxt_price = intercept
 
        for i, val in enumerate(self.chocolate_cache):
            nxt_price += val * coef_chocolate[i]

        for i, val in enumerate(self.strawberries_cache):
            nxt_price += val * coef_strawberries[i]
            
#        for i, val in enumerate(self.roses_cache[2:]):
#           nxt_price += val * coef_roses[i]


#        for i, val in enumerate(self.gift_basket_cache[2:]):
#            nxt_price += val * coef_gift_basket[i]


        return int(round(nxt_price))

            


    def calc_next_price_roses(self):
        coef_roses = [0.0032, -0.0001, -0.0076, -0.0055, 0.0018, 0.0153, 0.9926]
        intercept = 4.1872
        nxt_price = intercept
        for i, val in enumerate(self.roses_cache):
            nxt_price += val * coef_roses[i]

        return int(round(nxt_price))

    def calc_next_price_strawberries(self):
        coef_strawberries = [0.999892311120613]
        intercept = 0.4331290697859913
        nxt_price = intercept
        nxt_price += self.strawberries_cache[-1] * coef_strawberries[0]

        return int(round(nxt_price))









    
    
    def compute_orders_regression(self, product, order_depth, acc_bid, acc_ask, LIMIT,state: TradingState):
        orders: list[Order] = []

        osell = order_depth[product].sell_orders
        obuy = order_depth[product].buy_orders

        best_sell_pr = max(osell)
        best_buy_pr = min(obuy)

        cpos = self.get_position(product,state)

        for ask, vol in osell.items():
            if ((ask <= acc_bid) or ((self.get_position(product,state)<0) and (ask == acc_bid+1))) and cpos < LIMIT:
                order_for = min(-vol, LIMIT - cpos)
                cpos += order_for
                assert(order_for >= 0)
                orders.append(Order(product, math.floor(ask), order_for))

        undercut_buy = math.ceil(best_buy_pr + 1)
        undercut_sell = math.floor(best_sell_pr - 1)

        bid_pr = min(undercut_buy, acc_bid) # we will shift this by 1 to beat this price
        sell_pr = max(undercut_sell, acc_ask)

        if cpos < LIMIT:
            num = LIMIT - cpos
            orders.append(Order(product, bid_pr, num))
            cpos += num
        
        cpos = self.get_position(product,state)
        

        for bid, vol in obuy.items():
            if ((bid >= acc_ask) or ((self.get_position(product,state)>0) and (bid+1 == acc_ask))) and cpos > -LIMIT:
                order_for = max(-vol, -LIMIT-cpos)
                # order_for is a negative number denoting how much we will sell
                cpos += order_for
                assert(order_for <= 0)
                orders.append(Order(product, math.ceil(bid), order_for))

        if cpos > -LIMIT:
            num = -LIMIT-cpos
            orders.append(Order(product, sell_pr, num))
            cpos += num

        return orders
    
    def compute_orders_amethysts(self, product, order_depth, acc_bid, acc_ask, state: TradingState):
        orders: list[Order] = []

        osell = order_depth[product].sell_orders
        obuy = order_depth[product].buy_orders

        best_sell_pr = max(osell)
        best_buy_pr = min(obuy)

        cpos = self.get_position(product,state)

        mx_with_buy = -1

        for ask, vol in osell.items():
            if ((ask < acc_bid) or ((cpos<0) and (ask == acc_bid))) and cpos < self.position_limit[AMETHYSTS]:
                mx_with_buy = max(mx_with_buy, ask)
                order_for = min(-vol, self.position_limit[AMETHYSTS] - cpos)
                cpos += order_for
                assert(order_for >= 0)
                orders.append(Order(product, ask, order_for))


        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        bid_pr = min(undercut_buy, acc_bid-1) # we will shift this by 1 to beat this price
        sell_pr = max(undercut_sell, acc_ask+1)

        if (cpos < self.position_limit[AMETHYSTS]) and (self.get_position(product,state) < 0):
            num = min(40, self.position_limit[AMETHYSTS] - cpos)
            orders.append(Order(product, min(undercut_buy + 1, acc_bid-1), num))
            cpos += num

        if (cpos < self.position_limit[AMETHYSTS]) and (self.get_position(product,state) > 15):
            num = min(40, self.position_limit[AMETHYSTS] - cpos)
            orders.append(Order(product, min(undercut_buy - 1, acc_bid-1), num))
            cpos += num

        if cpos < self.position_limit[AMETHYSTS]:
            num = min(40, self.position_limit[AMETHYSTS] - cpos)
            orders.append(Order(product, bid_pr, num))
            cpos += num
        
        cpos = self.get_position(product,state)

        for bid, vol in obuy.items():
            if ((bid > acc_ask) or ((cpos>0) and (bid == acc_ask))) and cpos > -self.position_limit[AMETHYSTS]:
                order_for = max(-vol, -self.position_limit[AMETHYSTS]-cpos)
                # order_for is a negative number denoting how much we will sell
                cpos += order_for
                assert(order_for <= 0)
                orders.append(Order(product, bid, order_for))

        if (cpos > -self.position_limit[AMETHYSTS]) and (self.get_position(product,state) > 0):
            num = max(-40, -self.position_limit[AMETHYSTS]-cpos)
            orders.append(Order(product, max(undercut_sell-1, acc_ask+1), num))
            cpos += num

        if (cpos > -self.position_limit[AMETHYSTS]) and (self.get_position(product,state) < -15):
            num = max(-40, -self.position_limit[AMETHYSTS]-cpos)
            orders.append(Order(product, max(undercut_sell+1, acc_ask+1), num))
            cpos += num

        if cpos > -self.position_limit[AMETHYSTS]:
            num = max(-40, -self.position_limit[AMETHYSTS]-cpos)
            orders.append(Order(product, sell_pr, num))
            cpos += num

        return orders
    
    """ 
    def compute_orders_coconut_coupon(self, product, order_depth, acc_bid, acc_ask, state: TradingState):
        orders: list[Order] = []

        osell = order_depth[product].sell_orders
        obuy = order_depth[product].buy_orders

        best_sell_pr = acc_ask
        best_buy_pr = acc_bid

        cpos = self.get_position(product,state)

        mx_with_buy = -1

        for ask, vol in osell.items():
            if ((ask < acc_bid) or ((cpos<0) and (ask == acc_bid))) and cpos < self.position_limit[COCONUT_COUPON]:
                mx_with_buy = max(mx_with_buy, ask)
                order_for = min(-vol, self.position_limit[COCONUT_COUPON] - cpos)
                cpos += order_for
                assert(order_for >= 0)
                orders.append(Order(product, ask, order_for))


        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        bid_pr = min(undercut_buy, acc_bid-1) # we will shift this by 1 to beat this price
        sell_pr = max(undercut_sell, acc_ask+1)

        if (cpos < self.position_limit[COCONUT_COUPON]) and (self.get_position(product,state) < 0):
            num = min(40, self.position_limit[COCONUT_COUPON] - cpos)
            orders.append(Order(product, min(undercut_buy + 1, acc_bid-1), num))
            cpos += num

        if (cpos < self.position_limit[COCONUT_COUPON]) and (self.get_position(product,state) > 15):
            num = min(40, self.position_limit[COCONUT_COUPON] - cpos)
            orders.append(Order(product, min(undercut_buy - 1, acc_bid-1), num))
            cpos += num

        if cpos < self.position_limit[COCONUT_COUPON]:
            num = min(40, self.position_limit[COCONUT_COUPON] - cpos)
            orders.append(Order(product, bid_pr, num))
            cpos += num
        
        cpos = self.get_position(product,state)

        for bid, vol in obuy.items():
            if ((bid > acc_ask) or ((cpos>0) and (bid == acc_ask))) and cpos > -self.position_limit[COCONUT_COUPON]:
                order_for = max(-vol, -self.position_limit[COCONUT_COUPON]-cpos)
                # order_for is a negative number denoting how much we will sell
                cpos += order_for
                assert(order_for <= 0)
                orders.append(Order(product, bid, order_for))

        if (cpos > -self.position_limit[COCONUT_COUPON]) and (self.get_position(product,state) > 0):
            num = max(-40, -self.position_limit[COCONUT_COUPON]-cpos)
            orders.append(Order(product, max(undercut_sell-1, acc_ask+1), num))
            cpos += num

        if (cpos > -self.position_limit[COCONUT_COUPON]) and (self.get_position(product,state) < -15):
            num = max(-40, -self.position_limit[COCONUT_COUPON]-cpos)
            orders.append(Order(product, max(undercut_sell+1, acc_ask+1), num))
            cpos += num

        if cpos > -self.position_limit[COCONUT_COUPON]:
            num = max(-40, -self.position_limit[COCONUT_COUPON]-cpos)
            orders.append(Order(product, sell_pr, num))
            cpos += num

        return orders 
    
    
    def compute_orders_coconut_coupon(self, product, acc_bid, acc_ask, state: TradingState, isBid, isAsk):
        orders: list[Order] = []
        pos = self.get_position(product,state)

        if(isBid):
            orders.append(Order(product, acc_bid, self.position_limit[product]-pos))
        if(isAsk):
            orders.append(Order(product, acc_ask, -self.position_limit[product]-pos))

        return orders
    """
    
    def compute_orders_orchids2(self, product, state: TradingState):
        orders: list[Order] = []
        obs = state.observations.conversionObservations[ORCHIDS]

        sun = obs.sunlight
        hum = obs.humidity
        
        hum_diff = 0
        if self.past_humidity != None:
            if self.past_humidity >=80:
                if hum >= 80:
                    hum_diff = math.ceil((hum-80)/5)-math.ceil((self.past_humidity-80)/5)
                else:
                    hum_diff = math.floor((80 - self.past_humidity)/5)
            elif self.past_humidity <=60:
                if hum <=60:
                    hum_diff = math.ceil((60-hum)/5)-math.ceil((60-self.past_humidity)/5)
                else:
                    hum_diff = math.floor((self.past_humidity-60)/5)
            else:
                if hum <=60:
                    hum_diff = math.ceil((60-hum)/5)
                elif hum >= 80:
                    hum_diff = math.ceil((hum-80)/5)

        
        sun_diff = 0
        if self.past_sunlight != None:
            if self.past_sunlight >=7*365:
                if sun <= 7*365:
                    sun_diff = math.ceil((7*365-sun)*6/365)
            else:
                if sun <= 7*365:
                    sun_diff = math.ceil((7*365-sun)*6/365)-math.ceil((7*365-self.past_sunlight)*6/365)
                else:
                    sun_diff = math.floor((self.past_sunlight - 7*365)*6/365)

        

        imp    = obs.importTariff + obs.transportFees

        price1 = obs.askPrice/(0.98)**hum_diff
        price1 = price1/(0.96)**sun_diff

        price2 = obs.bidPrice/(0.98)**hum_diff
        price2 = price2/(0.96)**sun_diff

        self.past_humidity = hum
        self.past_sunlight = sun
        
    
        orders.append(Order(product, max(math.ceil(price1 + imp+1),math.ceil(obs.askPrice + imp+1)), -100))
        
        #if(min(math.floor(price2 -obs.exportTariff-1),math.floor(obs.bidPrice-obs.exportTariff-1)) < max(math.ceil(price1 + imp+1),math.ceil(obs.askPrice + imp+1))):
        orders.append(Order(product, min(math.floor(price2 -obs.exportTariff-1),math.floor(obs.bidPrice-obs.exportTariff-1)), 100))
        return orders

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        self.round += 1

        





        mid_price = self.get_mid_price(STARFRUIT, state)

        if len(self.starfruit_cache) == self.starfruit_dim:
            self.starfruit_cache.pop(0)
        self.starfruit_cache.append(mid_price)

        starfruit_lb = -float('inf')
        starfruit_ub = float('inf')

        if len(self.starfruit_cache) == self.starfruit_dim:
            starfruit_lb = self.calc_next_price_starfruit()-1
            starfruit_ub = self.calc_next_price_starfruit()+1





        
        
        mid_price = self.get_mid_price(CHOCOLATE, state)

        if len(self.chocolate_cache) == self.chocolate_dim:
            self.chocolate_cache.pop(0)
        self.chocolate_cache.append(mid_price)

        chocolate_lb = -float('inf')
        chocolate_ub = float('inf')

        if len(self.chocolate_cache) == self.chocolate_dim:
            chocolate_lb = self.calc_next_price_chocolate()-1
            chocolate_ub = self.calc_next_price_chocolate()+1

        
        
        
        
        
        



        mid_price = self.get_mid_price(ROSES, state)

        if len(self.roses_cache) == self.roses_dim:
            self.roses_cache.pop(0)
        self.roses_cache.append(mid_price)

        roses_lb = -float('inf')
        roses_ub = float('inf')

        if len(self.roses_cache) == self.roses_dim:
            roses_lb = self.calc_next_price_roses()-1
            roses_ub = self.calc_next_price_roses()+1
            
            
        

        


        

        mid_price = self.get_mid_price(STRAWBERRIES, state)

        if len(self.strawberries_cache) == self.strawberries_dim:
            self.strawberries_cache.pop(0)
        self.strawberries_cache.append(mid_price)

        strawberries_lb = -float('inf')
        strawberries_ub = float('inf')

        if len(self.strawberries_cache) == self.strawberries_dim:
            strawberries_lb = self.calc_next_price_strawberries()-1
            strawberries_ub = self.calc_next_price_strawberries()+1

        


        mid_price = self.get_mid_price(GIFT_BASKET, state)

        if len(self.gift_basket_cache) == self.gift_basket_dim:
            self.gift_basket_cache.pop(0)
        self.gift_basket_cache.append(mid_price)

        gift_basket_lb = -float('inf')
        gift_basket_ub = float('inf')

        if len(self.gift_basket_cache) == self.gift_basket_dim:
            gift_basket_lb = self.calc_next_price_gift_basket()-15
            gift_basket_ub = self.calc_next_price_gift_basket()+15





        

        






        amethysts_lb = 10000
        amethysts_ub = 10000
        result = {}
        try:
            result[AMETHYSTS] = self.compute_orders_amethysts(AMETHYSTS, state.order_depths, amethysts_lb, amethysts_ub, state)
        except Exception as e:
            print("Error in AMETHYSTS strategy")
            print(e)
        try:
            result[STARFRUIT] = self.compute_orders_regression(STARFRUIT, state.order_depths, starfruit_lb, starfruit_ub,self.position_limit[STARFRUIT], state)
        except Exception as e:
            print("Error in STARFRUIT strategy")
            print(e)
        try:
            result[ORCHIDS] = self.compute_orders_orchids2(ORCHIDS, state)
        except Exception as e:
            print("Error in ORCHID strategy")
            print(e)




          
        """
        try:
            result[ROSES] = self.compute_orders_regression(ROSES, state.order_depths, roses_lb, roses_ub,self.position_limit[ROSES], state)
        except Exception as e:
            print("Error in ROSES strategy")
            print(e)

        
        
        
        
        try:
            result[STRAWBERRIES] = self.compute_orders_regression(STRAWBERRIES, state.order_depths, strawberries_lb, strawberries_ub,self.position_limit[STRAWBERRIES], state)
        except Exception as e:
            print("Error in STRAWBERRIES strategy")
            print(e)
                
        
        
        
        try:
            result[CHOCOLATE] = self.compute_orders_regression(CHOCOLATE, state.order_depths, chocolate_lb, chocolate_ub,self.position_limit[CHOCOLATE], state)
        except Exception as e:
            print("Error in CHOCOLATE strategy")
            print(e)
        """

        try:
            result[GIFT_BASKET] = self.compute_orders_regression(GIFT_BASKET, state.order_depths, gift_basket_lb, gift_basket_ub,self.position_limit[GIFT_BASKET], state)
        except Exception as e:
            print("Error in GIFT_BASKET strategy")
            print(e)



        """         
        try:
            result[COCONUT] = self.compute_orders_regression(COCONUT, state.order_depths, coconut_lb, coconut_ub,self.position_limit[COCONUT], state)
        except Exception as e:
            print("Error in COCONUT strategy")
            print(e)
        """
        
        

        day = 5
        sigma = 0.0101132923
        T = (250 - day + (1000000 - state.timestamp) / 1000000) / 250
        mid_price_coconut = self.get_mid_price(COCONUT, state)
        mid_price_coconut_coupon = self.get_mid_price(COCONUT_COUPON, state)
        bs_price = self.get_bs_price_numpy(mid_price_coconut, T ,10000,0, sigma)
        
        
        
        
        
        
        predicted_price = 578.5008 + 1.0433*bs_price
        coconut_coupon_lb = int(predicted_price) -1
        coconut_coupon_ub = int(predicted_price) +1
        #resid = mid_price_coconut_coupon - predicted_price
        #resid_z = resid/S
        
        
        #if resid_z <= -2:
            #result["COCONUT_COUPON"] = self.compute_orders_coconut_coupon("COCONUT_COUPON", mid_price_coconut_coupon+1, mid_price_coconut_coupon-1, state, True, False)
        #if resid_z>= 2:
            #result["COCONUT_COUPON"] = self.compute_orders_coconut_coupon("COCONUT_COUPON", mid_price_coconut_coupon+1, mid_price_coconut_coupon-1, state, False, True)
        
        
        
        
        
        
        
        
        print("bs_price: {}".format(predicted_price))
        print("mid_price: {}".format(mid_price_coconut_coupon))
        
        
        try:
            result[COCONUT_COUPON] = self.compute_orders_regression(COCONUT_COUPON, state.order_depths, coconut_coupon_lb, coconut_coupon_ub,self.position_limit[COCONUT_COUPON], state)
        except Exception as e:
            print("Error in COCONUT_COUPON strategy")
            print(e)




        
        
        
        
        traderData = "SAMPLE" 
        conversions = -self.get_position(ORCHIDS, state)
        return result, conversions, traderData

                                                                               
                  
                                                                               
      
                                
                                     
                                 
                                                                               

import os
import sys

                 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from locust import task, tag
from tasks.base import BaseTaskSet
from utils.data_generator import DataGenerator, get_future_date


class QueryTasks(BaseTaskSet):
    'English text removed.'
    
                                                                               
          
                                                                               
    
    @task(10)
    @tag('travel', 'core', 'query')
    def query_high_speed_trips(self):
        'English text removed.'
        place_pair = DataGenerator.get_place_pair(is_high_speed=True)
        payload = {
            "startPlace": place_pair[0],
            "endPlace": place_pair[1],
            "departureTime": get_future_date(1)
        }
                    
        self.post_with_check(
            "/api/v1/travelservice/trips/left",
            json=payload,
            name="[Travel] Query High-Speed Trips",
            expect_data=False               
        )
    
    @task(8)
    @tag('travel', 'core', 'query')
    def query_normal_trips(self):
        'English text removed.'
        place_pair = DataGenerator.get_place_pair(is_high_speed=False)
        payload = {
            "startPlace": place_pair[0],
            "endPlace": place_pair[1],
            "departureTime": get_future_date(1)
        }
        self.post_with_check(
            "/api/v1/travel2service/trips/left",
            json=payload,
            name="[Travel] Query Normal Trips",
            expect_data=False
        )
    
                                                                               
          
                                                                               
    
    @task(5)
    @tag('order', 'core', 'query')
    def query_my_orders(self):
        'English text removed.'
        self.post_with_check(
            "/api/v1/orderservice/order/refresh",
            json={
                "loginId": self.user_id,
                "enableStateQuery": False,
                "enableTravelDateQuery": False,
                "enableBoughtDateQuery": False,
            },
            name="[Order] Query My Orders",
            expect_data=False
        )
    
    @task(5)
    @tag('order', 'core', 'query')
    def query_other_orders(self):
        'English text removed.'
        self.post_with_check(
            "/api/v1/orderOtherService/orderOther/refresh",
            json={
                "loginId": self.user_id,
                "enableStateQuery": False,
                "enableTravelDateQuery": False,
                "enableBoughtDateQuery": False,
            },
            name="[Order] Query Other Orders",
            expect_data=False
        )
    
                                                                               
          
                                                                               
    
    @task(3)
    @tag('payment', 'core', 'query')
    def query_payments(self):
        'English text removed.'
        self.get_with_check(
            "/api/v1/inside_pay_service/inside_payment/payment",
            name="[Payment] Query All Payments",
            expect_data=False
        )
    
                                                                               
            
                                                                               
    
    @task(2)
    @tag('basic', 'query')
    def query_stations(self):
        'English text removed.'
        self.get_with_check(
            "/api/v1/stationservice/stations",
            name="[Station] Get All Stations",
            expect_data=True           
        )
    
    @task(2)
    @tag('basic', 'query')
    def query_trains(self):
        'English text removed.'
        self.get_with_check(
            "/api/v1/trainservice/trains",
            name="[Train] Get All Trains",
            expect_data=True
        )
    
    @task(2)
    @tag('basic', 'query')
    def query_routes(self):
        'English text removed.'
        self.get_with_check(
            "/api/v1/routeservice/routes",
            name="[Route] Get All Routes",
            expect_data=True
        )
    
    @task(2)
    @tag('basic', 'query')
    def query_prices(self):
        'English text removed.'
        self.get_with_check(
            "/api/v1/priceservice/prices",
            name="[Price] Get All Prices",
            expect_data=False
        )
    
    @task(1)
    @tag('basic', 'query')
    def query_configs(self):
        'English text removed.'
        self.get_with_check(
            "/api/v1/configservice/configs",
            name="[Config] Get All Configs",
            expect_data=False
        )
    
                                                                               
            
                                                                               
    
    @task(3)
    @tag('contacts', 'query')
    def query_contacts(self):
        'English text removed.'
        self.get_with_check(
            f"/api/v1/contactservice/contacts/account/{self.user_id}",
            name="[Contact] Get My Contacts",
            expect_data=False              
        )
    
    @task(2)
    @tag('assurance', 'query')
    def query_assurance_types(self):
        'English text removed.'
        self.get_with_check(
            "/api/v1/assuranceservice/assurances/types",
            name="[Assurance] Get Types",
            expect_data=True
        )
    
                                                                               
          
                                                                               
    
    @task(2)
    @tag('food', 'query')
    def query_station_food(self):
        'English text removed.'
        self.get_with_check(
            "/api/v1/stationfoodservice/stationfoodstores",
            name="[Food] Get Station Food List",
            expect_data=False
        )
    
    @task(2)
    @tag('food', 'query')
    def query_train_food(self):
        'English text removed.'
        self.get_with_check(
            "/api/v1/trainfoodservice/trainfoods",
            name="[Food] Get Train Food List",
            expect_data=False
        )
    
                                                                               
          
                                                                               
    
    @task(2)
    @tag('consign', 'query')
    def query_consign_price(self):
        'English text removed.'
                            
        if os.getenv("TT_SKIP_CONSIGN_PRICE_CONFIG", "1") != "0":
            return
        self.get_with_check(
            "/api/v1/consignpriceservice/consignprice/config",
            name="[Consign] Get Price Config",
            expect_data=True
        )
    
    @task(2)
    @tag('consign', 'query')
    def query_my_consigns(self):
        'English text removed.'
                                                     
                        
        self.get_with_check(
            f"/api/v1/consignservice/consigns/account/{self.user_id}",
            name="[Consign] Get My Consigns",
            expect_data=False,
            allow_no_content=True             
        )


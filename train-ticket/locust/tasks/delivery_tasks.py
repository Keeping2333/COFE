                                                                               
              
                                                                               
       
                                        
 
      
                     
            
           
                                                                               

import logging
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from locust import task, tag
from tasks.base import BaseTaskSet
from utils.data_generator import get_future_date, random_uuid

logger = logging.getLogger(__name__)


class FoodDeliveryTasks(BaseTaskSet):
    'English text removed.'
    
              
    _cached_food_stores = []
    
    def on_start(self):
        'English text removed.'
        if not self.has_cached_orders:
            self.fetch_and_cache_orders(query_other=False)
        self._fetch_food_stores()
    
    def _fetch_food_stores(self):
        'English text removed.'
        success, data = self.get_with_check(
            "/api/v1/stationfoodservice/stationfoodstores",
            name="[FoodDelivery] Fetch Food Stores",
            expect_data=False
        )
        if success and data:
            self._cached_food_stores = data if isinstance(data, list) else []
    
    @task(1)
    @tag('fooddelivery', 'food', 'query')
    def food_delivery_welcome(self):
        'English text removed.'
        self.get(
            "/api/v1/fooddeliveryservice/welcome",
            name="[FoodDelivery] Welcome"
        )
    
    @task(3)
    @tag('fooddelivery', 'food', 'query')
    def get_all_orders(self):
        'English text removed.'
        self.get_with_check(
            "/api/v1/fooddeliveryservice/orders/all",
            name="[FoodDelivery] Get All Orders",
            expect_data=False
        )
    
    @task(1)
    @tag('fooddelivery', 'food', 'query')
    def get_order_by_store(self):
        'English text removed.'
        if not self._cached_food_stores:
            self._fetch_food_stores()
        
        if self._cached_food_stores:
            store = random.choice(self._cached_food_stores)
            store_id = store.get("id", random_uuid())
        else:
            store_id = random_uuid()
        
        self.get_with_check(
            f"/api/v1/fooddeliveryservice/orders/store/{store_id}",
            name="[FoodDelivery] Get Orders By Store",
            expect_data=False
        )
    
    @task(1)
    @tag('fooddelivery', 'food', 'write')
    def create_delivery_order(self):
        'English text removed.'
                    
        order = self.get_paid_order()
        if not order:
            logger.debug("No paid order available for food delivery, skipping")
            return
        
        order_id = order.get("id")
        trip_id = order.get("trainNumber", "D1345")
        
                 
        store_id = random_uuid()
        food_name = random.choice(["Rice", "Noodles", "Sandwich", "Burger"])
        
        if self._cached_food_stores:
            store = random.choice(self._cached_food_stores)
            store_id = store.get("id", store_id)
                        
            foods = store.get("foodList", [])
            if foods:
                food = random.choice(foods)
                food_name = food.get("foodName", food_name)
        
        payload = {
            "id": random_uuid(),                 
            "stationFoodStoreId": store_id,
            "foodName": food_name,
            "orderId": order_id,             
            "seatNo": f"{random.randint(1, 16)}{random.choice(['A', 'B', 'C', 'D', 'F'])}",
            "deliveryTime": get_future_date(1),
            "tripId": trip_id,          
            "price": random.randint(20, 100),
        }
        self.post_with_check(
            "/api/v1/fooddeliveryservice/orders",
            json=payload,
            name="[FoodDelivery] Create Order",
            expect_data=False
        )


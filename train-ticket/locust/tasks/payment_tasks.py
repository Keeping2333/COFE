                                                                               
              
                                                                               
       
                                          
 
      
                 
            
                                                                               

import logging
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from locust import task, tag
from tasks.base import BaseTaskSet
from utils.data_generator import random_uuid

logger = logging.getLogger(__name__)


class PaymentServiceTasks(BaseTaskSet):
    'English text removed.'
    
    def on_start(self):
        'English text removed.'
        if not self.has_cached_orders:
            self.fetch_and_cache_orders(query_other=False)
            self.fetch_and_cache_orders(query_other=True)
    
    @task(3)
    @tag('payment', 'query')
    def query_payments(self):
        'English text removed.'
        self.get_with_check(
            "/api/v1/paymentservice/payment",
            name="[PaymentSvc] Query Payments",
            expect_data=False
        )
    
    @task(1)
    @tag('payment', 'write')
    def add_money(self):
        'English text removed.'
                          
        order = self.get_unpaid_order()
        order_id = order.get("id") if order else random_uuid()
        
        payload = {
            "id": random_uuid(),
            "orderId": order_id,
            "userId": self.user_id,            
            "price": str(random.randint(100, 1000)),
            "type": "A",                 
        }
        self.post_with_check(
            "/api/v1/paymentservice/payment/money",
            json=payload,
            name="[PaymentSvc] Add Money",
            expect_data=False
        )
    
    @task(1)
    @tag('payment', 'write')
    def pay_order(self):
        'English text removed.'
                    
        order = self.get_unpaid_order()
        if not order:
            logger.debug("No unpaid order available for payment, skipping")
            return
        
        order_id = order.get("id")
        price = order.get("price", "100")
        
        payload = {
            "id": random_uuid(),
            "orderId": order_id,
            "userId": self.user_id,
            "price": str(price),
            "type": "P",           
        }
        
        success, _ = self.post_with_check(
            "/api/v1/paymentservice/payment",
            json=payload,
            name="[PaymentSvc] Pay Order",
            expect_data=False
        )
        
        if success:
            logger.info(f"External payment for order {order_id} submitted")


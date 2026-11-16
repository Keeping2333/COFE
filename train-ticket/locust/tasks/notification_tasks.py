                                                                               
              
                                                                               
       
                                      
 
      
            
                  
                                                                               

import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from locust import task, tag
from tasks.base import BaseTaskSet
from utils.data_generator import get_future_date, random_email, random_uuid

logger = logging.getLogger(__name__)


class NotificationTasks(BaseTaskSet):
    'English text removed.'
    
    def on_start(self):
        'English text removed.'
        if not self.has_cached_orders:
            self.fetch_and_cache_orders(query_other=False)
    
    def _get_order_info(self):
        'English text removed.'
        order = self.get_random_order(status_filter=(0, 1, 2))
        if order:
            return {
                "orderNumber": order.get("id"),
                "startPlace": order.get("from", "Shang Hai"),
                "endPlace": order.get("to", "Su Zhou"),
                "startTime": "08:00",
                "seatClass": str(order.get("seatClass", 2)),
                "seatNumber": order.get("seatNumber", "5A"),
                "price": str(order.get("price", 100)),
            }
        else:
                        
            return {
                "orderNumber": random_uuid(),
                "startPlace": "Shang Hai",
                "endPlace": "Su Zhou",
                "startTime": "08:00",
                "seatClass": "2",
                "seatNumber": "5A",
                "price": "100",
            }
    
    @task(1)
    @tag('notification', 'query')
    def notification_welcome(self):
        'English text removed.'
        self.get(
            "/api/v1/notifyservice/welcome",
            name="[Notify] Welcome"
        )
    
    @task(1)
    @tag('notification', 'write')
    def notify_preserve_success(self):
        'English text removed.'
        order_info = self._get_order_info()
        payload = {
            "email": random_email(),
            "username": "test_user",
            "date": get_future_date(1),
            **order_info,
        }
        self.post_with_check(
            "/api/v1/notifyservice/notification/preserve_success",
            json=payload,
            name="[Notify] Preserve Success",
            expect_data=False
        )
    
    @task(1)
    @tag('notification', 'write')
    def notify_order_create_success(self):
        'English text removed.'
        order_info = self._get_order_info()
        payload = {
            "email": random_email(),
            "username": "test_user",
            "date": get_future_date(1),
            **order_info,
        }
        self.post_with_check(
            "/api/v1/notifyservice/notification/order_create_success",
            json=payload,
            name="[Notify] Order Create Success",
            expect_data=False
        )
    
    @task(1)
    @tag('notification', 'write')
    def notify_order_changed_success(self):
        'English text removed.'
        order_info = self._get_order_info()
        payload = {
            "email": random_email(),
            "username": "test_user",
            "date": get_future_date(1),
            **order_info,
        }
        self.post_with_check(
            "/api/v1/notifyservice/notification/order_changed_success",
            json=payload,
            name="[Notify] Order Changed Success",
            expect_data=False
        )
    
    @task(1)
    @tag('notification', 'write')
    def notify_order_cancel_success(self):
        'English text removed.'
        order_info = self._get_order_info()
        payload = {
            "email": random_email(),
            "username": "test_user",
            "date": get_future_date(1),
            **order_info,
        }
        self.post_with_check(
            "/api/v1/notifyservice/notification/order_cancel_success",
            json=payload,
            name="[Notify] Order Cancel Success",
            expect_data=False
        )

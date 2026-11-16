                                                                               
                       
                                                                               
       
                             
                                   
                                 
 
      
                       
                       
            
                                                                               

import logging
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from locust import task, tag
from tasks.base import BaseTaskSet
from utils.data_generator import DataGenerator, get_future_date

logger = logging.getLogger(__name__)


class RebookTasks(BaseTaskSet):
    'English text removed.'
    
    def on_start(self):
        'English text removed.'
        self._refresh_orders_if_needed()
    
    def _refresh_orders_if_needed(self):
        'English text removed.'
        if not self.has_cached_orders:
            self.fetch_and_cache_orders(query_other=False)
            self.fetch_and_cache_orders(query_other=True)
    
    @task(1)
    @tag('rebook', 'order', 'query')
    def rebook_welcome(self):
        'English text removed.'
        self.get(
            "/api/v1/rebookservice/welcome",
            name="[Rebook] Welcome"
        )
    
    @task(2)
    @tag('rebook', 'order', 'write')
    def rebook_ticket(self):
        'English text removed.'
                 
        self._refresh_orders_if_needed()
        
                                  
        order = self.get_paid_order()
        if not order:
            logger.debug("No paid order available for rebook, skipping")
            return
        
        order_id = order.get("id")
        old_trip_id = order.get("trainNumber")
        
                  
        success, trips = self._query_available_trips()
        if not success or not trips:
            logger.debug("No trips available for rebook")
            return
        
                   
        new_trip = None
        for trip in trips:
            trip_id_obj = trip.get("tripId", {})
            if isinstance(trip_id_obj, dict):
                new_trip_id = f"{trip_id_obj.get('type', 'D')}{trip_id_obj.get('number', '1345')}"
            else:
                new_trip_id = str(trip_id_obj)
            
            if new_trip_id != old_trip_id:
                new_trip = new_trip_id
                break
        
        if not new_trip:
            new_trip = old_trip_id                           
        
        payload = {
            "orderId": order_id,
            "oldTripId": old_trip_id,
            "tripId": new_trip,
            "date": get_future_date(2),
            "seatType": str(random.choice([2, 3])),
        }
        
        self.post_with_check(
            "/api/v1/rebookservice/rebook",
            json=payload,
            name="[Rebook] Rebook Ticket",
            expect_data=False
        )
    
    def _query_available_trips(self):
        'English text removed.'
        place_pair = DataGenerator.get_place_pair(is_high_speed=True)
        payload = {
            "startPlace": place_pair[0],
            "endPlace": place_pair[1],
            "departureTime": get_future_date(2)
        }
        return self.post_with_check(
            "/api/v1/travelservice/trips/left",
            json=payload,
            name="[Rebook] Query Available Trips",
            expect_data=False
        )
    
    @task(1)
    @tag('rebook', 'order', 'write')
    def rebook_pay_difference(self):
        'English text removed.'
        self._refresh_orders_if_needed()
        
        order = self.get_paid_order()
        if not order:
            logger.debug("No paid order available for pay difference, skipping")
            return
        
        payload = {
            "orderId": order.get("id"),
            "oldTripId": order.get("trainNumber"),
            "tripId": "D1355",        
            "date": get_future_date(2),
            "seatType": str(random.choice([2, 3])),
        }
        self.post_with_check(
            "/api/v1/rebookservice/rebook/difference",
            json=payload,
            name="[Rebook] Pay Difference",
            expect_data=False
        )


class WaitOrderTasks(BaseTaskSet):
    'English text removed.'
    
    def on_start(self):
        'English text removed.'
        if not self.has_cached_contacts:
            self.fetch_and_cache_contacts()
    
    @task(1)
    @tag('waitorder', 'order', 'query')
    def wait_order_welcome(self):
        'English text removed.'
        self.get(
            "/api/v1/waitorderservice/welcome",
            name="[WaitOrder] Welcome"
        )
    
    @task(3)
    @tag('waitorder', 'order', 'query')
    def get_all_orders(self):
        'English text removed.'
        self.get_with_check(
            "/api/v1/waitorderservice/orders",
            name="[WaitOrder] Get All Orders",
            expect_data=False
        )
    
    @task(3)
    @tag('waitorder', 'order', 'query')
    def get_waitlist_orders(self):
        'English text removed.'
        self.get_with_check(
            "/api/v1/waitorderservice/waitlistorders",
            name="[WaitOrder] Get WaitList Orders",
            expect_data=False
        )
    
    @task(1)
    @tag('waitorder', 'order', 'write')
    def create_wait_order(self):
        'English text removed.'
               
        contact = self.get_random_contact()
        if not contact:
                        
            self.fetch_and_cache_contacts()
            contact = self.get_random_contact()
        
        contact_id = contact.get("id") if contact else None
        if not contact_id:
            logger.debug("No contact available for wait order, skipping")
            return
        
        place_pair = DataGenerator.get_place_pair()
        payload = {
            "accountId": self.user_id,
            "contactsId": contact_id,           
            "tripId": "D1345",
            "seatType": random.choice([2, 3]),
            "date": get_future_date(1),
            "from": place_pair[0],
            "to": place_pair[1],
            "price": str(random.randint(100, 500)),
        }
        self.post_with_check(
            "/api/v1/waitorderservice/order",
            json=payload,
            name="[WaitOrder] Create Order",
            expect_data=False
        )


class ExecuteTasks(BaseTaskSet):
    'English text removed.'
    
    def on_start(self):
        'English text removed.'
        if not self.has_cached_orders:
            self.fetch_and_cache_orders(query_other=False)
            self.fetch_and_cache_orders(query_other=True)
    
    @task(1)
    @tag('execute', 'order', 'query')
    def execute_welcome(self):
        'English text removed.'
        self.get(
            "/api/v1/executeservice/welcome",
            name="[Execute] Welcome"
        )
    
    @task(2)
    @tag('execute', 'order', 'write')
    def execute_ticket(self):
        'English text removed.'
                                   
        order = self.get_paid_order()
        if not order:
            logger.debug("No paid order available for execute, skipping")
            return
        
        order_id = order.get("id")
        
                    
        with self.client.get(
            f"/api/v1/executeservice/execute/execute/{order_id}",
            headers=self.get_headers(),
            name="[Execute] Execute Ticket",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if result.get("status") == 1:
                        response.success()
                        logger.info(f"Order {order_id} executed successfully")
                    else:
                                        
                        response.failure(f"Business error: {result.get('msg', 'Unknown')}")
                except Exception as e:
                    response.failure(f"JSON parse error: {e}")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(2)
    @tag('execute', 'order', 'write')
    def collect_ticket(self):
        'English text removed.'
                 
        order = self.get_paid_order()
        if not order:
            logger.debug("No paid order available for collect, skipping")
            return
        
        order_id = order.get("id")
        
        with self.client.get(
            f"/api/v1/executeservice/execute/collected/{order_id}",
            headers=self.get_headers(),
            name="[Execute] Collect Ticket",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if result.get("status") == 1:
                        response.success()
                        logger.info(f"Order {order_id} collected successfully")
                    else:
                        response.failure(f"Business error: {result.get('msg', 'Unknown')}")
                except Exception as e:
                    response.failure(f"JSON parse error: {e}")
            else:
                response.failure(f"HTTP {response.status_code}")


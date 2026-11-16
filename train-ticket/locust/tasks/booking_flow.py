                                                                               
          
                                                                               
                    
                                         
                                                                               

import logging
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from locust import task, tag
from tasks.base import BaseTaskSet
from utils.data_generator import DataGenerator, get_future_date, random_uuid

logger = logging.getLogger(__name__)


class BookingFlowTasks(BaseTaskSet):
    'English text removed.'
    
    def on_start(self):
        'English text removed.'
                          
        self.contact_id = None
        self.contact_name = None
        self.contact_document_number = None
    
                                                                               
          
                                                                               
    
    def _get_or_create_contact(self) -> tuple:
        'English text removed.'
        if self.contact_id:
            return self.contact_id, self.contact_name, self.contact_document_number
        
                  
        response = self.get(
            f"/api/v1/contactservice/contacts/account/{self.user_id}",
            name="[Flow] Get Contacts"
        )
        
        if response.status_code == 200:
            result = response.json()
            contacts = result.get("data", [])
            if contacts and len(contacts) > 0:
                contact = contacts[0]
                self.contact_id = contact.get("id")
                self.contact_name = contact.get("name")
                self.contact_document_number = contact.get("documentNumber")
                logger.info(f"Using existing contact: {self.contact_id}")
                return self.contact_id, self.contact_name, self.contact_document_number
        
                    
        contact_data = DataGenerator.generate_contact(self.user_id)
        response = self.post(
            "/api/v1/contactservice/contacts",
            json=contact_data,
            name="[Flow] Create Contact"
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("status") == 1:
                contact = result.get("data", {})
                self.contact_id = contact.get("id")
                self.contact_name = contact_data.get("name")
                self.contact_document_number = contact_data.get("documentNumber")
                logger.info(f"Created new contact: {self.contact_id}")
                return self.contact_id, self.contact_name, self.contact_document_number
        
        logger.warning("Failed to get/create contact")
        return None, None, None
    
    def _query_available_trips(self, is_high_speed: bool = True) -> list:
        'English text removed.'
        place_pair = DataGenerator.get_place_pair(is_high_speed)
        service = "travelservice" if is_high_speed else "travel2service"
        
        payload = {
            "startPlace": place_pair[0],
            "endPlace": place_pair[1],
            "departureTime": get_future_date(1)
        }
        
        response = self.post(
            f"/api/v1/{service}/trips/left",
            json=payload,
            name="[Flow] Query Trips"
        )
        
        if response.status_code == 200:
            result = response.json()
            trips = result.get("data", [])
                             
            return [(trip, place_pair, is_high_speed) for trip in trips if trip]
        
        return []
    
    def _preserve_ticket(self, trip_info: dict, place_pair: tuple, 
                         contact_id: str, is_high_speed: bool) -> str:
        'English text removed.'
        trip_id = trip_info.get("tripId", {})
        if isinstance(trip_id, dict):
            trip_id_str = f"{trip_id.get('type', 'D')}{trip_id.get('number', '1345')}"
        else:
            trip_id_str = str(trip_id)
        
        service = "preserveservice/preserve" if is_high_speed else "preserveotherservice/preserveOther"
        
        preserve_data = {
            "accountId": self.user_id,
            "contactsId": contact_id,
            "tripId": trip_id_str,
            "seatType": str(DataGenerator.get_seat_type(is_high_speed)),
            "date": get_future_date(1),
            "from": place_pair[0],
            "to": place_pair[1],
            "assurance": "0",
            "foodType": 0,
            "stationName": "",
            "storeName": "",
            "foodName": "",
            "foodPrice": "0",
            "handleDate": "",
            "consigneeName": "",
            "consigneePhone": "",
            "consigneeWeight": "0",
            "isWithin": False,
        }
        
        response = self.post(
            f"/api/v1/{service}",
            json=preserve_data,
            name="[Flow] Preserve Ticket"
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("status") == 1:
                                     
                data = result.get("data")
                                                     
                if isinstance(data, dict):
                    order_id = data.get("orderId")
                    if order_id:
                        logger.info(f"Ticket preserved, orderId: {order_id}")
                        return order_id
                                         
                logger.info("Preserve successful, need to query order")
                return "NEED_QUERY"
        
        logger.warning(f"Preserve failed: {response.text[:200]}")
        return None
    
    def _pay_order(self, order_id: str, trip_id: str) -> bool:
        'English text removed.'
        payment_data = {
            "orderId": order_id,
            "userId": self.user_id,
            "tripId": trip_id,
            "price": "100",
        }
        
        response = self.post(
            "/api/v1/inside_pay_service/inside_payment",
            json=payment_data,
            name="[Flow] Pay Order"
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("status") == 1:
                logger.info(f"Order paid: {order_id}")
                return True
        
        logger.warning(f"Payment failed: {response.text[:200]}")
        return False
    
    def _cancel_order(self, order_id: str) -> bool:
        'English text removed.'
        response = self.get(
            f"/api/v1/cancelservice/cancel/{order_id}/{self.user_id}",
            name="[Flow] Cancel Order"
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("status") == 1:
                logger.info(f"Order cancelled: {order_id}")
                return True
        
        return False
    
    def _get_latest_order(self, is_high_speed: bool = True) -> dict:
        'English text removed.'
        service = "orderservice/order" if is_high_speed else "orderOtherService/orderOther"
        
        response = self.post(
            f"/api/v1/{service}/refresh",
            json={"loginId": self.user_id},
            name="[Flow] Get Latest Order"
        )
        
        if response.status_code == 200:
            result = response.json()
            orders = result.get("data", [])
            if orders:
                            
                for order in orders:
                    if order.get("status") == 0:            
                        return order
                                
                return orders[0]
        
        return None
    
                                                                               
            
                                                                               
    
    @task(5)
    @tag('flow', 'booking', 'core')
    def complete_booking_flow(self):
        'English text removed.'
                     
        is_high_speed = random.choice([True, False])
        
                      
        trips = self._query_available_trips(is_high_speed)
        if not trips:
            logger.warning("No trips available")
            return
        
                   
        trip_info, place_pair, _ = trips[0]
        
                          
        contact_id, contact_name, document_number = self._get_or_create_contact()
        if not contact_id:
            logger.warning("No contact available")
            return
        
                      
        order_id = self._preserve_ticket(trip_info, place_pair, contact_id, is_high_speed)
        if not order_id:
            logger.warning("Preserve failed")
            return
        
                              
        if order_id == "NEED_QUERY":
            order = self._get_latest_order(is_high_speed)
            if order:
                order_id = order.get("id")
                logger.info(f"Found order: {order_id}")
        
        logger.info(f"Booking flow completed: order={order_id}")
    
    @task(3)
    @tag('flow', 'booking', 'payment', 'core')
    def complete_booking_with_payment(self):
        'English text removed.'
        is_high_speed = random.choice([True, True, False])          
        
                      
        trips = self._query_available_trips(is_high_speed)
        if not trips:
            return
        
        trip_info, place_pair, _ = trips[0]
        trip_id = trip_info.get("tripId", {})
        if isinstance(trip_id, dict):
            trip_id_str = f"{trip_id.get('type', 'D')}{trip_id.get('number', '1345')}"
        else:
            trip_id_str = str(trip_id)
        
                       
        contact_id, _, _ = self._get_or_create_contact()
        if not contact_id:
            return
        
                    
        order_id = self._preserve_ticket(trip_info, place_pair, contact_id, is_high_speed)
        if not order_id:
            return
        
                  
        if order_id == "NEED_QUERY":
            order = self._get_latest_order(is_high_speed)
            if order:
                order_id = order.get("id")
            else:
                return
        
                    
        self._pay_order(order_id, trip_id_str)
        
        logger.info(f"Booking with payment completed: order={order_id}")
    
    @task(1)
    @tag('flow', 'cancel')
    def booking_and_cancel_flow(self):
        'English text removed.'
        is_high_speed = True
        
                     
        trips = self._query_available_trips(is_high_speed)
        if not trips:
            return
        
        trip_info, place_pair, _ = trips[0]
        contact_id, _, _ = self._get_or_create_contact()
        if not contact_id:
            return
        
        order_id = self._preserve_ticket(trip_info, place_pair, contact_id, is_high_speed)
        if not order_id or order_id == "NEED_QUERY":
            order = self._get_latest_order(is_high_speed)
            if order:
                order_id = order.get("id")
            else:
                return
        
              
        self._cancel_order(order_id)
        logger.info(f"Booking and cancel flow completed: order={order_id}")
    
    @task(2)
    @tag('flow', 'query')
    def query_and_verify_orders(self):
        'English text removed.'
                
        response = self.post(
            "/api/v1/orderservice/order/refresh",
            json={"loginId": self.user_id},
            name="[Flow] Verify High-Speed Orders"
        )
        
        if response.status_code == 200:
            result = response.json()
            orders = result.get("data", [])
            logger.info(f"Found {len(orders)} high-speed orders")
        
                  
        response = self.post(
            "/api/v1/orderOtherService/orderOther/refresh",
            json={"loginId": self.user_id},
            name="[Flow] Verify Normal Orders"
        )
        
        if response.status_code == 200:
            result = response.json()
            orders = result.get("data", [])
            logger.info(f"Found {len(orders)} normal orders")
    
                                                                               
          
                                                                               
    
    def _create_consign(self, order_info: dict) -> str:
        'English text removed.'
        consign_data = {
            "accountId": self.user_id,
            "handleDate": get_future_date(0),      
            "targetDate": order_info.get("travelDate", get_future_date(1)),
            "from": order_info.get("from", "Shang Hai"),
            "to": order_info.get("to", "Su Zhou"),
            "orderId": order_info.get("id"),
            "consignee": f"Test_{random_uuid()[:8]}",
            "phone": "13800138000",
            "weight": str(random.randint(5, 20)),
            "id": "",
            "isWithin": False
        }
        
        response = self.put(
            "/api/v1/consignservice/consigns",
            json=consign_data,
            name="[Flow] Create Consign"
        )
        
        if response.status_code in [200, 201]:
            result = response.json()
            if result.get("status") == 1:
                logger.info(f"Consign created for order {order_info.get('id')}")
                return result.get("data", {}).get("id") if isinstance(result.get("data"), dict) else "SUCCESS"
        
        logger.warning(f"Create consign failed: {response.text[:200]}")
        return None
    
    def _query_my_consigns(self) -> list:
        'English text removed.'
        response = self.get(
            f"/api/v1/consignservice/consigns/account/{self.user_id}",
            name="[Flow] Query My Consigns"
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("status") == 1:
                consigns = result.get("data", [])
                logger.info(f"Found {len(consigns)} consign records")
                return consigns
        
        return []
    
    @task(2)
    @tag('flow', 'consign', 'write')
    def booking_with_consign_flow(self):
        'English text removed.'
        is_high_speed = True
        
                        
        trips = self._query_available_trips(is_high_speed)
        if not trips:
            return
        
        trip_info, place_pair, _ = trips[0]
        contact_id, _, _ = self._get_or_create_contact()
        if not contact_id:
            return
        
        order_id = self._preserve_ticket(trip_info, place_pair, contact_id, is_high_speed)
        if not order_id or order_id == "NEED_QUERY":
            order = self._get_latest_order(is_high_speed)
            if not order:
                return
            order_id = order.get("id")
            order_info = order
        else:
                           
            order_info = {
                "id": order_id,
                "from": place_pair[0],
                "to": place_pair[1],
                "travelDate": get_future_date(1)
            }
        
                      
        consign_id = self._create_consign(order_info)
        if not consign_id:
            logger.warning("Failed to create consign, but booking succeeded")
            return
        
                          
        consigns = self._query_my_consigns()
        if consigns:
            logger.info(f"Booking with consign flow completed: order={order_id}, consigns={len(consigns)}")
        else:
            logger.info(f"Booking with consign flow completed: order={order_id}, consign created")


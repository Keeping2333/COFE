                                                                               
            
                                                                               
               
  
       
                  
                                         
                  
                                                                               

import logging
import random
from typing import Optional, List, Dict, Tuple
from locust import TaskSet

logger = logging.getLogger(__name__)


class BaseTaskSet(TaskSet):
    'English text removed.'
    
                              
    _cached_orders: List[Dict] = []
    _cached_other_orders: List[Dict] = []
    _cached_contacts: List[Dict] = []
    
    def get_headers(self) -> dict:
        'English text removed.'
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
        }
        if hasattr(self.user, 'token') and self.user.token:
            headers['Authorization'] = f'Bearer {self.user.token}'
        return headers
    
                                                                               
                          
                                                                               
    
    def get(self, url: str, name: str = None, **kwargs):
        'English text removed.'
        return self.client.get(url, headers=self.get_headers(), name=name, **kwargs)
    
    def post(self, url: str, json=None, name: str = None, **kwargs):
        'English text removed.'
        return self.client.post(url, json=json, headers=self.get_headers(), name=name, **kwargs)
    
    def put(self, url: str, json=None, name: str = None, **kwargs):
        'English text removed.'
        return self.client.put(url, json=json, headers=self.get_headers(), name=name, **kwargs)
    
    def delete(self, url: str, name: str = None, **kwargs):
        'English text removed.'
        return self.client.delete(url, headers=self.get_headers(), name=name, **kwargs)
    
                                                                               
                      
                                                                               
    
    def get_with_check(self, url: str, name: str = None, 
                       expect_data: bool = False,
                       allow_no_content: bool = False, **kwargs) -> Tuple[bool, Optional[dict]]:
        'English text removed.'
        with self.client.get(url, headers=self.get_headers(), name=name, 
                             catch_response=True, **kwargs) as response:
            return self._check_business_response(response, expect_data, allow_no_content)
    
    def post_with_check(self, url: str, json=None, name: str = None,
                        expect_data: bool = False, **kwargs) -> Tuple[bool, Optional[dict]]:
        'English text removed.'
        with self.client.post(url, json=json, headers=self.get_headers(), 
                              name=name, catch_response=True, **kwargs) as response:
            return self._check_business_response(response, expect_data)
    
    def put_with_check(self, url: str, json=None, name: str = None,
                       expect_data: bool = False, **kwargs) -> Tuple[bool, Optional[dict]]:
        'English text removed.'
        with self.client.put(url, json=json, headers=self.get_headers(), 
                             name=name, catch_response=True, **kwargs) as response:
            return self._check_business_response(response, expect_data)
    
    def delete_with_check(self, url: str, name: str = None, **kwargs) -> Tuple[bool, Optional[dict]]:
        'English text removed.'
        with self.client.delete(url, headers=self.get_headers(), 
                                name=name, catch_response=True, **kwargs) as response:
            return self._check_business_response(response, expect_data=False)
    
    def _check_business_response(self, response, expect_data: bool = False,
                                    allow_no_content: bool = False) -> Tuple[bool, Optional[dict]]:
        'English text removed.'
        if response.status_code != 200:
            response.failure(f"HTTP {response.status_code}")
            return False, None
        
        try:
            result = response.json()
        except Exception as e:
            response.failure(f"Invalid JSON: {e}")
            return False, None
        
                
        status = result.get("status")
        msg = result.get("msg", "")
        
        if status != 1:
                                           
            if allow_no_content and ("No Content" in msg or "not found" in msg.lower()):
                response.success()
                return True, None
            
            response.failure(f"Business error: {msg or 'Unknown'}")
            return False, None
        
                  
        data = result.get("data")
        if expect_data and data is None:
            response.failure("Expected data but got None")
            return False, None
        
        response.success()
        return True, data
    
                                                                               
                           
                                                                               
    
    def fetch_and_cache_orders(self, query_other: bool = False) -> List[Dict]:
        'English text removed.'
        service = "orderOtherService/orderOther" if query_other else "orderservice/order"
        
        success, data = self.post_with_check(
            f"/api/v1/{service}/refresh",
            json={
                "loginId": self.user_id,
                "enableStateQuery": False,
                "enableTravelDateQuery": False,
                "enableBoughtDateQuery": False,
            },
            name="[Cache] Fetch Orders",
            expect_data=False               
        )
        
        if success and data:
            orders = data if isinstance(data, list) else []
            if query_other:
                self._cached_other_orders = orders
            else:
                self._cached_orders = orders
            return orders
        
        return []
    
    def get_random_order(self, status_filter: tuple = (0, 1), 
                         prefer_other: bool = False) -> Optional[Dict]:
        'English text removed.'
                     
        all_orders = []
        
        if prefer_other:
            all_orders = self._cached_other_orders + self._cached_orders
        else:
            all_orders = self._cached_orders + self._cached_other_orders
        
               
        filtered = [o for o in all_orders if o.get("status") in status_filter]
        
        if not filtered:
            return None
        
        return random.choice(filtered)
    
    def get_paid_order(self) -> Optional[Dict]:
        'English text removed.'
        return self.get_random_order(status_filter=(1,))
    
    def get_unpaid_order(self) -> Optional[Dict]:
        'English text removed.'
        return self.get_random_order(status_filter=(0,))
    
                                                                               
              
                                                                               
    
    def fetch_and_cache_contacts(self) -> List[Dict]:
        'English text removed.'
        success, data = self.get_with_check(
            f"/api/v1/contactservice/contacts/account/{self.user_id}",
            name="[Cache] Fetch Contacts",
            expect_data=False
        )
        
        if success and data:
            self._cached_contacts = data if isinstance(data, list) else []
            return self._cached_contacts
        
        return []
    
    def get_random_contact(self) -> Optional[Dict]:
        'English text removed.'
        if not self._cached_contacts:
            return None
        return random.choice(self._cached_contacts)
    
                                                                               
        
                                                                               
    
    @property
    def user_id(self) -> str:
        'English text removed.'
        return getattr(self.user, 'user_id', '') or ''
    
    @property
    def token(self) -> str:
        'English text removed.'
        return getattr(self.user, 'token', '') or ''
    
    @property
    def has_cached_orders(self) -> bool:
        'English text removed.'
        return len(self._cached_orders) > 0 or len(self._cached_other_orders) > 0
    
    @property
    def has_cached_contacts(self) -> bool:
        'English text removed.'
        return len(self._cached_contacts) > 0


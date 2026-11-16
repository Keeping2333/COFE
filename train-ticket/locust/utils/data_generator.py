                                                                               
         
                                                                               

import random
import string
import uuid
from datetime import datetime, timedelta
from typing import List, Any, Optional


def random_choice(items: List[Any]) -> Any:
    'English text removed.'
    if not items:
        return None
    return random.choice(items)


def random_string(length: int = 8) -> str:
    'English text removed.'
    return ''.join(random.choices(string.ascii_letters, k=length))


def random_phone() -> str:
    'English text removed.'
    return ''.join(random.choices(string.digits, k=11))


def random_email() -> str:
    'English text removed.'
    return f"{random_string(8)}@test.com"


def random_uuid() -> str:
    'English text removed.'
    return str(uuid.uuid4())


def random_id_card() -> str:
    'English text removed.'
    return ''.join(random.choices(string.digits, k=18))


def get_future_date(days_ahead: int = 1) -> str:
    'English text removed.'
    return (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")


class DataGenerator:
    'English text removed.'
    
                              
    HIGH_SPEED_PAIRS = [
        ("Shang Hai", "Su Zhou"),
        ("Su Zhou", "Shang Hai"),
        ("Nan Jing", "Shang Hai"),
        ("Shang Hai", "Nan Jing"),
    ]
    
             
    NORMAL_PAIRS = [
        ("Shang Hai", "Nan Jing"),
        ("Nan Jing", "Shang Hai"),
    ]
    
            
    HIGH_SPEED_SEAT_TYPES = [2, 3]                
    
              
    NORMAL_SEAT_TYPES = [2, 3, 4, 5, 6, 7]
    
                           
    HIGH_SPEED_TRIPS = ["D1345", "G1234", "G1236", "D1355"]
    NORMAL_TRIPS = ["K1345", "Z1234", "T1235"]
    
    @classmethod
    def get_place_pair(cls, is_high_speed: bool = True) -> tuple:
        'English text removed.'
        pairs = cls.HIGH_SPEED_PAIRS if is_high_speed else cls.NORMAL_PAIRS
        return random_choice(pairs)
    
    @classmethod
    def get_seat_type(cls, is_high_speed: bool = True) -> int:
        'English text removed.'
        types = cls.HIGH_SPEED_SEAT_TYPES if is_high_speed else cls.NORMAL_SEAT_TYPES
        return random_choice(types)
    
    @classmethod
    def get_trip_id(cls, is_high_speed: bool = True) -> str:
        'English text removed.'
        trips = cls.HIGH_SPEED_TRIPS if is_high_speed else cls.NORMAL_TRIPS
        return random_choice(trips)
    
    @classmethod
    def generate_contact(cls, account_id: str) -> dict:
        'English text removed.'
        return {
            "name": f"Test_{random_string(4)}",
            "accountId": account_id,
            "documentType": 1,         
            "documentNumber": random_id_card(),
            "phoneNumber": random_phone(),
        }
    
    @classmethod
    def generate_preserve_order(cls, account_id: str, contact_id: str,
                                 trip_id: str = None, is_high_speed: bool = True) -> dict:
        'English text removed.'
        place_pair = cls.get_place_pair(is_high_speed)
        if not trip_id:
            trip_id = cls.get_trip_id(is_high_speed)
        
        return {
            "accountId": account_id,
            "contactsId": contact_id,
            "tripId": trip_id,
            "seatType": str(cls.get_seat_type(is_high_speed)),
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
    
    @classmethod
    def generate_payment(cls, order_id: str, user_id: str, trip_id: str) -> dict:
        'English text removed.'
        return {
            "orderId": order_id,
            "userId": user_id,
            "tripId": trip_id,
            "price": "100",             
        }
    
    @classmethod
    def generate_consign(cls, account_id: str, order_id: str,
                         from_station: str, to_station: str) -> dict:
        'English text removed.'
        return {
            "accountId": account_id,
            "orderId": order_id,
            "handleDate": get_future_date(0),
            "targetDate": get_future_date(1),
            "from": from_station,
            "to": to_station,
            "consignee": random_string(6),
            "phone": random_phone(),
            "weight": str(random.randint(1, 30)),
            "isWithin": random.choice([True, False]),
        }


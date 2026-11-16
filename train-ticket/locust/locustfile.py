                                                                               
                           
                                                                               
                                    
 
       
                                                                                         
                                                                               

import logging
import os
import random
from datetime import datetime, timedelta
from locust import HttpUser, TaskSet, task, between, events, tag

      
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


                                                                               
      
                                                                               
HIGH_SPEED_PAIRS = [
    ("Shang Hai", "Su Zhou"),
    ("Su Zhou", "Shang Hai"),
    ("Nan Jing", "Shang Hai"),
]

NORMAL_PAIRS = [
    ("Shang Hai", "Nan Jing"),
    ("Nan Jing", "Shang Hai"),
]


def get_test_date(days_ahead=1):
    'English text removed.'
    return (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")


                                                                               
                            
                                                                               
class BaseTaskSet(TaskSet):
    'English text removed.'
    
    def get_headers(self):
        'English text removed.'
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
        }
        if hasattr(self.user, 'token') and self.user.token:
            headers['Authorization'] = f'Bearer {self.user.token}'
        return headers
    
    def get(self, url, name=None, **kwargs):
        'English text removed.'
        return self.client.get(url, headers=self.get_headers(), name=name, **kwargs)
    
    def post(self, url, json=None, name=None, **kwargs):
        'English text removed.'
        return self.client.post(url, json=json, headers=self.get_headers(), name=name, **kwargs)
    
    def put(self, url, json=None, name=None, **kwargs):
        'English text removed.'
        return self.client.put(url, json=json, headers=self.get_headers(), name=name, **kwargs)
    
    def delete(self, url, name=None, **kwargs):
        'English text removed.'
        return self.client.delete(url, headers=self.get_headers(), name=name, **kwargs)


                                                                               
                 
                                                                               
class CoreBusinessTasks(BaseTaskSet):
    'English text removed.'
    
    @task(10)
    @tag('travel', 'core')
    def query_trips(self):
        'English text removed.'
        place_pair = random.choice(HIGH_SPEED_PAIRS)
        payload = {
            "startPlace": place_pair[0],
            "endPlace": place_pair[1],
            "departureTime": get_test_date()
        }
        self.post(
            "/api/v1/travelservice/trips/left",
            json=payload,
            name="[Travel] Query High-Speed Trips"
        )
    
    @task(8)
    @tag('travel', 'core')
    def query_normal_trips(self):
        'English text removed.'
        place_pair = random.choice(NORMAL_PAIRS)
        payload = {
            "startPlace": place_pair[0],
            "endPlace": place_pair[1],
            "departureTime": get_test_date()
        }
        self.post(
            "/api/v1/travel2service/trips/left",
            json=payload,
            name="[Travel] Query Normal Trips"
        )
    
    @task(5)
    @tag('order', 'core')
    def query_my_orders(self):
        'English text removed.'
        self.post(
            "/api/v1/orderservice/order/refresh",
            json={
                "loginId": getattr(self.user, 'user_id', ''),
                "enableStateQuery": False,
                "enableTravelDateQuery": False,
                "enableBoughtDateQuery": False,
                "travelDateStart": None,
                "travelDateEnd": None,
                "boughtDateStart": None,
                "boughtDateEnd": None
            },
            name="[Order] Query My Orders"
        )
    
    @task(5)
    @tag('order', 'core')
    def query_other_orders(self):
        'English text removed.'
        self.post(
            "/api/v1/orderOtherService/orderOther/refresh",
            json={
                "loginId": getattr(self.user, 'user_id', ''),
                "enableStateQuery": False,
                "enableTravelDateQuery": False,
                "enableBoughtDateQuery": False,
            },
            name="[Order] Query Other Orders"
        )
    
    @task(2)
    @tag('payment', 'core')
    def query_payment(self):
        'English text removed.'
        self.get(
                                      
                                                        
                                                      
            "/api/v1/inside_pay_service/inside_payment/payment",
            name="[Payment] Query All Payments"
        )
    
    @task(1)
    @tag('basic')
    def query_stations(self):
        'English text removed.'
        self.get(
            "/api/v1/stationservice/stations",
            name="[Station] Get All Stations"
        )
    
    @task(1)
    @tag('basic')
    def query_trains(self):
        'English text removed.'
        self.get(
            "/api/v1/trainservice/trains",
            name="[Train] Get All Trains"
        )
    
    @task(1)
    @tag('basic')
    def query_routes(self):
        'English text removed.'
        self.get(
            "/api/v1/routeservice/routes",
            name="[Route] Get All Routes"
        )
    
    @task(1)
    @tag('basic')
    def query_configs(self):
        'English text removed.'
        self.get(
            "/api/v1/configservice/configs",
            name="[Config] Get All Configs"
        )


                                                                               
               
                                                                               
class SupportServiceTasks(BaseTaskSet):
    'English text removed.'
    
    @task(3)
    @tag('contacts')
    def query_contacts(self):
        'English text removed.'
        account_id = getattr(self.user, 'user_id', '')
        self.get(
            f"/api/v1/contactservice/contacts/account/{account_id}",
            name="[Contact] Get My Contacts"
        )
    
    @task(2)
    @tag('assurance')
    def query_assurances(self):
        'English text removed.'
        self.get(
            "/api/v1/assuranceservice/assurances/types",
            name="[Assurance] Get Types"
        )
    
    @task(2)
    @tag('price')
    def query_prices(self):
        'English text removed.'
        self.get(
            "/api/v1/priceservice/prices",
            name="[Price] Get All Prices"
        )
    
    @task(2)
    @tag('seat')
    def query_seats(self):
        'English text removed.'
                                                                           
                                    
        train_number = "D1345"
        travel_date = get_test_date()
        seat_type = 2

                                                               
        stations = None
        start_station = None
        dest_station = None
        total_num = None

        try:
            r_route = self.get(
                f"/api/v1/travelservice/routes/{train_number}",
                name="[Travel] Get Route By TripId (for Seat)"
            )
            if r_route.status_code == 200:
                j = r_route.json()
                data = (j or {}).get("data") or {}
                stations = data.get("stations")
                start_station = data.get("startStation")
                dest_station = data.get("endStation")
        except Exception:
                                           
            stations = ["Shang Hai", "Su Zhou"]
            start_station = "Shang Hai"
            dest_station = "Su Zhou"

        try:
            r_tt = self.get(
                f"/api/v1/travelservice/train_types/{train_number}",
                name="[Travel] Get TrainType By TripId (for Seat)"
            )
            if r_tt.status_code == 200:
                j = r_tt.json()
                data = (j or {}).get("data") or {}
                                                                   
                if seat_type == 2:
                    total_num = data.get("economyClass")
                else:
                    total_num = data.get("confortClass")
        except Exception:
            total_num = 200

        payload = {
            "travelDate": travel_date,
            "trainNumber": train_number,
            "startStation": start_station,
            "destStation": dest_station,
            "seatType": seat_type,
            "totalNum": int(total_num or 200),
            "stations": stations or [start_station, dest_station],
        }

        self.post(
            "/api/v1/seatservice/seats/left_tickets",
            json=payload,
            name="[Seat] Query Available Seats"
        )
    
    @task(1)
    @tag('security')
    def security_check(self):
        'English text removed.'
        account_id = getattr(self.user, 'user_id', '') or 'test-account'
        self.get(
            f"/api/v1/securityservice/securityConfigs/{account_id}",
            name="[Security] Check Account"
        )
    
    @task(1)
    @tag('basic')
    def query_travel_by_trip(self):
        'English text removed.'
                                                                 
           
                                
                              
                                   
                                                                                 
           
        train_number = "D1345"
        travel_date = get_test_date()

                                                                      
        route_id = None
        start_place = None
        end_place = None
        train_type_name = None

        r_route = self.get(
            f"/api/v1/travelservice/routes/{train_number}",
            name="[Travel] Get Route By TripId (for Basic)"
        )
        if r_route.status_code == 200:
            j = r_route.json()
            data = (j or {}).get("data") or {}
            route_id = data.get("id")
            start_place = data.get("startStation")
            end_place = data.get("endStation")

        r_tt = self.get(
            f"/api/v1/travelservice/train_types/{train_number}",
            name="[Travel] Get TrainType By TripId (for Basic)"
        )
        if r_tt.status_code == 200:
            j = r_tt.json()
            data = (j or {}).get("data") or {}
            train_type_name = data.get("name")

                                              
        if not start_place:
            start_place = "shanghai"
        if not end_place:
            end_place = "suzhou"
        if not train_type_name:
            train_type_name = "DongCheOne"
        if not route_id:
            route_id = ""

        payload = {
            "startPlace": start_place,
            "endPlace": end_place,
            "departureTime": travel_date,
            "trip": {
                "tripId": {
                    "type": train_number[0],
                    "number": train_number[1:],
                },
                "trainTypeName": train_type_name,
                "routeId": route_id,
            },
        }

        self.post(
            "/api/v1/basicservice/basic/travel",
            json=payload,
            name="[Basic] Query Travel Info"
        )


                                                                               
               
                                                                               
class FoodServiceTasks(BaseTaskSet):
    'English text removed.'
    
    @task(3)
    @tag('food')
    def query_foods(self):
        'English text removed.'
        self.get(
            f"/api/v1/foodservice/foods/{get_test_date()}/Shang Hai/Su Zhou/D1345",
            name="[Food] Query Foods"
        )
    
    @task(2)
    @tag('food')
    def query_station_food(self):
        'English text removed.'
        self.get(
                                    
                                                              
            "/api/v1/stationfoodservice/stationfoodstores",
            name="[Food] Get Station Food List"
        )
    
    @task(2)
    @tag('food')
    def query_train_food(self):
        'English text removed.'
        self.get(
            "/api/v1/trainfoodservice/trainfoods",
            name="[Food] Get Train Food List"
        )


                                                                               
               
                                                                               
class ConsignServiceTasks(BaseTaskSet):
    'English text removed.'
    
    @task(2)
    @tag('consign')
    def query_consign_price(self):
        'English text removed.'
             
                                                                                  
                                                                        
        if os.getenv("TT_SKIP_CONSIGN_PRICE_CONFIG", "1") != "0":
            return

        self.get(
            "/api/v1/consignpriceservice/consignprice/config",
            name="[Consign] Get Price Config"
        )
    
    @task(2)
    @tag('consign')
    def query_my_consigns(self):
        'English text removed.'
        account_id = getattr(self.user, 'user_id', '')
        self.get(
            f"/api/v1/consignservice/consigns/account/{account_id}",
            name="[Consign] Get My Consigns"
        )


                                                                               
                
                                                                               
class AdminServiceTasks(BaseTaskSet):
    'English text removed.'
    
    @task(2)
    @tag('admin')
    def admin_query_orders(self):
        'English text removed.'
        self.get(
            "/api/v1/adminorderservice/adminorder",
            name="[Admin] Get All Orders"
        )
    
    @task(2)
    @tag('admin')
    def admin_query_users(self):
        'English text removed.'
        self.get(
            "/api/v1/adminuserservice/users",
            name="[Admin] Get All Users"
        )
    
    @task(1)
    @tag('admin')
    def admin_query_contacts(self):
        'English text removed.'
        self.get(
            "/api/v1/adminbasicservice/adminbasic/contacts",
            name="[Admin] Get All Contacts"
        )
    
    @task(1)
    @tag('admin')
    def admin_query_stations(self):
        'English text removed.'
        self.get(
            "/api/v1/adminbasicservice/adminbasic/stations",
            name="[Admin] Get All Stations"
        )
    
    @task(1)
    @tag('admin')
    def admin_query_trains(self):
        'English text removed.'
        self.get(
            "/api/v1/adminbasicservice/adminbasic/trains",
            name="[Admin] Get All Trains"
        )
    
    @task(1)
    @tag('admin')
    def admin_query_routes(self):
        'English text removed.'
        self.get(
            "/api/v1/adminrouteservice/adminroute",
            name="[Admin] Get All Routes"
        )
    
    @task(1)
    @tag('admin')
    def admin_query_travels(self):
        'English text removed.'
        self.get(
            "/api/v1/admintravelservice/admintravel",
            name="[Admin] Get All Travels"
        )


                                                                               
          
                                                                               
class TrainTicketUser(HttpUser):
    'English text removed.'
    
    wait_time = between(1, 3)
    
    tasks = {
        CoreBusinessTasks: 5,
        SupportServiceTasks: 2,
        FoodServiceTasks: 1,
        ConsignServiceTasks: 1,
    }
    
          
    token = None
    user_id = None
    
    def on_start(self):
        'English text removed.'
        username = os.getenv("TT_USERNAME", "fdse_microservice")
        password = os.getenv("TT_PASSWORD", "")
        
        try:
            response = self.client.post(
                "/api/v1/users/login",
                json={"username": username, "password": password},
                headers={'Content-Type': 'application/json'},
                name="[Auth] Login"
            )
            if response.status_code == 200:
                result = response.json()
                if result.get("status") == 1:
                    data = result.get("data", {})
                    self.token = data.get("token")
                    self.user_id = data.get("userId")
                    logger.info(f"User {username} logged in, userId: {self.user_id}")
                else:
                    logger.warning(f"Login failed: {result.get('msg')}")
            else:
                logger.warning(f"Login HTTP error: {response.status_code}")
        except Exception as e:
            logger.error(f"Login exception: {e}")
    
    def on_stop(self):
        'English text removed.'
        self.token = None
        self.user_id = None


class TrainTicketAdminUser(HttpUser):
    'English text removed.'
    
    wait_time = between(2, 5)
    weight = 1
    
    tasks = {
                                                         
                                                                 
                                                   
        AdminServiceTasks: 3,
    }
    
    token = None
    user_id = None
    
    def on_start(self):
        'English text removed.'
        username = os.getenv("TT_ADMIN_USERNAME", "admin")
        password = os.getenv("TT_ADMIN_PASSWORD", "")
        
        try:
            response = self.client.post(
                "/api/v1/users/login",
                json={"username": username, "password": password},
                headers={'Content-Type': 'application/json'},
                name="[Auth] Admin Login"
            )
            if response.status_code == 200:
                result = response.json()
                if result.get("status") == 1:
                    data = result.get("data", {})
                    self.token = data.get("token")
                    self.user_id = data.get("userId")
                    logger.info(f"Admin {username} logged in, userId: {self.user_id}")
        except Exception as e:
            logger.error(f"Admin login exception: {e}")


                                                                               
      
                                                                               
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    logger.info("=" * 60)
    logger.info("Train-Ticket Load Test Started")
    logger.info(f"Target Host: {environment.host}")
    logger.info("=" * 60)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    logger.info("=" * 60)
    logger.info("Train-Ticket Load Test Completed")
    logger.info("=" * 60)


if __name__ == "__main__":
    print('English text removed.')

                                                                               
              
                                                                               
       
                                                  
                                              
 
             
                                                                               

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from locust import task, tag
from tasks.base import BaseTaskSet
from utils.data_generator import DataGenerator, get_future_date


class TravelPlanTasks(BaseTaskSet):
    'English text removed.'
    
    @task(1)
    @tag('travelplan', 'planning', 'query')
    def travel_plan_welcome(self):
        'English text removed.'
        self.get(
            "/api/v1/travelplanservice/welcome",
            name="[TravelPlan] Welcome"
        )
    
    @task(3)
    @tag('travelplan', 'planning', 'query')
    def get_cheapest(self):
        'English text removed.'
        place_pair = DataGenerator.get_place_pair()
        payload = {
            "startPlace": place_pair[0],
            "endPlace": place_pair[1],
            "departureTime": get_future_date(1),
        }
        self.post_with_check(
            "/api/v1/travelplanservice/travelPlan/cheapest",
            json=payload,
            name="[TravelPlan] Get Cheapest",
            expect_data=False
        )
    
    @task(3)
    @tag('travelplan', 'planning', 'query')
    def get_quickest(self):
        'English text removed.'
        place_pair = DataGenerator.get_place_pair()
        payload = {
            "startPlace": place_pair[0],
            "endPlace": place_pair[1],
            "departureTime": get_future_date(1),
        }
        self.post_with_check(
            "/api/v1/travelplanservice/travelPlan/quickest",
            json=payload,
            name="[TravelPlan] Get Quickest",
            expect_data=False
        )
    
    @task(2)
    @tag('travelplan', 'planning', 'query')
    def get_min_station(self):
        'English text removed.'
        place_pair = DataGenerator.get_place_pair()
        payload = {
            "startPlace": place_pair[0],
            "endPlace": place_pair[1],
            "departureTime": get_future_date(1),
        }
        self.post_with_check(
            "/api/v1/travelplanservice/travelPlan/minStation",
            json=payload,
            name="[TravelPlan] Get Min Station",
            expect_data=False
        )
    
    @task(2)
    @tag('travelplan', 'planning', 'query')
    def get_transfer_result(self):
        'English text removed.'
        payload = {
            "startStation": "Shang Hai",
            "endStation": "Bei Jing",
            "travelDate": get_future_date(1),
            "viaStation": "Nan Jing",
            "trainType": "D",
        }
        self.post_with_check(
            "/api/v1/travelplanservice/travelPlan/transferResult",
            json=payload,
            name="[TravelPlan] Get Transfer Result",
            expect_data=False
        )


class RoutePlanTasks(BaseTaskSet):
    'English text removed.'
    
    @task(1)
    @tag('routeplan', 'planning', 'query')
    def route_plan_welcome(self):
        'English text removed.'
        self.get(
            "/api/v1/routeplanservice/welcome",
            name="[RoutePlan] Welcome"
        )
    
    @task(3)
    @tag('routeplan', 'planning', 'query')
    def get_cheapest_route(self):
        'English text removed.'
        payload = {
            "startStation": "Shang Hai",
            "endStation": "Su Zhou",
            "num": 5,
            "travelDate": get_future_date(1),
        }
        self.post_with_check(
            "/api/v1/routeplanservice/routePlan/cheapestRoute",
            json=payload,
            name="[RoutePlan] Get Cheapest Route",
            expect_data=False
        )
    
    @task(3)
    @tag('routeplan', 'planning', 'query')
    def get_quickest_route(self):
        'English text removed.'
        payload = {
            "startStation": "Shang Hai",
            "endStation": "Su Zhou",
            "num": 5,
            "travelDate": get_future_date(1),
        }
        self.post_with_check(
            "/api/v1/routeplanservice/routePlan/quickestRoute",
            json=payload,
            name="[RoutePlan] Get Quickest Route",
            expect_data=False
        )
    
    @task(2)
    @tag('routeplan', 'planning', 'query')
    def get_min_stop_stations(self):
        'English text removed.'
        payload = {
            "startStation": "Shang Hai",
            "endStation": "Su Zhou",
            "num": 5,
            "travelDate": get_future_date(1),
        }
        self.post_with_check(
            "/api/v1/routeplanservice/routePlan/minStopStations",
            json=payload,
            name="[RoutePlan] Get Min Stop Stations",
            expect_data=False
        )


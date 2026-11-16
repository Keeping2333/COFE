                                                                               
             
                                                                               
             
                                                                               

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from locust import task, tag
from tasks.base import BaseTaskSet


class AdminTasks(BaseTaskSet):
    'English text removed.'
    
    @task(3)
    @tag('admin', 'order')
    def admin_query_orders(self):
        'English text removed.'
        self.get_with_check(
            "/api/v1/adminorderservice/adminorder",
            name="[Admin] Get All Orders",
            expect_data=False
        )
    
    @task(3)
    @tag('admin', 'user')
    def admin_query_users(self):
        'English text removed.'
        self.get_with_check(
            "/api/v1/adminuserservice/users",
            name="[Admin] Get All Users",
            expect_data=False
        )
    
    @task(2)
    @tag('admin', 'contact')
    def admin_query_contacts(self):
        'English text removed.'
        self.get_with_check(
            "/api/v1/adminbasicservice/adminbasic/contacts",
            name="[Admin] Get All Contacts",
            expect_data=False
        )
    
    @task(2)
    @tag('admin', 'station')
    def admin_query_stations(self):
        'English text removed.'
        self.get_with_check(
            "/api/v1/adminbasicservice/adminbasic/stations",
            name="[Admin] Get All Stations",
            expect_data=True
        )
    
    @task(2)
    @tag('admin', 'train')
    def admin_query_trains(self):
        'English text removed.'
        self.get_with_check(
            "/api/v1/adminbasicservice/adminbasic/trains",
            name="[Admin] Get All Trains",
            expect_data=True
        )
    
    @task(2)
    @tag('admin', 'route')
    def admin_query_routes(self):
        'English text removed.'
        self.get_with_check(
            "/api/v1/adminrouteservice/adminroute",
            name="[Admin] Get All Routes",
            expect_data=False
        )
    
    @task(2)
    @tag('admin', 'travel')
    def admin_query_travels(self):
        'English text removed.'
        self.get_with_check(
            "/api/v1/admintravelservice/admintravel",
            name="[Admin] Get All Travels",
            expect_data=False
        )
    
    @task(1)
    @tag('admin', 'config')
    def admin_query_configs(self):
        'English text removed.'
        self.get_with_check(
            "/api/v1/adminbasicservice/adminbasic/configs",
            name="[Admin] Get All Configs",
            expect_data=False
        )
    
    @task(1)
    @tag('admin', 'price')
    def admin_query_prices(self):
        'English text removed.'
        self.get_with_check(
            "/api/v1/adminbasicservice/adminbasic/prices",
            name="[Admin] Get All Prices",
            expect_data=False
        )


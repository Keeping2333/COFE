                                                                               
              
                                                                               
       
                                    
                                            
 
      
            
                        
                                                                               

import logging
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from locust import task, tag
from tasks.base import BaseTaskSet
from utils.data_generator import random_string, random_id_card, random_email

logger = logging.getLogger(__name__)

                
TEST_USER_PREFIX = "locust_test_"


class UserServiceTasks(BaseTaskSet):
    'English text removed.'
    
    @task(2)
    @tag('user', 'query')
    def get_all_users(self):
        'English text removed.'
        self.get_with_check(
            "/api/v1/userservice/users",
            name="[User] Get All Users",
            expect_data=False
        )
    
    @task(2)
    @tag('user', 'query')
    def get_user_by_name(self):
        'English text removed.'
        self.get_with_check(
            "/api/v1/userservice/users/fdse_microservice",
            name="[User] Get User By Name",
            expect_data=False
        )
    
    @task(1)
    @tag('user', 'query')
    def get_user_by_id(self):
        'English text removed.'
        if self.user_id:
            self.get_with_check(
                f"/api/v1/userservice/users/id/{self.user_id}",
                name="[User] Get User By Id",
                expect_data=False
            )
    
    @task(1)
    @tag('user', 'write')
    def register_user(self):
        'English text removed.'
                            
        if os.getenv("TT_DISABLE_USER_REGISTRATION", "0") == "1":
            return
        
        username = f"{TEST_USER_PREFIX}{random_string(6)}"
        payload = {
            "userName": username,
            "password": os.getenv("TT_NEW_USER_PASSWORD", ""),
            "email": random_email(),
            "gender": random.randint(0, 1),
            "documentType": 1,
            "documentNum": random_id_card(),
        }
        
        success, _ = self.post_with_check(
            "/api/v1/userservice/users/register",
            json=payload,
            name="[User] Register User",
            expect_data=False
        )
        
        if success:
            logger.info(f"Test user registered: {username}")


class VerifyCodeTasks(BaseTaskSet):
    'English text removed.'
    
    @task(3)
    @tag('verifycode', 'auth', 'query')
    def generate_code(self):
        'English text removed.'
                                      
        response = self.get(
            "/api/v1/verifycode/generate",
            name="[VerifyCode] Generate"
        )
                      
        if response.status_code != 200:
            logger.warning(f"Generate verify code failed: {response.status_code}")
    
    @task(2)
    @tag('verifycode', 'auth', 'query')
    def verify_code(self):
        'English text removed.'
                    
        response = self.get(
            "/api/v1/verifycode/verify/1234",
            name="[VerifyCode] Verify"
        )
        if response.status_code != 200:
            logger.warning(f"Verify code failed: {response.status_code}")

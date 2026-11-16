                                                                               
                                      
                                                                               
                                         
 
           
                                             
                                  
                                
 
       
          
                                                                              
 
                        
                                                                                    
 
                    
                                                                  
 
           
                      
                                                
 
            
                                               
 
           
                                                
                                                                               

import logging
import os
import sys

            
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from locust import HttpUser, between, events

from tasks import (
          
    QueryTasks, 
    BookingFlowTasks, 
    AdminTasks,
          
    UserServiceTasks,
    VerifyCodeTasks,
        
    PaymentServiceTasks,
          
    RebookTasks,
    WaitOrderTasks,
    ExecuteTasks,
          
    TravelPlanTasks,
    RoutePlanTasks,
        
    FoodDeliveryTasks,
        
    NotificationTasks,
)
from utils import AuthMixin, cleanup_on_test_end

      
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

                        
_admin_token = None


                                                                               
       
                                                                               

class TrainTicketUser(AuthMixin, HttpUser):
    'English text removed.'

    wait_time = between(1, 3)
    weight = int(os.getenv("TT_W_DEFAULT", "3"))

    tasks = {
        QueryTasks: 3,
        BookingFlowTasks: 2,
    }
    
    def on_start(self):
        'English text removed.'
        username = os.getenv("TT_USERNAME", "fdse_microservice")
        password = os.getenv("TT_PASSWORD", "")
        
        if not self.login(username, password):
            logger.error(f"User {username} login failed!")
    
    def on_stop(self):
        'English text removed.'
        self.token = None
        self.user_id = None


class TrainTicketQueryUser(AuthMixin, HttpUser):
    'English text removed.'
    
    wait_time = between(1, 2)
    weight = int(os.getenv("TT_W_QUERY", "2"))

    tasks = {
        QueryTasks: 1,
    }
    
    def on_start(self):
        username = os.getenv("TT_USERNAME", "fdse_microservice")
        password = os.getenv("TT_PASSWORD", "")
        self.login(username, password)
    
    def on_stop(self):
        self.token = None
        self.user_id = None


class TrainTicketFlowUser(AuthMixin, HttpUser):
    'English text removed.'
    
    wait_time = between(2, 5)            
    weight = int(os.getenv("TT_W_FLOW", "1"))

    tasks = {
        BookingFlowTasks: 1,
    }
    
    def on_start(self):
        username = os.getenv("TT_USERNAME", "fdse_microservice")
        password = os.getenv("TT_PASSWORD", "")
        self.login(username, password)
    
    def on_stop(self):
        self.token = None
        self.user_id = None


class TrainTicketAdminUser(AuthMixin, HttpUser):
    'English text removed.'
    
    wait_time = between(2, 5)
    weight = int(os.getenv("TT_W_ADMIN", "1"))

    tasks = {
        AdminTasks: 1,
    }
    
    def on_start(self):
        global _admin_token
        username = os.getenv("TT_ADMIN_USERNAME", "admin")
        password = os.getenv("TT_ADMIN_PASSWORD", "")
        
        if self.admin_login(username, password):
            _admin_token = self.token                 
    
    def on_stop(self):
        self.token = None
        self.user_id = None


class TrainTicketFullCoverageUser(AuthMixin, HttpUser):
    'English text removed.'
    
    wait_time = between(1, 3)
    weight = int(os.getenv("TT_W_FULL", "1"))

    tasks = {
        QueryTasks: 3,
        BookingFlowTasks: 2,
              
        UserServiceTasks: 1,
        VerifyCodeTasks: 1,
        PaymentServiceTasks: 1,
        RebookTasks: 1,
        TravelPlanTasks: 2,
        RoutePlanTasks: 2,
                                                    
                                                               
                                                            
        NotificationTasks: 1,
        ExecuteTasks: 1,
    }
    
    def on_start(self):
        username = os.getenv("TT_USERNAME", "fdse_microservice")
        password = os.getenv("TT_PASSWORD", "")
        
        if not self.login(username, password):
            logger.error(f"User {username} login failed!")
    
    def on_stop(self):
        self.token = None
        self.user_id = None


                                                                               
      
                                                                               

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    logger.info("=" * 70)
    logger.info("Train-Ticket Load Test (Modular Version v2) Started")
    logger.info(f"Target Host: {environment.host}")
    logger.info("=" * 70)
    logger.info('English text removed.')
    logger.info('English text removed.')
    logger.info('English text removed.')
    logger.info('English text removed.')
    logger.info("=" * 70)
    logger.info("User Types:")
    logger.info('English text removed.')
    logger.info('English text removed.')
    logger.info('English text removed.')
    logger.info('English text removed.')
    logger.info('English text removed.')
    logger.info("=" * 70)
    
              
    logger.info("Environment Config:")
    logger.info(f"  TT_CLEANUP_ON_END: {os.getenv('TT_CLEANUP_ON_END', '0')} (1=English text removed.)")
    logger.info(f"  TT_DISABLE_USER_REGISTRATION: {os.getenv('TT_DISABLE_USER_REGISTRATION', '0')} (1=English text removed.)")
    logger.info(f"  TT_SKIP_CONSIGN_PRICE_CONFIG: {os.getenv('TT_SKIP_CONSIGN_PRICE_CONFIG', '1')} (1=English text removed.)")
    logger.info("=" * 70)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    logger.info("=" * 70)
    logger.info("Train-Ticket Load Test (Modular Version v2) Completed")
    logger.info("=" * 70)
    
                  
    if os.getenv("TT_CLEANUP_ON_END", "0") == "1":
        if _admin_token:
            cleanup_on_test_end(environment.host, _admin_token)
        else:
            logger.warning("No admin token available for cleanup. Run with TrainTicketAdminUser.")


                                                                               
     
                                                                               

if __name__ == "__main__":
    print('English text removed.')

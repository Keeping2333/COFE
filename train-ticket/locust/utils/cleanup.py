                                                                               
        
                                                                               
                    
 
     
                              
                   
                 
                                                                               

import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)

          
TEST_USER_PREFIX = "locust_test_"
TEST_CONTACT_PREFIX = "LocustTest_"

      
CLEANUP_CONFIG = {
    "cleanup_test_users": False,                   
    "cleanup_test_contacts": False,                 
    "cleanup_test_orders": False,                      
}


class DataCleanup:
    'English text removed.'
    
    def __init__(self, host: str, admin_token: str):
        'English text removed.'
        self.host = host.rstrip('/')
        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': f'Bearer {admin_token}'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def cleanup_test_users(self) -> int:
        'English text removed.'
        cleaned = 0
        try:
                    
            response = self.session.get(f"{self.host}/api/v1/userservice/users")
            if response.status_code != 200:
                logger.error(f"Failed to get users: {response.status_code}")
                return cleaned
            
            result = response.json()
            if result.get("status") != 1:
                logger.error(f"Business error: {result.get('msg')}")
                return cleaned
            
            users = result.get("data", [])
            
                    
            test_users = [u for u in users if u.get("userName", "").startswith(TEST_USER_PREFIX)]
            
            for user in test_users:
                user_id = user.get("userId")
                username = user.get("userName")
                
                      
                del_response = self.session.delete(
                    f"{self.host}/api/v1/userservice/users/{user_id}"
                )
                
                if del_response.status_code == 200:
                    logger.info(f"Deleted test user: {username}")
                    cleaned += 1
                else:
                    logger.warning(f"Failed to delete user {username}: {del_response.status_code}")
            
        except Exception as e:
            logger.error(f"Error cleaning up users: {e}")
        
        return cleaned
    
    def cleanup_test_contacts(self, user_id: str) -> int:
        'English text removed.'
        cleaned = 0
        try:
                      
            response = self.session.get(
                f"{self.host}/api/v1/contactservice/contacts/account/{user_id}"
            )
            if response.status_code != 200:
                logger.error(f"Failed to get contacts: {response.status_code}")
                return cleaned
            
            result = response.json()
            if result.get("status") != 1:
                return cleaned
            
            contacts = result.get("data", [])
            
                     
            test_contacts = [c for c in contacts if c.get("name", "").startswith(TEST_CONTACT_PREFIX)]
            
            for contact in test_contacts:
                contact_id = contact.get("id")
                contact_name = contact.get("name")
                
                       
                del_response = self.session.delete(
                    f"{self.host}/api/v1/contactservice/contacts/{contact_id}"
                )
                
                if del_response.status_code == 200:
                    logger.info(f"Deleted test contact: {contact_name}")
                    cleaned += 1
                else:
                    logger.warning(f"Failed to delete contact {contact_name}: {del_response.status_code}")
            
        except Exception as e:
            logger.error(f"Error cleaning up contacts: {e}")
        
        return cleaned
    
    def get_cleanup_summary(self) -> dict:
        'English text removed.'
        summary = {
            "test_users": 0,
            "test_contacts": 0,
        }
        
        try:
                    
            response = self.session.get(f"{self.host}/api/v1/userservice/users")
            if response.status_code == 200:
                result = response.json()
                if result.get("status") == 1:
                    users = result.get("data", [])
                    summary["test_users"] = len([
                        u for u in users 
                        if u.get("userName", "").startswith(TEST_USER_PREFIX)
                    ])
        except Exception as e:
            logger.error(f"Error getting user summary: {e}")
        
        return summary


def cleanup_on_test_end(host: str, admin_token: Optional[str] = None):
    'English text removed.'
    import os
    
              
    if os.getenv("TT_CLEANUP_ON_END", "0") != "1":
        logger.info("Data cleanup disabled (set TT_CLEANUP_ON_END=1 to enable)")
        return
    
    if not admin_token:
        logger.warning("No admin token provided, skipping cleanup")
        return
    
    logger.info("=" * 60)
    logger.info("Starting test data cleanup...")
    
    cleanup = DataCleanup(host, admin_token)
    
           
    summary = cleanup.get_cleanup_summary()
    logger.info(f"Found {summary['test_users']} test users to cleanup")
    
          
    if CLEANUP_CONFIG["cleanup_test_users"]:
        cleaned = cleanup.cleanup_test_users()
        logger.info(f"Cleaned {cleaned} test users")
    
    logger.info("Data cleanup completed")
    logger.info("=" * 60)



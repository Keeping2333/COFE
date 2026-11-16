                                                                               
        
                                                                               

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class AuthMixin:
    'English text removed.'
    
                   
    token: Optional[str] = None
    user_id: Optional[str] = None
    
    def login(self, username: Optional[str] = None,
              password: Optional[str] = None) -> bool:
        'English text removed.'
        username = username or os.getenv("TT_USERNAME", "fdse_microservice")
        password = password if password is not None else os.getenv("TT_PASSWORD", "")

        login_payload = {
            "username": username,
            "password": password,
        }
        
        try:
            response = self.client.post(
                "/api/v1/users/login",
                json=login_payload,
                headers={'Content-Type': 'application/json'},
                name="[Auth] Login"
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("status") == 1:
                    data = result.get("data", {})
                    self.token = data.get("token")
                    self.user_id = data.get("userId")
                    logger.info(f"Login successful: user={username}, userId={self.user_id}")
                    return True
                else:
                    logger.warning(f"Login failed: {result.get('msg')}")
                    return False
            else:
                logger.warning(f"Login HTTP error: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Login exception: {e}")
            return False
    
    def admin_login(self, username: Optional[str] = None,
                    password: Optional[str] = None) -> bool:
        'English text removed.'
        username = username or os.getenv("TT_ADMIN_USERNAME", "admin")
        password = password if password is not None else os.getenv("TT_ADMIN_PASSWORD", "")
        return self.login(username, password)
    
    def get_auth_headers(self) -> dict:
        'English text removed.'
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
        }
        if self.token:
            headers['Authorization'] = f'Bearer {self.token}'
        return headers
    
    def ensure_authenticated(self) -> bool:
        'English text removed.'
        if not self.token:
            return self.login()
        return True

from fastapi import Depends, HTTPException, status

from api.auth import get_current_user
from models.schemas import UserInfo

async def get_current_active_user(current_user: UserInfo = Depends(get_current_user)) -> UserInfo:
    if current_user.disabled:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user")
    return current_user

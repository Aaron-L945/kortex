"""
JWT 鉴权层。

简化实现：用户信息存储在内存中（生产环境替换为数据库）。
包含：
  - create_access_token
  - verify_token（FastAPI Dependency）
  - fake_users_db（演示用户）
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

from config import settings
from models.schemas import UserInfo

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


# ─── 演示用户数据库（生产环境替换为真实数据库）────────────────────────────────

FAKE_USERS_DB = {
    "admin": {
        "username": "admin",
        "hashed_password": pwd_context.hash("admin"),
        "user_id": "u001",
        "user_group": "admin",
        "department": "it",
        "permission_level": 4,     # 最高权限
    },
    "hr_user": {
        "username": "hr_user",
        "hashed_password": pwd_context.hash("hr"),
        "user_id": "u002",
        "user_group": "hr",
        "department": "hr",
        "permission_level": 2,     # 内部权限
    },
    "employee": {
        "username": "employee",
        "hashed_password": pwd_context.hash("emp"),
        "user_id": "u003",
        "user_group": "staff",
        "department": "engineering",
        "permission_level": 1,     # 公开权限
    },
}


# ─── 核心函数 ─────────────────────────────────────────────────────────────────

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def authenticate_user(username: str, password: str) -> Optional[dict]:
    user = FAKE_USERS_DB.get(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user


def create_access_token(data: dict) -> str:
    payload = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
    )
    payload["exp"] = expire
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserInfo:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无效凭证，请重新登录",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user_data = FAKE_USERS_DB.get(username)
    if user_data is None:
        raise credentials_exception

    return UserInfo(
        user_id=user_data["user_id"],
        username=user_data["username"],
        user_group=user_data["user_group"],
        department=user_data["department"],
        permission_level=user_data["permission_level"],
    )

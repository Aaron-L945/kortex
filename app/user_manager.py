import sqlite3
import jwt
import datetime
from passlib.context import CryptContext
from loguru import logger
import bcrypt
import os

# --- 补丁必须在 passlib 之前 ---
if not hasattr(bcrypt, "__about__"):
    bcrypt.__about__ = type('About', (object,), {'__version__': bcrypt.__version__})
from passlib.context import CryptContext


# 1. 获取当前文件所在目录的上一级，即项目根目录
# __file__ 是 user_manager.py，它的 parent 是 app/，再 parent 就是根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 2. 强制指定数据库在根目录下的 data/users.db
DB_PATH = os.path.join(BASE_DIR, "sql_data", "users.db")

# 密码哈希配置
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# JWT 密钥（生产环境请存放在环境变量中）
SECRET_KEY = "enterprise-super-secret-key"
ALGORITHM = "HS256"

class UserManager:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._create_table()

    def _get_conn(self):
        return sqlite3.connect(self.db_path)

    def _create_table(self):
        """初始化用户表"""
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    dept TEXT NOT NULL,
                    role TEXT NOT NULL
                )
            """)
            conn.commit()

    def register_user(self, username, password, dept, role="internal"):
        """用户注册：哈希密码并存入 SQLite"""
        hashed = pwd_context.hash(password)
        try:
            with self._get_conn() as conn:
                conn.execute(
                    "INSERT INTO users (username, password_hash, dept, role) VALUES (?, ?, ?, ?)",
                    (username, hashed, dept, role)
                )
                conn.commit()
            logger.success(f"用户 {username} 注册成功！部门: {dept}")
            return True
        except sqlite3.IntegrityError:
            logger.error(f"注册失败：用户名 {username} 已存在")
            return False

    def authenticate_user(self, username, password):
        """登录校验并签发 JWT"""
        with self._get_conn() as conn:
            cursor = conn.execute(
                "SELECT password_hash, dept, role FROM users WHERE username = ?", (username,)
            )
            user = cursor.fetchone()

        if user and pwd_context.verify(password, user[0]):
            # 校验通过，准备 Payload
            payload = {
                "user_id": username,
                "dept": user[1],
                "role": user[2],
                "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=8) # 8小时有效期
            }
            token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
            return token
        return None

    @staticmethod
    def decode_token(token):
        """解析 Token"""
        try:
            return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        except jwt.ExpiredSignatureError:
            return "EXPIRED"
        except jwt.PyJWTError:
            return None
import sqlite3
import jwt
import datetime
from passlib.context import CryptContext
from loguru import logger
import bcrypt
import os
import hashlib  # 新增导入

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
        """用户注册：先预哈希再存入 SQLite"""
        # 将原始密码预哈希，确保交给 bcrypt 的永远是固定长度的字符串
        # 这样即便用户输入 100 位密码也不会报错
        prepared_password = hashlib.sha256(password.encode()).hexdigest()
        hashed = pwd_context.hash(prepared_password)
        
        try:
            with self._get_conn() as conn:
                conn.execute(
                    "INSERT INTO users (username, password_hash, dept, role) VALUES (?, ?, ?, ?)",
                    (username, hashed, dept, role)
                )
                conn.commit()
            logger.success(f"用户 {username} 注册成功！")
            return True
        except sqlite3.IntegrityError:
            return False

    def authenticate_user(self, username, password):
        """登录校验：先预哈希再 verify"""
        with self._get_conn() as conn:
            cursor = conn.execute(
                "SELECT password_hash, dept, role FROM users WHERE username = ?", (username,)
            )
            user = cursor.fetchone()

        if not user:
            return None

        # 核心修复点：对输入的明文密码进行同样的预哈希
        prepared_password = hashlib.sha256(password.encode()).hexdigest()
        
        # 日志调试（生产环境请删除）
        logger.debug(f"校验用户: {username}, 预哈希长度: {len(prepared_password)}")

        if pwd_context.verify(prepared_password, user[0]):
            payload = {
                "user_id": username,
                "dept": user[1],
                "role": user[2],
                "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=8)
            }
            return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
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
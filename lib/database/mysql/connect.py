import time
from ...loader import lazy_import
from loguru import logger
import hashlib


def auto_reconnect(func):
    def wrapper(self, *args, **kwargs):
        retries = 3
        for attempt in range(retries):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                if "Duplicate entry" in str(e):
                    logger.warning(f" @@ Duplicate entry: {e}.")
                    break
                if attempt < retries - 1:
                    logger.warning(f" !! Connection error: {e}, retrying ({attempt+1}/{retries})...")
                    self._safe_reconnect()
                    continue
                raise e

        return None
    return wrapper


class SafeCursor:
    def __init__(self, connection, dict_cursor=True):
        self.connection = connection
        self.dict_cursor = dict_cursor
        self.cursor = None

    def __enter__(self):
        try:
            self.cursor = self.connection.cursor(dictionary=self.dict_cursor)
            return self.cursor
        except Exception as e:
            logger.error(f"Cursor creation failed: {e}")
            raise
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.cursor:
                self.cursor.close()
        except Exception as close_error:
            logger.debug(f"Cursor close error: {close_error}")
        if exc_type:
            logger.error(f"Cursor operation error: {exc_val}")


class MysqlConnection:
    # import mysql.connector
    MySqlConnector = None

    def __init__(self, username, password, host, port, database):
        if self.MySqlConnector is None:
            type(self).MySqlConnector = lazy_import('mysql.connector')

        self.host = host
        self.port = port
        self.database = database
        self.username = username
        self.password = password

        logger.debug(f" @@ Connecting to MySQL database {self.database}, host: {self.host}, port: {self.port}, username: {self.username}.")

        self.db = self.make_connection()
        if self.db is None:
            logger.error(f" !! Error connecting to MySQL database: {self.database}, host: {self.host}, port: {self.port}")
            raise Exception("Error connecting to MySQL database.")
        else:
            logger.debug(f" @@ Connected to MySQL database {self.database} successfully, host: {self.host}, port: {self.port}")
        self.default_cursor = self.get_cursor(in_dict=True)

    def make_connection(self):
        try_time = 3
        db_obj = self.connect()
        while (db_obj is None or db_obj.is_closed()) and try_time > 0:
            logger.debug(f" @@ Reconnecting to MySQL database {self.database}, host: {self.host}, port: {self.port}, username: {self.username}.")
            db_obj = self.connect()
            time.sleep(1)
            try_time -= 1
        return db_obj

    def start_transaction(self):
        if self.db.in_transaction:
            self.db.rollback()
        self.db.start_transaction()

    def commit(self):
        self.db.commit()
    
    def rollback(self):
        self.db.rollback()

    def connect(self):
        db = None
        connection_params = {
            "host": self.host,
            "port": self.port,
            "user": self.username,
            "password": self.password,
            "database": self.database,
            "autocommit": True,
            "allow_local_infile": True,
            "connect_timeout": 30,          # 增加连接超时
            "pool_size": 15,                 # 连接池大小
            "pool_name": f"mysql_pool_{self.database}",       # 连接池名称
            "pool_reset_session": True      # 重置会话
        }

        if hasattr(self.MySqlConnector, 'CONNECTION_POOL'):
            connection_params.update({
                "pool_heartbeat_interval": 300,  # 5分钟心跳
                "pool_validation_interval": 60    # 每分钟验证
            })

        try:
            db = self.MySqlConnector.connect(**connection_params)
        except Exception as e:
            logger.exception(f" !! Error connecting to MySQL database. {e}")
        
        time.sleep(2)
        if db is None:
            try:
                db = self.MySqlConnector.connect(**connection_params)
            except Exception as e:
                logger.exception(f" !! Error connecting to MySQL database. {e}")

        return db

    def close(self):
        try:
            self.db.close()
        except:
            logger.exception(" !! Error closing MySQL database.")

    def __del__(self):
        self.close()

    def _safe_reconnect(self):
        try:
            self.db.close()
        except Exception as close_error:
            logger.debug(f"Error closing old connection: {close_error}")
        finally:
            self.db = self.make_connection()
            self.default_cursor = self.get_cursor(in_dict=True)

    def mysql_connection_check(self):
        try:
            if self.db.is_connected():
                with self.db.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    cursor.fetchall()
                return True

        except (self.MySqlConnector.errors.InterfaceError, 
            self.MySqlConnector.errors.OperationalError,
            AttributeError) as e:
            logger.warning(f" !! Connection check failed: {e}, attempting reconnect...")
            for _ in range(3):
                try:
                    self.db.ping(reconnect=True, attempts=3, delay=1)
                    self.default_cursor = self.get_cursor(in_dict=True)
                    logger.info("Connection reestablished successfully")
                    return True
                except Exception as ping_error:
                    logger.error(f"Reconnection attempt failed: {ping_error}")
                    time.sleep(1)
        
            logger.error(" !! Reconnection failed, try to reconnect mysql.")
            try:
                self._safe_reconnect()
                return True
            except Exception as e:
                logger.error(" !! Error connecting to MySQL database under <mysql_connection_check>.")
                raise ConnectionError("Failed to reconnect after multiple attempts")

    def _execute(self, query, value, cursor=None, fetchall=False):
        if cursor is None:
            with SafeCursor(self.db) as cursor:
                cursor.execute(query, value)
                if fetchall:
                    return cursor.fetchall()
                else:
                    return True
        else:
            cursor.execute(query, value)
            if fetchall:
                return cursor.fetchall()
            else:
                return True

    @auto_reconnect
    def insert_item(self, data_dict, table, cursor=None):
        # self.mysql_connection_check()
        key_list = list(data_dict.keys())
        for index, key in enumerate(key_list):
            key_list[index] = f'`{key}`'

        query = f"INSERT INTO {table} ({', '.join(key_list)}) VALUES ({', '.join(['%s'] * len(data_dict))})"
        values = list(data_dict.values())
        self._execute(query, values, cursor)
        return True

    @auto_reconnect
    def update_item(self, set_dict, conditions, table, logical="AND", cursor=None):
        # self.mysql_connection_check()

        key_str_list = list()
        value_list = list()
        for k, v in set_dict.items():
            key_str_list.append(f"{k} = %s")
            value_list.append(v)

        set_str = ", ".join(key_str_list)
        
        query = ""
        
        if conditions:
            query += " WHERE "

            condition_str_list = list()
            
            for condition in conditions:
                condition_str_list.append(f"{condition[0]} {condition[2]} %s")
                value_list.append(str(condition[1]))
            query += (' ' + logical + ' ').join(condition_str_list)

        update_query = f"UPDATE {table} SET {set_str} {query}"
        # 执行更新操作
        self._execute(update_query, value_list, cursor)
        return True
    
    @auto_reconnect
    def batch_upsert_with_composite_keys(
        self,
        table: str,
        data: list[dict],
        key_fields: list[str],
        cursor=None
    ) -> bool:
        """修正VALUES()嵌套问题的批量upsert方案"""
        if not data or not key_fields:
            return True

        try:
            cursor = cursor or self.db.cursor()
            temp_table = f"temp_{table}_upsert_{hashlib.md5(str(key_fields).encode()).hexdigest()[:6]}"
            
            # 1. 创建临时表（保持原逻辑）
            columns = list(data[0].keys())
            pk_def = ', '.join(
                f'{k} BIGINT' if isinstance(data[0][k], (int, float))
                else f'{k} VARCHAR(255)'
                for k in key_fields
            )
            cursor.execute(f"""
                CREATE TEMPORARY TABLE {temp_table} (
                    {pk_def},
                    PRIMARY KEY ({', '.join(key_fields)}),
                    {', '.join(f'{col} TEXT' for col in columns if col not in key_fields)}
                ) ENGINE=InnoDB
            """)
            
            # 2. 批量插入临时表（保持原逻辑）
            cursor.executemany(
                f"REPLACE INTO {temp_table} ({', '.join(columns)}) VALUES ({', '.join(['%s']*len(columns))})",
                [tuple(str(item[col]) for col in columns) for item in data]
            )
            
            # 3. 修正UPDATE子句构造
            update_cols = [col for col in columns if col not in key_fields]
            update_clause = ', '.join(f'{col}=VALUES({col})' for col in update_cols)
            
            cursor.execute(f"""
                INSERT INTO {table} ({', '.join(columns)})
                SELECT {', '.join(columns)} FROM {temp_table}
                ON DUPLICATE KEY UPDATE {update_clause}
            """)
            
            return True
            
        except Exception as e:
            self.db.rollback()
            raise RuntimeError(f"Batch upsert failed: {str(e)}")
        finally:
            if cursor and not hasattr(cursor, 'external_owner'):
                cursor.close()

    @auto_reconnect
    def select_item(self, conditions, table, logical="AND", cursor=None, for_update=False):
        """
        :param conditions: list [[key1, value1, operator1], [key2, value2, operator2], ...]
        :param logical: str
        :return: list
        """
        # self.mysql_connection_check()
        assert logical in ["AND", "OR"], " !! logical must be 'AND' or 'OR'"
        condition_value_list = list()
        if conditions:
            query = f"SELECT * FROM {table} WHERE "

            condition_str_list = list()
            
            for condition in conditions:
                if condition[2] == 'in':
                    temp_str = "(" + ', '.join(['%s'] * len(condition[1])) + ")"
                    condition_str_list.append(f"{condition[0]} {condition[2]} {temp_str}")
                    condition_value_list.extend(condition[1])
                else:
                    condition_str_list.append(f"{condition[0]} {condition[2]} %s")
                    condition_value_list.append(str(condition[1]))
            query += (' ' + logical + ' ').join(condition_str_list)
        else:
            query = f"SELECT * FROM {table}"

        if for_update:
            query += " FOR UPDATE"

        result = self._execute(query, condition_value_list, cursor, fetchall=True)
        return result

    @auto_reconnect
    def item_exists(self, conditions, table, logical="AND", cursor=None):
        result = self.select_item(conditions, table, logical, cursor)
        return len(result) > 0

    @auto_reconnect
    def is_not_null(self, key, table, cursor=None):
        # self.mysql_connection_check()
        query = f"SELECT * FROM {table} WHERE {key} IS NOT NULL"

        result = self._execute(query, [], cursor, fetchall=True)
        return result

    @auto_reconnect
    def delete_item(self, conditions, table, logical='AND', cursor=None):
        # self.mysql_connection_check()
        query = ""
        
        if conditions:
            query += " WHERE "

            condition_str_list = list()
            value_list = list()
            
            for condition in conditions:
                condition_str_list.append(f"{condition[0]} {condition[2]} %s")
                value_list.append(str(condition[1]))
            query += (' ' + logical + ' ').join(condition_str_list)

        del_query = f"DELETE FROM {table} {query}"
        self._execute(del_query, value_list, cursor)

    def get_cursor(self, in_dict=True):
        self.mysql_connection_check()
        return self.db.cursor(dictionary=in_dict)

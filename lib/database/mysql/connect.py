from ...utils import lazy_import


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

        self.db = self.connect()
        self.default_cursor = self.get_cursor(in_dict=True)

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
        err = 'connect failed.'

        try:
            db = self.MySqlConnector.connect(host=self.host, port=self.port, user=self.username, password=self.password, database=self.database, autocommit=True)
        except Exception as err:
            pass

        if db is None:
            raise RuntimeError(f" !! Error connecting to MySQL database: {err}")
        else:
            print(f" @@ Connected to MySQL database {self.database} successfully.")

        return db

    def close(self):
        self.db.close()

    def insert_item(self, data_dict, table, cursor=None):
        if cursor is None:
            cursor = self.default_cursor

        query = f"INSERT INTO {table} ({', '.join(data_dict.keys())}) VALUES ({', '.join(['%s'] * len(data_dict))})"
        cursor.execute(query, list(data_dict.values()))

    def update_item(self, set_dict, conditions, table, logical="AND", cursor=None):

        if cursor is None:
            cursor = self.default_cursor

        try:
            self.start_transaction()
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
            cursor.execute(update_query, value_list)
            self.commit()
            return True
        except Exception as e:
            self.rollback()
            print(f" !! Error updating item: {e}")
            return False

    def select_item(self, conditions, table, logical="AND", cursor=None, for_update=False):
        """
        :param conditions: list [[key1, value1, operator1], [key2, value2, operator2], ...]
        :param logical: str
        :return: list
        """
        assert logical in ["AND", "OR"], " !! logical must be 'AND' or 'OR'"

        if cursor is None:
            cursor = self.default_cursor

        condition_value_list = list()
        if conditions:
            query = f"SELECT * FROM {table} WHERE "

            condition_str_list = list()
            
            for condition in conditions:
                condition_str_list.append(f"{condition[0]} {condition[2]} %s")
                condition_value_list.append(str(condition[1]))
            query += (' ' + logical + ' ').join(condition_str_list)
        else:
            query = f"SELECT * FROM {table}"


        if for_update:
            query += " FOR UPDATE"

        cursor.execute(query, condition_value_list)
        result = cursor.fetchall()
        return result

    def delete_item(self, conditions, table, logical='AND', cursor=None):
        if cursor is None:
            cursor = self.default_cursor

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
        cursor.execute(del_query, value_list)

    
    def get_cursor(self, in_dict=True):
        return self.db.cursor(dictionary=in_dict)

from uuid import uuid4

import psycopg2
from pymongo import MongoClient
import os

IS_DEV = True


def get_client():
    if IS_DEV:
        MONGO_URI = 'mongodb://localhost:27017/'
        client = MongoClient(MONGO_URI)
    else:
        MONGO_URI = 'mongodb://arcUser:cam4TAV-znh*myn0umt@SG-arc-mongo-52775.servers.mongodirector.com:27017/arcEtlDB?ssl=true'
        client = MongoClient(MONGO_URI, ssl=True, tls=True, tlsCAFile=os.path.dirname(os.path.abspath(__file__))+"/certificates/mongodb.pem",
                                     tlsAllowInvalidCertificates=True)
    return client


def get_collection(document_name):
    client = get_client()
    if IS_DEV:
        db = client.anodelabs
    else:
        db = client.arcEtlDB
    return db[document_name]


def get_refresh_token(vendor):
    collection = get_collection('credentials')
    result = collection.find_one({})
    return result.get(vendor)


def set_refresh_token(vendor, refresh_token):
    collection = get_collection('credentials')
    newvalues = {"$set": {vendor: refresh_token}}
    collection.update_one({}, newvalues)


def get_connection():
    if IS_DEV:
        conn = psycopg2.connect(dbname='anodelabs_new',
                                user='postgres',
                                password='12345',
                                host='localhost',
                                port='5432')
    else:
        conn = psycopg2.connect(dbname='arcDB',
                                user='arcUser',
                                password='jhy6RXH*ywu!epn9wmn',
                                host='SG-arc-rdbms-3742-pgsql-master.servers.mongodirector.com',
                                port='6432',
                                sslmode='prefer',
                                sslrootcert=os.path.dirname(os.path.abspath(__file__))+"/certificates/sgArcDB.pem")
    return conn


def insert_thermostat_data(**kwargs):
    conn = get_connection()
    cur = conn.cursor()
    query = f"""INSERT INTO thermostatdata (device_name, humidity, ambient_temp, temp, mode, device_type, created_at, ref_id) 
                                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s);"""
    cur.execute(query,
                (kwargs['device_name'], kwargs['humidity'], kwargs['ambient_temp'],
                 kwargs['heat_temp'] if kwargs['mode'] == 'HEAT' else kwargs['cool_temp'],
                 kwargs['mode'], kwargs['device_vendor'], kwargs['created_at'], kwargs['ref_id']))
    conn.commit()
    cur.close()
    conn.close()


def insert_weather_data(**kwargs):
    conn = get_connection()
    cur = conn.cursor()
    table = 'weatherdata'
    query = f"""INSERT INTO {table} (humidity, temperature, temperature_min, temperature_max, feels_like, pressure, zip_code, created_at, ref_id) 
                                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);"""
    cur.execute(query,
                (kwargs['humidity'], kwargs['temperature'], kwargs['temperature_min'], kwargs['temperature_max'], kwargs['feels_like'],
                 kwargs['pressure'], kwargs['zip_code'], kwargs['created_at'], kwargs['ref_id']))
    conn.commit()
    cur.close()
    conn.close()


def insert_weather_forecast_data(**kwargs):
    conn = get_connection()
    cur = conn.cursor()
    table = 'weatherforecastnoaa'
    query = f"""INSERT INTO {table} (end_date, temperature, wind_speed, zip_code, created_at, ref_id) 
                                        VALUES (%s, %s, %s, %s, %s, %s);"""
    cur.execute(query,
                (kwargs['end_date'], kwargs['temperature'], kwargs['wind_speed'], kwargs['zip_code'], kwargs['created_at'], kwargs['ref_id']))
    conn.commit()
    cur.close()
    conn.close()


def insert_weather_forecast_openweather_data(**kwargs):
    conn = get_connection()
    cur = conn.cursor()
    table = 'weatherforecastopenweather'
    query = f"""INSERT INTO {table} (humidity, temperature, temperature_min, temperature_max, feels_like, pressure, zip_code, created_at, ref_id) 
                                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);"""
    cur.execute(query,
                (kwargs['humidity'], kwargs['temperature'], kwargs['temperature_min'], kwargs['temperature_max'],
                 kwargs['feels_like'],
                 kwargs['pressure'], kwargs['zip_code'], kwargs['created_at'], kwargs['ref_id']))
    conn.commit()
    cur.close()
    conn.close()


def insert_emporia_data(**kwargs):
    conn = get_connection()
    cur = conn.cursor()
    table = 'emporia'
    query = f"""INSERT INTO {table} (device_name, device_id, usage, created_at, ref_id) 
                                        VALUES (%s, %s, %s, %s, %s);"""
    cur.execute(query, (kwargs['device_name'], kwargs['device_id'], kwargs['usage'], kwargs['created_at'], kwargs['ref_id']))
    conn.commit()
    cur.close()
    conn.close()


def insert_logger_data(type, is_success, data, created_at):
    conn = get_connection()
    cur = conn.cursor()
    table = 'logger'
    query = f"""INSERT INTO {table} (type, is_success, data, created_at) 
                             VALUES (%s, %s, %s, %s);"""
    cur.execute(query, (type, is_success, data, created_at))
    conn.commit()
    cur.close()
    conn.close()


def make_unique(string):
    ident = uuid4().__str__()
    return f'{ident}-{string}'


def insert_emporia_device_data(**kwargs):
    conn = get_connection()
    cur = conn.cursor()
    query = f"""INSERT INTO emporia_device (device_name, device_id, zip_code, utility, created_at) 
                                        VALUES ('{kwargs['device_name']}', '{kwargs['device_id']}', '{kwargs['zip_code']}', '{kwargs['utility']}', '{kwargs['created_at']}');"""
    cur.execute(query)
    conn.commit()
    cur.close()
    conn.close()


def update_emporia_device_data(**kwargs):
    conn = get_connection()
    cur = conn.cursor()
    table = 'emporia_device'
    query = f"""UPDATE {table} SET device_name = '{kwargs['device_name']}'   Where device_id = '{kwargs['device_id']}'"""
    cur.execute(query)
    conn.commit()
    cur.close()
    conn.close()


def delete_emporia_device(**kwargs):
    conn = get_connection()
    cur = conn.cursor()
    query = f"""Delete from emporia_device where device_id='{kwargs['device_id']}';"""
    cur.execute(query)
    conn.commit()
    cur.close()
    conn.close()


def get_emporia_device():
    conn = get_connection()
    cur = conn.cursor()
    query = f"""SELECT * FROM emporia_device;"""
    cur.execute(query)
    rows = cur.fetchall()
    result_list = [row[1] for row in rows]
    conn.commit()
    cur.close()
    conn.close()
    return result_list


def get_emporia_device_zip():
    conn = get_connection()
    cur = conn.cursor()
    query = f"""SELECT * FROM emporia_device;"""
    cur.execute(query)
    rows = cur.fetchall()
    result_list = [row[2] for row in rows if row[2] is not None]
    conn.commit()
    cur.close()
    conn.close()
    return set(result_list)


def check_record_exist_or_not(**kwargs):
    conn = get_connection()
    cur = conn.cursor()
    query = f"""SELECT count(*) FROM weatherforecastnoaa where end_date='{kwargs['end_date']}' and zip_code = '{kwargs['zip_code']}';"""
    cur.execute(query)
    rows = cur.fetchall()
    result_list = rows[0][0]
    conn.commit()
    cur.close()
    conn.close()
    return result_list


def check_openweather_record_exist_or_not(**kwargs):
    conn = get_connection()
    cur = conn.cursor()
    query = f"""SELECT count(*) FROM weatherforecastnoaa where created_at='{kwargs['created_at']}' and zip_code = '{kwargs['zip_code']}';"""
    cur.execute(query)
    rows = cur.fetchall()
    result_list = rows[0][0]
    conn.commit()
    cur.close()
    conn.close()
    return result_list


def insert_user_data(**kwargs):
    conn = get_connection()
    cur = conn.cursor()
    query = f"""INSERT INTO users (name, email, password, is_admin) 
                                        VALUES ('{kwargs['name']}', '{kwargs['email']}', '{kwargs['password']}', '{kwargs['is_admin']}');"""
    cur.execute(query)
    conn.commit()
    cur.close()
    conn.close()


def check_user_record_exist_or_not(**kwargs):
    conn = get_connection()
    cur = conn.cursor()
    query = f"""SELECT count(*) FROM users where email='{kwargs['email']}';"""
    cur.execute(query)
    rows = cur.fetchall()
    result_list = rows[0][0]
    conn.commit()
    cur.close()
    conn.close()
    return result_list

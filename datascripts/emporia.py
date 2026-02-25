import datetime
import math
import os
import time
import grpc
from google.protobuf.json_format import MessageToDict
from helper import get_collection, insert_logger_data
from emporia_client import *

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))
except ImportError:
    pass

USERNAME = os.environ.get('EMPORIA_USERNAME', '')
PASSWORD = os.environ.get('EMPORIA_PASSWORD', '')
PARTNER_API_ENDPOINT = 'partner.emporiaenergy.com:50052'


def initialize():
    get_access()


def get_access():
    now = datetime.datetime.now()
    try:
        creds = grpc.ssl_channel_credentials()
        channel = grpc.secure_channel(PARTNER_API_ENDPOINT, creds)

        # client stub (blocking)
        stub = api.PartnerApiStub(channel)
        request = AuthenticationRequest()
        request.partner_email = USERNAME
        request.password = PASSWORD
        auth_response = stub.Authenticate(request=request)
        auth_token = auth_response.auth_token
        record_id = get_device_list(auth_token, stub)
        insert_logger_data('emporia', True, f'Data Loaded to mongoDB ({record_id})', now)
    except Exception as e:
        insert_logger_data('emporia', False, str(e), now)


def get_device_list(auth_token, stub):
    inventoryRequest = DeviceInventoryRequest()
    inventoryRequest.auth_token = auth_token
    inventoryResponse = stub.GetDevices(inventoryRequest)
    devices_dict = MessageToDict(inventoryResponse)
    dt = datetime.datetime.now()
    now = math.ceil(datetime.datetime.timestamp(dt.replace(second=0)))
    twenty_min_off = now - 1200
    start_time = twenty_min_off - 300
    end_time = twenty_min_off
    tmf_dt = datetime.datetime.fromtimestamp(start_time)
    for index, dev in enumerate(devices_dict['devices']):
        if dev['model'] == 'Vue2':
            deviceUsageRequest = DeviceUsageRequest()
            deviceUsageRequest.auth_token = auth_token
            deviceUsageRequest.start_epoch_seconds = start_time
            deviceUsageRequest.end_epoch_seconds = end_time
            deviceUsageRequest.scale = DataResolution.Minutes
            deviceUsageRequest.channels = DeviceUsageRequest.UsageChannel.MAINS
            deviceUsageRequest.manufacturer_device_ids.append(dev['manufacturerDeviceId'])
            usageResponse = stub.GetUsageData(deviceUsageRequest)
            uses_dict = MessageToDict(usageResponse)

            devices_dict['devices'][index]['usage'] = uses_dict
    return dump_data_to_mongodb(devices_dict, tmf_dt)


def dump_data_to_mongodb(data, dt):
    emporia_data = get_collection('emporia')
    record = {
        'data': data,
        'is_processed': False,
        'datetime': dt
    }
    record_id = emporia_data.insert_one(record).inserted_id
    print(record_id)
    return record_id

initialize()




import datetime
import math
import grpc
from google.protobuf.json_format import MessageToDict
from .emporia_client import *

PARTNER_API_ENDPOINT = 'partner.emporiaenergy.com:50052'


def get_emporia_data(username=None, password=None):
    now = datetime.datetime.now()
    try:
        creds = grpc.ssl_channel_credentials()
        channel = grpc.secure_channel(PARTNER_API_ENDPOINT, creds)

        # client stub (blocking)
        stub = api.PartnerApiStub(channel)
        request = AuthenticationRequest()
        request.partner_email = username
        request.password = password
        auth_response = stub.Authenticate(request=request)
        auth_token = auth_response.auth_token
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
            print(DeviceInventoryResponse.Device.DeviceModel.Vue2)
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
                for device_uses in uses_dict['deviceUsages']:
                    device_uses['bucketEpochSeconds'].pop(-1)
                    for i in device_uses['channelUsages']:
                        i['usages'].pop(-1)
                devices_dict['devices'][index]['usage'] = uses_dict
        return {'data': devices_dict, 'datetime': tmf_dt}
    except Exception as e:
        return {'error': str(e)}




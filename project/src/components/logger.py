import mycommons.mqtt_wrap as mqtt
import mycommons.data_structure as ds
import traceback
import logging

settings_file = 'settings.json'
allowed_stdout = False


def write_log(client, log:ds.Log):
    try:
        client.file_logs.write(str(log)+'\n')
        if allowed_stdout:
            client.logger.log(log.type, log.client + ' -> ' + log.result)
    except Exception:
        traceback.print_exc()


def init(client:mqtt.MQTTClient):
    client.file_logs = open('../logs/logs.txt', 'a+')

def end(client:mqtt.MQTTClient):
    if client and hasattr(client, 'file_logs'):
        client.file_logs.write('-----------------------------------------\n')
        client.file_logs.close()


def main(process='logger', id='0'):
    mqtt.MQTTClient(process_type=process, funct=write_log, init=init, end=end, id=id, type=mqtt.LOGGER)


if __name__=='__main__':
    main()





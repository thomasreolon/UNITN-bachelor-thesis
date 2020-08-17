import paho.mqtt.client as mqtt
from functools import wraps
import mycommons.data_structure as ds
import json
import traceback
import time
import logging
import os

log_level = logging.ERROR
FORMAT = '%(levelname)s: %(name)s -> %(message)s'
logging.basicConfig(format=FORMAT)

###################### SETTINGS #########################################

settings_file = os.path.dirname(os.path.realpath(__file__))+'/../settings.json'
publish_log = True

####################### SUPPORT FUNCTIONS ###############################
# what to print on successful connection
def _on_connect(client, userdata, flags, rc):
    outcome = ds.Log(
        timestamp=str(time.time()),
        client=client._client_id.decode("utf-8"),
        type=ds.CN,
        result=f"connected with result code {rc}"
    )

    if publish_log:
        client.publish(topic=client.log_topic, payload=str(outcome))
    client.logger.critical(outcome.result)


######################## general client #######################################

# parameters used to set which behaviour the client should have
BASE = 1
CLASSIFIER = 101
PREPROCESSOR = 102
STORAGE = 104
LOGGER = 107
FILTER = 111
CLUSTER = 200


# simple wrapper for mqtt client
class MQTTClient(mqtt.Client):
    """
    a mqtt client that will take care of message passing
    process_type -> the name to search in settings.json
    funct        -> function(client, userdata, msg) to call once you have the data
    init         -> called on initiation    params: client
    id           -> distingush from other components
    """

    def __init__(self, process_type: str, funct=None, init=None, end=None, id='0', type=CLASSIFIER):

        # try to get settings to startup, if can not shuts down
        with open(settings_file, 'r') as fin:
            settings = json.load(fin)

        # common info for all kinds of classifiers
        broker_host = settings['broker_host']
        comp = settings['components'][process_type]
        self.type = type
        self.logger = logging.getLogger(process_type)
        self.logger.setLevel(log_level)
        self.count, self.published = 0, 0


        # selecting input topic, logger vs others
        if type == LOGGER:                              # logger topic is global
            self.in_topics = [settings['log_topic']]
        else:
            self.log_topic = settings['log_topic']
            self.in_topics = comp['input_topics']
            n_input = len(self.in_topics)

        # storage and logger do not have on output topic
        if type == CLASSIFIER or type == PREPROCESSOR or type == FILTER or type == CLUSTER or type==BASE:
            self.out_topic = comp['output_topic']

        if type == CLASSIFIER or type == STORAGE or type == FILTER:
            # check if fields (what fields to take in  and send) exists
            req_u = comp['required_user'] if 'required_user' in comp else None
            req_a = comp['required_activity'] if 'required_activity' in comp else None
            if type == STORAGE:
                req_u, req_a, n_input = None, None, 1
            # used to coordinate/clean messages
            self.msg_handler = ds.UserMessageHandler(req_u, req_a, n_input)
            on_message = self.check_user_wrapper(funct)
        elif type == LOGGER:
            on_message = self.check_log_wrapper(funct)
        elif type == PREPROCESSOR:
            on_message = self.preprocessor_wrapper(funct)
        elif type==CLUSTER:
            on_message = self.cluster_wrapper(funct)
        elif type==BASE:
            on_message = self.base_wrapper(funct)
        else:
            on_message = lambda c, x, m: self.logger.debug(f'received: {m.payload}')  #

        super(MQTTClient, self).__init__(
            client_id=process_type + '-' + id,
            clean_session=False
        )
        if init and callable(init):
            init(self)
        self.on_connect = _on_connect
        self.on_message = on_message
        self.connect(broker_host)  # can't find broker --> errno 111

        # subscribe (qos==2 ==> exactly one message)
        for topic in self.in_topics:
            self.logger.debug('listening to (topic): '+topic)
            self.subscribe(topic=topic, qos=2)

        # start listening for messages
        try:
            self.loop_forever()
        except KeyboardInterrupt:
            self.logger.critical('closing...{}'.format(self._client_id.decode("utf-8")))
            if end:
                end(self)
            exit(0)


    def publish(self, topic, payload='{}', qos=0, retain=False, properties=None):
        if topic is None or payload is None:
            raise ValueError('called publish specifing topic/payload')
        return super().publish(topic, payload=payload, qos=qos, retain=retain, properties=properties)

    # automatically checks if user is ok and publish
    def pub(self, data, qos=1):
        try:
            if isinstance(data, ds.Message):
                self.publish(topic=self.out_topic, payload=str(data))
            elif isinstance(data, dict):
                self.publish(topic=self.out_topic, payload=json.dumps(data), qos=qos)
            # self.loop_stop()
        except Exception as e:
            print(e)
            traceback.print_exc()

    def check_user_wrapper(self, funct):
        @wraps(funct)
        def wrapper(cli, ud, msg):
            level = logging.DEBUG
            result = 'waiting for the next message'
            client = self._client_id.decode("utf-8")
            timestamp = str(time.time())
            user = None
            self.count += 1
            self.logger.debug('received a message')
            try:
                # ask to data structure to check if it is completed
                user, ok = self.msg_handler.handle(msg.payload)

                # if it is completed, call funct to process it
                # else: publish it as incomplete
                if user is not None:
                    if ok:  # user respect requirements
                        user_to_send, result = funct(self, user)

                        if user_to_send and (self.type==CLASSIFIER or self.type==FILTER):  # correctly processed
                            user_to_send.id, user_to_send.token, user_to_send.latest_activity = user.id, user.token, user.latest_activity
                            self.pub(user_to_send)
                            self.published += 1
                        elif self.type==FILTER or self.type==STORAGE:  # filter wants to drop or saved on db
                            level = logging.CRITICAL
                            if 'update' in result:
                                level = logging.INFO
                        else:
                            raise ds.UnableToProcessUser('classifier returned None')
                    elif self.type==STORAGE:
                        result=f'incompatible user for storage , {user}'
                    else:
                        raise ds.UnableToProcessUser(f'user {user.id} did not have the required fields')
            except ds.UnableToProcessUser as e:
                # if classifier was not able to process data
                level = logging.WARNING
                result = e.__repr__()
                if self.type == CLASSIFIER and user:
                    self.pub(ds.User(id=user.id, token=user.token))
            except Exception as e:
                # if errors occurred, publish the message as incomplete
                level = logging.ERROR
                result = f'failed calling {funct.__name__} -> ' + str(e) + '\n' + traceback.format_exc()
                if self.type == CLASSIFIER and user:
                    self.pub(ds.User(id=user.id, token=user.token))

            finally:
                result += f'({self.published}/{self.count})'
                # write the log: on stdout or /logs topic
                self.logger.log(level, result)
                if publish_log:
                    #### check connection
                    outcome = ds.Log(
                        client=client,
                        result=result,
                        type=level,
                        timestamp=timestamp
                    )
                    self.publish(topic=self.log_topic, payload=str(outcome))

        return wrapper

    def check_log_wrapper(self, funct):
        @wraps(funct)
        def wrapper(cli, ud, msg):
            try:
                myLog = ds.Log(msg.payload)
                funct(self, myLog)
            except Exception as e:
                self.logger.warning(traceback.format_exc())

        return wrapper

    # if the function fails, execute the except
    def preprocessor_wrapper(self, fun):
        @wraps(fun)
        def f2(cli, ud, msg):
            level = logging.DEBUG
            result = 'preprocessed succesfully'
            client = self._client_id.decode("utf-8")
            timestamp = str(time.time())
            try:
                # preprocessor
                users_list, result = fun(self, msg.payload)
                if isinstance(users_list, dict):
                    users_list = [v for k, v in users_list.items()]
                for user in users_list:
                    self.count += 1
                    if user.id and user.token and len(user.activities)>20 and user.latest_activity:  # check for minimal fields
                        self.pub(user)
                        self.published += 1
            except Exception as e:
                result = f'failed calling {fun.__name__} -> ' + str(e) + '\n' + traceback.format_exc()
                level = logging.ERROR
            finally:
                result += f'tried to process {self.count} users. Published {self.published}'
                self.logger.log(level, result)
                if publish_log:
                    outcome = ds.Log(
                        client=client,
                        result=result,
                        type=type,
                        timestamp=timestamp
                    )
                    self.publish(topic=self.log_topic, payload=str(outcome))
        return f2

    def cluster_wrapper(self, fun):
        @wraps(fun)
        def f2(cli, ud, msg):
            cli.logger.debug('received a message')
            level = logging.DEBUG
            result = 'success'
            client = self._client_id.decode("utf-8")
            timestamp = str(time.time())

            try:
                input = json.loads(msg.payload)
                input = ds.Cluster(input)
            except Exception:
                input = ds.Cluster(token=msg.payload.decode("utf-8"))

            try:
                out, result = fun(self, input)
                if out:
                    self.pub(out)
            except Exception as e:
                result = f'failed calling {fun.__name__} -> ' + str(e) + '\n' + traceback.format_exc()
                level = logging.ERROR
            finally:
                self.logger.log(level, result)
                if publish_log:
                    outcome = ds.Log(
                        client=client,
                        result=result,
                        type=type,
                        timestamp=timestamp
                    )
                    self.publish(topic=self.log_topic, payload=str(outcome))

        return f2

    def base_wrapper(self, fun):
        @wraps(fun)
        def f2(cli, ud, msg):
            cli.logger.debug('received a message')
            level = logging.DEBUG
            result = 'success'
            client = self._client_id.decode("utf-8")
            timestamp = str(time.time())

            try:
                _, result = fun(self, msg.payload)
            except Exception as e:
                result = f'failed calling {fun.__name__} -> ' + str(e) + '\n' + traceback.format_exc()
                level = logging.ERROR
            finally:
                exit(0)
                self.logger.log(level, result)
                if publish_log:
                    outcome = ds.Log(
                        client=client,
                        result=result,
                        type=type,
                        timestamp=timestamp
                    )
                    self.publish(topic=self.log_topic, payload=str(outcome))
        return f2




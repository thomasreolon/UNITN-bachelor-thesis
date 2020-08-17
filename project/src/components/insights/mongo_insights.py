import mycommons.mqtt_wrap as mqtt
import mycommons.data_structure as ds
import pymongo
import pymongo.errors as e
from threading import Thread, Event
import signal
import time

import requests
import tempfile
import webbrowser

host = 'localhost'
port = 27017
max_reconnect = 3
time_to_wait_for_automatic_clustering = 60*5

# threading sleep

def wrap(client):
    def set_exit_event(signo, _frame):
        client.exit_event.set()
        client.tr.join()
        client.logger.critical('closing...{}'.format(client._client_id.decode("utf-8")))
        exit(0)
    return set_exit_event

def ii(cli):
    cli.prev = {}

def evaluator(client, user:ds.User):

    res = 'ok'
    def ss(old, new, text):
        a = text.split(old)
        return new.join(a)

    if user.id in client.prev:
        n, old = client.prev[user.id]
        print('ALREADY', n, old.id)

        if n==2:
            link = 'https://twitter.com/' + old.username
            gender = 'M' if old.gender>0.5 else 'F'
            html = '<a href="link">link</a><h1 style="color:white">age</h1><h1  style="color:white">gen</h1>'
            html = ss('age', str(old.age), html)
            html = ss('gen', gender, html)
            html = ss('link', link, html)

            with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html') as f:
                url = 'file://' + f.name
                f.write(html)
            webbrowser.open(url)
            res=html
            del client.prev[old.id]
        else:
            old.load_json(user)
            client.prev[user.id] = (n+1, old)
    else:
        client.prev[user.id] = (1, user)
        print('-----> ADDED NEW ',user.id)

    if len(client.prev) > 10:
        n, old = client.prev[list(client.prev.keys())[0]]
        link = 'https://twitter.com/' + old.username
        gender = 'M' if old.gender > 0.5 else 'F'
        html = '<a href="link">link</a><h1 style="color:white">age</h1><h1  style="color:white">gen</h1>'
        html = ss('age', str(old.age), html)
        html = ss('gen', gender, html)
        html = ss('link', link, html)

        with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html') as f:
            url = 'file://' + f.name
            f.write(html)
        webbrowser.open(url)
        res = html
        del client.prev[old.id]



    return None, res



times_gotten = {}
class CheckPossibleClustering(Thread):
    def __init__(self, client):
        Thread.__init__(self)
        self.client = client
    def run(self):
        while self.client.acceso and not self.client.exit_event.is_set():
            self.client.exit_event.wait(60*10)
            to_del = []
            for token, last_updated in times_gotten.items():
                print(time.time()-last_updated, 'INSIGHTS-CHECk')
                if time.time()-last_updated > time_to_wait_for_automatic_clustering:
                    to_del.append(token)
                    self.client.publish('/cluster/token', token)
            for token in to_del:
                del times_gotten[token]




####################### FUNCTIONS #############################################

def init(client):
    client.users = pymongo.MongoClient(host, port)['personas']['users']
    client.insights_to_save = ['interests','platform', 'mbti', 'ocean', 'needs', 'img_topics', 'nlu', 'age', 'gender', 'language']
    client.acceso = True
    client.exit_event = Event()
    set_exit_event = wrap(client)
    signal.signal(signal.SIGINT, set_exit_event)
    client.tr = CheckPossibleClustering(client)
    client.tr.start()

def process_user(client:mqtt.MQTTClient, data:ds.User):
    user = {
        '_id': data.id,
        'token': data.token,
        'latest_activity': data.latest_activity,
        'platform': data.platform,
        'averaged':False
    }
    times_gotten[data.token] = time.time()


    res = 'insert'
    reconnect=0
    for k in client.insights_to_save:
        if k in data.__dict__:
            user[k] = data.__dict__[k]

    try:
        client.users.insert_one(user)                # create
    except e.AutoReconnect:
        reconnect+=1
        if reconnect>=max_reconnect:
            raise ConnectionError(f'CONN ERROR: not able to save user:{user}')
    except e.DuplicateKeyError:
        res = 'update'
        my_query = {"_id": data.id}
        old = client.users.find(my_query)[0]
        if old['token'] == user['token']:
            for f in old:
                if f in user and user[f] is None:
                    user[f] = old[f]
            del user['_id']
            client.users.update_one(my_query, {"$set":user})
        else:
            raise ValueError("same id but different token")
    return None, f"{res} user in the db:{data.id}"


def main(process_type='mongo_insights', id='0'):
    mqtt.MQTTClient(process_type=process_type, funct=process_user, init=init,id=id, type=mqtt.STORAGE)


if __name__ == '__main__':
    main()

import mycommons.mqtt_wrap as mqtt
import mycommons.data_structure as ds
import json
import requests
import pymongo
from whoosh.index import create_in
from whoosh.fields import *
from whoosh.scoring import BM25F
from whoosh import qparser
from threading import Thread, Event
import signal
import time

index_folder =  '../../../models' if __name__=='__main__' else '../models'
categ_path = '../../../models/tapoi.json' if __name__=='__main__' else '../models/tapoi.json'
cred_path = '../../../credentials/tapoi.json' if __name__=='__main__' else '../credentials/tapoi.json'
to_process = {}
verify = True

# threading sleep
def wrap(client):
    def set_exit_event(signo, _frame):
        client.exit_event.set()
        client.tr.join()
        client.logger.critical('closing...{}'.format(client._client_id.decode("utf-8")))
        exit(0)
    return set_exit_event


class WaitForTapoi(Thread):
    def __init__(self, client):
        Thread.__init__(self)
        self.client = client

    def run(self):
        self.client.exit_event.wait(120)
        while not self.client.exit_event.is_set():
            for resource_id, data in list(to_process.items()):
                remain, user = data
                if remain > 0:
                    try:
                        response = get_insights(resource_id, self.client.host, self.client.default_headers,
                                                self.client.instance)
                        entities = response['entity-map']['data']['occMap']
                        interests = {x: 0 for x in self.client.categ}
                        pages_found = 0
                        i = 1
                        while len(entities) > 20:
                            i += 1
                            keys = list(entities.keys())
                            for k in keys:
                                if entities[k] < i:
                                    del entities[k]

                        for ent, num in entities.items():
                            #self.client.exit_event.wait(1)  #### try not to overcharge wikipedia
                            text_wiki = get_text(ent)
                            if text_wiki:
                                pages_found += 1
                                score_categories = search(self.client, text_wiki)[:2]
                                for cat, score in score_categories:
                                    interests[cat] += score * num
                        for k in interests.keys():
                            interests[k] /= (pages_found + 1)
                        interests = {k: v for k, v in sorted(interests.items(), key=lambda item: 100 / (item[1] + 1))}

                        if hasattr(self.client, 'logger'):
                            self.client.logger.debug(f'interests found using tapoi: {interests} for https://twitter.com/{user.username}')
                        else:
                            print(f'////////////////////\n/////////////////\ninterests found ---> {interests}')

                        interests = norm_interests(interests)
                        self.client.pub(
                            ds.User(interests=interests, latest_activity=user.latest_activity, id=user.id,
                                    token=user.token))

                        del to_process[resource_id]
                    except Exception as e:
                        print(f'FAILED {user.real_id} {user.username} {e}')
                        to_process[resource_id] = (remain-1, user)
                else:
                    del to_process[resource_id]
            self.client.exit_event.wait(60*12)


def get_asset(foreign_id, host, headers, instance):
    url = host+'/instance/{}/asset/'.format(instance)
    content = {
        "foreignId": str(foreign_id),
        "tags": [],
        "metadata": {}
    }

    res_asset = requests.post(url, data=json.dumps(content), headers=headers, verify=verify)

    res_asset = json.loads(res_asset.content)
    if 'assetId' in res_asset:
        asset_id = res_asset['assetId']
    else:
        asset_id = res_asset['message'].split('[')[1].split('}]')[0]
    return asset_id

def get_resource(asset_id, foreign_id, host, headers, instance):
    url = host + '/instance/{}/asset/{}/resource/'.format(instance, asset_id)
    response = requests.get(url, headers=headers, verify=verify).text
    response = json.loads(response)
    for resource in response:
        if 'twitter' in resource['resourceType']:
            return resource['resourceId']

    # create resource
    url = 'https://core.tapoi.me/v1/instance/{}/asset/{}/resource/'.format(instance, asset_id)

    content = [
        {
            "type": "twitter.user",
            "foreignId": str(foreign_id),
            "properties": {}
        }
    ]
    res_resource = requests.post(url, data=json.dumps(content), headers=headers, verify=verify)
    resource_id = json.loads(res_resource.text)['successful'][0]['resourceId']
    collect(asset_id, host, headers, instance)
    return resource_id

def collect(asset_id, host, headers, instance):
    # collect data
    url = host + '/instance/{}/asset/{}/collect/'.format(instance,asset_id)
    requests.post(url, data={}, headers=headers, verify=verify)

def get_insights(resource_id, host, headers, instance):
    # analyze data - topics
    analisis = ['entity-map']#'activity-count']#, 'topic-map', 'topicWithEntity-map']
    results = {}

    for a in analisis:
        url = host + f'/computation/{a}/'
        content = {
            "period": {"type":"allTime"},
            "target": {
                "type": "resource",
                "resourceId": resource_id,
                "instanceId": instance,
                'unify': True

            },
            "profile": 'all-activities'
        }
        r = json.loads(requests.post(url, data=json.dumps(content), headers=headers, verify=verify).content)

        results['resourceId'] = resource_id
        results['twitter_username'] = 'toagodfather'
        if 'data' in r:
            results[a] = r

    return results



def process(client, user:ds.User):
    # check if exists
    interests = None
    lang = user.language if isinstance(user.language, str) else sorted(user.language.items(), key=lambda item: 100/(item[1]+1))[0][0]
    if lang in ['it', 'en', 'un']:
        try:
            asset_id = get_asset(user.real_id, client.host, client.default_headers, client.instance)
            resource_id = get_resource(asset_id, user.real_id, client.host, client.default_headers, client.instance)
            if not client.mongo_tapoi.find_one({'_id':user.real_id}):
                client.mongo_tapoi.insert_one({'_id':user.real_id, 'assetId':asset_id})

            to_process[resource_id] = (5, user)

        except Exception as e:
            import traceback
            traceback.print_exc()
            if hasattr(client, 'logger'):
                client.logger.error(e)
            else:
                print(e)
            interests = None

    if not interests:
        text = [act['text'] for act in user.activities]
        text = ' '.join(text)[:5000]
        score_categories = search(client, text)
        interests={x:0 for x in client.categ}
        for cat, score in score_categories:
            interests[cat] = score
        tmp = search(client, user.description)
        if len(tmp)>0:
            cc, ss = tmp[0]
            interests[cc] = max(interests[cc], ss)
        interests = norm_interests(interests)

    return ds.User(interests=interests), f'for https://twitter.com/{user.username} found interests:{interests}'



def get_text(link):
    if 'en' not in link:
        page = requests.get(link).text
        link = page.split('lang="en" hreflang="en" class="interlanguage-link-target">')[0]
        link = link.split('href="')
        link = link[len(link)-1]
        link = link.split('" title')[0]

    try:
        cat = link.split('wiki/')[1]
        url = f'https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&exintro&explaintext&redirects=1&titles={cat}'
        res = json.loads(requests.get(url).text)
        x = res['query']['pages']
        x = x[list(x.keys())[0]]['extract']
    except:
        return None
    return x

def norm_interests(interests):
    tot= sum([v for k,v in interests.items()]) + 0.1
    for k in interests:
        interests[k] = interests[k]/tot
    return interests

def search(client, query):
    q = client.parser.parse(query)
    results = client.searcher.search(q)
    r, tot = [], 0
    for res in results:
        tot += res.score
    for res in results:
        r.append((res['title'], res.score/(tot+1)))
    return r

def init(client):
    with open(cred_path, 'r') as credentials_file:
        cred = json.load(credentials_file)
    client.default_headers = {
        'X-ApiKey': cred['X-ApiKey'],
        'X-ApiSecret': cred['X-ApiSecret'],
        'Content-Type': 'application/json',
    }
    client.host = cred['host']
    client.instance = cred['instance']
    client.mongo_tapoi = pymongo.MongoClient('localhost', 27017)['personas']['tapoi']
    init_whoosh(client)

    client.exit_event = Event()
    set_exit = wrap(client)
    client.tr = WaitForTapoi(client)
    signal.signal(signal.SIGINT, set_exit)
    client.tr.start()


def init_whoosh(client):
    with open(categ_path, 'r') as fin:
        categories = json.load(fin)

    schema = Schema(title=TEXT(stored=True), content=TEXT)
    ix = create_in(index_folder, schema)
    writer = ix.writer()

    for category, document in categories.items():
        writer.add_document(title=category, content=document)
    writer.commit()
    client.searcher = ix.searcher(weighting=BM25F(B=1, K1=1.2))
    client.categ = categories.keys()

    # query con or tra ogni parola
    client.parser = qparser.QueryParser(
        "content", schema, group=qparser.OrGroup.factory(0.9))

def main(name, id='0'):
    mqtt.MQTTClient(name, funct=process, init=init, id=id, type=mqtt.CLASSIFIER)

if __name__ == '__main__':
    cli = lambda: None
    init(cli)

    process(cli, ds.User(real_id=175277917, language={'en': 1}))
    process(cli, ds.User(real_id=1274523421, language={'en':1}))

    time.sleep(60*30)
    #x, res = process(cli, ds.User(real_id=577518356, language={'en':1}))
    # x, res = process(cli, ds.User(real_id=144210805, language={'it':1}))
    # x, res = process(cli, ds.User(real_id=1446281586, language={'it':1}))
    #print(res)

    # with open(categ_path, 'r') as fin:
    #     categories = json.load(fin)
    # for cat in categories:
    #     '_'.join(cat.split(' '))
    #     t = get_text('https://en.wikipedia.org/wiki/'+cat)
    #     if t:
    #         print(t)
    #         categories[cat] = t
    # json.dump(categories, open(categ_path, 'w'), indent=2)

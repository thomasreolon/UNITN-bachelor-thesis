import requests
import json
import mycommons.mqtt_wrap as mqtt
import mycommons.data_structure as ds

credentials_file = '../credentials/ibm.json'
classes_json = '../models/nlu_classes.json'
ibm_service = 'nlu'
headers = {'Content-Type': 'application/json'}


def init(client:mqtt.MQTTClient):
    with open(credentials_file, 'r') as fin:
        credentials = json.load(fin)[ibm_service]
    key = 'apikey:{}'.format(credentials['key'])
    client.url = 'https://{}@{}/v1/analyze?version=2019-07-12'.format(key, credentials['url'])

    with open(classes_json, 'r') as fin:
        cls = json.load(fin)
    tmp = {}
    for k,v in cls.items():
        tmp[k] = set(v)
    client.categories = tmp

def process_nlu(client:mqtt.MQTTClient, user:ds.User):
    # get all posts in english
    if user.nlu is not None:
        raise ds.UnableToProcessUser('already completed')
    posts, crashed = [], False
    for post in user.activities:
        if post.text:
            posts.append(post.text)

    data_str = json.dumps({
        "text": ".\n".join(posts)[:15000],
        "features": {
            "categories": {"limit": 15}
        }
    })

    if len(user.activities)>20:
        r = requests.post(client.url, headers=headers, data=data_str)
        res = json.loads(r.content)
        tot = 1

        categories = {k:0 for k in client.categories}
        if 'categories' in res:
            cc = [c['label'] for c in res['categories']]
            for c in cc:
                for cat in c.split('/'):
                    for name, modelcat in client.categories.items():
                        if cat in modelcat:
                            categories[name] += 1
                            tot+=1
            for k in categories:
                categories[k] /= tot
    else:
        raise ds.UnableToProcessUser(f'not enough posts for user:{user.id}')

    return ds.User(nlu={'categories':categories}), f"processed text using ibm for:{user.id}"


def main(process_type='ibm_nlu', id='0'):
    mqtt.MQTTClient(process_type=process_type, funct=process_nlu, init=init, id=id, type=mqtt.CLASSIFIER)


if __name__ == '__main__':
    main()

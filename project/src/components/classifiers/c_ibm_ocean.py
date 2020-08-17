import requests
import json
import mycommons.mqtt_wrap as mqtt
import mycommons.data_structure as ds

credentials_file = '../credentials/ibm.json'
ibm_service = 'ocean'
headers = {'Content-Type': 'text/plain;charset=utf-8', 'Accept': 'application/json'}


def init(client:mqtt.MQTTClient):
    with open(credentials_file, 'r') as fin:
        credentials = json.load(fin)[ibm_service]
    key = 'apikey:{}'.format(credentials['key'])
    client.url = 'https://{}@{}/v3/profile?version=2017-10-13'.format(key, credentials['url'])


def process_ocean(client:mqtt.MQTTClient, user:ds.User):
    if user.ocean is not None:
        raise ds.UnableToProcessUser('had already a value')

    # get all posts in english
    posts, is_eng, crashed = [], False, False
    if user.language == 'en':
        is_eng = True
    for post in user.activities:
        if post.text and (is_eng or (post.language == 'en')):
            posts.append(post['text'])

    # if there are enough posts (at least 600 words)
    # group them in a string and call the API
    ocean = {}
    needs = {}
    result = f'processing ocean using ibm API for:{user.id}'
    if len(posts) > 20:
        doc = '.\n'.join(posts).encode('utf-8')
        r = requests.post(client.url, data=doc, headers=headers)

        res = json.loads(r.content)
        client.logger.debug(res)
        for trait in res['personality']:
            if 'open' in trait['trait_id']:
                ocean['O'] = trait['percentile']
            if 'cons' in trait['trait_id']:
                ocean['C'] = trait['percentile']
            if 'extra' in trait['trait_id']:
                ocean['E'] = trait['percentile']
            if 'agree' in trait['trait_id']:
                ocean['A'] = trait['percentile']
            if 'neuro' in trait['trait_id']:
                ocean['N'] = trait['percentile']

        try:
            needs['conservative'] = res['values'][0]['percentile']
            needs['need_expression'] = res['needs'][9]['percentile']
            needs['need_love'] = res['needs'][8]['percentile']
            needs['anger'] = res['personality'][4]['children'][0]['percentile']
            needs['depression'] = res['personality'][4]['children'][2]['percentile']
            needs['stress'] = res['personality'][4]['children'][5]['percentile']
            needs['intellect'] = res['personality'][0]['children'][4]['percentile']
        except Exception as e:
            result = 'error in calculating needs' + str(e)
            needs=None
    else:
        raise ds.UnableToProcessUser(f'not enough posts for {user.id}')

    # if everything went well, return the user to send + the log result
    return ds.User(ocean=ocean, needs=needs), result


def main(process_type='ibm_ocean',id='0'):
    mqtt.MQTTClient(process_type=process_type, funct=process_ocean, init=init,id=id, type=mqtt.CLASSIFIER)


if __name__ == '__main__':
    main()

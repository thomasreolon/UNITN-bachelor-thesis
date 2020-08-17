import requests
import mycommons.mqtt_wrap as mqtt
import mycommons.data_structure as ds
import json

credentials_file = '../credentials/ibm.json'
ibm_service = 'images'
max_images = 1

def init(client:mqtt.MQTTClient):
    with open(credentials_file, 'r') as fin:
        credentials = json.load(fin)[ibm_service]
    client.key = 'apikey:{}'.format(credentials['key'])
    client.instance = credentials['url']


def process_imgs(client:mqtt.MQTTClient, user:ds.User):
    if user.img_topics is not None:
        raise ds.UnableToProcessUser('already completed')

    # get a unique list of images' urls
    images = []
    for act in user.activities:
        images += act.images

    # for each image (until max_images): get info from API
    topics, count = [], 0
    for image in images:
        count += 1
        url = "https://{}@gateway.watsonplatform.net/visual-recognition/api/v3/classify?url={}&version=2018-03-19".format(client.key, image)
        r = requests.get(url)
        res = json.loads(r.content)
        for cl in res['images'][0]['classifiers'][0]['classes']:
            topics.append(cl['class'])
        if count >= max_images:
            break

    # if got no topics, set error, than send
    if len(topics) == 0:
        raise ds.UnableToProcessUser('did not found images')

    return ds.User(img_topics=topics), f"sucessfully called IBM API to compute photos for:{user.id}"


# test API
def main(process_type='ibm_vision', id='0'):
    mqtt.MQTTClient(process_type=process_type, funct=process_imgs, init=init, id=id, type=mqtt.CLASSIFIER)


if __name__ == '__main__':
    main()


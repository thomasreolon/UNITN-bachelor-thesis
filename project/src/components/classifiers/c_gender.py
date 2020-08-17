import mycommons.mqtt_wrap as mqtt
import mycommons.data_structure as ds
import json

names_file = '../models/names_and_genders.json'
tr = 0.9 # treshold of previous age classifier (m3inf)

def init(client:mqtt.MQTTClient):
    with open(names_file, 'r') as fin:
        client.names = json.load(fin)

def process_gender(client:mqtt.MQTTClient, user:ds.User):
    gender = None

    names = [user.name]
    if user.username:
        names.append(user.username)

    if user.gender and user.gender > 0.3 and user.gender < 0.7:
        for name in names:
            parts = []
            for i, c in enumerate(name):
                if ord(c) < 90:
                    parts += [name[:i].lower(), name[i:].lower()]
            name = name.lower()
            parts += name.split('.') + name.split(' ') + name.split('_')
            for part in parts:
                if part in client.names:
                    gender = 1. if client.names[part]=='M' else 0.
                    cause = f'{part} is a name of a {client.names[part]}'
                    break
    elif isinstance(user.gender, float):
        gender = user.gender
        cause = 'm3 inference'

    if not gender:
        raise ds.UnableToProcessUser('could not find a gender')

    return ds.User(gender=gender), f" [{gender}|{cause}] https://twitter.com/{user.username} |({user.gender})"


def main(settings, id='0'):
    mqtt.MQTTClient(process_type=settings, funct=process_gender, init=init, id=id, type=mqtt.CLASSIFIER)




if __name__ == '__main__':
    client = lambda :None
    user={"profile":{"name":"Luca Casagrande"}, "insights":{}}
    init(client)
    process_gender(client, user)
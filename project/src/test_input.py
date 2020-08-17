import paho.mqtt.client as mqtt
import json
import random
from time import sleep

raw = '/raw_data'
filename = '../data/NutellaUSA.json'
fromid = 0

def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))

if __name__ == '__main__':
    client = mqtt.Client()
    client.on_connect = on_connect
    client.connect('localhost')
    list = True

    print("waiting for command:\n-> insights <filename> <num>\n-> cluster <token>")
    while list:
        accounts = ['economicsfest', 'ilfestivalsport', 'WishShopping', 'zalando_uk', 'patmcgrathreal', 'theffballers',]
        mx = 1
        cmd = input().split(' ')
        if cmd[0]=='insights':
            for par in cmd[1:]:
                try:
                    mx = int(par)
                except ValueError:
                    accounts.append(par)
            accounts = [x for x in reversed(accounts)][:mx]

            count = 0
            for x in accounts:
                with open(f'../data/{x}.json', 'r') as tw_file:
                    profiles = json.load(tw_file)
                for prof in profiles:
                    client.publish(raw, payload=json.dumps(prof), retain=False, qos=0)
                    count +=1
                    # if count==100:
                    #     exit(0)

                    print(f'published-{count}: {prof["data_profile"][0]["screen_name"]}')

                sleep(60*7)
        elif cmd[0]=='cluster':
            topic = '/cluster/token'
            client.publish(topic, payload=cmd[1], retain=False, qos=1)
            print(f'published-token: {cmd[1]}  for clustering (on {topic})')
        elif cmd[0]=='collect':
            topic = '/tw_collector'
            client.publish(topic, payload=json.dumps({'token':cmd[1]}), retain=False, qos=1)
            print(f'published-token: {cmd[1]}  for collection etc. (on {topic})')
        else:
            list=False
            print('closing')


import mycommons.data_structure as ds
import mycommons.mqtt_wrap as mqtt
import tempfile
import webbrowser
import random
import os
from datetime import date
import pymongo


template_file = '../models/dashboard.html'
summary_file = '../models/personas_summary_onefile.html'
photos = os.path.dirname(os.path.realpath(__file__))+'/../../../photos/'

if __name__=='__main__':
    template_file = '../../' + template_file
    summary_file = '../../' + summary_file

class Averageator():
    def __init__(self, onlypos=True):
        self.onlypos=onlypos
    def add(self, other, me=None):
        if hasattr(other, '__dict__'):
            other = other.__dict__
        if not me:
            me=self.__dict__
        elif hasattr(me, '__dict__'):
            me = me.__dict__

        for k, v in other.items():
            if k not in me or not me[k]:
                me[k] = v
                me[str(k)+'count'] = 1
                if isinstance(v, dict):
                    self.settoone(me[k])
            elif isinstance(v, (int, float)) and (v>=0 or not self.onlypos):
                v = max(0,v)
                me[k] += v
                me[str(k) + 'count'] += 1
            elif isinstance(v, dict):
                self.add(v, me[k])
    def settoone(self, dic:dict):
        toadd = []
        for k, v in dic.items():
            toadd.append(str(k) + 'count')
            if isinstance(v, dict):
                self.settoone(v)
        for k in toadd:
            dic[k]=1

    def __repr__(self):
        new = {'count':0}
        todo = [self.__dict__]
        while len(todo)>0:
            tmp = todo[0]
            del todo[0]
            for k,v in tmp.items():
                if isinstance(v, (int,float)):
                    if 'count' in k:
                        new['count'] = new['count'] if new['count']>tmp[k] else tmp[k]
                    else:
                        new[k] = v
                elif isinstance(v, dict):
                    todo.append(v)
        tot = new['count']
        for k,v in new.items():
            new[k] = v/tot
        del new['count']
        return str(new)


def save_insights(client:mqtt.MQTTClient, cluster:ds.Cluster):
    client.logger.debug('saving ' + cluster.type)
    rr='ok'
    with open('/tmp/clsave'+cluster.token+cluster.type[:7], 'w') as f:
        f.write(str(cluster))
        rr = 'done'


    return None, rr


def open_html(html):
    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html') as f:
        url = 'file://' + f.name
        f.write(html)
    webbrowser.open(url)

def average(list_of_insights):
    avg = Averageator()
    for ins in list_of_insights:
        avg.add(ins)

    final = {}
    ####### DEMOS #########
    final['age'] = int(avg.age) // avg.agecount
    final['gender'] = 'Male' if avg.gender / avg.gendercount > 0.5 else 'Female'
    final['rel'] = len(list_of_insights)

    ###### PERSONALITY ######
    E = min(1,avg.mbti['E'] / avg.mbti['Ecount'])
    S = min(1,avg.mbti['S'] / avg.mbti['Scount'])
    T = min(1,avg.mbti['T'] / avg.mbti['Tcount'])
    J = min(1,avg.mbti['J'] / avg.mbti['Jcount'])

    final['ext'] = E
    final['int'] = S
    final['thi'] = T
    final['thi'] = J
    E = 'E' if E>0.5 else 'I'
    S = 'S' if S>0.5 else 'N'
    T = 'T' if T>0.5 else 'F'
    J = 'J' if J>0.5 else 'P'
    final['personality'] = personalityToAdjective[E+S+T+J]

    p, l = 0,'undefined'
    for k,v in avg.language.items():
        if v>p and 'count' not in k:
            p=v
            l=k
    final['language'] = l

    ###### INTERESTS #######
    # indoor = avg.nlu['categories']['indoor'] / avg.nlu['categories']['indoorcount']
    # outdoor = avg.nlu['categories']['outdoor'] / avg.nlu['categories']['outdoorcount']
    # final['activities'] = 'Indoor' if indoor>outdoor else 'Outdoor'

    topics = {}
    for d in [avg.interests, avg.img_topics]:
        for k, v in d.items():
            if 'count' not in k:
                topics[k] = v / d[str(k)+'count']
    topics = list(topics.items())
    topics.sort(key=lambda x:10/(x[1]+2))

    final['interests'] = [x[0] for x in topics][:4]
    final['money'] = final['age']//10

    poss = []
    av = final['gender'][0] + str(final['age']//10)
    for pic in avatars:
        if av in pic:
            poss.append(pic)

    if final['language'] == 'it':
        final['name'] = random.choice(names[final['gender'][0]+'it'])
    else:
        final['name'] = random.choice(names[final['gender'][0]])
    if final['language'] in surnames:
        final['surname'] = random.choice(surnames[final['language']])
    else:
        final['surname'] = random.choice(surnames['un'])

    return final



def upper_name(name):
    return name[0].upper() + name[1:].lower()

def personas_to_dict(final:dict,tot, client):
    pers = {}
    pers['pic-src'] = get_src(client, {'type':'face', 'age':final['age'], 'gender':final['gender']})
    pers['name'] = upper_name(final['name']) + ' ' + upper_name(final['surname'])
    pers['money'] = final['money']*'$'
    pers['rel'] = int(final['rel']/tot*100)
    pers['ext'] = int(final['ext']*100)
    pers['int'] = int(final['int']*100)
    pers['thi'] = int(final['thi']*100)
    pers['mbti'] = final['personality']
    pers['age'] = str(final['age'])
    pers['gender'] = final['gender']
    pers['lang'] = final['language'].upper()
    pers['i1'] = final['interests'][0]
    pers['src-i1'] = get_src(client, {'type':'interests', 'name':final['interests'][0]})
    pers['i2'] = final['interests'][1]
    pers['src-i2'] = get_src(client, {'type':'interests', 'name':final['interests'][1]})
    pers['i3'] = final['interests'][2]
    pers['src-i3'] = get_src(client, {'type':'interests', 'name':final['interests'][2]})

    return pers

def sub(old, new, string):
    a = string.split(old)
    return str(new).join(a)

personalityToAdjective={
    "ISTJ": "The Inspector",
    "ISTP": "The Crafter",
    "ISFJ": "The Protector",
    "ISFP": "The Artist",
    "INFJ": "The Advocate",
    "INFP": "The Mediator",
    "INTJ": "The Architect",
    "INTP": "The Thinker",
    "ESTP": "The Persuader",
    "ESTJ": "The Director",
    "ESFP": "The Performer",
    "ESFJ": "The Caregiver",
    "ENFP": "The Champion",
    "ENFJ": "The Giver",
    "ENTP": "The Debater",
    "ENTJ": "The Commander",
}
avatars = {'M0-1.jpg', 'M1-1.jpg', 'M2-1.jpg', 'M2-2.jpg', 'M2-3.jpg', 'M2-4.jpg','M3-1.jpg', 'M3-2.jpg', 'M3-3.jpg', 'M3-4.jpg', 'M4-1.jpg','M4-2.jpg', 'M5-1.jpg', 'M5-2.jpg', 'M5-3.jpg', 'M5-4.jpg', 'F0-1.jpg','F0-2.jpg', 'F1-1.jpg','F1-2.jpg','F2-1.jpg','F2-2.jpg','F3-1.jpg','F3-2.jpg','F4-1.jpeg','F4-2.jpeg','F5-1.jpg','F5-2.jpg',}
src_images={
    'animals':'animal.jpg',
    'music':'musig.jpg',
    'hospital':'hospital.jpg',
    'war':'war.jpeg',
    'clothes':'clothes.jpeg',
    'sport':'sport.jpg',
    'alcohol':'beer.jpg',
    'cars':'rich.png',
    'costumes':'cowboyhat.jpg',
    'office':'coffe.jpeg',
    'holiday':'holyyday.jpg',
    'poor':'poor.jpeg',
    'rich':'poor.jpeg',
    'baby':'baby.png',
    'tech':'tech.jpg',
    'family':'family.png',
    'politics':'politics.jpeg',
    'electronics':'Electronics.jpeg',
    'university': 'university.jpg',
    'animation' : 'animation.jpeg',
    'american football':'afoot.png',
    'soccer':'soccer.jpeg',
    'sports':'sport.jpg',
    'dance':'dance.jpeg',
    'activism':'activism.jpeg',
    'nature':'nature.jpeg',
    'Food':'food.jpeg',
    'basketball':'basket.jpeg',
    'religion':'religion.png',
    'video games':'videogames.jpeg',
    'artificial_intelligence':'ai.jpeg',
    'journalism':'journalism.jpeg',
    'design':'design.jpeg',
}
names = {
    'Mit':['andrea', 'antonino', 'bartolomeo', 'cosmo', 'faustino', 'matteo', 'luca', 'simone', 'samuel', 'giovanni','alessandro'],
    'Fit':['sara','alessia','eleonora','elena','sabrina','simaona','margherita','daniela','lucia','margaret','susanna','marica'],
    'M':['james', 'john', 'robert', 'michael', 'william', 'david', 'richard', 'joseph', 'thomas', 'charles', 'christopher', 'daniel', 'matthew'],
    'F':['marijayne', 'myaa', 'mary', 'patricia', 'jennifer', 'linda', 'elisabeth', 'barbara', 'susan', 'jessica', 'sarah', 'karen', 'nancy', 'margaret', 'lisa', 'betty']
}
surnames = {
    'en':['smith', 'jhonson', 'garcia', 'lopez', 'miller', 'browns'],
    'it':['rossi', 'russo', 'ferrari', 'esposito', 'bianchi', 'romano'],
    'fr':['martin','bernard','dubois','robert','laurent'],
    'sp':['muller','schmidt','schneider','fischer'],
    'un':['smith', 'da silva', 'ali', 'martin']
}

class CheckTooSimilar():
    def __init__(self):
        self.finals = []
    def __iter__(self):
        return self.finals.__iter__()
    def add(self, new):
        ok = True
        for old in self.finals:
            if new['language']==old['language'] and abs(new['age']-old['age'])<3 and new['gender']==old['gender'] and any([x in old['interests'] for x in new['interests']]):
                ok = False
        if ok:
            self.finals.append(new)
        else:
            old['rel'] += new['rel']


def get_avatar(age, gen):
    if gen == 'Male':
        if age<22:
            return 'mm2.jpg'
        elif age<35:
            return 'mm3.jpg'
        else:
            return 'mm4.webp'
    else:
        if age < 22:
            return 'ff2.jpeg'
        elif age < 35:
            return 'ff3.jpg'
        else:
            return 'ff4.jpeg'

def process_cluster(client:mqtt.MQTTClient, cluster:ds.Cluster):
    to_examinate = 0
    for clus in cluster.cluster:
        to_examinate+=len(clus)
    tot = to_examinate
    to_examinate = to_examinate // 2
    max_personas=4

    personas = CheckTooSimilar()
    every = Averageator() #########

    for clus in cluster.cluster:
        for x in clus:  #######
            every.add(x) #########
        final = average(clus)
        personas.add(final)
        to_examinate -= len(clus)
        if to_examinate < 0 or max_personas <= len(personas.finals):
            break

    to_save = {}
    to_save['data'] = str(date.today())
    to_save['token'] = cluster.token
    to_save['avg_age'] = int(every.age / every.agecount)
    to_save['avg_male'] = int(every.gender*100 / every.gendercount)
    to_save['personas'] = []
    to_save['type'] = cluster.type

    for final in personas.finals:
        to_save['personas'].append(personas_to_dict(final, tot, client))

    res = client.personas.find_one({'token':to_save['token'], 'data':to_save['data']})
    if not res or res['type']!=to_save['type']:
        client.personas.insert_one(to_save)
        client.logger.critical('added a personas group')
    else:
        client.personas.delete_one(res)
        client.personas.insert_one(to_save)
        client.logger.critical('updated a personas group')


    load_personas_html(client, cluster.token)

    return None, 'finished'


def load_personas_html(client, token):
    with open(template_file, 'r') as fin:
        html = fin.read()

    all_personas = []
    cursor = client.personas.find({'token':token})
    for pers in cursor:
        all_personas.append(pers)
    all_personas.reverse()

    # sidebar
    links = html.split('<!-- Persona Link -->')
    link = links[1]
    block = [links[0]]
    ages, dates  =[], []
    for i, p in enumerate(all_personas):
        ages.append(p['avg_age'])
        dates.append(p['data'])
        id = '#l' + chr(ord('a') + i % 20)
        tmp = sub('##id##', f'{id}',link)
        tmp = sub('##data##', p['data'], tmp)
        block.append(tmp)
    block.append(links[2])
    html = ' '.join(block)

    # group of personas per date
    links = html.split('<!-- Persona Title -->')
    title = links[1]                                  # Title
    personas = links[2].split('<!-- Personas -->')
    end = personas[2]                                 # Footer
    personas = personas[1]                             # Persona
    block = [links[0]]
    for i, p in enumerate(all_personas):
        id = 'l' + chr(ord('a') + i%20)
        tmp = sub('##id##', f'{id}', title)
        tmp = sub('##data##', p['data'], tmp)
        tmp = sub('##type##', '', tmp)

        block2 = [tmp]
        for pers in p['personas']:
            tmp = sub('##name##', pers['name'], personas)
            del pers['name']
            for field in pers:
                tmp = sub(f'##{field}##', pers[field], tmp)
            rand_href = ''.join(random.choice(['a','b','c','d','e','f']) for i in range(5))
            tmp = sub('personaCollapse', rand_href, tmp)

            block2.append(tmp)
        block.append(' '.join(block2))
    block.append(end)

    html = ' '.join(block)
    if len(all_personas)>0:
        M = all_personas[0]['avg_male']
        html = sub('##M##', M, html)
        html = sub('##F##', 100-M, html)
        html = sub('##ages##', list(reversed(ages)), html)
        html = sub('##dates##', list(reversed(dates)), html)

    with open(f'/home/thomas/Scrivania/dump-{token}.html', 'w') as ff:
        ff.write(html)


def get_src(client, info):
    if info['type']=='face':
        num = max(min(int(info['age'])//10, 5), 1)
        name = info['gender'][0].lower() + str(num)
        img = client.images.find_one({"name":name})
        if img:
            return img['url']
        else:
            return 'https://cdn.pixabay.com/photo/2014/04/02/14/11/male-306408_1280.png'
    elif info['type']=='interests':
        name = '_'.join(info['name'].split(' ')).lower()
        img = client.images.find_one({"name": name})
        if img:
            return img['url']
        else:
            return 'https://tbm-studentville.s3.amazonaws.com/app/uploads/2018/12/iStock-646900184.jpg'
    else:
        return 'https://tbm-studentville.s3.amazonaws.com/app/uploads/2018/12/iStock-646900184.jpg'


def init(client):
    mongo = pymongo.MongoClient('localhost', 27017)['personas']
    client.personas = mongo['pers']
    client.images = mongo['images']


def main(name, id='0'):
    mqtt.MQTTClient(process_type=name, funct=save_insights, init=init, type=mqtt.CLUSTER, id=id)

if __name__=='__main__':
    cli = lambda :None
    init(cli)
    load_personas_html(cli, 'theffballers')
    print('Done')

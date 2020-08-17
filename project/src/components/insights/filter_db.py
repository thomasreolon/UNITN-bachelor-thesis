import mycommons.mqtt_wrap as mqtt
import mycommons.data_structure as ds
import pymongo
import pymongo.errors as e

host = 'localhost'
port = 27017


####################### FUNCTIONS #############################################

def init(client):
    client.users = pymongo.MongoClient(host, port)['personas']['users']


def process_user(client:mqtt.MQTTClient, user:ds.User):
    my_query = {
        '_id': user.id
    }

    old = client.users.find_one(my_query)
    if old:                                                   # already in DB
        lat_act = user.latest_activity
        user.load_json(old)
        user.latest_activity = lat_act

        if user.mbti and user.img_topics and user.interests and old['latest_activity']==user.latest_activity:
            if not 'Artificial' in user.interests:
                user = None
            result = 'user already in DB and completed'
        else:
            result = f'user to re-proecss:{user.id}'
                                                              ## select only new activities
        # new_act, lat_act = [], old['latest_activity']
        # for act in user.activities:
        #     if lat_act < act.date:                          ### take activities more recent than the old latest activity
        #         new_act.append(act)
        # user.activities = new_act

    else:                                                       # not in db
        #return None, 'dropping: mode only update'
        result = f'new user:{user.id}'                          ## already good

    return user, result



def main(process_type='filter_db', id='0'):
    mqtt.MQTTClient(process_type=process_type, funct=process_user, init=init,id=id, type=mqtt.FILTER)


if __name__ == '__main__':
    main()





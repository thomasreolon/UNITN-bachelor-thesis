import mycommons.mqtt_wrap as mqtt
import mycommons.data_structure as ds
import json
import time
import hashlib

big_int = 9223372036854775807  # sys.max_index .... or something like that


####################### FUNCTIONS #############################################

def get_id(id, token):
    st = str(id) + str(token)
    return int(hashlib.sha1(st.encode('utf-8')).hexdigest(), 16) % big_int


def get_timestamp_tw(date):
    tmp = time.strptime(date, '%a %b %d %H:%M:%S +0000 %Y')
    return time.strftime('%Y-%m-%d %H:%M:%S', tmp)


def auto_complete(dest: dict, src: dict, list_fields: list):
    for new_field, field in list_fields:
        if field in src and src[field] is not None:
            dest[new_field] = src[field]

def tw_profile_photo_big(url:str):
    tmp = [url, '']
    for fmt in ['_normal', '_bigger', '_mini']:
        if fmt in url:
            tmp = url.split(fmt)
            break
    return tmp[0]+tmp[1]          # remove normal

def preprocess_twitter(json_dict: dict):
    users = dict()
    if 'token' in json_dict and len(json_dict['token']) > 0:
        token = json_dict['token']

        for profile in json_dict['data_profile']:
            the_id = get_id(profile['id'], token)

            my_user = ds.User(
                token=token,
                id=the_id,
                real_id=profile['id'],
                platform='twitter',
                username=profile['screen_name'],
                name=profile['name'],
                latest_activity='0',
                photo=tw_profile_photo_big(profile['profile_image_url_https']),
                bg_photo=profile['profile_background_image_url_https'],
                description=profile['description'],
                site_url=profile['url'],
                n_friends=profile['friends_count'],
                n_followers=profile['followers_count'],
                language=profile['lang'],
                location=profile['location'],
                created=get_timestamp_tw(profile['created_at']),
            )
            if not my_user.language and 'status' in profile:
                my_user.language = profile['status']['lang']
            users[the_id] = my_user

        for post in json_dict['data_posts']:
            id = get_id(post['user']['id'], token)
            ### mandatory post fields ###
            my_activity = ds.Activity(
                text=post['text'],
                likes=post['favorite_count'],
                shares=post['retweet_count'],
                hashtags=[x['text'] for x in post['entities']['hashtags']],
                date=get_timestamp_tw(post['created_at']),
                in_response=post['in_reply_to_status_id'] is not None,
                language=post['lang']
            )
            ### optional fields ###
            if 'extended_entities' in post:
                for m in post['extended_entities']['media']:
                    if m['type'] == 'photo':
                        my_activity.images.append(m['media_url'])
                    elif m['type'] == 'video':
                        my_activity.videos.append(m['media_url'])
                    else:
                        my_activity.gifs.append(m['media_url'])

            if id in users:
                users[id].activities.append(my_activity)
                if users[id].latest_activity < my_activity.date:
                    users[id].latest_activity = my_activity.date

    return users


def process_raw_data(client:mqtt.MQTTClient, msg:str):
    data = json.loads(msg)
    users = None
    if 'twitter' in data['platform']:
        users = preprocess_twitter(data)
    elif 'instagram' in data['platform']:
        pass

    if not users:
        raise ValueError(f'could not parse correctly the file received. file={msg}')

    return users, "users_found:" + ",".join([str(x) for x in users.keys()])


def main(process_type='preprocessor', id='0'):
    mqtt.MQTTClient(process_type=process_type, funct=process_raw_data, id=id, type=mqtt.PREPROCESSOR)


if __name__ == '__main__':
    main()
    #print(tw_profile_photo_big('https://pbs.twimg.com/profile_images/1059540888169467905/QO8-90n__normal.jpg'))

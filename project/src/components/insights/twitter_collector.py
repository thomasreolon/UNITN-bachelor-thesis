import json
import paho.mqtt.client as mqtt
import tweepy
from flask import request
from flask import Flask
import threading
app = Flask(__name__)
threads = []

twapi_path = '../../../credentials/twitter.json' if __name__=='__main__' else '../credentials/twitter.json'



class TwitterAPI():
    def __init__(self, consumer, c_secret, token, t_secret):
        auth = tweepy.OAuthHandler(consumer, c_secret)
        auth.set_access_token(token, t_secret)

        self.api = tweepy.API(auth, wait_on_rate_limit=True, retry_count=4, retry_delay=10)

    def get_users_from_account(self, account_screenname, to_get):
        batch = []

        for account in self.api.followers_ids(screen_name=account_screenname):
            batch.append(account)
            to_get -= 1
            if len(batch)==100:
                # lookup users
                try:
                    total_profiles = [x._json for x in self.api.lookup_users(batch)]
                    for pr in total_profiles:
                        print('one')
                        theid = pr['id']
                        try:
                            # lookup messages
                            total_activities = [x._json for x in self.api.user_timeline(id=theid, count=500)]
                            # format message
                            my_data = {
                                "token": account_screenname,
                                "platform": 'twitter',
                                "data_profile": [pr],
                                "data_posts": total_activities
                            }
                            if len(total_activities) > 20:
                                yield my_data
                            else:
                                raise ValueError(f'not enough activities {len(total_activities)}')

                        except tweepy.TweepError as e2:
                            pass
                        except KeyboardInterrupt as e:
                            raise e
                        except Exception as e:
                            pass

                except tweepy.TweepError as e2:
                    pass
                except KeyboardInterrupt as e:
                    raise e
                batch = []

            if to_get<=0: break


            if len(batch)>0:
                # lookup users
                try:
                    total_profiles = [x._json for x in self.api.lookup_users(batch)]
                    for pr in total_profiles:
                        theid = pr['id']
                        try:
                            # lookup messages
                            total_activities = [x._json for x in self.api.user_timeline(id=theid, count=500)]
                            # format message
                            my_data = {
                                "token": account_screenname,
                                "platform": 'twitter',
                                "data_profile": [pr],
                                "data_posts": total_activities
                            }
                            if len(total_activities) > 20:
                                yield my_data
                            else:
                                raise ValueError(f'not enough activities {len(total_activities)}')

                        except tweepy.TweepError as e2:
                            pass
                        except KeyboardInterrupt as e:
                            raise e
                        except Exception as e:
                            pass

                except tweepy.TweepError as e2:
                    pass



def init(client):
    with open(twapi_path, 'r') as fin:
        cr = json.load(fin)
    client.twapi = TwitterAPI(cr['consumer'], cr['c_secret'], cr['token'], cr['t_secret'])



@app.route('/personas')
def personas():
    token = request.args.get('token')
    if len(threads)>2:
        threads[0].join()
        del threads[0]
    tr = threading.Thread(target=process, args=[token])
    tr.start()
    threads.append(tr)
    return f'<h1>success</h1><p>{token} added</p>'



def process(token):
    cc=0
    print(token)
    client = mqtt.Client()
    client.connect('localhost')
    init(client)

    try:
        for message in client.twapi.get_users_from_account(token, 300):
            cc+=1
            print(f'pub{cc}')
            client.publish('/raw_data',json.dumps(message), qos=0)
    except KeyboardInterrupt as e:
        print('closing...collector')
        exit(0)

    return None, f'successfully published {cc} accounts in the pipeline'


def main(name, id='0'):
    app.run()

if __name__ == '__main__':
    main()

incompatible = 'incompatible'
min_post = 1
import json
from typing import List, Dict


#       #### --> needed field

class Message(object):
    def __init__(self, user=None, **kw):
        if user is not None:
            self.load_json(user)
        self.load_json(kw)

    def load_json(self, user):
        if isinstance(user, bytes) or isinstance(user, bytearray):
            user = json.loads(user.decode('utf-8'))
        elif isinstance(user, str):
            user = json.loads(user)
        elif hasattr(user, '__dict__'):
            user = user.__dict__

        if user != {}:
            me = self.__dict__
            for key in me:
                if key in user and user[key] is not None:
                    me[key] = user[key]
            self.__dict__ = me

    def __contains__(self, item):
        return item in self.__dict__

    def __getitem__(self, item):
        res = None
        if item in self.__dict__:
            res = self.__dict__[item]
        return res

    def __setitem__(self, key, value):
        if key in self and value is not None:
            self.__dict__[key] = value

    def cleared(self):
        without_none = {}
        for k, v in self.__dict__.items():
            if v is not None:
                without_none[k] = v
        return without_none

    def __repr__(self, drop_none=True):
        if drop_none:  # if True, class.__repr__() will not contain None values
            without_none = {}
            for k, v in self.__dict__.items():
                if v is not None:
                    without_none[k] = v
            s = json.dumps(without_none, default=lambda x: x.cleared())
        else:
            s = json.dumps(self.__dict__, default=lambda x: x.__dict__)
        return s


# class representing a post
class Activity(Message):
    def __init__(self, data=None, **kw):
        # from preprocessor
        self.text: str = None
        self.language: str = None
        self.likes: int = None
        self.shares: int = None
        self.hashtags: List[str] = []
        self.images: List[str] = []
        self.gifs: List[str] = []
        self.videos: List[str] = []
        self.date: str = None
        self.in_response: bool = None

        # loads data if provided
        super().__init__(data, **kw)

    def __hash__(self):
        res = 0
        if self.shares:
            res += self.shares * 10000000
        if self.likes:
            res += self.likes * 100000
        if self.text:
            for c in self.text:
                try:
                    res += ord(c) * 100
                except Exception:
                    pass
        if self.images:
            res += len(self.images) * 10
        if self.videos:
            res += len(self.videos)
        return res


# class representing characteristics of a user
class User(Message):
    def __init__(self, user=None, **kw):
        # from preprocessor
        self.token: str = None  ####
        self.id: int = None  ####
        self.real_id = None
        self.platform: str = None
        self.username: str = None
        self.name: str = None
        self.photo: str = None
        self.bg_photo: str = None
        self.description: str = None
        self.site_url: str = None
        self.n_friends: int = None
        self.n_followers: int = None
        self.language: Dict[str, int] = None
        self.location: str = None
        self.created: str = None
        self.latest_activity: str = None  ####

        # from classifiers
        self.mbti: Dict[str, int] = None
        self.ocean: Dict[str, float] = None
        self.needs: Dict[str, float] = None
        self.img_topics: List[str] = None
        self.nlu: Dict[str, List[str]] = None
        self.age: int = None
        self.gender: float = None
        self.interests = None

        # activities
        self.activities: List[Activity] = []  ####

        # loads data if provided
        self.load_json(user)
        self.load_json(kw)

    def load_json(self, user: dict, set_of_activities: set = None):
        if user is not None:
            if isinstance(user, bytes) or isinstance(user, bytearray):
                user = json.loads(user.decode('utf-8'))
            elif isinstance(user, str):
                user = json.loads(user)
            elif isinstance(user, User):
                user = user.__dict__
            me = self.__dict__
            activities = user['activities'] if ('activities' in user) else []
            user.__delitem__('activities') if ('activities' in user) else None
            for key in me:
                if key in user and user[key] is not None:
                    me[key] = user[key]
            self.__dict__ = me
            for s in activities:
                act = Activity(s)
                if set_of_activities:
                    if act not in set_of_activities:
                        set_of_activities.add(act)
                        self.activities.append(act)
                else:
                    self.activities.append(act)


# types of LOG
CN = 1000  # on connection #info
FN = 1001  # on message processed #debug
ERR = 1002  # on error #warn
ST = 1003  # on end (either user written in db or dropped) #crit


class Log(Message):
    def __init__(self, data=None, **kw):
        # from preprocessor
        self.client: str = None
        self.timestamp: str = None
        self.type: int = None
        self.result: str = None

        # loads data if provided
        super().__init__(data, **kw)


class Insights(Message):
    def __init__(self, data=None, **kw):
        self.mbti = {"T": -1, "S": -1, "E": -1, "J": -1}
        #self.ocean = {"O": -1, "C": -1, "E": -1, "A": -1, "N": -1, }
        #self.needs = {"conservative": -1, "need_expression": -1, "need_love": -1, "anger": -1, "depression": -1,
        #              "stress": -1, "intellect": -1}
        #self.nlu = {"categories": {"indoor": -1, "outdoor": -1, "cars": -1, "tech": -1, "family": -1, "politics": -1}}
        self.img_topics = {"Animals": -1, "Music": -1, "Healt Problems": -1, "War": -1, "Clothing": -1, "Sports": -1,
                           "Drinks": -1, "Rich": -1, "Cosplay": -1, "Office": -1, "Travel": -1, "Family": -1, "Baby": -1}
        self.age = -1
        self.language = {"it":-1,"en":-1,"fr":-1,"de":-1,"sp":-1, "un":-1}
        self.gender = -1
        self.interests = {'Tech': 0, 'Academic': 0, 'Animation': 0, 'American Football': 0, 'Soccer': 0, 'Basketball': 0, 'Tennis':0, 'Biking':0, 'Baseball':0, 'Dance': 0, 'Activism': 0, 'Nature': 0, 'Animals': 0, 'Food': 0, 'Religion': 0, 'Cars': 0, 'Video Games': 0, 'AI': 0, 'Design': 0, 'Politics': 0}

        super().__init__(data, **kw)
        if isinstance(self.language, str):
            tmp = self.language
            if tmp not in ["it","en","fr","de","sp"]:
                tmp = 'un'
            self.language = {"it": 0, "en": 0, "fr": 0, "de": 0, "sp": 0, "un":0, "und":0}
            self.language[tmp] = 1

    def get_point(self):
        point = []
        iterables = [self.__dict__]
        while len(iterables) > 0:
            value = iterables[0]
            del iterables[0]
            if isinstance(value, (int, float)):
                point.append(float(value))
            elif isinstance(value, list):
                iterables += value
            elif isinstance(value, dict):
                for x, y in value.items():
                    iterables.append(y)
            else:
                raise ValueError(f'type {type(value)} | {value} not allowed')

        return point


class Cluster(Message):
    def __init__(self, data=None, **kw):
        self.token = None
        self.cluster: List[List[Insights]] = []
        self.type = 'undefined'
        super().__init__(data, **kw)

    def append(self, a: Insights, indice=0):
        self.cluster[indice].append(a)

    def set_type(self, type):
        self.type = type

    def append_to_new_cluster(self, a: Insights):
        self.cluster.append([a])

    def get_points(self):
        res = []
        for p in self.cluster[0]:
            res.append(p.get_point())
        return res


if __name__ == '__main__':
    # test data format
    act1 = {"text": "ciaooo", "videos": ["a", "b"], "altro": None}
    act1 = Activity(act1)

    act2 = {"text": "heyyy", "videos": ["c", "b"], "date": 29}
    act2 = Activity(act2)

    user = User()
    user.name = 'luca'
    user.activities = [act1, act2]
    user.username = 99

    print(str(user))
    s = str(user)

    newuser = User(s, age=44, gender='male', sex='a lot')
    newuser.ocean = 1001
    print(newuser)


class UserMessageHandler(object):
    """
    class to handle messages and multiple dependencies
    """

    def __init__(self, req_u: List[str] = None, req_a: List[str] = None, n_input: int = 1):
        # set which fields are needed:
        self.set_requirements(req_u=req_u, req_a=req_a)

        # set the function to call to check messages
        self.n_input = n_input
        if n_input == 1:
            self.handle = self.handle_single
        else:
            self.handle = self.handle_multiple
            self.users = {}  # stores a user for that id
            self.count = {}  # stores how many messages have been receiver for that id
            self.u_activities = {}  # stores the hash of the activities, so that they are not duplicated

    # makes sets
    def set_requirements(self, req_u, req_a):
        # check that all required fields in a User are good
        available_fields = set(User().__dict__.keys())
        always_needed = {'id', 'token', 'latest_activity'}
        if req_u is None:
            self.req_u = always_needed
        else:
            self.req_u = set(req_u).union(always_needed)
        if len(self.req_u - available_fields) > 0:
            raise ValueError(
                f'required fields not allowd: {self.req_u - available_fields}--{self.req_u}--{available_fields}')

        # check that all required fields in an Activity are good
        available_fields = set(Activity().__dict__.keys())
        always_needed = set()
        if req_a is None:
            self.req_a = always_needed
        else:
            self.req_a = set(req_a).union(always_needed)
        if len(self.req_a - available_fields) > 0:
            raise ValueError(f'required fields not allowd: {self.req_a - available_fields}')

    # read a message, if it does not contain everything, set incompatibility
    def handle_single(self, user: dict):
        if isinstance(user, User):
            user = user.__dict__
        elif isinstance(user, bytes):
            user = json.loads(user.decode('utf-8'))
        ok, acts = True, []
        for required_user_field in self.req_u:
            if required_user_field not in user or user[required_user_field] is None:
                ok = False
                break
        if ok:
            for act in user['activities']:
                if all(req_act_f in act for req_act_f in self.req_a):
                    acts.append(act)
            user['activities'] = acts
            if len(acts) == 0 and len(self.req_a) > 0:
                ok = False
        return User(user), ok

    def handle_multiple(self, user: dict):
        if isinstance(user, User):
            user = user.__dict__
        elif isinstance(user, bytes):
            user = json.loads(str(user))
        id = user['id']
        if id not in self.users:
            self.users[id] = User(user)
            self.count[id] = 1
            self.u_activities[id] = set()
        else:
            self.users[id].load_json(user, self.u_activities[id])
            self.count[id] += 1

        res, ok = None, True
        if self.count[id] == self.n_input:
            res, ok = self.handle_single(self.users[id])
            del self.users[id]
            del self.count[id]
        return res, ok


if __name__ == '__main__':
    # test message handler
    mh = UserMessageHandler(['id', 'age'], ['text'])

    u, r = mh.handle(User().__dict__)
    print(r, '---', u)  # false

    u, r = mh.handle(User(id=12, name='luca', token='asd', latest_activity='mai').__dict__)
    print(r, '---', u)  # false

    u, r = mh.handle(
        User(id=12, age=9, name='luca', token='asd', latest_activity='mai', activities=[{'text': 'ciao'}]).__dict__)
    print(r, '---', u)  # true

    mh = UserMessageHandler(['id', 'age'], ['text'], 3)

    u, r = mh.handle(User(id=12, token='asd', latest_activity='neva').__dict__)
    print(r, '---', u)  # tr, none

    u, r = mh.handle(User(id=12, age='luca').__dict__)
    print(r, '---', u)  # tr, none

    u, r = mh.handle(
        User(id=12, latest_activity='mai', activities=[{'text': 'ciao'}]).__dict__)
    print(r, '---', u)  # true, user

    mh = UserMessageHandler(['id', 'age'], ['text'], 2)

    u, r = mh.handle(User(id=12, token='asd', latest_activity='neva').__dict__)
    print(r, '---', u)  # tr, none

    u, r = mh.handle(User(id=12, age='luca').__dict__)
    print(r, '---', u)  # fl, res

    u, r = mh.handle(
        User(id=12, latest_activity='mai', activities=[{'text': 'ciao'}]).__dict__)
    print(r, '---', u)  # true, none


class UnableToProcessUser(Exception):
    def __init__(self, msg=None):
        self.msg = msg if msg else 'could not process user'

    def __repr__(self):
        return str(self.msg)

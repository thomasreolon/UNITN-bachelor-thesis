import json

file = '../src/settings.json'
non_classifiers = {'preprocessor', 'logger'}

# ogni classifier ha un coso nei settings  v
# ogni coso nei setting ha un classifier   x
# il classifier è unico                    X




if __name__=='__main__':
    with open(file, 'r') as fin:
        settings = json.load(fin)

    # controlla che i campi insights corrispondano
    ins = settings['user_structure']['insights']
    num_to_have = len(ins)
    num_got = 0
    non_corresp = []
    for comp, set in settings['components'].items():
        if comp not in non_classifiers:
            num_got += len(set['fields']['output']['insights'])
            for x in set['fields']['output']['insights']:
                if x not in ins:
                    non_corresp.append(comp)

    if num_got != num_to_have or len(non_corresp)>0:
        raise ValueError(f'got: {num_got}/{num_to_have}      non_corresp: {", ".join(non_corresp)}')

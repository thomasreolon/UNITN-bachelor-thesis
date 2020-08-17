import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import components
import argparse
import mycommons.mqtt_wrap as wr
import logging

levels = {
    'D':logging.DEBUG,
    'I':logging.INFO,
    'W':logging.WARNING,
    'E':logging.ERROR,
    'C':logging.CRITICAL
}

if __name__ == '__main__':
    cmp_factory = components.ComponentsFactory()
    parser = argparse.ArgumentParser()
    parser.add_argument("component", help="the name of the component as in the settings.json")
    parser.add_argument("--id", help="used to distiguish two istances of the same component")
    parser.add_argument("--log", help="D debug, I info, W warnings, E errors (default), C critical")
    args = parser.parse_args()

    if args.log:
        wr.log_level = levels[args.log]

    if args.id:
        cmp_factory[args.component](args.component, args.id)
    else:
        cmp_factory[args.component](args.component)





#!/bin/bash

[ "$(docker ps | grep personas-broker)" ] && docker stop personas-broker || echo "personas-broker"
[ "$(docker ps | grep personas-mongo)" ] &&  docker stop personas-mongo || echo "personas-mongo"
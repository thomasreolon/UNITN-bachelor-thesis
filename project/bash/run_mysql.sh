#!/bin/bash
# allow ctrl+C to stop
trap "exit" INT


# if it is not running
if [ ! "$(docker ps | grep mysql-personas)" ]
then
  # make container if does not exist
  [ ! "$(docker ps -a | grep mysql-personas)" ] &&
      echo "creating mysql container" &&
      docker create \
        -e MYSQL_ROOT_PASSWORD=ciao \
        -v /home/data/mysql:/var/lib/mysql \
        -p 3306:3306 \
        --name mysql-personas \
        mysql/mysql-server
  # starts docker
  echo "starting mysql"
  docker start mysql-personas
fi

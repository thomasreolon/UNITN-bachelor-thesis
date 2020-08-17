#!/bin/bash
# allow ctrl+C to stop
trap "exit" INT
	# create directory for volumes if does not exist
	DIRNAME="/home/data/personas/mongodb/data"
	if [[ ! -d $DIRNAME ]]
	then
		echo 'need permission to create' $DIRNAME
		sudo mkdir -p --mode=777 $DIRNAME
		cd $DIRNAME && cd .. && sudo chown "$USER" -R . || exit       # set user as owner
	fi

	# if it is not running
	if [ ! "$(docker ps | grep personas-mongo)" ]
	then
		# make container if does not exist
		[ ! "$(docker ps -a | grep personas-mongo)" ] &&
		    echo "creating mongo container" &&
        docker create \
          -p 27017:27017 \
          -v $DIRNAME:/data/db \
          --privileged \
          --name personas-mongo \
          mongo
		# starts docker
		echo "starting mongodb"
		docker start personas-mongo
	fi

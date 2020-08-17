#!/bin/bash
# allow ctrl+C to stop
trap "exit" INT

	# create directories for volumes if do not exist
	MYCONF="/home/data/personas/mosquitto"
	MYDATA="/home/data/personas/mosquitto/data"

	# mosquitto conf
	MOSQUITTOFILE="
		persistence true
		persistence_location /mosquitto/data/
		persistence_file mosquitto.db
		max_queued_messages 200
		persistent_client_expiration 1d
	"

  # create voleme folder if does not exist
	if [[ ! -d $MYDATA ]]
	then
		echo 'need permission to create' $MYDATA
		sudo mkdir -p $MYDATA ; sudo chmod 777 $MYDATA             # gives permissions
		cd $MYDATA && cd ../.. && sudo chown "$USER" -R . || exit     # set user as owner
		echo "$MOSQUITTOFILE" > "$MYCONF/mosquitto.conf"
	fi

	# run it, if it is not
	if [ ! "$(docker ps | grep personas-broker)" ]
	then
		# make container if does not exist
		[ ! "$(docker ps -a | grep personas-broker)" ] &&
		    echo "creating broker container" &&
				docker create  -p 1883:1883 \
						-p 9001:9001 \
						--privileged \
						-v $MYCONF:/mosquitto/config :ro \
						-v $MYDATA:/mosquitto/data \
						--name personas-broker \
						eclipse-mosquitto
		echo "starting mosquitto"
		docker start personas-broker
	fi

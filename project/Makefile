.PHONY: stop runmongo runmosquitto runcomponents all
.SILENT: stop runmongo runmosquitto runcomponents all

all: runmosquitto runmongo runcomponents

runmongo:
	bash bash/run_mongo.sh


runmosquitto:
	bash bash/run_mosquitto.sh


runcomponents:
	bash bash/run_components.sh

stop:
	bash bash/run_stop.sh

process:
	cd src && python test_input.py






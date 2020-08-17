# Personas System

The code for the system.
It does not work because it leaks the models and the APIs keys

### Adding a Component

**settings.json**
this file specifies the name of a component and which topics it should use

```
  "component_name": {
    "type": ["classifier", "active"],
    "fields": {
      "required": {
        "profile": ["age", "gender"]
      }
    },
    "input_topics": ["/data/formatted"],
    "output_topic": "/data/age"
  }
```

**components/\_\_init\_\_.py**
this file tells the system which components to import when the key \_\_\_ is given as input

```
  if key=="preprocessor":
      from components.insights.preprocessor import main
      return main
```

**components/insights/preprocessor.py**
this file implements the component.
a component is built as follows:

1. a function _funct(environment, message)_ is declared, every time a message is received, this function is called. It must return the JSON object that will be sent in the output topic
2. a MQTTclient is initializated and _funct_ is passed as parameter. the client initializes itself using the setting.json file and then begin listening to the MQTTbroker

```

def main(process_type='preprocessor', id='0'):
    mqtt.MQTTClient(process_type=process_type, funct=process_raw_data, id=id, type=mqtt.PREPROCESSOR)

```

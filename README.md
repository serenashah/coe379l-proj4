# coe379l-proj4
## Using the container
__To run the inference server, pull and run the Docker image with the Alt Lenet model:__  
```
docker pull serenashah/proj04-api
docker run -it --rm -p 5000:5000 serenashah/proj04-api
```

## HTTP requests
__To get information about the model:__ 
```
curl localhost:5000/models
```

This would be the expected output:
```
Content           : {
                      "version": "v1",
                      "name": "models",
                      "description": "Predict flood levels using information about location, elevation, and Standard Engineering Storm Events",
                      "output": "Array corresponding to inputs that give the prediction of flooding level"
                    }
```
__To get information on the expected input, run this in a Python script/Jupyter Notebook:__ 
```
rsp = requests.get("http://172.17.0.1:5000/input_example")
rsp.json()
```
__where the expected response should look as follows:__ 
```{'data':
      [{
      'BM_ELEV': 58.37, 
      'SE10YR': 56.2, 
      'SE50YR': 58.7, 
      'SE100YR': 59.1, 
      'SE500YR': 60.5, 
      'POINT_X': -95.49876634, 
      'POINT_Y': 29.67809883
      }, 
      { 
      'BM_ELEV': 58.37, 
      'SE10YR': 56.2, 
      'SE50YR': 58.7, 
      'SE100YR': 59.1, 
      'SE500YR': 60.5, 
      'POINT_X': -95.49876634, 
      'POINT_Y': 29.67809883 
       }] 
}
```

__Variable descriptions:__
`BM_ELEV`: benchmark elevation (ft) 
`SE10YR`: Storm Event 10 Years (ft) 
`SE50YR`: Storm Event 50 Years (ft) 
`SE100YR`: Storm Event 100 Years (ft) 
`SE500YR`: Storm Event 500 Years (ft) 
`POINT_X`: Latitude
`POINT_Y`: Longitude

__To get a flood level prediction for your input of parameters shown above for a specific classification model, run this in a Python script/Jupyter Notebook:__ 
```
rsp = requests.post("http://172.17.0.1:5000/models/<model_type>", json={"data": input_data})
rsp.json()
```
where `input_data` is a list of dictionaries with values for all aforementioned columns. 

The `model_type`s are as follows:
KNN: `knn` 
Decision Tree: `dt` 
Naive Bayes: `nb` 
XGBoost: `xgb`

Note that `Flask` is a required module.  
__The expected output for this is a prediction array that corresponds to the inputs of data and should have the form of the output below:__   
`{'XGBoost Prediction': ['Low Flood Level', 'High Flood Level'], 'Medium Flood Level'}`  

## Interpreting Results


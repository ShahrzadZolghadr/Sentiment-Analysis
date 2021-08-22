# Sentiment-Analysis
Sentiment Analysis on Snappfood dataset

To test webApp:
in webApp folder, 

to build image:docker build -t webapp .

to run:docker run -p 5000:5000 webapp

to predict:curl --location --request POST 'http://0.0.0.0:5000/predict' --header 'content-type:application/json' --data-raw '{"rawtext":"جمله مورد نظر"}'

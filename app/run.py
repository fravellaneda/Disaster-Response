import pandas as pd
import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from adicionalFeatures import TextLengthExtractor

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine
import joblib

app = Flask(__name__)

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_response', engine)

# load model
model = joblib.load("../models/classifier.pkl")
count_words = pd.read_csv('../models/count_words.csv')

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    top_20_words_names = count_words['words'][0:20].values
    top_20_words_values = count_words['frequency'][0:20].values

    category = df.iloc[:, 4:]
    category_series = category[category == 1].count().sort_values(ascending=False)
    top_20_category_names = category_series.index[:20]
    top_20_category_count = category_series.values[:20]
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    text=[x for x in genre_names],
                    marker=dict(
                        color='rgb(112,142,161)'
                    ),
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=top_20_words_names,
                    y=top_20_words_values,
                    text=[x for x in top_20_words_names],
                    marker=dict(
                         color='rgb(112,142,161)'
                    )
                )
            ],

            'layout': {
                'title': 'Top 20 Distribution of Words Without Stopwords',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Top 20 Words",
                    'tickangle': 30,
                    'tickfont': {
                        'size': 10
                    }
                }
            }
        },
        {
            'data': [
                Bar(
                    x=top_20_category_names,
                    y=top_20_category_count,
                    text=[x for x in top_20_category_names],
                    marker=dict(
                         color='rgb(112,142,161)'
                    )
                )
            ],

            'layout': {
                'title': 'Top 20 Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Top 20 Category",
                    'tickangle': 30,
                    'tickfont': {
                        'size': 10
                    }
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='127.0.0.1', port=5000, debug=True)


if __name__ == "__main__":
    main()
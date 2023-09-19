# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

import category_encoders as ce
import time
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings(action='ignore')

import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import base64
import plotly.graph_objs as go
import plotly.tools as tls
from dash.exceptions import PreventUpdate



data = pd.read_csv("C:/Users/vinee/Downloads/car_price_prediction.csv")
original_data = data.copy()

# Starting Data Preprocessing
data.columns = data.columns.str.lower().str.replace(" ","_")
data.rename(columns={'prod._year': 'prod_year'}, inplace=True)
data.drop_duplicates(inplace=True)
data.drop('id', axis=1, inplace=True)

data['levy'] = data['levy'].replace(['-'], ['0'])
data['levy'] = data['levy'].astype(str).astype(int)

# Making a new turbo column which is either positive 1 or negative 0
data['turbo'] = data['engine_volume'].str.contains("Turbo").map({False: 0, True: 1})
data['engine_volume'] = data['engine_volume'].str.replace('Turbo', '')
data['engine_volume'] = data['engine_volume'].astype(str).astype(float)

# Removing km and converting into int
data["mileage"] = data["mileage"].str[:-2]
data['mileage'] = data['mileage'].astype('int64')

data = data.replace({'doors': {'04-May': '4-5', '02-Mar': '2-3'}})

cat_feature = [feature for feature in data.columns if data[feature].dtype == 'O']
data[cat_feature].head()

num_feature = [feature for feature in data.columns if data[feature].dtype != 'O']
data[num_feature].head()


 
def plot_outliers_bef(column):
    plt.figure(figsize=(20, 3))
    plt.suptitle('Distribution before handling outliers')
    plt.subplot(1, 2, 1)
    plt.title(f'Car {column} Distribution Plot')
    sns.distplot(data[f'{column}'], color='red')
    plt.subplot(1, 2, 2)
    plt.title('Car Price Box Plot')
    sns.boxplot(y=column, data=data)


def plot_outliers_aft(column):
    plt.figure(figsize=(20, 3))
    plt.suptitle('Distribution after handling outliers')
    plt.subplot(1, 2, 1)
    plt.title(f'Car {column} Distribution Plot')
    sns.distplot(data[f'{column}'], color='red')
    plt.subplot(1, 2, 2)
    plt.title('Car Price Box Plot')
    sns.boxplot(y=column, data=data)


Q1 = data[num_feature].quantile(0.25)
Q3 = data[num_feature].quantile(0.75)
IQR = Q3 - Q1

# Calculate the lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers in each numerical column
outliers = ((data[num_feature] < lower_bound) | (data[num_feature] > upper_bound)).sum()

# Calculate the outlier percentage
outlier_percentage = (outliers / len(data[num_feature])) * 100

out_feature = ['price', 'levy', 'engine_volume', 'mileage']


# Solving outliers
def iqr_handle(column):
    df_new = data.copy()
    q1 = data[f'{column}'].quantile(0.25)
    q3 = data[f'{column}'].quantile(0.75)
    iqr = q3 - q1  # IQR
    fence_low = q1 - 1.5 * iqr
    fence_high = q3 + 1.5 * iqr
    df_new.loc[:, column] = np.where(data[f'{column}'] > fence_high, fence_high,
                                    np.where(data[f'{column}'] < fence_low, fence_low,
                                            data[f'{column}']))
    return df_new


def visualize_outliers(column):
    outlier_percentage_table = html.Table(
    [
        html.Tr([html.Th("Column"), html.Th("Outlier Percentage")])
    ] + [
        html.Tr([html.Td(col), html.Td(f"{outlier_percentage[col]:.2f}%")])
        for col in outlier_percentage.index
    ],
    className="outlier-percentage-table"

    )
    plot_outliers_bef(column)
    data_new = iqr_handle(column)
    plot_outliers_aft(column)
    
    fig_bef_density = go.Figure()
    fig_bef_density.add_trace(go.Histogram(x=data[column], name='Before'))
    fig_bef_density.update_layout(title_text=f"Density Plot of {column} (Before)")

    fig_aft_density = go.Figure()
    fig_aft_density.add_trace(go.Histogram(x=data_new[column], name='After'))
    fig_aft_density.update_layout(title_text=f"Density Plot of {column} (After)")
    
    fig_bef_box = px.box(data, y=column, title=f"Boxplot of {column} (Before)")
    fig_aft_box = px.box(data_new, y=column, title=f"Boxplot of {column} (After)")
    
    return [
        dcc.Graph(figure=fig_bef_density),
        dcc.Graph(figure=fig_aft_density),
        dcc.Graph(figure=fig_bef_box),
        dcc.Graph(figure=fig_aft_box)
    ]

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)



# Define the callback function to update the modified data display and observations
@app.callback(
    [Output('modified-data', 'children'), Output('data-observations', 'children')],
    [Input('dataset-selector', 'value')]
)

def update_modified_data(dataset):
    if dataset == 'cleaned':
        # Return the modified data as a scrollable table
        modified_table = dash_table.DataTable(
            id='modified-data-table',
            columns=[{'name': col, 'id': col} for col in data.columns],
            data=data.head().to_dict('records'),
            style_table={'height': '100%'},
            style_data={'whiteSpace': 'pre', 'height': 'auto'},
            style_cell={'textAlign': 'left'},
        )
        observations = [
            html.P('ID column was removed'),
            html.P('There were 313 duplicate items which were removed.'),
            html.P('After this modification shape of data is (18924,17).'),
            html.P('Some basic preprocessing null values like - replaced with 0 in levy. Mileage (removed kms), doors(had some random data)'),
            html.P('Turbo column added was added and finally uniform column names were added'),
            
        ]  
        return modified_table, observations
    else:
        # Show the original data set
        original_table = dash_table.DataTable(
            id='original-data-table',
            columns=[{'name': col, 'id': col} for col in original_data.columns],
            data=original_data.head().to_dict('records'),
            style_table={'height': '100%'},
            style_data={'whiteSpace': 'normal', 'height': 'auto'},
            style_cell={'textAlign': 'left'},
        )
        observations = [
            html.P('ID: There are 19,237 IDs in the dataset'),
            html.P('Price: There is high variation in the price of cars.'),
            html.P('Levy: There are 30% of cars with 0 levy, and most of the levy ranges between 500 and 1000. There is high variation in the levy.'),
            html.P('Manufacturer: 20% of the cars are Hyundai, 19% are Toyota, and 11% are Mercedes-Benz.'),
            html.P('Model: There are 1590 models in the car dataset.'),
            html.P('Prod. Year: The dataset contains cars produced between the years 1939 and 2020.'),
            html.P('Category: There are 11 categories of cars in the dataset.'),
            html.P('Leather Interior: It is a binary variable containing "Yes" or "No".'),
            html.P('Fuel Type: There are 7 types of fuel for cars.'),
            html.P('Engine Volume: There are 60 distinct values of engine volume.'),
            html.P('Mileage: Most cars have a mileage between 100,000 and 200,000.'),
            html.P('Cylinders: 74.7% of the cars have 4 cylinders, and the remaining are distributed between 1 and 16. It is highly positively correlated with engine volume.'),
            html.P('Gear Box Type: There are 4 types of gears for cars, and 70% of all cars have automatic gear type.'),
            html.P('Drive Wheels: There are 3 types of drive wheels.'),
            html.P('Doors: 95% of the cars have 4-5 doors, and there are less than 1% of cars with more than 5 doors.'),
            html.P('Wheel: It indicates left-hand and right-hand drive cars. 92% of the cars are left-hand drive.'),
            html.P('Color: There are 16 colors of cars.'),
            html.P('Airbags: There are 17 distinct values of airbags in cars.')
        ]
        return original_table, observations
    
@app.callback(
    Output('preprocess-output', 'children'),
    Input('preprocess-selector', 'value')
)
def preprocess_column(column):
    if column is not None:
        graphs = visualize_outliers(column)
        return html.Div(graphs)
    else:
        return html.Div()


numeric_data = data.select_dtypes(include=[np.number])
# Calculate the correlation matrix
correlation_matrix = numeric_data.corr()

# Create the heatmap figure
fig_corr = go.Figure(data=go.Heatmap(
    z=correlation_matrix.values,
    x=correlation_matrix.columns,
    y=correlation_matrix.columns,
      # Replace 'coolwarm' with a valid colorscale option
    colorbar=dict(title='Correlation')
))


# Set the layout for the heatmap
fig_corr.update_layout(
    title='Correlation Matrix',
    height=600,
    width=800,
)


@app.callback(
    Output('visualization-content', 'children'),
    Input('visualization-selector', 'value')
)

def update_visualization_content(selected_value):
    if selected_value == 'correlation':
        return html.Div([
            html.H3('Correlation Matrix'),
            dcc.Graph(figure=fig_corr),
            html.H5('Positive correlations:'),
            html.P('Price is positively correlated with prod_year (0.31), engine_volume (0.08), and turbo (0.17).'),
            html.P('Levy is positively correlated with prod_year (0.41), engine_volume (0.35), and cylinders (0.23).'),
            html.P('Engine_volume is positively correlated with cylinders (0.72).'),
            html.P('Airbags is positively correlated with prod_year (0.23), engine_volume (0.27), and cylinders (0.17).'),
            html.H5('Negative correlations:'),
            html.P('Price is negatively correlated with mileage (-0.22) and airbags (-0.05).'),
            html.P('Mileage is negatively correlated with price (-0.22).')
        ])
    elif selected_value == 'count-barplot':
        num_feature_dropdown = dcc.Dropdown(
            id='num-feature-dropdown',
            options=[{'label': feature, 'value': feature} for feature in ['manufacturer','model','category','prod_year','engine_volume','cylinders','airbags','turbo']],
            placeholder='Select a numerical feature'
        )
        count_barplot = dcc.Graph(id='count-barplot')
        return html.Div([
            html.H3('Count vs Attribute'),
            html.Div([
                html.Label('Select a numerical feature:'),
                num_feature_dropdown
            ]),
            html.Div([
                count_barplot
            ])
        ])
    
    elif selected_value == 'year-wise':
        lcv = data[cat_feature]
        lcv.drop(columns=['manufacturer', 'model'], inplace=True)
        cat_feature_dropdown = dcc.Dropdown(
            id='cat-feature-dropdown',
            options=[{'label': feature, 'value': feature} for feature in lcv],
            placeholder='Select a Feature to plot a line plot'
        )
        return html.Div([
        cat_feature_dropdown,
        dcc.Graph(id='year-lineplot'),
        html.H5("Observations:"),
        html.P("The dataset contains data of last 54 years from 2020 From distribution plot ,we come to know that there is high frequency of cars manufactured between 2010 and 2017. There is also skewness in the production years"),
        
    ])

    elif selected_value == 'price':
        lcv = data[cat_feature]
        lcv.drop(columns=['manufacturer', 'model'], inplace=True)
        feature_dropdown = dcc.Dropdown(
            id='feature-dropdown',
            options=[{'label': feature, 'value': feature} for feature in lcv.columns],
            placeholder='Select a Feature to plot a line plot'
        )
        return html.Div([
            feature_dropdown,
            dcc.Graph(id='price'),
            html.P("We get an idea that:"),
            html.P("Leather seats always tend to have higher price"),
            html.P("Recently tiptrotinic type gearboxes is Highly used and Affects the price a lot"),
            html.P("Doors having more than 5 doors are recently introduced and they affect the price. Apparently 2-3 door cars have a higher price.")
        ])

    
    # Add other visualization options (year wise - mean price vs attributes, price vs attributes) here
    
    else:
        return html.Div('No visualization selected')



# Create a callback function for the dropdown menu
@app.callback(
    Output('count-barplot', 'figure'),
    Input('num-feature-dropdown', 'value')
)
def update_count_barplot(feature):
    if feature is None:
        raise PreventUpdate  # Skip callback update if no feature is selected
    count_data = data[feature].value_counts().reset_index()
    count_data.columns = ['Value', 'Count']
    fig = px.bar(count_data, x='Value', y='Count', labels={'Value': feature, 'Count': 'Count'})
    fig.update_layout(title=f'Count of {feature}')
    return fig

@app.callback(
    Output('year-lineplot', 'figure'),
    Input('cat-feature-dropdown', 'value')
)
def update_year_lineplot(feature):
    ax = sns.lineplot(data=data, x='prod_year', y='price', hue=feature, errorbar=None)
    fig = tls.mpl_to_plotly(ax.figure)
    fig.update_layout(title=f'Mean Price vs Year of Production - {feature}')
    return fig

@app.callback(
    Output('price', 'figure'),
    Input('feature-dropdown', 'value')
)
def update_price(feature):
    if feature is None:
        return go.Figure()
    
    ax = sns.barplot(x =data[feature], y= data['price'],palette="mako",errorbar=None)
    fig = tls.mpl_to_plotly(ax.figure)
    fig.update_layout(title=f'Price of {feature}')
    return fig



#Feature engineering to prepare the model
datan=data.copy()
datan= iqr_handle('price')
datan= iqr_handle('levy')
datan= iqr_handle('engine_volume')
datan= iqr_handle('mileage')
datan.drop(['color'], axis = 1, inplace = True)

dfenc= datan.copy()
targetEncod = ce.LeaveOneOutEncoder()
ordinalEncoder = OrdinalEncoder()

dfenc['manufacturer']=ordinalEncoder.fit_transform(data[['manufacturer']])
dfenc['model'] = ordinalEncoder.fit_transform(data[['model']])
dfenc['category'] = ordinalEncoder.fit_transform(data[['category']])
dfenc['leather_interior'] = data['leather_interior'].map({"Yes":1,"No" : 0})
dfenc['doors'] = data['doors'].map({'4-5':4, '2-3':2, '>5':5})
dfenc['fuel_type'] = ordinalEncoder.fit_transform(data[['fuel_type']])
dfenc['gear_box_type'] = ordinalEncoder.fit_transform(data[['gear_box_type']]) 
dfenc['drive_wheels'] = ordinalEncoder.fit_transform(data[['drive_wheels']])
dfenc['wheel'] = data['wheel'].map({"Left wheel":0,"Right-hand drive" : 1})
dfclean= dfenc.copy()
dfclean.drop_duplicates(inplace= True)


x= dfclean.drop('price',axis=1)
y= dfclean['price']
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.25, random_state=42)

def train_linear_regression():
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_gradient_boosting():
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    return model

def train_random_forest():
    Rf = RandomForestRegressor(n_estimators = 300,max_depth=20, max_features='sqrt',random_state=1)
    Rf.fit(X_train, y_train)
    return Rf


import dash_table

@app.callback(
    Output("model-results", "children"),
    Input("model-selector", "value")
)
def train_and_evaluate_model(model_name):
    if model_name is None:
        return html.Div("Select a model to see the results.")

    # Train the selected model
    if model_name == "linear":
        model = train_linear_regression()
    elif model_name == "gradient":
        model = train_gradient_boosting()
    elif model_name == "random_forest":
        model = train_random_forest()
    elif model_name == "result":
                results_table = html.Div(
            [
                html.H3("Model Evaluation Results"),
                dash_table.DataTable(
                    data=[
                        {"Model": "LinearRegression", "Score With Cross-Validation": 0.338743, "Time": 0.116328,
                         "Test_Score(R2)": 0.337986, "Train_Score": 0.341033},
                        {"Model": "GradientBoostingRegressor", "Score With Cross-Validation": 0.647933,
                         "Time": 6.327972, "Test_Score(R2)": 0.679125, "Train_Score": 0.671431},
                        {"Model": "RandomForestRegressor", "Score With Cross-Validation": 0.740040, "Time": 26.714316,
                         "Test_Score(R2)": 0.775485, "Train_Score": 0.951882},
                    ],
                    columns=[
                        {"name": "Model", "id": "Model"},
                        {"name": "Score With Cross-Validation", "id": "Score With Cross-Validation"},
                        {"name": "Time", "id": "Time"},
                        {"name": "Test_Score", "id": "Test_Score"},
                        {"name": "Train_Score", "id": "Train_Score"},
                    ],
                    style_table={"overflowX": "auto"},
                ),
                html.P("From these results, we understand that Linear Regression is the worst model."),
                html.P("To get the average results, we prefer the Gradient Boosting Model."),
                html.P("If we want the best results and don't care about training time, then Random Forest is the best model to use."),
            ]
        )
                return results_table

    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Create a DataFrame with the test and prediction values
    results_df = pd.DataFrame({"Test": y_test, "Prediction": y_pred})

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
 

    # Display the results
    results = html.Div(
        [
            html.H3(f"{model_name} Model Evaluation"),
            html.H4("Top Few Values of Test and Prediction:"),
            dash_table.DataTable(
                data=results_df.head().to_dict("records"),
                columns=[{"name": col, "id": col} for col in results_df.columns],
                style_table={"overflowX": "auto"},
            ),
            html.H4("Model Evaluation Metrics:"),
            html.Table(
                [
                    html.Thead(html.Tr([html.Th("Metric"), html.Th("Value")])),
                    html.Tbody(
                        [
                            html.Tr([html.Td("Mean Absolute Error"), html.Td(mae)]),
                            html.Tr([html.Td("Mean Squared Error"), html.Td(mse)]),
                            html.Tr([html.Td("Root Mean Squared Error"), html.Td(rmse)]),
                        ]
                    ),
                ]
            ),
        ]
    )

    return results


# Define the layout of the app
app.layout = dbc.Container(
    [
        html.H1("Car Price Dashboard", className="header"),
        dbc.Tabs(
            [
                dbc.Tab(
                    label=["Data Set"],
                    children=[
                        dcc.RadioItems(
                            id='dataset-selector',
                            options=[
                                {'label': '  Original Data Set', 'value': 'original'},
                                {'label': ' Cleaned Database', 'value': 'cleaned'}
                            ],
                            value='original'
                            #labelStyle={'display': 'inline-block'}
                        ),
                        html.H3("Data Set:"),
                        html.Div(id='modified-data'),
                        html.H3("Observations:"),
                        html.Div(id='data-observations')
                    ],
                ),
                dbc.Tab(
                    label=[(" Visualisation")],
                    children=[
                        dcc.RadioItems(
                            id='visualization-selector',
                            options=[
                                {'label': 'Correlation Matrix', 'value': 'correlation'},
                                {'label': 'Count vs Attributes', 'value': 'count-barplot'},
                                {'label': 'Year wise - Mean Price vs Attributes', 'value': 'year-wise'},
                                {'label': 'Price vs Attributes', 'value': 'price'},
                            ],
                            value='correlation',
                            labelStyle={'display': 'block'},
                            style={'margin': '10px'}
                        ),
                        # Replace the existing 'visualization-content' div with the updated version
                        html.Div(id='visualization-content', children= update_visualization_content('correlation'))

                    ],
                ),
                dbc.Tab(
                    label=[(" Preprocessing")],
                    children=[
                        html.H4("Outlier Percentage:"),
                        html.Table(
                            [
                                html.Tr([html.Th("Column"), html.Th("Outlier Percentage")]),
                            ] + [
                                html.Tr([html.Td(col), html.Td(f"{outlier_percentage[col]:.2f}%")])
                                for col in outlier_percentage.index
                            ],
                            className="outlier-percentage-table",
                        ),
                        html.P("Even though the outlier percentage for the cylinder is high, it is totally normal because cylinders are not a continuous variable. Similarly for others."),
                        dcc.Dropdown(
                            id='preprocess-selector',
                            options=[{'label': col, 'value': col} for col in out_feature],
                            placeholder='Select a column to preprocess'
                        ),
                        html.Div(id='preprocess-output')
                    ]
                ),
                dbc.Tab(
                    label="Models",
                    children=[
                        html.H2("Choose a Model"),
                        dcc.RadioItems(
                            id="model-selector",
                            options=[
                                {"label": "Linear Regression", "value": "linear"},
                                {"label": "Gradient Boost Regressor", "value": "gradient"},
                                {"label": "Random Forests", "value": "random_forest"},
                                {"label": "Results", "value": "result"}
                            ],
                            value=None,
                            labelStyle={"display": "block"},
                            style={"margin": "10px"},
                        ),
                        html.Div(id="model-results"),
                    ]
                )

            ]
        )
    ],
    className="main-container",
)


app.css.append_css(
    {
        "external_url": "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css"
    }
)


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

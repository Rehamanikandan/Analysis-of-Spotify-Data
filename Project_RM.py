import dash
import dash
import numpy as np
import plotly.express as px
from dash import dcc
from dash.dependencies import Input,Output
from dash import html
import pandas as pd
from statsmodels.graphics.gofplots import qqplot
from sklearn.preprocessing import StandardScaler
from numpy import linalg as la
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kstest
from scipy.stats import shapiro
from scipy.stats import normaltest
import plotly.graph_objects as go

#Load Dataset
pd.set_option('max_columns', 19)
data=pd.read_csv('data.csv')

#Load external stylesheets and assign tab styles
external_stylesheets=["https://unpkg.com/purecss@2.1.0/build/pure-min.css" ]
my_app=dash.Dash('My App',external_stylesheets=external_stylesheets)
tabs_styles = {
    'height': '45px'
}
tab_style = {
    'borderBottom': '2px solid #0DF66C',
    'borderTop': '2px solid #0DF66C',
    'borderRight': '2px solid #0DF66C',
    'fontWeight': 'bold',
    'backgroundColor': '#111111',
    'textAlign': 'center',
    'color':'#0DF66C',
    'padding': '6px'

}

tab_selected_style = {
    'borderTop': '2px solid #111111',
    'borderBottom': '2px solid #111111',
    'borderRight':'2px solid #111111',
    'padding': '6px',
    'fontWeight': 'bold',
    'backgroundColor': '#0DF66C',
    'textAlign':'center',
    'color': '#111111'
}

#Design the tab layout
my_app.layout=html.Div(style={'backgroundColor':'#111111',
                              'textAlign':'center','color':'#0DF66C'},
                       children=[html.Img(src='https://wallpaperaccess.com/full/1373294.jpg', style={'display':'inline-block','height':'2.5%', 'width':'2.5%'}),
                                 html.H1(' Analysis of Spotify Data ',style={'display':'inline-block'}),
                                 html.Img(src='https://wallpaperaccess.com/full/1373294.jpg', style={'display':'inline-block','height':'2.5%', 'width':'2.5%'}),
                                 dcc.Tabs(id='tabs1',children=[
                                     dcc.Tab(label='Know your Data',value='Know your Data',style=tab_style, selected_style=tab_selected_style),
                                     dcc.Tab(label='Oultier Analysis',value='Outlier Analysis',style=tab_style, selected_style=tab_selected_style),
                                     dcc.Tab(label='PCA',value='PCA',style=tab_style, selected_style=tab_selected_style),
                                     dcc.Tab(label='Normality Tests',value='Normality Tests',style=tab_style, selected_style=tab_selected_style),
                                     dcc.Tab(label='Heatmap',value='Heatmap',style=tab_style, selected_style=tab_selected_style),
                                     #dcc.Tab(label='Statistics',value='Statistics',style=tab_style, selected_style=tab_selected_style),
                                     dcc.Tab(label='Analysis',value='Analysis',style=tab_style, selected_style=tab_selected_style),
                                     dcc.Tab(label='Dashboard',value='Dashboard',style=tab_style, selected_style=tab_selected_style),
                                     dcc.Tab(label='Summary',value='Summary',style=tab_style, selected_style=tab_selected_style)],style=tabs_styles,
                                          value='Know your Data'),
                                 html.Div(id='layout',style={'backgroundColor':'#0DF66C','color':'#111111'})

                                 ])

#Layput for first tab
tab1_layout=html.Div(style={'backgroundColor':'#0DF66C','color':'#111111'},
                     children=[html.Br()
                         ,html.H3('About the Dataset:',style={'margin': '0','textAlign':'left'
}),
                               html.P('The dataset contains more than 160,000 songs collected from Spotify Web API. The dataset is from Spotify and contains 169k songs from the year 1921 to year 2020. Each year got top 100 songs.',style={'display':'inline-block','margin': '0','textAlign':'left'}),
                               html.Hr(style={'border': '1px solid black'}),
                               html.H3('More about Data:',style={'margin': '1px','textAlign':'left'}),
                               html.P('Click one option to understand the basic information about the data!',style={'margin': '1px','textAlign':'left'}),
                               dcc.RadioItems(id='infos',options=[
                                   {'label':'Column Names','value':'Column'},
                                   {'label':'Count of rows','value':'rows'},
                                   {'label':'Count of columns','value':'columns'
                               }],
                               value='Column',inputStyle={"margin-left": "20px"}),
                               html.Plaintext(id='datainfo',style = {'backgroundColor':'#111111','color':'#0DF66C','font-size':'15px'}),
                               html.Hr(style={'border': '1px solid black'}),
                               html.H3('Data Preprocessing:',style={'margin': '1px','textAlign':'left'}),
                               html.P('Choose an option:',style={'margin': '1px','textAlign':'left'}),
                               dcc.RadioItems(id='cleans', options=[
                                   {'label': 'Check for Null values', 'value': 'nulls'},

                                   {'label': 'Statistics', 'value': 'stats'},
                                   {'label':'Display data','value':'head'
                                    }],value='Column',inputStyle={"margin-left": "20px"}),
                               html.Plaintext(id='preprocess',style = {'backgroundColor':'#111111','color':'#0DF66C','font-size':'15px'}),
                               html.Hr(style={'border': '1px solid black'}),
                               html.H3('Download Data:', style={'margin': '1px', 'textAlign': 'left'}),
                               html.P('Click to download the dataset!', style={'margin': '1px', 'textAlign': 'left'}),
                               html.Button("Download CSV", id="btn_csv",style={'background-color':'#111111','color':'#0DF66C'}),
                               dcc.Download(id="download-dataframe-csv")

])

#Outlier detection and removal
data1 = data.copy()
cols_out = ['danceability','duration_ms','instrumentalness','tempo','liveness','loudness','speechiness']
for i in cols_out:

    q1_h, q2_h, q3_h = data1[i].quantile([0.25, 0.5, 0.75])

    IQR_h = q3_h - q1_h
    lower1 = q1_h - 1.5 * IQR_h
    upper1 = q3_h + 1.5 * IQR_h
    data1 = data1[(data1[i] > lower1) & (data1[i] < upper1)]
    print(f'Q1 and Q3 of the {i} is {q1_h:.2f}  & {q3_h:.2f} \n IQR for the {i} is {IQR_h:.2f} \nAny {i} < {lower1:.2f}  and {i} > {upper1:.2f}  is an outlier')


#Design for second tab layout
tab2_layout=html.Div(style={'backgroundColor':'#0DF66C','color':'#111111'},
                     children=[html.Br(),
                        html.H3('Outlier Detection: An analysis of numeric variables using boxplot',style={'margin': '0','textAlign':'left'}),
                        html.P('Choose a variables to view the boxplot:',style={'margin': '1px','textAlign':'left'}),
                        dcc.Dropdown(id='drop1',
                                            options=[
                                                {'label': 'acousticness', 'value': 'acousticness'},
                                                {'label': 'danceability', 'value': 'danceability'},
                                                {'label': 'energy', 'value': 'energy'},
                                                {'label': 'duration_ms', 'value': 'duration_ms'},
                                                {'label': 'instrumentalness', 'value': 'instrumentalness'},
                                                {'label': 'valence', 'value': 'valence'},
                                                {'label': 'tempo', 'value': 'tempo'},
                                                {'label': 'liveness', 'value': 'liveness'},
                                                {'label': 'loudness', 'value': 'loudness'},
                                                {'label': 'speechiness', 'value': 'speechiness'},

                                            ], value='acousticness', clearable=False,style={'width': '200px'}),
                               html.Br(),
                               dcc.Graph(id='graphbox1',style={'width':'800px','height':'500px'}),
                               html.Hr(style={'border': '1px solid black'}),
                               html.H3('Outlier Removal:IQR method', style={'margin': '1px', 'textAlign': 'left'}),
                               html.P('The outliers from following variables were removed: danceability, duration_ms, instrumentalness, tempo, liveness, loudness, speechiness',style = {'backgroundColor':'#111111','color':'#0DF66C','font-size':'15px'}),
                               html.Hr(style={'border': '1px solid black'}),
                        html.H3('Outlier Removal: An analysis of numeric variables using boxplot',style={'margin': '0','textAlign':'left'}),
                        html.P('Choose a variables to view the boxplot:',style={'margin': '1px','textAlign':'left'}),
                        dcc.Dropdown(id='drop2',
                                            options=[
                                                {'label': 'acousticness', 'value': 'acousticness'},
                                                {'label': 'danceability', 'value': 'danceability'},
                                                {'label': 'energy', 'value': 'energy'},
                                                {'label': 'duration_ms', 'value': 'duration_ms'},
                                                {'label': 'instrumentalness', 'value': 'instrumentalness'},
                                                {'label': 'valence', 'value': 'valence'},
                                                {'label': 'tempo', 'value': 'tempo'},
                                                {'label': 'liveness', 'value': 'liveness'},
                                                {'label': 'loudness', 'value': 'loudness'},
                                                {'label': 'speechiness', 'value': 'speechiness'},

                                            ], value='acousticness', clearable=False,style={'width': '200px'}),
                               html.Br(),
                               dcc.Graph(id='graphbox2',style={'width':'800px','height':'500px'}),
])


#PCA
Features=data._get_numeric_data().columns.to_list()[:-1]
x=data[data._get_numeric_data().columns.to_list()[:-1]]

x=x.values
x=StandardScaler().fit_transform(x)

pca=PCA(n_components='mle',svd_solver='full')
pca.fit(x)
x_pca=pca.transform(x)

#plot of cumsum
number_of_components=np.arange(1,len(np.cumsum(pca.explained_variance_ratio_))+1)
fig=px.line(x=number_of_components,y=np.cumsum(pca.explained_variance_ratio_))
fig.update_layout(title='Cumulative Explained Variance')

#svd and condition number
H=np.matmul(x.T,x)
_,d,_=np.linalg.svd(H)


#svd and condition number-tranformed
H_pca=np.matmul(x_pca.T,x_pca)
_,d_pca,_=np.linalg.svd(H_pca)

#PCA correlation matrix
fig1=px.imshow(pd.DataFrame(x_pca).corr())
#Better visuals
plt.figure(figsize=(20,20))
sns.heatmap(pd.DataFrame(x_pca).corr(), annot=True)
plt.title('correlation plot of PCA features')
plt.show()

#Design for third tab
tab3_layout=html.Div(style={'backgroundColor':'#0DF66C','color':'#111111'},
                     children=[html.Br(),
                               html.H3('Principal Component Analysis',
                                       style={'margin': '0', 'textAlign': 'left'}),
                               html.P('Choose options to view outputs of PCA:',
                                      style={'margin': '1px', 'textAlign': 'left'}),
                               dcc.RadioItems(id='checkpca',options=[
                                   {'label':'Original Space','value':'Original'},
                                   {'label':'Transformed Space','value':'tranformed'}],value='Original',inputStyle={"margin-left": "20px"}),
                               html.Plaintext(id='pcaout',style = {'backgroundColor':'#111111','color':'#0DF66C','font-size':'15px'}),
                               html.Hr(style={'border': '1px solid black'}),
                               html.H3('Cumulative Explained Variance:',
                                       style={'margin': '0', 'textAlign': 'left'}),
                               html.Br(),
                               dcc.Graph(figure=fig,style={'width':'800px','height':'500px'}),
                               html.Hr(style={'border': '1px solid black'}),
                               html.H3('PCA features correlation matrix:',
                                       style={'margin': '0', 'textAlign': 'left'}),
                               html.Br(),
                               dcc.Graph(figure=fig1, style={'width': '800px', 'height': '500px'})
                               ])

#Design for tab4
tab4_layout=html.Div(style={'backgroundColor':'#0DF66C','color':'#111111'},
                     children=[html.H3('Normality Tests',style={'margin': '0', 'textAlign': 'left'}),
                               html.Br(),
                        html.P('Choose variable:',style={'margin': '1px', 'textAlign': 'left'}),
                        dcc.Dropdown(id='dropvar',
                            options = [
                                          {'label': 'acousticness', 'value': 'acousticness'},
                                          {'label': 'danceability', 'value': 'danceability'},
                                          {'label': 'energy', 'value': 'energy'},
                                          {'label': 'duration_ms', 'value': 'duration_ms'},
                                          {'label': 'instrumentalness', 'value': 'instrumentalness'},
                                          {'label': 'valence', 'value': 'valence'},
                                          {'label': 'tempo', 'value': 'tempo'},
                                          {'label': 'liveness', 'value': 'liveness'},
                                          {'label': 'loudness', 'value': 'loudness'},
                                          {'label': 'speechiness', 'value': 'speechiness'},
                                          {'label': 'popularity', 'value': 'popularity'},

                                      ], value = 'acousticness',style={'width': '200px'},clearable=False),
                        html.Br(),
                        html.P('Choose the test',style={'margin': '1px', 'textAlign': 'left'}),
                        dcc.Dropdown(id='droptest',options=[
                            {'label':'normaltest','value':'normaltest'},
                            {'label': 'kstest', 'value': 'kstest'},
                            {'label': 'shapiro', 'value': 'shapiro'}
                        ],value='normaltest',style={'width': '200px'}),
                               html.Br(),
                        html.Plaintext(id='ntout',style = {'backgroundColor':'#111111','color':'#0DF66C','font-size':'15px'}),
                        html.Hr(style={'border': '1px solid black'}),

])

#Correlation coefficient matrices
fig2=px.imshow(data.corr())

#Design tab5 layout
tab5_layout=html.Div(style={'backgroundColor':'#0DF66C','color':'#111111'},
                     children=[html.H3('Pearson correlation coefficient: Heatmap',style={'margin': '0', 'textAlign': 'left'}),
                               html.Br(),
                               dcc.Graph(figure=fig2,style={'width': '800px', 'height': '600px'})])

#Design for tab6
tab6_layout=html.Div(style={'backgroundColor':'#0DF66C','color':'#111111'},
                     children=[html.H3('Visualize data using various plots',style={'margin': '0', 'textAlign': 'left'}),
                               html.P('Choose variable:',style={'margin': '1px', 'textAlign': 'left','display':'inline-block'}),
                            dcc.Dropdown(id='dropline',
                            options = [
                                          {'label': 'acousticness', 'value': 'acousticness'},
                                          {'label': 'danceability', 'value': 'danceability'},
                                          {'label': 'energy', 'value': 'energy'},
                                          {'label': 'duration_ms', 'value': 'duration_ms'},
                                          {'label': 'instrumentalness', 'value': 'instrumentalness'},
                                          {'label': 'valence', 'value': 'valence'},
                                          {'label': 'tempo', 'value': 'tempo'},
                                          {'label': 'liveness', 'value': 'liveness'},
                                          {'label': 'loudness', 'value': 'loudness'},
                                          {'label': 'speechiness', 'value': 'speechiness'},
                                          {'label': 'popularity', 'value': 'popularity'},

                                      ], value = 'acousticness',style={'width': '200px','display':'inline-block'},clearable=False),
                        html.P('Choose variable to color:',style={'display':'inline-block','margin': '1px', 'textAlign': 'left'}),
                        dcc.Dropdown(id='dropcolor',
                            options = [
                                          {'label': 'mode', 'value': 'mode'},
                                          {'label': 'explicit', 'value': 'explicit'},
                                          {'label': 'key', 'value': 'key'},

                                      ], value = 'mode',style={'display':'inline-block','width': '200px'},clearable=False),
                        html.Br(),

                        html.P('Line Plot:',style={'margin': '1px', 'textAlign': 'left'}),
                        dcc.Graph(id='line', style={'width': '800px', 'height': '400px'}),
                        html.Br(),
                        html.P('Histogram:',style={'margin': '1px', 'textAlign': 'left'}),
                        dcc.Slider(id='bins',min=20,max=100,value=50,tooltip={"placement": "bottom", "always_visible": True}),
                        dcc.Graph(id='bar', style={'width': '800px', 'height': '400px'}),
                        html.P('Distplots:',style={'margin': '1px', 'textAlign': 'left'}),
                        html.P("Select Distribution:",style={'margin': '1px', 'textAlign': 'left'}),
                        dcc.RadioItems(
                        id='distribution',
                        options=[
                                    {'label':'box','value':'box'},
                                   {'label':'violin','value':'violin'},
                            {'label':'rug','value':'rug'}
                        ],

                        value='box',inputStyle={"margin-left": "20px"}),
                        dcc.Graph(id="graphd",style={'width': '800px', 'height': '400px'}),
                        ])

#Design for Tab7

#First column
figmode=px.pie(data,names='mode',values='popularity')
figexp=px.pie(data,names='explicit',values='popularity')
figkey=px.pie(data,names='key',values='popularity')

#Second column
f=['acousticness','danceability','instrumentalness','popularity','liveness']
figscatter=px.scatter_matrix(data,
                      dimensions=f,
                      color='mode',
                             labels={
'acousticness':'acoustic','danceability':'dance','instrumentalness':'instrument','popularity':'popular','liveness':'live'
                             })
figscatter.update_layout(yaxis=dict(tickangle = 45),xaxis=dict(tickangle = 45))

#scatterplot
figscatter2 = px.scatter(data,y='energy',x='popularity',color='mode',trendline='ols')

figscatter3 = px.scatter(data,y='liveness',x='popularity',color='mode',trendline='ols')


tab7_layout=html.Div([html.H3('Welcome to Dashboard!',style={'margin': '0', 'textAlign': 'left'}),
                      html.Br(),
    html.Div(style={'backgroundColor':'#0DF66C','color':'#111111','padding':5, 'flex': 1},
        children=[
            html.P('Pie Chart of Mode based on Popularity'),
            dcc.Graph(figure=figmode),
            html.P('Pie Chart of Explicit based on Popularity'),
            dcc.Graph(figure=figexp),
            html.P('Pie Chart of Key based on Popularity'),
            dcc.Graph(figure=figkey),

        ],
    ),
html.Div(style={'backgroundColor':'#0DF66C','color':'#111111','padding':5, 'flex': 1},
        children=[
            html.P('Scatter matrix'),
            dcc.Graph(figure=figscatter),
            html.P('Scatter Plot'),
            dcc.Graph(figure=figscatter2),
            dcc.Graph(figure=figscatter3),
        ]
    ),
],style={'display': 'flex', 'flex-direction': 'row','backgroundColor':'#0DF66C','color':'#111111'})


#Design for tab8
tab8_layout=html.Div(style={'backgroundColor':'#0DF66C','color':'#111111'},
                     children=[html.H3('Summary',style={'margin': '0', 'textAlign': 'left'}),
                               html.Br(),
                               html.P('The developed dash application helps user to visualize the various aspects of songs from spotify collections.\n Every year has top 100 songs and analysis of available features helps in predicting the popularity of the song. The users can navigate through different tabs to understand the data from the stage of preprocessing to analysis of features at different levels. ',style={'margin': '1px', 'textAlign': 'left'}),
                               html.Hr(style={'border': '1px solid black'}),
                               html.H3('References',style={'margin': '0', 'textAlign': 'left'}),
                               html.Br(),
                               html.Plaintext(' 1.https://dash.plotly.com/dash-core-components \n 2.https://dash.plotly.com/dash-html-components \n 3.https://dash.plotly.com/advanced-callbacks \n 4.https://plotly.com/python/box-plots/ \n 5.https://plotly.com/python/histograms/ \n 6. https://plotly.com/python/distplot/',style={'margin': '1px', 'textAlign': 'left'}),
                               html.Hr(style={'border': '1px solid black'}),
                               html.H3('Author Information',style={'margin': '0', 'textAlign': 'left'}),

                               html.Plaintext('Please feel free to drop an email if you have any questions or suggestions to improve the app!\nCreated by: Rehapriadarsini Manikandasamy\nEmail:rehamanikandan@gwu.edu',style = {'backgroundColor':'#111111','color':'#0DF66C','font-size':'15px'}),
                               html.Plaintext('Data Source: Kaggle \n Link to Dataset: https://www.kaggle.com/datasets/ektanegi/spotifydata-19212020 \n *The app has been created for 6401-Visulaization of Complex Data coursework at The George Washington University*',style = {'backgroundColor':'#111111','color':'#0DF66C','font-size':'15px'})
                               ]
                     )

#Main callback for the main layout
@my_app.callback(Output(component_id='layout',component_property='children'),
                 [Input(component_id='tabs1',component_property='value')
                ])

def update_layout(tabselect):
    if tabselect=='Know your Data':
        return tab1_layout
    elif tabselect=='Outlier Analysis':
         return tab2_layout
    elif tabselect=='PCA':
         return tab3_layout
    elif tabselect=='Normality Tests':
         return tab4_layout
    elif tabselect=='Heatmap':
         return tab5_layout
    elif tabselect=='Analysis':
         return tab6_layout
    elif tabselect=='Dashboard':
        return tab7_layout
    elif tabselect=='Summary':
        return tab8_layout

#Callback for tab1
@my_app.callback(Output(component_id='datainfo',component_property='children'),
                 [Input(component_id='infos',component_property='value')])
def update_graph(input):
    if input=='Column':
        cols=data.columns
        return ['\n'+j for j in cols]
    elif input=='rows':
         i=len(data)
         return f'Number of rows:{i}'
    elif input=='columns':
         cols=data.columns
         return f'Number of columns:{len(cols)}'

@my_app.callback(Output(component_id='preprocess',component_property='children'),
                 [Input(component_id='cleans',component_property='value')])
def update_graph(input):
    if input=='nulls':
        d=data.isnull().sum()
        return f'{d}\nDataset is cleaned already!'
    if input=='stats':
        return f'{data.describe()}'
    elif input=='head':
        pd.set_option('max_columns',6)
        return f'{data.head()}'

@my_app.callback(
    Output("download-dataframe-csv", "data"),
    Input("btn_csv", "n_clicks"),
    prevent_initial_call=True,
)
def func(n_clicks):
    return dcc.send_data_frame(data.to_csv, "Spotify_data.csv")


#Callbacks fro tab2 components
@my_app.callback(Output(component_id='graphbox1',component_property='figure'),
                 [Input(component_id='drop1',component_property='value')])
def update_graph(input):
    fig = px.box(data,y=input)
    fig.update_layout(title='Box plot')
    return fig

@my_app.callback(Output(component_id='graphbox2',component_property='figure'),
                 [Input(component_id='drop2',component_property='value')])
def update_graph(input):
    fig = px.box(data1,y=input)
    fig.update_layout(title='Box plot')
    return fig

#PCA callbacks
@my_app.callback(Output(component_id='pcaout',component_property='children'),
                 [Input(component_id='checkpca',component_property='value')])
def update_graph(input):
    if input=='Original':
        return f'Features:{Features[:6]}\n{Features[6:]}\n\nOriginal Shape:{x.shape}\n\nSingular values:{d}\n\nCondition number:{la.cond(x)}'
    elif input=='tranformed':
        return f'Transformed shape:{x_pca.shape}\n\nSingular values:{d_pca}\n\nCondition number:{la.cond(x_pca)}\n\nExplained Variance Ratio:{pca.explained_variance_ratio_}'

#Normality callbacks
@my_app.callback(
    Output(component_id='ntout',component_property='children'),
    [Input(component_id='dropvar',component_property='value'),
     Input(component_id='droptest',component_property='value')]
)
def tests(inp,inp2):
    f1=data[inp]
    if inp2=='normaltest':
        return f'Normal test:{normaltest(f1)}'
    elif inp2=='kstest':
        res=kstest(f1,'norm')
        return f'KS test:{res}'
    else:
        return f'Shapiro Wilk Test:{shapiro(f1)}'
#Analysis callbacks
@my_app.callback(Output(component_id='line',component_property='figure'),
                Output(component_id='bar',component_property='figure'),
                Output(component_id='graphd',component_property='figure'),
                 [Input(component_id='dropline',component_property='value'),
                  Input(component_id='dropcolor',component_property='value'),
                  Input(component_id='bins',component_property='value'),
                  Input(component_id='distribution',component_property='value')])
def update_graph(input,inp2,inp3,inp4):
    fig = px.line(data,x='year',y=input,color=inp2)
    fig1=px.histogram(data,x=input,nbins=inp3)
    fig3 = px.histogram(data, x='popularity', y=input, color=inp2,
        marginal=inp4)
    return fig,fig1,fig3


my_app.run_server(port=8100, host='0.0.0.0')
import dash
from dash.dash_table.Format import Group
import dash_table
from dash import dcc
from dash import html
import plotly.graph_objs as go
from dash.dependencies import State, Input, Output
from dash.exceptions import PreventUpdate
import geopandas as gpd
import plotly.graph_objects as go
import json
import pandas as pd
import os
import dash_bootstrap_components as dbc
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
from dash_table import DataTable
import colorsys
import geopandas as gpd
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN
import pandas as pd
import datetime
import requests
import plotly.express as px
import datetime
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


data = pd.read_csv("geocoded_data.csv")

app = dash.Dash(
    __name__,
    meta_tags=[
        {
            "name": "viewport",
            "content": "width=device-width, initial-scale=1, maximum-scale=1.0, user-scalable=no",
        }
    ],
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)
app.title = "Flood Monitoring Analytics"
server = app.server

app.config["suppress_callback_exceptions"] = True

def build_upper_left_panel():
    return html.Div(
        id="upper-left",
        className="six columns",
        style={
            "backgroundColor": "#e0f7e9",  # light green background
            "padding": "20px",
            "borderRadius": "10px",
            "border": "2px solid #32CD32"
        },
        children=[
            html.H3(
                className="section-title",
                children="Please Select the Station",
                style={
                    "color": "#2E8B57",
                    "fontFamily": "Arial",
                    "textAlign": "center",
                    "marginBottom": "15px"
                }
            ),
            html.P(
                "Welcome to our station selection panel! Enjoy a vibrant green-themed interface designed to enhance your experience.",
                style={
                    "color": "#006400",
                    "fontSize": "14px",
                    "fontFamily": "Arial",
                    "margin": "10px 0"
                }
            ),
            html.Img(src=app.get_asset_url("logo.gif")),  # Replace with a valid image path
            html.Div(
                className="control-row-1",
                children=[
                    html.Div(
                        id="county-select-outer",
                        style={
                            "backgroundColor": "#ccffcc",
                            "padding": "15px",
                            "borderRadius": "5px",
                            "marginBottom": "15px"
                        },
                        children=[
                            html.Label(
                                "Choose from the dropdown below:",
                                style={
                                    "color": "#228B22",
                                    "fontWeight": "bold",
                                    "fontSize": "16px"
                                }
                            ),
                            dcc.Dropdown(
                                id="station-select",
                                options=[
                                    {"label": str(station), "value": str(station)}
                                    for station in data["station_id"].unique() if pd.notna(station)
                                ],
                                value=str(data["station_id"].unique()[0]),
                                style={
                                    "backgroundColor": "#98fb98",
                                    "color": "black",
                                    "border": "2px solid #32CD32",
                                    "borderRadius": "5px",
                                    "fontFamily": "Arial",
                                    "fontSize": "16px",
                                },
                            )
                        ],
                    ),
                ],
            ),
            html.P(
                "After selecting a station, you'll be able to view detailed data and analytics on the right panel. "
                "We continuously update our platform to provide you with the best insights.",
                style={
                    "color": "#2E8B57",
                    "fontSize": "14px",
                    "fontFamily": "Arial",
                    "margin": "10px 0"
                }
            ),
            html.P(
                "For any questions or further assistance, please contact our support team. We hope you enjoy exploring our data!",
                style={
                    "color": "#006400",
                    "fontSize": "12px",
                    "fontFamily": "Arial",
                    "margin": "10px 0",
                    "fontStyle": "italic"
                }
            )
        ],
    )


def generate_colors(n):
    colors = []
    for i in range(n):
        hue = i / n
        rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        colors.append(hex_color)
    return colors

def generate_geo_map(plot_data):
    geojson_file = 'uk.geojson'
    gdf = gpd.read_file(geojson_file)
    center_lat = gdf.geometry.centroid.y.mean()
    center_lon = gdf.geometry.centroid.x.mean()
    
    scatter_data = go.Scattermapbox(
        lat=plot_data['lat'],
        lon=plot_data['long'],
        mode='markers',
        marker=dict(
            size=8,
            color=plot_data['point_color'],
            opacity=plot_data['opacity'],
            showscale=False,
        ),
        text=plot_data.apply(
            lambda row: f"Town: {row['town']}<br>"
                        f"Label: {row['label']}<br>"
                        f"Station ID: {row['station_id']}<br>"
                        f"Notation: {row['notation']}<br>"
                        f"Lat: {row['lat']:.4f}<br>"
                        f"Long: {row['long']:.4f}",
            axis=1
        ),
        hoverinfo='text',
        customdata=plot_data['station_id']
    )
    
    layout = go.Layout(
        margin=dict(l=10, r=10, t=20, b=10, pad=5),
        plot_bgcolor="black",
        paper_bgcolor="black",
        clickmode="event+select",
        hovermode="closest",
        showlegend=False,
        mapbox=go.layout.Mapbox(
            center=go.layout.mapbox.Center(
                lat=center_lat,
                lon=center_lon
            ),
            zoom=6,
            style="carto-darkmatter"
        )
    )
    
    return {"data": [scatter_data], "layout": layout}



app.layout = html.Div(
    className="container scalable",
    children=[
        html.Div(
            id="banner",
            className="banner",
            children=[
                html.Img(src=app.get_asset_url("defra.svg"),className="logo"),
                html.H4("Real time flood monitoring analytics"),

            ],
        ),
        html.Div(
            id="upper-container",
            className="row",
            children=[
                build_upper_left_panel(),
                html.Div(
                    id="geo-map-outer",
                    className="six columns",
                    children=[
                            html.Div(id="station-stats-table"),
                        html.Div(
                            id="geo-map-loading-outer",
                            children=[
                                dcc.Loading(
                                    id="loading",
                                    children=dcc.Graph(
                                        id="geo-map",
                                        figure={
                                            "data": [],
                                            "layout": dict(
                                                plot_bgcolor="#171b26",
                                                paper_bgcolor="white",
                                            ),
                                        },
                                    ),
                                )
                            ],
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            id="lower-container",
            children=[
                dcc.Graph(
                    id="line-plot",
                    

                )
            ],
        ),
    ],
)



@app.callback(
    [Output("geo-map", "figure"),
     Output("station-select", "value"),  # Update the dropdown value when a point is clicked
     Output("station-stats-table", "children")],
    [Input("geo-map", "clickData"),
     Input("station-select", "value")]
)
def update_geo_map_and_ui(click_data, dropdown_value):
    # Determine which input triggered the callback
    ctx = dash.callback_context
    triggered_input = ctx.triggered[0]["prop_id"].split('.')[0] if ctx.triggered else None

    # Prioritize map click only if it was the triggering event
    if triggered_input == "geo-map" and click_data is not None:
        selected_station = click_data['points'][0]['customdata']
    else:
        selected_station = dropdown_value

    data_modified = data.copy()

    def convert_coord(val, index):
        if isinstance(val, str) and ',' in val:
            parts = val.strip("[]").split(",")
            return float(parts[index].strip())
        else:
            return float(val)

    data_modified['lat'] = data_modified['lat'].apply(lambda x: convert_coord(x, 0))
    data_modified['long'] = data_modified['long'].apply(lambda x: convert_coord(x, 1))

    # Clustering
    X = data_modified[['lat', 'long']].to_numpy()
    clustering = DBSCAN(eps=0.01, min_samples=1).fit(X)
    data_modified['cluster'] = clustering.labels_
    unique_clusters = data_modified['cluster'].unique()
    cluster_colors = generate_colors(len(unique_clusters))
    color_mapping = dict(zip(unique_clusters, cluster_colors))
    data_modified['point_color'] = data_modified['cluster'].map(color_mapping)
    data_modified['opacity'] = 1.0

    # Update the map points' opacity and color based on the selected station
    data_modified.loc[data_modified['station_id'] != selected_station, 'opacity'] = 0.2
    data_modified.loc[data_modified['station_id'] == selected_station, "point_color"] = 'white'

    # Update the table with the selected station's details
    station_df = data[data["station_id"] == selected_station]
    selected_columns = ["notation", "riverName", "town", "label", "station_id"]
    filtered_df = station_df[selected_columns]

    table_data = filtered_df.to_dict("records")
    table_columns = [{"name": col, "id": col} for col in selected_columns]

    table_component = html.Div(
        DataTable(
            id="station-stats-table-table",
            columns=table_columns,
            data=table_data,
            filter_action="native",
            page_size=5,
            style_cell={
                "background-color": "#2e4a3a",  # Dark green background
                "color": "#c1d8c5",  # Light greenish text color
            },
            style_header={
                "background-color": "#1b3521",  # Darker green header background
                "color": "#ffffff",  # White text for the header
                "padding": "0px 5px",
            },
        )
    )

    # Return the updated map, the selected station for the dropdown, and the table component
    return generate_geo_map(data_modified), selected_station, table_component


@app.callback(
    Output("line-plot", "figure"),
    [Input("station-select", "value")]
)
def update_line_chart(selected_station):
    if not selected_station:
        raise PreventUpdate
    
    now_utc = datetime.datetime.utcnow()
    since_str = (now_utc - datetime.timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    url = f"https://environment.data.gov.uk/flood-monitoring/id/stations/{selected_station}/readings?_sorted&since={since_str}&_limit=1000"
    response = requests.get(url)
    
    if response.status_code != 200:
        return go.Figure()
    
    data = response.json()
    
    def get_measure_type(measure_url):
        return measure_url.split('/')[-1].split('-')[2]
    
    readings_list = [
        {
            "dateTime": item["dateTime"],
            "measure_type": get_measure_type(item["measure"]),
            "value": item["value"],
        }
        for item in data.get("items", [])
    ]
    
    df = pd.DataFrame(readings_list)
    
    if df.empty:
        return go.Figure()
    
    df["dateTime"] = pd.to_datetime(df["dateTime"])
    measure_types = df["measure_type"].unique()
    
    fig = make_subplots(
        rows=len(measure_types), 
        cols=1, 
        shared_xaxes=True,
        subplot_titles=[f"{m_type}" for m_type in measure_types]
    )
    
    for i, m_type in enumerate(measure_types, start=1):
        df_subset = df[df["measure_type"] == m_type]
        fig.add_trace(
            go.Scatter(
                x=df_subset["dateTime"],
                y=df_subset["value"],
                mode="lines+markers",
                marker=dict(color="green"),
                name=m_type,
            ),
            row=i,
            col=1,
        )
        fig.update_yaxes(title_text="Measurement Value", row=i, col=1)
    
    fig.update_xaxes(title_text="DateTime", tickangle=-45)
    fig.update_layout(
        title_text=f"Flood Monitoring Readings for Station {selected_station}",
        template="plotly_dark",
        hovermode="x unified",
        height=300 * len(measure_types) 
    )
    
    return fig



if __name__ == "__main__":
    app.run_server(debug=True)

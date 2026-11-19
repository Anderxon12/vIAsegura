from dash import Dash, dcc, html
import plotly.graph_objects as go

app = Dash(__name__)

fig = go.Figure(go.Scattermap())
fig.update_layout(
    map=dict(
        style="open-street-map",
        center=dict(lat=4.6534, lon=-74.0836),
        zoom=11,
    ),
    margin=dict(l=0, r=0, t=0, b=0),
    paper_bgcolor="#0D1117",
)

app.layout = html.Div([
    dcc.Graph(
        id="mapa",
        figure=fig,
        config={"scrollZoom": True, "displayModeBar": False},
        style={"height": "100vh"},
    )
])

if __name__ == "__main__":
    app.run(debug=True)

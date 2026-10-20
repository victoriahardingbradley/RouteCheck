"""Plotly figures; no network tiles or map API keys needed."""
import plotly.graph_objects as go
from plotly.subplots import make_subplots

GREEN, GRAY, ORANGE = "#147D68", "#98A8B7", "#E99B42"


def style(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=20,r=20,t=35,b=25),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Arial",color="#324A5C",size=12),
                      legend=dict(orientation="h",y=1.12,x=0),
                      hoverlabel=dict(bgcolor="white"))
    fig.update_xaxes(gridcolor="#E8EDF1", zeroline=False)
    fig.update_yaxes(gridcolor="#E8EDF1", zeroline=False)
    return fig


def route_map(result, dataset):
    fig = go.Figure()
    for label, key, color, dash in [("Before", "before", GRAY, "dot"), ("Current", "after", GREEN, "solid")]:
        trace = result[key]["trace"]
        fig.add_trace(go.Scatter(x=[p["lon"] for p in trace], y=[p["lat"] for p in trace],
            mode="lines+markers", name=label, line=dict(color=color,width=2,dash=dash),
            marker=dict(size=6), text=[f'{p["sequence"]} · {p["stop_id"]}' for p in trace],
            hovertemplate="%{text}<extra>%{fullData.name}</extra>"))
    affected = [s for s in dataset["stops"] if s["stop_id"] in {i["stop_id"] for i in dataset["issues"]} and s["lat"] is not None and s["lon"] is not None]
    fig.add_trace(go.Scatter(x=[s["lon"] for s in affected], y=[s["lat"] for s in affected],
        mode="markers",name="Affected stop",marker=dict(size=13,color=ORANGE,symbol="circle-open",line=dict(width=2)),
        text=[s["stop_id"] for s in affected],hovertemplate="%{text}<extra>Affected stop</extra>"))
    fig.add_trace(go.Scatter(x=[dataset["depot"]["lon"]],y=[dataset["depot"]["lat"]],mode="markers",name="Depot",
        marker=dict(size=15,color="#182E3D",symbol="square")))
    fig.update_xaxes(title="Longitude", tickformat=".2f")
    fig.update_yaxes(title="Latitude", tickformat=".2f",scaleanchor="x",scaleratio=1.35)
    return style(fig,440)


def comparisons(result):
    metrics = [("Estimated distance · km","distance_km"),("Estimated time · min","duration_min"),
               ("Estimated on time · %","on_time_pct"),("Stops scheduled","completed")]
    fig = make_subplots(rows=2,cols=2,subplot_titles=[x[0] for x in metrics],vertical_spacing=.3)
    for i, (_, key) in enumerate(metrics):
        for label, version, color in [("Before","before",GRAY),("Current","after",GREEN)]:
            fig.add_trace(go.Bar(x=[label],y=[result[version][key]],name=label,marker_color=color,
                text=[result[version][key]],textposition="auto",showlegend=i==0,legendgroup=label),row=i//2+1,col=i%2+1)
    return style(fig,440)


def history_chart(rows, metric):
    fig = go.Figure()
    for label, version, color in [("Before","before",GRAY),("Current","after",GREEN)]:
        fig.add_trace(go.Scatter(x=[r["created_at"] for r in rows],y=[r["result"][version][metric] for r in rows],
            mode="lines+markers",name=label,line=dict(color=color,width=3),
            text=[r["id"] for r in rows],hovertemplate="%{x}<br>%{y}<br>Run %{text}<extra>%{fullData.name}</extra>"))
    return style(fig,370)


def study_plot(frame, condition):
    fig=go.Figure()
    for enabled,label,color in [(False,'Without countermeasure',GRAY),(True,'With countermeasure',GREEN)]:
        rows=frame[(frame.condition==condition)&(frame.countermeasures==enabled)].groupby('stage_pct',as_index=False).duration_min.mean()
        fig.add_trace(go.Scatter(x=rows.stage_pct,y=rows.duration_min,mode='lines+markers',name=label,line=dict(color=color,width=3)))
    fig.update_xaxes(title='Simulated degraded-feed share per type (%)')
    fig.update_yaxes(title='Reference replay duration (min)')
    return style(fig,400)

import plotly.graph_objects as go

months = ['True','False']
before = []
after  = []

fig = go.Figure()
fig.add_trace(go.Bar(
    x=months,
    y=[20, 0.8514851485148515],
    name='Accuracy'))
fig.add_trace(go.Bar(
    x=months,
    y=[19, 0.9454545454545454],
    name='Recall'
))
fig.add_trace(go.Bar(
    x=months,
    y=[20, 0.8125],
    name='Precision'))
fig.add_trace(go.Bar(
    x=months,
    y=[19, 14],
    name='F1'
))
fig.add_trace(go.Bar(
    x=months,
    y=[19, 14],
    name='Time'
))


# Here we modify the tickangle of the xaxis, resulting in rotated labels.
fig.update_layout(barmode='group', xaxis_tickangle=-45)
fig.update_layout(title='Wpływ selective_pruning', title_x=0.5, font=dict(size=20))
fig.show()

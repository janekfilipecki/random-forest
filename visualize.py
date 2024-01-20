import plotly.graph_objects as go

months = ['True','False','lol']
false = [0,8613861386138614,
0,9454545454545454,
0,8253968253968254,
0,8813559322033897, 1,531]
true  = [0,8547854785478548,
0,9454545454545454,
0,8167539267015707,
0,8764044943820224, 1,5166
]

fig = go.Figure()
fig.add_trace(go.Bar(
    x=months,
    y=[true[0], false[0]],
    name='Accuracy'))
fig.add_trace(go.Bar(
    x=months,
    y=[true[1], false[1]],
    name='Recall'
))
fig.add_trace(go.Bar(
    x=months,
    y=[true[2], false[2]],
    name='Precision'))
fig.add_trace(go.Bar(
    x=months,
    y=[true[3], false[3]],
    name='F1'
))
fig.add_trace(go.Bar(
    x=months,
    y=[true[4], false[4]],
    name='Time'
))


# Here we modify the tickangle of the xaxis, resulting in rotated labels.
fig.update_layout(barmode='group', xaxis_tickangle=-45)
fig.update_layout(title='Wpływ selective_pruning', title_x=0.5, font=dict(size=20))
fig.show()

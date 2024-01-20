import plotly.graph_objects as go

# Default
# months = ['Accuracy', 'Recall', 'Precision', 'F1', 'Time']
# false_data = [0.8613,
# 0.9454,
# 0.8253,
# 0.88135, 1.531]
# true_data  = [0.8547,
# 0.94545,
# 0.81675,
# 0.87640, 1.5166
# ]

# fig = go.Figure()
# fig.add_trace(go.Bar(
#     x=months,
#     y=true_data,
#     name='True'))
# fig.add_trace(go.Bar(
#     x=months,
#     y=false_data,
#     name='False'
# ))


# Max_depth:
months = [13,12,11,10,9]
acc = [0.8448844884488449, 0.8184818481848185, 0.8745874587458746, 0.834983498349835]
recall = [0.9696969696969697, 0.7393939393939394, 0.8666666666666667, 0.9515151515151515]
precision = [0.7920792079207921, 0.9104477611940298, 0.89937106918239, 0.7889447236180904]
f1 = [0.8719346049046321, 0.8160535117056856, 0.8827160493827161, 0.8626373626373627]
time=[14.010942, 13.881686, 11.230621, 8.765649]

fig = go.Figure()
fig.add_trace(go.Bar(
    x=months,
    y=acc,
    name='Accuracy'))
fig.add_trace(go.Bar(
    x=months,
    y=recall,
    name='Recall'
))
fig.add_trace(go.Bar(
    x=months,
    y=precision,
    name='Precision'
))
fig.add_trace(go.Bar(
    x=months,
    y=f1,
    name='F1'
))
fig.add_trace(go.Bar(
    x=months,
    y=time,
    name='Time'
))

fig.update_layout(barmode='group', xaxis_tickangle=-45)
fig.update_layout(title='Wpływ Max_depth na wartość parametrów', title_x=0.5, font=dict(size=20))
fig.show()

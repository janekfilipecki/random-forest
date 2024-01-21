import plotly.graph_objects as go
import numpy as np

secondsInMin = 60
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

# # Max_depth:
# months = [13,12,11,10,9]
# acc = np.round([0.858085808580858, 0.8712871287128713, 0.8448844884488449, 0.8613861386138614, 0.8547854785478548],3)
# recall = np.round([0.9393939393939394, 0.9757575757575757, 0.9212121212121213, 0.9333333333333333, 0.9454545454545454], 3)
# precision = np.round([0.824468085106383, 0.8214285714285714, 0.8172043010752689, 0.8324324324324325, 0.8167539267015707], 3)
# f1 = np.round([0.878186968838527, 0.8919667590027699, 0.8660968660968662, 0.88, 0.8764044943820224],3)
# time= [118.459764, 103.431641, 91.761967, 102.340069, 96.286574]
# time_minutes = np.round([x / secondsInMin for x in time], 3)

# fig = go.Figure()
# fig.add_trace(go.Bar(
#     x=months,
#     y=acc,
#     name='Accuracy',
#     text=acc,
#     textposition='outside'))
# fig.add_trace(go.Bar(
#     x=months,
#     y=recall,
#     name='Recall',
#     text=recall,
#     textposition='outside'
# ))
# fig.add_trace(go.Bar(
#     x=months,
#     y=precision,
#     name='Precision',
#     text=precision,
#     textposition='outside'
# ))
# fig.add_trace(go.Bar(
#     x=months,
#     y=f1,
#     name='F1',
#     text=f1,
#     textposition='outside'
# ))
# fig.add_trace(go.Bar(
#     x=months,
#     y=time_minutes,
#     name='Time [min]',
#     text=time_minutes,
#     textposition='outside'
# ))

# fig.update_layout(barmode='group', xaxis_tickangle=-45)
# fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
# fig.update_layout(title='Wpływ Max_depth na wartość parametrów, 50 drzew', title_x=0.5, font=dict(size=20))
# fig.show()



# Split_density:
months = [15,12,9,6,3]
acc = np.round([0.8514851485148515, 0.8811881188118812, 0.8844884488448845, 0.8514851485148515, 0.8448844884488449],3)
recall = np.round([0.9393939393939394, 0.9454545454545454, 0.9333333333333333, 0.9393939393939394, 0.9696969696969697], 3)
precision = np.round([0.8157894736842105, 0.8524590163934426, 0.8651685393258427, 0.8157894736842105, 0.7920792079207921], 3)
f1 = np.round([0.8732394366197183, 0.8965517241379309, 0.8979591836734695, 0.8732394366197183, 0.8719346049046321],3)
time= [68.767935, 58.563442, 41.894149, 29.331601, 14.128595]
time_minutes = np.round([x / secondsInMin for x in time], 3)





fig = go.Figure()
fig.add_trace(go.Bar(
    x=months,
    y=acc,
    name='Accuracy',
    text=acc,
    textposition='outside'))
fig.add_trace(go.Bar(
    x=months,
    y=recall,
    name='Recall',
    text=recall,
    textposition='outside'
))
fig.add_trace(go.Bar(
    x=months,
    y=precision,
    name='Precision',
    text=precision,
    textposition='outside'
))
fig.add_trace(go.Bar(
    x=months,
    y=f1,
    name='F1',
    text=f1,
    textposition='outside'
))
fig.add_trace(go.Bar(
    x=months,
    y=time_minutes,
    name='Time [min]',
    text=time_minutes,
    textposition='outside'
))

fig.update_layout(barmode='group', xaxis_tickangle=-45)
fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
fig.update_layout(title='Wpływ split_density na wartość parametrów, 50 drzew', title_x=0.5, font=dict(size=20))
fig.show()
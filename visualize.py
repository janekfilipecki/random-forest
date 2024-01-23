import plotly.graph_objects as go
import plotly.express as px
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
# months = [13,11,9,7,5,3]
# acc = np.round([0.8448844884488449, 0.8646864686468647, 0.8679867986798679, 0.8679867986798679, 0.8481848184818482, 0.8415841584158416],3)
# recall = np.round([0.9515151515151515, 0.9333333333333333, 0.9515151515151515, 0.9515151515151515, 0.9393939393939394, 0.9272727272727272], 3)
# precision = np.round([0.8010204081632653, 0.8369565217391305, 0.8306878306878307, 0.8306878306878307, 0.8115183246073299, 0.8095238095238095], 3)
# f1 = np.round([0.8698060941828254, 0.8825214899713467, 0.887005649717514, 0.887005649717514, 0.8707865168539326, 0.864406779661017],3)
# time= [46.120747, 48.077306, 46.397289, 47.65144, 42.894451, 25.848207]
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



# # Split_density:
# months = [15,12,9,6,3]
# acc = np.round([0.8514851485148515, 0.8811881188118812, 0.8844884488448845, 0.8514851485148515, 0.8448844884488449],3)
# recall = np.round([0.9393939393939394, 0.9454545454545454, 0.9333333333333333, 0.9393939393939394, 0.9696969696969697], 3)
# precision = np.round([0.8157894736842105, 0.8524590163934426, 0.8651685393258427, 0.8157894736842105, 0.7920792079207921], 3)
# f1 = np.round([0.8732394366197183, 0.8965517241379309, 0.8979591836734695, 0.8732394366197183, 0.8719346049046321],3)
# time= [68.767935, 58.563442, 41.894149, 29.331601, 14.128595]
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
# fig.update_layout(title='Wpływ split_density na wartość parametrów, 50 drzew', title_x=0.5, font=dict(size=20))
# fig.show()


# # gini_impurity:
# months = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
# acc = np.round([0.8877887788778878, 0.8481848184818482, 0.8448844884488449, 0.8250825082508251, 0.759075907590759, 0.759075907590759, 0.759075907590759],3)
# recall = np.round([0.9333333333333333, 0.9696969696969697, 0.9454545454545454, 0.9333333333333333, 0.7636363636363637, 0.7636363636363637, 0.7636363636363637], 3)
# precision = np.round([0.8700564971751412, 0.7960199004975125, 0.8041237113402062, 0.7857142857142857, 0.7875, 0.7875, 0.7875], 3)
# f1 = np.round([0.9005847953216375, 0.8743169398907105, 0.8690807799442897, 0.8531855955678669, 0.7753846153846152, 0.7753846153846152, 0.7753846153846152],3)
# time= [45.474342, 37.951341, 28.774322, 14.732761, 4.43687, 4.444344, 4.42526]
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
# fig.update_layout(title='Wpływ gini_impurity na wartość parametrów, 50 drzew', title_x=0.5, font=dict(size=20))
# fig.show()


# bootstrapping:
# months = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
# acc = np.round([0.8514851485148515, 0.8877887788778878, 0.8679867986798679, 0.8811881188118812, 0.8778877887788779, 0.8943894389438944, 0.8745874587458746],3)
# recall = np.round([0.9333333333333333, 0.9333333333333333, 0.9393939393939394, 0.9393939393939394, 0.9515151515151515, 0.9575757575757575, 0.9333333333333333], 3)
# precision = np.round([0.8191489361702128, 0.8700564971751412, 0.8378378378378378, 0.856353591160221, 0.8440860215053764, 0.8633879781420765, 0.850828729281768], 3)
# f1 = np.round([0.8725212464589235, 0.9005847953216375, 0.8857142857142858, 0.8959537572254335, 0.8945868945868947, 0.9080459770114941, 0.8901734104046243],3)
# time= [46.070653, 47.922621, 44.809524, 43.214698, 41.008881, 39.663027, 38.090268]
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
# fig.update_layout(title='Wpływ bootstrapping na wartość parametrów, 50 drzew', title_x=0.5, font=dict(size=20))
# fig.show()

# # feature_bagging:
# months = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
# acc = np.round([0.8448844884488449, 0.8745874587458746, 0.8811881188118812, 0.8712871287128713, 0.8646864686468647],3)
# recall = np.round([0.9696969696969697, 0.9393939393939394, 0.9272727272727272, 0.9636363636363636, 0.9454545454545454], 3)
# precision = np.round([0.7920792079207921, 0.8469945355191257, 0.864406779661017, 0.828125, 0.8297872340425532], 3)
# f1 = np.round([0.8719346049046321, 0.8908045977011495, 0.8947368421052632, 0.8907563025210083, 0.8838526912181301],3)
# time= [52.441393, 47.820612, 36.943312, 32.516728, 27.450116]
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
# fig.update_layout(title='Wpływ feature_bagging na wartość parametrów, 50 drzew', title_x=0.5, font=dict(size=20))
# fig.show()



# jeszce min_rows

# max_depth powtorz bez selective_pruning

ours = {'acc': [0.8026315789473685, 0.8157894736842105, 0.8289473684210527, 0.881578947368421, 0.881578947368421, 0.881578947368421], 'recall': [0.7906976744186046, 0.9767441860465116, 0.9767441860465116, 0.9767441860465116, 0.9767441860465116, 0.9767441860465116], 'precision': [0.85, 0.7636363636363637, 0.7777777777777778, 0.84, 0.84, 0.84], 'f1': [0.8192771084337349, 0.8571428571428571, 0.8659793814432991, 0.9032258064516129, 0.9032258064516129, 0.9032258064516129], 'time': [1.690095, 15.84103, 31.554793, 67.810172, 131.27629, 263.943582]}
scikit = {'acc': [0.8026315789473685, 0.881578947368421, 0.8552631578947368, 0.8552631578947368, 0.881578947368421, 0.8552631578947368], 'recall': [0.813953488372093, 0.9302325581395349, 0.8372093023255814, 0.9069767441860465, 0.9069767441860465, 0.9069767441860465], 'precision': [0.8333333333333334, 0.8695652173913043, 0.9, 0.8478260869565217, 0.8863636363636364, 0.8478260869565217], 'f1': [0.8235294117647058, 0.898876404494382, 0.8674698795180723, 0.8764044943820224, 0.896551724137931, 0.8764044943820224], 'time': [0.014231, 0.090359, 0.168686, 0.3226, 0.611167, 1.207758]}

# Traces with lines: My vs Scikit 
x = [10, 100, 200, 400, 800, 1600]
y = [1.690095, 15.84103, 31.554793, 67.810172, 131.27629, 263.943582]
fig = px.scatter().update_traces(mode="lines+markers")
for i in (ours.keys()):
    fig.add_trace(go.Scatter(x=x, y=ours[i], name=i))
for i in (scikit.keys()):
    fig.add_trace(go.Scatter(x=x, y=scikit[i], name=i+'_scikit'))
# fig = px.scatter(x=x, y=y).update_traces(mode="lines+markers")
fig.update_layout(title='Porównanie parametrów w zależności od liczby drzew', title_x=0.5, font=dict(size=20))
fig.show()
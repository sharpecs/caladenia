"""
caladenia.py

A application of libraries available from scikit-learn for the predictive analysis on the distribution of Spider Orchids around Perth.

References: 
1. https://scikit-learn.org/stable/auto_examples/applications/plot_species_distribution_modeling.html (Species distribution modelling)
2. https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_iris.htm (Principal Component Analysis)

"""
import time
start_time = time.time()

"""
1. LOAD: Pipe in relevant data from different sources.
"""
import pandas as pd

d = pd.read_json('data/djoobak.json')
dSpecies = pd.DataFrame(d['djoobak'][0]['species'])
dRecords = pd.DataFrame(d['djoobak'][0]['records'])

f = pd.read_json('data/mycota.json')
fSpecies = pd.DataFrame(f['mycota'][0]['species'])
fRecords = pd.DataFrame(f['mycota'][0]['records']).assign(c='["none"]')

dRecords.c = dRecords.c.apply(tuple)
fRecords.c = fRecords.c.apply(tuple)

records = dRecords.merge(fRecords)
records.c = records.c.apply(list)

"""
2. PROFILE: Identify environmental characteristics from data which can be transformed to learning features.
"""
# only fungi which are basidios and jelly like.
jelly = fSpecies[fSpecies['c'].str.contains('JELLY')]

# merge species details
dRecords = dRecords[dRecords['s'].isin(dSpecies['i'])]
fRecords = fRecords[fRecords['s'].isin(jelly['i'])] 

jelly_count = fRecords.groupby(['l','s']).size().reset_index(name='counts')
jelly_idx = jelly_count.groupby(['l']).agg('sum').reset_index()

# set the scientific name for species. 
def set_name(id):
    fname = dSpecies.loc[dSpecies['i'] == id]['n'].to_string(index=False)
    sname = fname.split(" ") 
    return f"{sname[0][0]}.{sname[1]}"

# natural is the tag used to describe the natural environment in OpenStreetMap.
def set_natural(features):
    natural = list(filter(lambda x: x in ['natural=wood','natural=wetland','natural=water','natural=scrub','natural=heath','natural=hill'], features)) 
    return 'unknown' if len(natural) == 0 else natural[0].split("=")[1] 

# set fungi index according to jelly basidios observed in locality.
def set_fungi(locality):
    idx = jelly_idx[jelly_idx['l'].str.contains(locality)]['counts']
    return int(idx.iloc[0]) if list(jelly_idx.l.unique()).count(locality) > 0 else 0

# identify spiders as seperate group of Caladenia having longer narrower sepals.
def set_spider(id):
    broad_sepal_species = ['latifolia','flava'] # typical broad sepal orchids observed.
    fname = dSpecies.loc[dSpecies['i'] == id]['n'].to_string(index=False)
    sname = fname.split(" ") 
    return 0 if sname[1] in broad_sepal_species else 1

# build feature set.
dRecords['v'] = dRecords['c'].apply(set_natural) # set natural land cover vegetation.
dRecords['fungi'] = dRecords['l'].apply(set_fungi)
dRecords['name'] = dRecords['s'].apply(set_name)
dRecords['spider'] = dRecords['s'].apply(set_spider)
dRecords['vegetation'], p_index = pd.Series(dRecords['v'].factorize())
dRecords['habitat'], p_index = pd.Series(dRecords['h'].factorize())
dRecords['locality'], p_index = pd.Series(dRecords['l'].factorize())

# build observations on the genus Caladenia.
caladenia = dSpecies[dSpecies['g'].str.contains('Caladenia')]
observations = dRecords[dRecords['s'].isin(caladenia['i'])].copy()

observations.rename(columns={'s': 'species'}, inplace=True)
observations.rename(columns={'d': 'date'}, inplace=True)

"""
EXPLORATORY. Transform data into learning features with a binary target classifier.
"""
import matplotlib.pyplot as plt
import numpy as np
import math

# Figure B. Chart the observations for Caladenia species in ranking order.
caladenia = observations.groupby(['name','spider']).size().reset_index(name='counts')

plt.figure(1, figsize=(10, 7))
ax1 = plt.subplot(111)
rank = caladenia.groupby(['name','spider'], as_index=False).agg({'counts': 'sum'}).sort_values(by = 'counts', ascending = True)

y = np.array(rank['name'])
x = np.array(rank['counts'])
s = np.array(rank['spider'])

colours = ['brown' if val==1 else 'blue' for val in s]

plt.barh(y, x, color=colours, alpha=0.5)

perc = rank.groupby(['spider'], as_index=False).agg({'counts': 'sum'}).sort_values(by = 'counts', ascending = True)

for a in [1, 0]:
    p1 = math.ceil(perc[perc['spider']==1]['counts'] / perc['counts'].sum() * 100)
    p0 = 100-p1
    ax1.scatter([], [], c='blue' if a==0 else 'brown', alpha=0.5, s=300, label=f"Other {p0}%" if a==0 else f"Spider {p1}%")
ax1.legend(scatterpoints=1, frameon=False, labelspacing=1, loc='lower right')

ax1.set_title('Ranking of Caladenia Species', fontweight="bold", fontsize=18)
ax1.set_xlabel('Number of Observations')

features = ['date','fungi','vegetation','habitat','species','locality','x','y'] # X-value
target = ['spider'] # y-value


# MACHINE A
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
from sklearn.decomposition import PCA

MachineA = PCA(n_components=3)

X = observations[target + features]
y = observations.h.unique()
X_reduced = MachineA.fit_transform(X)

# visualise Machine Analysis
plt.figure(2, figsize=(10, 10))
ax2 = plt.subplot(111, projection="3d", elev=-150, azim=110)
scatter = ax2.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2],  c=X['habitat'], s=80)
ax2.set(xlabel="1st Eigenvector", ylabel="2nd Eigenvector", zlabel="3rd Eigenvector")
ax2.set_title("MACHINE A ANALYSIS", fontweight="bold", fontsize=18)

# Add a legend
leg = ax2.legend(scatter.legend_elements()[0], y, loc="upper right", title="Classes")
ax2.add_artist(leg)

plt.show()

"""
LEARNING: Apply a supervised learning approach over the feature set.
"""
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from imblearn.over_sampling import SMOTE

oData = observations[target + features + ['name']].copy()
X_train, X_test, y_train, y_test = train_test_split(oData[features], oData[target], test_size=0.2, random_state=42)

# MACHINE B
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html 
from sklearn.linear_model import LogisticRegression

X_train_machA = MachineA.fit_transform(X_train) # reduce X with Machine A.
X_test_machA = MachineA.transform(X_test)

# For multiclass problems, 'multi_class' can be 'ovr' (One-vs-Rest) or 'multinomial'.
# 'lbfgs' is a common and robust solver for multinomial logistic regression.
MachineB = LogisticRegression(multi_class='auto', solver='liblinear', C=1.0, random_state=42, class_weight='balanced')
MachineB.fit(X_train_machA, y_train)

pred_machB = MachineB.predict(X_test_machA)
accu_machB = accuracy_score(y_test, pred_machB)

# MACHINE C
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
from sklearn.ensemble import RandomForestClassifier

smt = SMOTE(random_state=42) # Apply over-sampling technique to balance out.
X_train_resampled, y_train_resampled = smt.fit_resample(X_train, y_train)

MachineC = RandomForestClassifier(n_estimators=100, random_state=42)
MachineC.fit(X_train_resampled, y_train_resampled)

pred_machC = MachineC.predict(X_test)
accu_machC = accuracy_score(y_test, pred_machC)

# MACHINE D
# https://scikit-learn.org/stable/auto_examples/svm/plot_oneclass.html
from sklearn import svm

dFeatures = ['date','fungi','vegetation','habitat','species'] # X-value
dataD = observations[target + dFeatures + ['l','x','y']].copy()

dataD[dFeatures] = (dataD[dFeatures] - dataD[dFeatures].mean()) / dataD[dFeatures].std() # standardise features.

X_train, X_test, y_train, y_test = train_test_split(dataD[dFeatures], dataD[target], test_size=0.2, random_state=42)

MachineD = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.5)  # Gaussuan Kernal. Radial Basis Function handles non-linear relationships
MachineD.fit(X_train)                                        # transforming data into space where boundaries can be drawn.

pred_machD = MachineD.predict(X_test) #
accu_machD = accuracy_score(y_test, pred_machD)


"""
EVALUATION: Rate fit-for-purpose operating performance.
"""
from sklearn.metrics import roc_curve, auc

eval_df = pd.DataFrame({'True': np.squeeze(y_test[target].values), 'Machine C': pred_machC, 'Machine B': pred_machB, 'Machine D': pred_machD})

plt.figure(figsize=(15, 7))

for model in ['Machine B','Machine C','Machine D']:
    fpr, tpr, _ = roc_curve(eval_df['True'], eval_df[model])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{model} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'r--', label='Random Guess')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('MACHINE OPERATING PERFORMANCE', fontweight="bold", fontsize=18)
plt.legend()
plt.show()


"""
INDUCTION: Use machines to predict on selected localities from dataset.
"""

localities = ['KALAMUNDA','MC_WETLAND','BEELU','BANYOWLA','LS_BUSHLAND','KORUNG',"JULIMAR"]

def build_hotspots():
    #
    # Use Machine D to predict hotspots, spread and park density.
    #
    y_pred_train = MachineD.predict(X_train) # first 80% or rows
    y_pred_test = MachineD.predict(X_test) # remaining rows.

    y_pred_est = pd.Series(MachineD.decision_function(X_test))
    y_pred_train_est = pd.Series(MachineD.decision_function(X_train))

    y_pred = pd.Series(y_pred_test).replace([-1,1],[1,0])
    y_pred_train = pd.Series(y_pred_train).replace([-1,1],[1,0])

    # You can also reset the index to get a continuous integer index
    y_prediction = pd.concat([y_pred_train, y_pred], ignore_index=True)
    y_estimate = pd.concat([y_pred_train_est, y_pred_est], ignore_index=True)

    dataD['pred'] = y_prediction
    dataD['decision'] = y_estimate 

    # build hotspots of spiders with points and scaling for likely distribution from decision.
    hotspots = dataD[(dataD['pred'] == 1) ].copy()
    decision = hotspots['decision']
    mean = decision.mean(axis=0)
    std = decision.std(axis=0)
    hotspots['prob'] = decision * -100
    hotspots['z-score'] = (decision - mean) / std # scoring the decision function result for scaling.

    return hotspots

def build_estimates(hotspots):
    #
    # Use Machine B, C, D to predict likelihood estimates of presence. 
    #
    prob_machD = []
    eval_df = pd.DataFrame()

    for l in localities:
        lRow = observations[observations.l.str.contains(l)].iloc[0]
        eval_df = pd.concat([eval_df, lRow.to_frame().T], ignore_index=True)
        proba = hotspots[hotspots['l'] == l]['prob'].max()
        prob_machD.append(proba)

    eval_data = eval_df[features].copy()
    data_machA = MachineA.transform(eval_data) # reduce data dimensions with Machine A.

    pMachB = pd.Series(MachineB.predict_proba(data_machA)[:, 1]*100).astype(int)
    pMachC = pd.Series(MachineC.predict_proba(eval_data)[:, 1]*100).astype(int)
    pMachD = pd.Series(prob_machD).astype(int)

    prob = pd.DataFrame({
        'locality': np.squeeze(eval_df.l.values), 
        'Machine B': pMachB, 
        'Machine C': pMachC, 
        'Machine D': pMachD,
        'Total': pMachB.add(pMachC, fill_value=0).add(pMachD, fill_value=0)
    })

    return prob.sort_values(by = 'Total', ascending = True)

hotspots = build_hotspots()
likelihood = build_estimates(hotspots)

#
# A. Graph Likelihood Estimates for selected localities. 
#
estimates = {'Machine B': likelihood['Machine B'], 'Machine C': likelihood['Machine C'], 'Machine D': likelihood['Machine D']}
localities = likelihood['locality']

fig, ax = plt.subplots(figsize=(15, 7))
plabels = np.zeros(7)
width = 0.6  # the width of the bars: can also be len(x) sequence

for machine, percentage in estimates.items():
    p = plt.barh(localities, percentage, width, label=machine, left=plabels, alpha=0.5)
    plabels += percentage

    ax.bar_label(p, label_type='center')

ax.tick_params(
    axis='x',          # Apply to the x-axis
    which='both',      # Affects both major and minor ticks
    bottom=False,      # Turn off ticks along the bottom edge
    labelbottom=False  # Turn off labels along the bottom edge
)

ax.set_title('MACHINE BASED ESTIMATES FOR SPIDER ORCHIDS', fontweight="bold", fontsize=18)
ax.legend()
plt.show()


#
# B. Map hotspot predictions to visualise spread around Perth.
#
from mpl_toolkits.basemap import Basemap
from scipy.interpolate import griddata
from matplotlib.patches import Polygon
from perth_parks import BanyowlaRP, BeeluNP, JandakotRP, JohnForrestNP, JulimarSF, KalamundaNP, KurongNP, MundyRP, WungongRP

plt.figure(figsize=(15,15))
width = 28000000; lon_0 = 116.090882; lat_0 = -31.785461

perth = Basemap(projection='lcc', lat_0=lat_0, lon_0=lon_0, width=100000, height=100000, resolution='h')
perth.drawcoastlines(color='gray')
perth.fillcontinents(color="#FFF5F0", lake_color='#DDEEFF')
perth.drawmapboundary(fill_color="#DDEEFF")

# 2. Map some points of interest for scaling purposes.
x, y = perth(115.888488, -32.075336)
plt.text(x, y, '  M.Carroll Park', fontsize=10)

x, y = perth(115.827957, -31.857827)
plt.text(x, y, ' L.Swamp Bushland', fontsize=10)

x, y = perth(116.166090, -31.957252)
plt.text(x, y, '    Mundaring Weir', fontsize=10)

x, y = perth(115.921766, -32.050434)
plt.text(x, y, 'E.Brook Valley', fontsize=10)

x, y = perth(116.314078, -31.425385)
plt.text(x, y, 'Julimar Forest', fontsize=10)

x, y = perth(116.051425, -32.148775)
plt.text(x, y, '  Settlers Common', fontsize=10)

x, y = perth(115.905683, -32.206059)
plt.text(x, y, '  Wandi Reserve', fontsize=10)

x, y = perth(116.100100, -31.880328)
plt.text(x, y, '  Hovea Falls', fontsize=10)

x, y = perth(116.086815, -31.973989)
plt.text(x, y, '  Jorgensen Park', fontsize=10)

x, y = perth(115.947909, -31.926734)
plt.text(x, y, 'Gooseberry Hill', fontsize=10)

x, y = perth(116.07773, -32.031231)
plt.text(x, y, 'Victoria Reservoir', fontsize=10)

# a) data coordinates
lats = hotspots['x']
lons = hotspots['y']

# b) grid data coordinates.
glons, glats = np.meshgrid(lons, lats)

# c) interpolate significant areas using z-scores from hotspots.
x, y = perth(glons, glats)
z = griddata((lons, lats), hotspots['z-score'] ** 2, (glons, glats), method='linear')

# d) map distribution with contour.
clevels = np.arange(-2, 2.1, 0.2) # Example contour levels
cs = perth.contourf(x, y, z, clevels, cmap=plt.cm.RdBu_r)

cbar = perth.colorbar(cs, location='bottom', pad=0.05, fraction=0.057)

# e) map out presence-only spider observations.
plats = dataD['x']
plons = dataD['y']
perth.scatter(plons, plats, latlon=True, marker='o', c='#67000D', zorder=1, alpha=0.03)

# f) map out pridicted hotspots.
plats = hotspots.loc[hotspots['spider']==0]['x']
plons = hotspots.loc[hotspots['spider']==0]['y']
perth.scatter(plons, plats, latlon=True, marker='o', c='#EC9374', zorder=1, alpha=0.3)

# g) map out significant areas using density levels on z-scores.
parks = {
    'JOHN_FOREST': JohnForrestNP(perth), 
    'KALAMUNDA':  KalamundaNP(perth),
    'BEELU': BeeluNP(perth),
    'KORUNG': KurongNP(perth),
    'MUNDY': MundyRP(perth),
    'BANYOWLA': BanyowlaRP(perth),
    'WUNGONG': WungongRP(perth),
    'JANDAKOT': JandakotRP(perth),
    'JULIMAR': JulimarSF(perth)
}
density = hotspots[['l','z-score']].groupby('l', as_index=False).mean('z-score').sort_values(by = 'z-score', ascending = True)

for locale in density.l:
    if locale not in list(parks.keys()):
        continue
    p = parks[locale]
    perp = 0.1 * density[density.l==locale]['z-score'].iloc[0]**2 + 0.08
    polp = Polygon(p.coordinates(),facecolor='brown',edgecolor='black',linewidth=1, alpha=perp, label=locale)
    plt.gca().add_patch(polp)    


plt.legend(scatterpoints=1, frameon=False, labelspacing=1, loc='upper left', fontsize=12, title='PREDICTED TOP LOCATIONS', title_fontsize='large')
plt.title('MACHINE D HOTSPOT ANALYSIS\n',fontweight="bold", fontsize=18)
plt.show()

end_time = time.time() # or time.perf_counter()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.4f} seconds")


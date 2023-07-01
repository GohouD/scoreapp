import streamlit as st
import pickle
import shap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from matplotlib.patches import Circle
from matplotlib.patches import Wedge 
from matplotlib.patches import Rectangle
from numba import njit
from numba import generated_jit
from numba import types
from numba import jit
import numba as nb

###############################################################

# Load the necessary data and models
X_train_scaled = pickle.load(open('X_train_scaled.pkl', 'rb'))
X_test_scaled = pickle.load(open('X_test_scaled.pkl', 'rb'))
feats = pickle.load(open('feats.pkl', 'rb'))
fitted_model = pickle.load(open('model_features.pkl', 'rb'))
X_test_dash = pickle.load(open('X_test_dash.pkl', 'rb'))
predicted_prob = pickle.load(open('predicted_prob.pkl', 'rb'))
predicted_classes = pickle.load(open('predicted_classes.pkl', 'rb'))
X_test_scaled_REVERTEDBACK = pickle.load(open('X_test_scaled_REVERTEDBACK.pkl', 'rb'))


######################################


def print_client_info(SK_ID_CURR, X_test_dash):
    client_id = X_test_dash["SK_ID_CURR"].values
    id = 0
    for i in range(len(client_id)):
        if client_id[i] == SK_ID_CURR:
            id = client_id[i]
            break

    age_value = X_test_dash["DAYS_BIRTH"].values
    age = 0
    for i in range(len(age_value)):
        if client_id[i] == SK_ID_CURR:
            age = age_value[i]
            break

    gender_value = X_test_dash["CODE_GENDER"].values
    gender = "Unknown"
    for i in range(len(gender_value)):
        if client_id[i] == SK_ID_CURR:
            gender = "Female" if gender_value[i] == 0 else "Male"
            break

    marital_status = X_test_dash["NAME_FAMILY_STATUS_Married"].values
    status = "Unknown"
    for i in range(len(marital_status)):
        if client_id[i] == SK_ID_CURR:
            status = "Married" if marital_status[i] == 1 else "Not married"
            break

    education_level = X_test_dash["NAME_EDUCATION_TYPE_Highereducation"].values
    level = "Unknown"
    for i in range(len(education_level)):
        if client_id[i] == SK_ID_CURR:
            level = "Yes" if education_level[i] == 1 else "No"
            break

    type_revenus = X_test_dash["NAME_EDUCATION_TYPE_Highereducation"].values
    revenus = "Unknown"
    for i in range(len(type_revenus)):
        if client_id[i] == SK_ID_CURR:
            revenus = "Working" if type_revenus[i] == 1 else "Business or other"
            break

    # Display all elements in a single box
    st.info("Client Information:")
    st.markdown("- **Client ID:** {}\n- **Age:** {}\n- **Gender:** {}\n- **Marital status:** {}\n- **Higher Education:** {}\n- **Type de revenus:** {}".format(id, age, gender, status, level, revenus))



###############################################################

# Function to display the gauge
@jit(nopython=False)
def gauge(labels=['LOW','MEDIUM','HIGH','VERY HIGH','EXTREME'], \
          colors='jet_r', arrow=1, title='', fname=False): 
    
    # internal functions
    def degree_range(n): 
        start = np.linspace(0,180,n+1, endpoint=True)[0:-1]
        end = np.linspace(0,180,n+1, endpoint=True)[1::]
        mid_points = start + ((end-start)/2.)
        return np.c_[start, end], mid_points

    def rot_text(ang): 
        rotation = np.degrees(np.radians(ang) * np.pi / np.pi - np.radians(90))
        return rotation
    
    
    """
    some sanity checks first
    
    """
    N = len(labels)
    
    if arrow > 180: 
        raise Exception("\n\nThe category ({}) is greated than \
        the length\nof the labels ({})".format(arrow, N))
  
    """
    if colors is a string, we assume it's a matplotlib colormap
    and we discretize in N discrete colors 
    """
    if isinstance(colors, str):
        cmap = cm.get_cmap(colors, N)
        cmap = cmap(np.arange(N))
        colors = cmap[::-1,:].tolist()
    if isinstance(colors, list): 
        if len(colors) == N:
            colors = colors[::-1]
        else: 
            raise Exception("\n\nnumber of colors {} not equal \
            to number of categories{}\n".format(len(colors), N))

    """
    begins the plotting
    """
    fig, ax = plt.subplots(figsize=(5, 3))
    ang_range, mid_points = degree_range(N)
    labels = labels[::-1]
    
    """
    plots the sectors and the arcs
    """
    patches = []
    for ang, c in zip(ang_range, colors): 
        # sectors
        patches.append(Wedge((0.,0.), .4, *ang, facecolor='w', lw=2))
        # arcs
        patches.append(Wedge((0.,0.), .4, *ang, width=0.10, facecolor=c, lw=2, alpha=0.8))
    
    [ax.add_patch(p) for p in patches]
   
    """
    set the labels (e.g. 'LOW','MEDIUM',...)
    """
    for mid, lab in zip(mid_points, labels): 

        ax.text(0.35 * np.cos(np.radians(mid)), 0.35 * np.sin(np.radians(mid)), lab, \
            horizontalalignment='center', verticalalignment='center', fontsize=14, \
            fontweight='bold', rotation = rot_text(mid))

    """
    set the bottom banner and the title
    """
    r = Rectangle((-0.4,-0.1),0.8,0.1, facecolor='w', lw=2)
    ax.add_patch(r)
    
    ax.text(0, -0.09, title, horizontalalignment='center', \
         verticalalignment='center', fontsize=50, fontweight='bold')

    """
    plots the arrow now
    """
    pos = arrow
    
    ax.arrow(0, 0, 0.200 * np.cos(np.radians(pos)), 0.200 * np.sin(np.radians(pos)), \
                 width=0.01, head_width=0.03, head_length=0.1, fc='k', ec='k')
    
    ax.add_patch(Circle((0, 0), radius=0.02, facecolor='k'))
    ax.add_patch(Circle((0, 0), radius=0.01, facecolor='w', zorder=11))

    """
    removes frame and ticks, and makes axis equal and tight
    """
    ax.set_frame_on(False)
    ax.axes.set_xticks([])
    ax.axes.set_yticks([])
    ax.axis('equal')
    sns.set_style("darkgrid")
    plt.tight_layout()
    if fname:fig.savefig(fname, dpi=200) 


###############################################################

# Calculate explainer
def calculate_explainer(fitted_model, X_train_scaled, feats):
    explainer = shap.TreeExplainer(fitted_model, X_train_scaled, feature_names=np.array(feats))
    
    return explainer

###############################################################    

# Sidebar to select "ithreshold"
st.sidebar.title("Treshold")
threshold = st.sidebar.selectbox("Threshold", [0.11, 0.11])

# Streamlit UI and sidebar to select "id"
st.sidebar.title("ID Row")
selected_id_row = X_test_dash["id_row"].values
id_row = st.sidebar.selectbox("ID_row", selected_id_row)

# Sidebar to select "id_row"
st.sidebar.title("Client ID")  
selected_id = X_test_dash.loc[X_test_dash["id_row"] == int(id_row), "SK_ID_CURR"].values
SK_ID_CURR = st.sidebar.selectbox("SK_ID_CURR", selected_id)


###############################################################    
    
# Define the Streamlit app
def main():
    
    # Print client information
    print_client_info(SK_ID_CURR, X_test_dash)

    c1, c2 = st.columns((3,7))

    with c1:
        st.info('Decision Maker')
        #st.markdown('#### Decision Maker')
        arrow_val = 180 - predicted_prob.loc[SK_ID_CURR].probabilites * 100 * 1.8 - (50 - threshold * 100) * 1.8
        title_val = '\n {:.2%}'.format(predicted_prob.loc[SK_ID_CURR].probabilites)
        gauge(labels=['Accord', 'Refus'],colors=sns.color_palette("Blues", 2),arrow=arrow_val, title=title_val)
              #arrow_val = 180 - predicted_prob.loc[SK_ID_CURR].probabilites * 100 * 1.8 - (50 - threshold * 100) * 1.8
              #arrow=180 - predicted_prob.loc[SK_ID_CURR].probabilites * 100 * 1.8 - (50 - threshold * 100) * 1.8,
              #title='\n {:.2%}'.format(predicted_prob.loc[SK_ID_CURR].probabilites))


    with c2:
        st.info('Explainer')
        #st.title("Explicabilité")
        if st.button("Generate Bar Plot"):
            explainer = calculate_explainer(fitted_model, X_train_scaled, feats)
            shap_values = explainer(X_test_scaled)
            shap.bar_plot(explainer.shap_values(X_test_scaled[selected_id_row]), feature_names=np.array(feats),max_display=10)
            sns.set_style("darkgrid")


# Run the Streamlit app
if __name__ == "__main__":
    main()


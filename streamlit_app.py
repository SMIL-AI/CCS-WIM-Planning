import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import pickle

# Display logos
st.sidebar.image("./data/GDOT.PNG", use_column_width=True) 
st.sidebar.image("./data/UGA.PNG", use_column_width=True)  
st.title('Strategic Planning of CCS and WIM Sites')


if 'display_mode' not in st.session_state:
    st.session_state['display_mode'] = 'CCS'
if 'focus' not in st.session_state:
    st.session_state['focus'] = {'CCS': 'rural', 'WIM': 'rural'}
if 'desired_sensor_num' not in st.session_state:
    st.session_state['desired_sensor_num'] = {'CCS': 5, 'WIM': 7}
if 'KL_weight' not in st.session_state:
    st.session_state['KL_weight'] = {'CCS': 0.6, 'WIM': 0.2}
if 'FC_weight' not in st.session_state:
    st.session_state['FC_weight'] = {'CCS': 0.2, 'WIM': 0.2}
if 'AADT_weight' not in st.session_state:
    st.session_state['AADT_weight'] = {'CCS': 0.2, 'WIM': 0.2}
if 'functional_classes' not in st.session_state:
    st.session_state['functional_classes'] = {'CCS': [1, 5, 6], 'WIM': [1, 6]}
if 'display_map' not in st.session_state:
    st.session_state['display_map'] = False
if 'map_placeholder' not in st.session_state:
    st.session_state['map_placeholder'] = st.empty()


################################################################### function line ###################################################################
def multi_criteria_decision(kl_weight, fc_class_weight, aadt_weight, fc_class_string, unvisited_index, desired_sensor_num, df, kl_divergences, selection, mode, sensor_index):
    
    df_copy = df.copy()
    target_df = df_copy.iloc[unvisited_index]


    # select based on rural/uban/both
    if selection == "urban":
        filtered_urban_df = target_df[target_df['MPO_NAME'].str.strip() != '']
        target_df = filtered_urban_df
    elif selection == "rural":
        filtered_rural_df = target_df[target_df['MPO_NAME'].str.strip() == '']
        target_df = filtered_rural_df

    # acquire the fc numbers from the input string e.g. "146"
    fc_class_choice = []

    for char in fc_class_string:
        if char.isdigit(): 
            fc_class_choice.append(int(char))  

    filtered_df = target_df[target_df['FC2015'].isin(fc_class_choice)]

    target_indices = filtered_df.index.tolist() 

    sorted_values = filtered_df['FC2015'].sort_values().unique()
    # print(sorted_values)
    # mapping the sorted values
    value_mapping = {value: index + 1 for index, value in enumerate(sorted_values)}
    # print("mapped FC values are: ", value_mapping)
    # assign a new column with the mapped values
    filtered_df['New_FC2015'] = filtered_df['FC2015'].map(value_mapping)

    fc_class = filtered_df["New_FC2015"]

    if mode == 'CCS':
        aadt = filtered_df["TOTAL_V"]
    
    if mode == 'WIM':
        aadt = filtered_df["FRGHT_V"]
    # filter the KL, acquire accordingly
    # Find the positions of elements in target_indices within unvisited_index
    positions = [unvisited_index.index(idx) for idx in target_indices]
    kls_updated = [kl_divergences[pos] for pos in positions]
    kl = kls_updated
    # print(len(kl))
    # print(kl)
    fc_class_normalized = fc_class / fc_class.max()
    aadt_normalized = aadt / aadt.max()
    kl_normalized = kl / max(kl_divergences)

    weights = {
                "kl_weight": kl_weight,
                "fc_class_weight": fc_class_weight,
                "aadt_weight": aadt_weight, 
                }

    final_scores = (
                    weights["kl_weight"] * kl_normalized  # the lower, the better: infomation loss 
                    + weights["fc_class_weight"] * fc_class_normalized  # the lower, the better: lower -> highway
                    - weights["aadt_weight"] * aadt_normalized  # the higher, the better: higher volume captured
                    )
    
    final_scores_list = final_scores.tolist()
    # print(len(final_scores_list))
    # print("Final Scores:")
    # print(final_scores_list)

    sorted_score = sorted(range(len(list(final_scores_list))), key=lambda i: final_scores_list[i])
    lowest_desired_indices = sorted_score[:desired_sensor_num]

    # print(f"\nIndices of the lowest {desired_sensor_num} scores:", lowest_desired_indices)

    # given the lowest indices(order), get the corresponding element in the "unvisited_index"
    corresponding_elements_given_score = [target_indices[i] for i in lowest_desired_indices]

    # print("\nCorresponding elements:", corresponding_elements_given_score)

    return corresponding_elements_given_score

################################################################### function line ###################################################################

# Define a function to load data
def load_data(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)


# Load and preprocess common data
df = pd.read_csv('./data/state_all.csv')
df = df.assign(lat=(df['x_end'] + df['y_end']) / 2, lon=(df['x_start'] + df['y_start']) / 2)

current_display_mode = st.session_state.get('display_mode', 'CCS')

def display_map(display_mode):
    desired_sensor_num = st.session_state['desired_sensor_num'][display_mode]
    KL_weight = st.session_state['KL_weight'][display_mode]
    FC_weight = st.session_state['FC_weight'][display_mode]
    AADT_weight = st.session_state['AADT_weight'][display_mode]
    FC_string = ''.join(str(fc) for fc in st.session_state['functional_classes'][display_mode])
    focus = st.session_state['focus'][display_mode]

    # Initialize the map
    m = folium.Map(location=[32.97835537534753, -83.62878788223315], zoom_start=6.5)
    

    # Conditional logic based on the session state
    if display_mode == 'CCS':
        # Load and process CCS data
        df_ccs = pd.read_csv('./data/ccs_state_all.csv')
        ccs_list_df = df_ccs['NO'].to_list()
        locations_global = load_data('./data/state_2nd_approach_1000nodes_locations_global_new.pkl')
        locations_target = load_data('./data/state_2nd_approach_1000nodes_locations_target_new.pkl')
        kl_divergences = load_data('./data/state_2nd_approach_KL_divergences_CCS_new.pkl')
        unvisited_index = load_data('./data/unvisited_index_state_2nd_approach_1000nodes_new.pkl')
        
        # Perform CCS-specific processing
        final_choice_ccs = multi_criteria_decision(KL_weight, FC_weight, AADT_weight, FC_string, unvisited_index, desired_sensor_num, df, kl_divergences, focus, display_mode, df_ccs)
        # Check if the number of segments matches the desired number
        if len(final_choice_ccs) != desired_sensor_num:
            warning_message = "⚠️Warning: The number of available sites is less than the desired number of sites."
            st.markdown(f"<span style='color: red; font-size: 20px;'>{warning_message}</span>", unsafe_allow_html=True)

        locations_current_CCS = [(row['lat'], row['lon']) for index, row in df.loc[ccs_list_df].iterrows()]
        locations_score_ccs = [(row['lat'], row['lon']) for index, row in df.loc[final_choice_ccs].iterrows()]
         
        # Create FeatureGroups
        fg_unvisited = folium.FeatureGroup(name='Target Segments')
        fg_best_choice = folium.FeatureGroup(name='Best Choice')
        fg_current = folium.FeatureGroup(name='Current CCS Stations')
        
        ccs_data = pd.DataFrame({
            'Segment ID': final_choice_ccs,
            'Location': locations_score_ccs
        })
        # Display the DataFrame as a table
        st.table(ccs_data)

        # Plot unvisited locations
        for idx, location in zip(unvisited_index, locations_target):
            popup_text = f'Unvisited Index: {idx, (location)}'
            folium.CircleMarker(location=location, popup=popup_text, radius=5, color='blue', fill_color='blue').add_to(fg_unvisited)

        # Plot the best choice locations
        for idx, location in zip(final_choice_ccs, locations_score_ccs):
            popup_text = f'Best Choice Index: {idx, (location)}'
            folium.Marker(location=location, popup=popup_text, icon=folium.Icon(color='purple')).add_to(fg_best_choice)

        # Plot the current CCS locations
        for location in locations_current_CCS:
            folium.RegularPolygonMarker(location=location, number_of_sides=3, radius=7, color='green', fill_color='green').add_to(fg_current)

        # Add FeatureGroups to the map
        m.add_child(fg_unvisited)
        m.add_child(fg_best_choice)
        m.add_child(fg_current)

        # Add LayerControl
        m.add_child(folium.LayerControl())


    elif display_mode == 'WIM':
        # Load and process WIM data
        df_wim = pd.read_csv('./data/wim_state_all.csv')
        wim_list_df = df_wim['NO'].to_list()
        locations_global = load_data('./data/state_2nd_approach_1000nodes_locations_global_WIM_new.pkl')
        locations_target = load_data('./data/state_2nd_approach_1000nodes_locations_target_WIM_new.pkl')
        kl_divergences = load_data('./data/state_2nd_approach_KL_divergences_WIM_new.pkl')
        unvisited_index = load_data('./data/unvisited_index_state_2nd_approach_1000nodes_forWIM_new.pkl')
        
        # Perform WIM-specific processing
        final_choice_wim = multi_criteria_decision(KL_weight, FC_weight, AADT_weight, FC_string, unvisited_index, desired_sensor_num, df, kl_divergences, focus, display_mode, df_wim)
        
        if len(final_choice_wim) != desired_sensor_num:
            warning_message = "⚠️Warning: The number of available sites is less than the desired number of sites."
            st.markdown(f"<span style='color: red; font-size: 20px;'>{warning_message}</span>", unsafe_allow_html=True)
        
        locations_current_WIM = [(row['lat'], row['lon']) for index, row in df.loc[wim_list_df].iterrows()]
        locations_score_wim = [(row['lat'], row['lon']) for index, row in df.loc[final_choice_wim].iterrows()]

        # Create FeatureGroups
        fg_unvisited = folium.FeatureGroup(name='Target Segments')
        fg_best_choice = folium.FeatureGroup(name='Best Choice')
        fg_current = folium.FeatureGroup(name='Current WIM Stations')
        # Create a DataFrame for displaying
        wim_data = pd.DataFrame({
            'Segment ID': final_choice_wim,
            'Location': locations_score_wim
        })

        # Display the DataFrame as a table
        st.table(wim_data)
        
        # Plot unvisited locations
        for idx, location in zip(unvisited_index, locations_target):
            popup_text = f'Unvisited Index: {idx, (location)}'
            folium.CircleMarker(location=location, popup=popup_text, radius=5, color='blue', fill_color='blue').add_to(fg_unvisited)

        # Plot the best choice locations
        for idx, location in zip(final_choice_wim, locations_score_wim):
            popup_text = f'Best Choice Index: {idx, (location)}'
            folium.Marker(location=location, popup=popup_text, icon=folium.Icon(color='purple')).add_to(fg_best_choice)

        # Plot the current WIM locations
        for location in locations_current_WIM:
            folium.RegularPolygonMarker(location=location, number_of_sides=4, radius=7, color='red', fill_color='red').add_to(fg_current)  

        # Add FeatureGroups to the map
        m.add_child(fg_unvisited)
        m.add_child(fg_best_choice)
        m.add_child(fg_current)

        # Add LayerControl
        m.add_child(folium.LayerControl())

    
    # Display the map
    folium_static(m,  width=800, height=800)

with st.sidebar:
    with st.form(key='user_settings_form'):
        st.header('User Settings')

        # # Use session state to store the previous mode. If it doesn't exist, initialize it to 'CCS'.
        if 'previous_mode' not in st.session_state:
            st.session_state['previous_mode'] = 'CCS'

        temp_mode = st.radio('Choose Sensor Type:', ['CCS', 'WIM'], 
                             index=0 if st.session_state['previous_mode'] == 'CCS' else 1)
        # Check if the mode has been changed and update the default values accordingly
        if st.session_state['previous_mode'] != temp_mode:
            if temp_mode == 'WIM':
                # Set the default values for WIM
                st.session_state['desired_sensor_num']['WIM'] = 7
                st.session_state['KL_weight']['WIM'] = 0.6
                st.session_state['FC_weight']['WIM'] = 0.2
                st.session_state['AADT_weight']['WIM'] = 0.2
                st.session_state['functional_classes']['WIM'] = [1, 6]
                st.session_state['focus']['WIM'] = 'rural'
            else:
                # Set the default values for CCS
                st.session_state['desired_sensor_num']['CCS'] = 5
                st.session_state['KL_weight']['CCS'] = 0.6
                st.session_state['FC_weight']['CCS'] = 0.2
                st.session_state['AADT_weight']['CCS'] = 0.2
                st.session_state['functional_classes']['CCS'] = [1, 5, 6]
                st.session_state['focus']['CCS'] = 'rural'

            # Update the previous_mode to the current mode
            st.session_state['previous_mode'] = temp_mode
            # # Redisplay the map with the default settings for the new mode
            # st.session_state.map_placeholder.empty()
            # display_map(temp_mode)

        # Other settings widgets go here...
        temp_desired_sensor_num = st.slider('Desired number of sensors', 1, 10, 
                                            st.session_state['desired_sensor_num'][temp_mode])
        temp_focus = st.selectbox('Select Area Type(s):', ['urban', 'rural', 'both'], 
                          index=['urban', 'rural', 'both'].index(st.session_state['focus'][temp_mode]))
        temp_KL_weight = st.number_input('Weight for Network Coverage', min_value=0.0, max_value=1.0, 
                                         value=st.session_state['KL_weight'][temp_mode], step=0.01)
        temp_FC_weight = st.number_input('Weight for Functional Class', min_value=0.0, max_value=1.0, 
                                         value=st.session_state['FC_weight'][temp_mode], step=0.01)
        temp_AADT_weight = st.number_input('Weight for AADT', min_value=0.0, max_value=1.0, 
                                           value=st.session_state['AADT_weight'][temp_mode], step=0.01)
        temp_functional_classes = st.multiselect('Select Functional Classes', [1, 2, 3, 4, 5, 6], 
                                                 default=st.session_state['functional_classes'][temp_mode])

        # Form submit button
        submitted = st.form_submit_button('Apply Changes')

# Handling the form submission
if submitted:
    # Check if the sensor type has changed and if so, reset to default values for that type
    if st.session_state['display_mode'] != temp_mode:
        st.session_state['display_mode'] = temp_mode
        # If we're switching to WIM mode, load the default WIM settings
        if temp_mode == 'WIM':
            st.session_state['desired_sensor_num']['WIM'] = 7  # Default number for WIM
            st.session_state['KL_weight']['WIM'] = 0.6  # Default KL weight for WIM
            st.session_state['FC_weight']['WIM'] = 0.2  # Default FC weight for WIM
            st.session_state['AADT_weight']['WIM'] = 0.2  # Default AADT weight for WIM
            st.session_state['functional_classes']['WIM'] = [1, 6]  # Default classes for WIM
            st.session_state['focus']['WIM'] = 'rural'
        else:
            # If switching back to CCS, reset to CCS defaults
            st.session_state['desired_sensor_num']['CCS'] = 5  # Default number for CCS
            st.session_state['KL_weight']['CCS'] = 0.6  # Default KL weight for CCS
            st.session_state['FC_weight']['CCS'] = 0.2  # Default FC weight for CCS
            st.session_state['AADT_weight']['CCS'] = 0.2  # Default AADT weight for CCS
            st.session_state['functional_classes']['CCS'] = [1, 5, 6]  # Default classes for CCS
            st.session_state['focus']['CCS'] = 'rural'
    else:
        # Update the session state with the form submission values
        st.session_state['desired_sensor_num'][temp_mode] = temp_desired_sensor_num
        st.session_state['KL_weight'][temp_mode] = temp_KL_weight
        st.session_state['FC_weight'][temp_mode] = temp_FC_weight
        st.session_state['AADT_weight'][temp_mode] = temp_AADT_weight
        st.session_state['functional_classes'][temp_mode] = temp_functional_classes
        st.session_state['focus'][temp_mode] = temp_focus

    
    # Update the previous mode
    st.session_state['previous_mode'] = temp_mode
    
    # Clear the existing map and display the new map
    st.session_state.map_placeholder.empty()
    display_map(st.session_state['display_mode'])


# Check if the map has been initialized and display it if not
if not st.session_state.get('map_initialized', False):
    # display_map(st.session_state['display_mode'])
    st.session_state['map_initialized'] = True

# Display the map based on the current mode without waiting for the form submission
# This is for the scenario when a user switches the sensor type
if st.session_state['display_mode'] and not submitted:
    display_map(st.session_state['display_mode'])


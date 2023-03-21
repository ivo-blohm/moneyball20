###################### Terminal commands:

# In terminal for the folder Business_Analytics_Capstone:
    # pip install streamlit
    # streamlit run fifa_app.py

# In Console: 
    # pip install streamlit-pandas-profiling

###################### Overall setup

    
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px
import matplotlib.pyplot as plt
import pickle
import plotly.graph_objs as go



###################### General Properties of the App

st.set_page_config(page_title="Moneyball App", 
                   page_icon="⚽",
                   layout="wide")

st.title("Moneyball App")

st.write("⚽ This is the Moneyball App. In the first section below, there are visualizations of the evolution of individual skills of FIFA soccer players, of an overall distribution of positions and skills.  Further, there are three interactive tools. The tool **'Player'** allows to retrieve more information on specific soccer players. The tool **'Search'** enables football club owners to search for replacements when a position has been opened. It returns the most undervalued player that fulfills specific characteristics. Lastly, the tool **'Clubs'** facilitates a club search. There, all players of a club are displayed. Further, one will see which players are over- and undervalued. For the most overvalued player, the tool returns a potential replacement.⚽")




###################### Define functions to load the data and model



# Remove the comments if you wish to use a different machine learning technique.
#@st.cache()
#def load_model_OLS():
#    filename_ols="fifa_ols_model.sav"
#    loaded_ols = pickle.load(open(filename_ols,"rb"))
#    return(loaded_ols)

#@st.cache()
#def load_model_tree():
#    filename_tree="fifa_tree_opt_model.sav"
#    loaded_tree = pickle.load(open(filename_tree,"rb"))
#    return(loaded_tree)

@st.cache(allow_output_mutation=True)
def load_model_forest():
    filename_forest="fifa_forest_opt_model.sav"
    loaded_forest = pickle.load(open(filename_forest,"rb"))
    return(loaded_forest)

def load_data():
    data=pd.read_csv("fifa_15_21.csv")
    return(data)


data = load_data()
#model_OLS = load_model_OLS()
#model_tree = load_model_tree()
model_forest = load_model_forest()




###################### Prediction


new_data = pd.read_csv("fifa_22.csv")


# Filter out variables needed for the prediction
    
        
prediction_data = new_data[["potential","age","league_level","skill_moves","preferred_foot","international_reputation","pace","shooting","passing","dribbling","defending"]]
prediction_data = pd.get_dummies(prediction_data,drop_first=True)

# OLS work flow

#predictions_OLS = model_OLS.predict(sm.add_constant(prediction_data))
    
# Save the predicted value and the value difference to the new_data
#new_data["predictions_OLS"] = np.exp(predictions_OLS)
#new_data["value_difference_OLS"] = new_data["predictions_OLS"] - new_data["value_eur"]


# Optimised Decision Tree work flow

#predictions_tree = model_tree.predict(prediction_data)
    
# Save the predicted value and the value difference to the new_data
#new_data["predictions_tree"] = np.exp(predictions_tree)
#new_data["value_difference_tree"] = new_data["predictions_tree"] - new_data["value_eur"]


# Optimised Random Forest work flow

predictions_forest = model_forest.predict(prediction_data)
    
#Save the predicted value and the value difference to the new_data
new_data["predictions_forest"] = np.exp(predictions_forest)
new_data["value_difference_forest"] = new_data["predictions_forest"] - new_data["value_eur"]






###################### The App

### Section 1 of the app: Filters

row1_col1, row1_col2 = st.columns([1,1])


player_options_top = data["short_name"].unique()
player_top = row1_col1.selectbox("Select a player", player_options_top)


variable_options_top = ["potential","pace","shooting","passing","dribbling","defending"]
variable_top = row1_col2.selectbox("Select variable to compare", variable_options_top)



### Section 2 of the app: Overview of the data set and visualizations

row2_col1, row2_col2, row2_col3 = st.columns([2,1,1])

potential_data = pd.DataFrame(data.groupby('position_0')['potential'].mean().reset_index().values, columns=["position","average"])
pace_data = pd.DataFrame(data.groupby('position_0')['pace'].mean().reset_index().values, columns=["position","average"])
shooting_data = pd.DataFrame(data.groupby('position_0')['shooting'].mean().reset_index().values, columns=["position","average"])
dribbling_data = pd.DataFrame(data.groupby('position_0')['dribbling'].mean().reset_index().values, columns=["position","average"])
defending_data = pd.DataFrame(data.groupby('position_0')['defending'].mean().reset_index().values, columns=["position","average"])
passing_data = pd.DataFrame(data.groupby('position_0')['passing'].mean().reset_index().values, columns=["position","average"])

potential_data["skill"]="potential"
pace_data["skill"]="pace"
shooting_data["skill"]="shooting"
dribbling_data["skill"]="dribbling"
defending_data["skill"]="defending"
passing_data["skill"]="passing"

polar = pd.concat([potential_data,pace_data,shooting_data,dribbling_data,defending_data,passing_data])


# Create skill evolution plot
with row2_col1:    
    st.subheader("Evolution of {}'s {}:".format(player_top,variable_top))

    # Define Data

    data_lineplot = data.loc[(data["short_name"]==player_top)]
        
    data_lineplot_one_year = data_lineplot.loc[(data_lineplot["year"]==2021)]
        
    position_line_data = data_lineplot_one_year["position_0"].iloc[-1]
    number_line_data = variable_top

    average_line_data = polar.loc[(polar["position"]==position_line_data) & (polar["skill"]==number_line_data)]
    y_value = average_line_data["average"].iloc[-1]
    
    my_text = "average value for {} of position {}".format(number_line_data,position_line_data)
    
    fig = px.line(data_lineplot, x="year", y=variable_top,
                  color_discrete_sequence=["#d8576b"])
    fig.add_hline(y = y_value,line_dash="dash", 
                  line_color="#fb9f3a",
                  annotation_text=my_text, 
                  annotation_position="top right")
    st.plotly_chart(fig, use_container_width=True)
    
    if st.button("Line graph description"):
        st.write("This chart displays the evolution of {} over time. The red line shows how his {} changed over time. The dashed orange line gives more context by visualizing the average score for {} for all players with the position {}.".format(player_top,number_line_data,number_line_data,position_line_data))
    else:
        st.write("")


# Plot distribution of positions
with row2_col2:
    st.subheader("Distribution of positions")

    # Define Data
    pie_data = pd.DataFrame(data["position_0"].value_counts().reset_index().values, columns=["position","count"])


    fig = px.pie(pie_data, values='count', 
                 names='position',
                 color_discrete_sequence= px.colors.sequential.Plasma_r)
    st.plotly_chart(fig, use_container_width=True)

    if st.button("Pie chart description"):
        st.write("This pie chart displays the distribution of the positions in the dataset of FIFA players between 2015 and 2021.")
    else:
        st.write("")

# Plot average skills for each position
with row2_col3:
    st.subheader("Average skills per positions")
    
 

    # Create a polar plot
    
    fig = px.line_polar(polar, r="average", theta="position", color="skill", line_close=True,
                    color_discrete_sequence=px.colors.sequential.Plasma_r)
    st.plotly_chart(fig, use_container_width=True)

    if st.button("Polar chart description"):
        st.write("This polar chart displays the average score of all players for the cumulated years 2015-2021. It enables a comparison between the different skills - ppace, shooting, dribbling, defenfing, passing and potential -  as it visualizes the general level that players achieve in this skill.")
    else:
        st.write("")







#######################################################################################################################################

### Section 3 of the app: Players, Search, Clubs


## Players ############################################################################################################

tab1, tab2, tab3 = st.tabs(["Players", "Search", "Clubs"])

with tab1:
    # Let the user chose a player
    
        
    row3_col1, row3_col2 = st.columns([1,1])

    player_options = data["long_name"].unique()
    player = row3_col1.selectbox("Select a player", player_options)

    year_options = data["year"].unique()

    year = row3_col2.selectbox("Please select the year for which you want to visualise the player profile",
                           year_options)
    try:
    
        data_radar = data.loc[(data["long_name"]==player)& (data["year"]==year),]

    
        player_image = data_radar["player_face_url"].iloc[-1]
        player_name = data_radar["long_name"].iloc[-1]
        
        club_image = data_radar["club_logo_url"].iloc[-1]
        club_name = data_radar["club_name"].iloc[-1]
    
        nation_image = data_radar["nation_logo_url"].iloc[-1]

    
        col1, col2, col3 = st.columns([3,1,1])
    
    # Display player image 
        try:
            # Display Image
            with col1:
                st.image(player_image)
                st.title(player_name)
        
        except:
            with col1:
                st.title(player_name)
    
        st.markdown("Tags: {}".format(data_radar["player_traits"].iloc[-1]))    
    

        # Display player characteristics
        row4_col1, row4_col2, row4_col3, row4_col4 = st.columns([1,1,1,1])

        pace = data_radar["pace"].iloc[-1]
        shooting = data_radar["shooting"].iloc[-1]
        passing = data_radar["passing"].iloc[-1]
        dribbling = data_radar["dribbling"].iloc[-1]
        defending = data_radar["defending"].iloc[-1]
    
    
        df = pd.DataFrame(dict(
            r = [pace,shooting,passing,dribbling,defending],
            theta=["pace","shooting","passing","dribbling","defending"]))
    
        fig2 = px.line_polar(df, r='r', theta='theta', line_close=True,color_discrete_sequence=["#d8576b"])
        fig2.update_traces(fill='toself')
    
        # Put figure in col 1 
        row4_col1.subheader("Skills")
        row4_col1.plotly_chart(fig2, use_container_width=True)
    
        # Display profile information in column 2
    
        reputation_stars = int(data_radar["skill_moves"].iloc[-1])*"⭐"
        moves_stars = int(data_radar["international_reputation"].iloc[-1])*"⭐"
        
        with row4_col2:
            st.subheader("Profile")
        
            st.markdown("- Age: {} years".format(int(data_radar["age"].iloc[-1])))
            st.markdown("- Height: {} cm".format(data_radar["height_cm"].iloc[-1]))
            st.markdown("- Preferred Foot: {}".format(data_radar["preferred_foot"].iloc[-1]))

        
            st.markdown("Player ratings")
            st.markdown("- International Reputation: {}".format(moves_stars))
            st.markdown("- Skill Moves: {}".format(reputation_stars))
        
        # Display information on the club
        with row4_col3:
            st.subheader("Club: {}".format(club_name))
            
            try:
                st.image(club_image)
            except:
                pass
            
            st.markdown("- Club contract valid until: {}".format(int(data_radar["club_contract_valid_until"].iloc[-1])))
            st.markdown("- League Level: {}".format(int(data_radar["league_level"].iloc[-1])))

            
            try:
                st.image(nation_image)
                st.markdown("{} also plays on the national team.".format(player_name))
            except:
                pass

                    
        # Display a soccer field
        
        positions = {'position_name': ["LS","ST","RS",
                                       "LW","LF","CF","RF","RW",
                                       "LAM","CAM","RAM",
                                       "LM","LCM","CM","RCM","RM",
                                       "LWB","LDM","CDM","RDM","RWB",
                                       "LB","LCB","CB","RCB","RB",
                                       "GK"], 
                     'x_position': [62,93,124,
                                    31,62,93,124,155,
                                    62,93,124,
                                    31,62,93,124,155,
                                    31,62,93,124,155,
                                    31,62,93,124,155,
                                    93],  
                     'y_position': [33.875,33.875,33.875,
                                    67.75,67.75,67.75,67.75,67.75,
                                    101.625,101.625,101.625,
                                    135.5,135.5,135.5,135.5,135.5,
                                    169.375,169.375,169.375,169.375,169.375,
                                    203.25,203.25,203.25,203.25,203.25,
                                    237.125]}
        df_positions = pd.DataFrame(data=positions)
        
        my_list = []
    
        for i in data_radar['player_positions'].str.split(',').iloc[-1]:
            my_list.append(i)
        
        new_list = [item.strip() for item in my_list]
        
        
        test_positions = df_positions[df_positions["position_name"].isin(new_list)]
        
        
        
        img = plt.imread("soccer_field.jpeg")
        fig, ax = plt.subplots()
        ax.scatter(test_positions.x_position, test_positions.y_position, linewidth=5, color='#d8576b')
    
        ax.axis('off')
        
        # zip joins x and y coordinates in pairs
        for x,y in zip(test_positions.x_position,test_positions.y_position):
        
            filtered_df = test_positions.loc[ (test_positions["x_position"] ==x) & (test_positions["y_position"] ==y),]
            label = filtered_df["position_name"].iloc[-1,]

            plt.annotate(label, # this is the text
                         (x,y), # these are the coordinates to position the label
                         textcoords="offset points", # how to position the text
                         xytext=(0,10), # distance from text to points (x,y)
                         ha='center') # horizontal alignment can be left, right or center
        
        ax.imshow(img)
        
        fig.savefig("field_with_positions.png")
    
        row4_col4.subheader("Player position")
    
        from PIL import Image
        image = Image.open('field_with_positions.png')
    
        row4_col4.image(image)

    except IndexError:
        st.error('**No data available.** It looks like there is no data for {} for the year of {}. Try to change your selection.'.format(player,year), icon="🚨")           



    try:
        # Add in the prediction
        row5_col1, row5_col2, row5_col3 = st.columns([1,1,1])
        
        data_prediction = new_data.loc[(new_data["long_name"]==player),]
        
        row5_col1.subheader("Prediction for {}".format(player))
        row5_col1.metric(label="Predicted Value", 
                         value=round(data_prediction["predictions_forest"].iloc[-1]), 
                         delta=round(data_prediction["value_difference_forest"]).iloc[-1])
        
        
        euros = data_prediction["value_difference_forest"].iloc[-1]*0.000001
                
        if euros <= 0:
            x = "overvalued"
        elif euros > 0:
            x = "undervalued"
        else:
            pass
        
        row5_col1.markdown("{} is {} by ca. {} million euros.".format(player_name,
                                                                      x,
                                                                      round(euros)))
    
    except IndexError:
        st.error('**No prediction available for {}.** '.format(player), icon="🚨")           



    



## Search  ############################################################################################################

with tab2:
    st.header("Predicting Actual Player Values")


## Allow the user to filter for the most undervalued players
    
    st.subheader("Most undervalued player: filters")
    
    col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1])


    # Let the user choose a position
    
    position_options = ["LS","ST","RS",
                        "LW","LF","CF","RF","RW",
                        "LAM","CAM","RAM",
                        "LM","LCM","CM","RCM","RM",
                        "LWB","LDM","CDM","RDM","RWB",
                        "LB","LCB","CB","RCB","RB",
                        "GK"]

    position = col1.selectbox("Please select the position",
                           position_options)

    # Let the user chose a maximum value

    value_slider = col2.slider('Choose a maximum value (in millions)',
                                 new_data["value_eur"].min()*0.000001,
                                 new_data["value_eur"].max()*0.000001,
                                 70.0)
    
    # Let the user chose a league level
    
    league_options = new_data["league_level"].unique()

    league = col3.selectbox("Please select the league",
                           league_options)

    # Let the user chose a minimum potential
    
    potential_slider = col4.slider('Choose a minimum potential',
                                 int(new_data["potential"].min()),
                                 int(new_data["potential"].max()),
                                 65)
    
    
    # Let the user chose until when the club contract should be valid
    

    contract_slider = col5.slider('Choose until when the current contract should be valid the latest',
                             int(new_data["club_contract_valid_until"].min()),
                             int(new_data["club_contract_valid_until"].max()),
                             2028)


    try: 
        data_new_radar = new_data.loc[(new_data["value_eur"]<=value_slider*1000000) & 
                                      (new_data["league_level"]==league) & 
                                      (new_data["potential"]>=potential_slider) &
                                      (new_data["club_contract_valid_until"]<=contract_slider),]
    
        data_new_radar = data_new_radar.loc[(new_data["position_0"]==position) | 
                                            (new_data["position_1"]==position) | 
                                            (new_data["position_2"]==position) | 
                                            (new_data["position_3"]==position),]
    
        data_new_everyone = data_new_radar.copy()
    
        x = data_new_radar["value_difference_forest"].max()
    
        data_new_radar = data_new_radar.loc[data_new_radar["value_difference_forest"]==x]

    
    
########### Display the most undervalued player

        tab2_1_col1, tab2_1_col2, tab2_1_col3 = st.columns(3)
        
    
    
        player_image = data_new_radar["player_face_url"].iloc[-1]
        player_name = data_new_radar["long_name"].iloc[-1]
        
        club_image = data_new_radar["club_logo_url"].iloc[-1]
        club_name = data_new_radar["club_name"].iloc[-1]
    
        nation_image = data_new_radar["nation_logo_url"].iloc[-1]
    
  
        tab2_1_col1.subheader("Most undervalued player: {}".format(data_new_radar["short_name"].iloc[-1]))


        tab2_1_col2.metric(label="Predicted Value", value=round(data_new_radar["predictions_forest"]), delta=round(x))   

        if st.checkbox("Show profile of {}".format(data_new_radar["short_name"].iloc[-1]), False):
            
            col1, col2, col3 = st.columns([3,1,1])

            
            # Display player image 
    
            try:
                # Display Image
                with col1:
                    st.image(player_image)
                    st.title(player_name)
        
            except:
                with col1:
                    st.title(player_name)
    
            try:
                st.markdown("Tags: {}".format(data_new_radar["player_traits"].iloc[-1]))   
            except:
                pass
 
    

            # Display player characteristics
            row4_col1, row4_col2, row4_col3, row4_col4 = st.columns([1,1,1,1])

            pace = data_new_radar["pace"].iloc[-1]
            shooting = data_new_radar["shooting"].iloc[-1]
            passing = data_new_radar["passing"].iloc[-1]
            dribbling = data_new_radar["dribbling"].iloc[-1]
            defending = data_new_radar["defending"].iloc[-1]
 
 
            df = pd.DataFrame(dict(
                r = [pace,shooting,passing,dribbling,defending],
                theta=["pace","shooting","passing","dribbling","defending"]))
 
            fig2 = px.line_polar(df, r='r', theta='theta', line_close=True,color_discrete_sequence=["#d8576b"])
            fig2.update_traces(fill='toself')
 
            # Put figure in col 1 
            row4_col1.subheader("Skills")
            row4_col1.plotly_chart(fig2, use_container_width=True)
 
            # Display profile information in column 2
 
            reputation_stars = int(data_new_radar["skill_moves"].iloc[-1])*"⭐"
            moves_stars = int(data_new_radar["international_reputation"].iloc[-1])*"⭐"
     
            with row4_col2:
                st.subheader("Profile")
                
                st.markdown("- Age: {} years".format(int(data_new_radar["age"].iloc[-1])))
                st.markdown("- Height: {} cm".format(data_new_radar["height_cm"].iloc[-1]))
                st.markdown("- Preferred Foot: {}".format(data_new_radar["preferred_foot"].iloc[-1]))

     
                st.markdown("Player ratings")
                st.markdown("- International Reputation: {}".format(moves_stars))
                st.markdown("- Skill Moves: {}".format(reputation_stars))
     
            # Display information on the club
            with row4_col3:
                st.subheader("Club: {}".format(club_name))
         
                try:
                    st.image(club_image)
                except:
                    pass
         
                st.markdown("- Club contract valid until: {}".format(int(data_new_radar["club_contract_valid_until"].iloc[-1])))
                st.markdown("- League Level: {}".format(int(data_new_radar["league_level"].iloc[-1])))

         
                try:
                    st.image(nation_image)
                    st.markdown("{} also plays on the national team.".format(player_name))
                except:
                    pass

            # Display soccer field with positions

            my_list = []
            
            for i in data_new_radar['player_positions'].str.split(',').iloc[-1]:
                my_list.append(i)
        
            new_list = [item.strip() for item in my_list]
        
        
            prediction_positions = df_positions[df_positions["position_name"].isin(new_list)]
        
        
        
            img = plt.imread("soccer_field.jpeg")
            fig, ax = plt.subplots()
            ax.scatter(prediction_positions.x_position, prediction_positions.y_position, linewidth=5, color='#d8576b')
        
            ax.axis('off')
        
            # zip joins x and y coordinates in pairs
            for x,y in zip(prediction_positions.x_position,prediction_positions.y_position):
        
                filtered_df = prediction_positions.loc[ (prediction_positions["x_position"] ==x) & (prediction_positions["y_position"] ==y),]
                label = filtered_df["position_name"].iloc[-1,]

                plt.annotate(label, # this is the text
                             (x,y), # these are the coordinates to position the label
                             textcoords="offset points", # how to position the text
                             xytext=(0,10), # distance from text to points (x,y)
                             ha='center') # horizontal alignment can be left, right or center
        
            ax.imshow(img)
        
            fig.savefig("field_with_positions_prediction.png")
    
            row4_col4.subheader("Position")
    
            image = Image.open('field_with_positions_prediction.png')
    
            row4_col4.image(image)
    
    
    
    # Display the data set with the filtered data
    

    
            if st.checkbox("Show all of the players that match the filters", False):
                st.subheader("All Matches")
                st.write(data_new_everyone)


    except IndexError:
         st.error('**No matches found.** Please change your filters, for instance by allowing the contract to be valid for a longer period of time.', icon="🚨")           
        
 
    
 
    
 
    





## Clubs ############################################################################################################


with tab3:
    
    tab3_col1, tab3_col2, tab3_col3 = st.columns([2,1,1])


# Let the user choose a soccer club 

    club_options = new_data["club_name"].unique()

    club = tab3_col1.selectbox("Please select a club",
                               club_options)
    
    club_data = new_data.loc[(new_data["club_name"]==club)]
    
    st.write(club_data)
    
    fig = go.Figure(data=go.Heatmap(
        z=club_data["value_difference_forest"],
        x=club_data["short_name"],
        y=club_data["position_0"]))

    
    # Put figure in tab3_col1
    tab3_col1.plotly_chart(fig, use_container_width=True)
    

# Let the user choose a potential replacement
   
    x = club_data["value_difference_forest"].min()
    
    most_overvalued = club_data.loc[club_data["value_difference_forest"]==x]
    
    
    with tab3_col2:
        st.write("Your most overvalued player is {}".format(most_overvalued["short_name"].iloc[-1]))

    # Potential replacement for most overvalued player
    
        replacement_data = new_data.loc[(new_data["position_0"]==most_overvalued["position_0"].iloc[-1]) | 
                                        (new_data["position_1"]==most_overvalued["position_0"].iloc[-1]) | 
                                        (new_data["position_2"]==most_overvalued["position_0"].iloc[-1]) |  
                                        (new_data["position_3"]==most_overvalued["position_0"].iloc[-1])
                                        ,]
    
        replacement_data = replacement_data.loc[(replacement_data["potential"]<=most_overvalued["potential"].iloc[-1] -10)  &
                                                (replacement_data["pace"]<=most_overvalued["pace"].iloc[-1] -10)  & 
                                                (replacement_data["shooting"]<=most_overvalued["shooting"].iloc[-1] -10)  &
                                                (replacement_data["passing"]<=most_overvalued["passing"].iloc[-1] -10)  &
                                                (replacement_data["dribbling"]<=most_overvalued["dribbling"].iloc[-1] -10)  &
                                                (replacement_data["defending"]<=most_overvalued["defending"].iloc[-1] -10)  &
                                                (replacement_data["league_level"]==most_overvalued["league_level"].iloc[-1])
                                                ,]
    
    
        x = replacement_data["value_difference_forest"].max()
    
        replacement = replacement_data.loc[replacement_data["value_difference_forest"]==x]

    
    # Ask if the user wants to find a replacement:
        if st.button("Click here for a potential replacement for {}".format(most_overvalued["short_name"].iloc[-1])):
            player_name = replacement["long_name"].iloc[-1]
                        
            try:
                player_image = replacement["player_face_url"].iloc[-1]
                
                # Display Image
                with tab3_col2:
                    st.image(player_image)
                    st.subheader(player_name)
        
            except:
                with tab3_col2:
                    st.subheader(player_name)
    
            if str(replacement["player_traits"].iloc[-1]) != "nan":
                tab3_col2.markdown("Tags: {}".format(replacement["player_traits"].iloc[-1]))   
            else:
                pass
            
            tab3_col2.write(replacement)

            try:

                pace_replacement = replacement["pace"].iloc[-1]
                shooting_replacement = replacement["shooting"].iloc[-1]
                passing_replacement = replacement["passing"].iloc[-1]
                dribbling_replacement = replacement["dribbling"].iloc[-1]
                defending_replacement = replacement["defending"].iloc[-1]
     
                pace_overvalued = most_overvalued["pace"].iloc[-1]
                shooting_overvalued = most_overvalued["shooting"].iloc[-1]
                passing_overvalued = most_overvalued["passing"].iloc[-1]
                dribbling_overvalued = most_overvalued["dribbling"].iloc[-1]
                defending_overvalued = most_overvalued["defending"].iloc[-1]   
                
                overvalued_name = most_overvalued["short_name"].iloc[-1]
                                                                     
                categories = ["pace","shooting","passing","dribbling","defending"]

                fig = go.Figure()

                fig.add_trace(go.Scatterpolar(
                    r=[pace_replacement,shooting_replacement,passing_replacement,dribbling_replacement,defending_replacement],
                    theta=categories,
                    fill='toself',
                    name=player_name,
                    marker=dict(color= "#fb9f3a")
                    ))
                
                
                fig.add_trace(go.Scatterpolar(
                    r=[pace_overvalued,shooting_overvalued,passing_overvalued,dribbling_overvalued,defending_overvalued],
                    theta=categories,
                    fill='toself',
                    name=overvalued_name,
                    marker=dict(color= "#d8576b")
                    ))

     
                # Put figure in col 1 
                with tab3_col3:
                    st.plotly_chart(fig, use_container_width=True)
            
            except:
                pass

            
        else:
            st.write("You can also find a replacement using the Search Tool")
    
    
    
    
    
    
    
    

import streamlit as st
import os
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import duckdb
import joblib
from pathlib import Path
from scipy.integrate import trapezoid

if 'count' not in st.session_state:
    st.session_state.count = 0
if "list_batch" not in st.session_state:
    st.session_state.list_batch = ("No batch produced yet")
    list_batch = st.session_state.list_batch
        
## First Page
st.title("Adeno-Associated Virus production data generation and analysis")
    
st.container()
         
def bioprocess_model(y, t, params):
    #dependent variables definition
    X, G, L, A, P = y
    #parameters definition (max growth rate, reaction rate constant (Ki_A (Ammonia), Ki_L (Lactate)), conversion rate)
    mu_max, Ks, Ki_A, Ki_L, Y_xg, Y_lg, Y_ag, q_p = params
    
    #equations for growth, consumption and production
    mu = mu_max * (G / (Ks + G)) * (Ki_A / (Ki_A + A)) * (Ki_L / (Ki_L + L))
    
    dXdt = mu * X
    dGdt = -(1/Y_xg) * mu * X
    dLdt = Y_lg * (1/Y_xg) * mu * X
    dAdt = Y_ag * (1/Y_xg) * mu * X
   # Hybrid Production Logic
    if t > 72:
        if t > 230:
            # Final Plateau: Efficiency drops as growth (mu) slows down
            # This forces the S-curve shape at the very end
            dPdt = q_p * X * (mu / mu_max)   
        else:
            dPdt = q_p * X  # Peak Production phase (Linear/Aggressive)
    else:
        dPdt = 0  # Lag/Growth phase (No production)
        
    return [dXdt, dGdt, dLdt, dAdt, dPdt]

def generate_sabotaged_data(num_batches, n_iteration):
    all_telemetry = []
    all_outcomes = []
    t_eval = np.linspace(0, 240, 241) 
    
    #defining batches
    for i in range(num_batches):
        batch_id = f"BATCH_{n_iteration:03d}"
        status = "Golden"

        #default values for parameters and initial value for variables
        # mu_max, Ks, Ki_A, Ki_L, Y_xg, Y_lg, Y_ag, q_p
        params = [0.04, 0.5, 15.0, 40.0, 0.4, 0.6, 0.1, 0.0025]        
        y0 = [0.5, 50.0, 0.0, 0.0, 0.0]

        #ouptu value between 0.0 and 1.0
        dice_roll = np.random.random()

        # as the dice roll should have equal probability for all results,
        # there is a 15 % chance to start with lower glucose
        if dice_roll < 0.15: 
            status = "OOS_Glucose_Fail"
            y0[1] = 30.0 

        # 15 % chance to have a toxic drift, 
        # Yag of 0.4 instead of 0.1 means that for the same amount of biomass
        # there is 1.8 times more ammonia produced
        elif dice_roll < 0.30: 
            status = "OOS_Toxic_Drift"
            params[5] = 0.18
            
        # 10 % chance of transfection fail,
        # low transfection is translated by low productivity rate (30 % lower)
        elif dice_roll < 0.40: 
            status = "OOS_Transfection_Fail"
            params[6] = 0.035 
            
        #solving equation system    
        sol = odeint(bioprocess_model, y0, t_eval, args=(params,))

        #creating random normaly distributed noise for all data
        noise = np.random.normal(0, 0.015, sol.shape)# random noise centered at 0 with 1.5 % standard deviation populated accross the shape of the sol matrix
        sol_noisy = sol + (sol * noise) #adding the noise fitted to the actual data with the data
        
        batch_df = pd.DataFrame(sol_noisy, columns=['VCD', 'Glucose', 'Lactate', 'Ammonia', 'Product'])

        base_ph = 7.2
        ph_drop = 0.05 * batch_df['Lactate']
        ph_buffer = 0.01 * batch_df['Ammonia']
        batch_df['pH'] = base_ph - ph_drop + ph_buffer + np.random.normal(0, 0.01, len(t_eval))
        
        batch_df['Hour'] = t_eval
        batch_df['Batch_ID'] = batch_id
        all_telemetry.append(batch_df)

        base_eff = 0.35
        if status == "OOS_Toxic_Drift":
            eff = base_eff * 0.6
        elif status == "OOS_Transfection_Fail":
            eff = base_eff * 0.3
        else:
            eff = base_eff * np.random.normal(1, 0.05)

        total_titer = sol[-1, 4] #fetching the last value for the 5th variable
        full_titer = total_titer * eff
        pct_full = eff * 100

        all_outcomes.append({
            'Batch_ID': batch_id,
            'Total_Titer': total_titer,
            'Full_Titer': full_titer,
            'Percent_Full': pct_full
            #,'Status': status (deactivating status for application)
        })
        
    return pd.concat(all_telemetry), pd.DataFrame(all_outcomes)

def generate_dirty_data(telemetry_df):
    dirty_rows = []
    
    for batch_id in telemetry_df['Batch_ID'].unique():
        batch_data = telemetry_df[telemetry_df['Batch_ID'] == batch_id]
        
        # 1. High Frequency: Ammonia & Lactate (Every 2 hours)
        # We simulate probe data with jittery timestamps
        ammonia_data = batch_data.iloc[::2][['Hour', 'pH', 'Ammonia', 'Lactate', 'Batch_ID']].copy()
        ammonia_data['Hour'] += np.random.normal(0, 0.1, len(ammonia_data)) 
        
        # 2. Low Frequency: VCD & Glucose (Every 24 hours)
        # Mimics a technician coming in once a day
        vcd_data = batch_data.iloc[::24][['Hour', 'VCD', 'Glucose', 'Product', 'Batch_ID']].copy()
        vcd_data['Hour'] += np.random.normal(0, 0.5, len(vcd_data)) 
        
        # Combine into a "messy" long-form table
        dirty_rows.append(ammonia_data)
        dirty_rows.append(vcd_data)
        
    return pd.concat(dirty_rows).sort_values(['Batch_ID', 'Hour'])

def align_telemetry(group):
    # Define a clean 1-hour grid
    clean_grid = np.arange(0, 241, 1)
    
    # We isolate the numeric columns so 'Batch_ID' doesn't cause a warning
    cols_to_sync = ['VCD', 'Glucose', 'Lactate', 'Ammonia', 'Product', 'pH']
    
    # 1. Combine existing hours with the target grid
    target_indices = pd.Index(clean_grid, name='Hour')
    combined_indices = group.set_index('Hour').index.union(target_indices)
    
    # 2. Reindex to the combined set (preserves all original data points)
    return (group.set_index('Hour')[cols_to_sync]
                 .reindex(combined_indices)
                 .sort_index()
                 .interpolate(method='linear') # Interpolates using the float distances
                 .ffill()
                 .bfill()
                 .loc[clean_grid]) # Finally, keep only the exact hours from your grid

@st.cache_resource
def get_db_connection():
    return duckdb.connect()

def generate_data(): 
    con = get_db_connection()
    
    df_telemetry, df_outcomes = generate_sabotaged_data(1,st.session_state.count)
    st.session_state.count += 1
    df_messy = generate_dirty_data(df_telemetry)
    df_cleaned = (
        df_messy
        .groupby('Batch_ID')
        .apply(align_telemetry, include_groups=False)
        .reset_index()
                 )
    # Calculate instantaneous rates in the aligned dataframe
    df_cleaned = df_cleaned.sort_values(['Batch_ID', 'Hour'])
    df_cleaned['dt'] = df_cleaned.groupby('Batch_ID')['Hour'].diff()
    
    # Growth Rate (mu): (ΔVCD / VCD) / Δt
    df_cleaned['Growth_Rate'] = df_cleaned.groupby('Batch_ID')['VCD'].diff() / (df_cleaned['VCD'] * df_cleaned['dt'])
        
    # Specific Production Rate (qp): ΔP / (VCD * Δt)
    df_cleaned['Production_Rate'] = df_cleaned.groupby('Batch_ID')['Product'].diff() / (df_cleaned['VCD'] * df_cleaned['dt'])
        
    # Specific Consumption Rate (qg): ΔG / (VCD * Δt)
    df_cleaned['Consumption_Rate'] = df_cleaned.groupby('Batch_ID')['Glucose'].diff() / (df_cleaned['VCD'] * df_cleaned['dt'])
    df_cleaned = df_cleaned.drop("dt",axis=1)
    
    con.execute("DROP VIEW IF EXISTS temp.telemetry_aligned")
    con.execute("""CREATE TABLE IF NOT EXISTS telemetry_aligned(
            Batch_ID VARCHAR, 
            Hour DOUBLE,
            VCD DOUBLE, 
            Glucose DOUBLE, 
            Lactate DOUBLE, 
            Ammonia DOUBLE, 
            Product DOUBLE,
            pH DOUBLE,
            Growth_Rate DOUBLE,
            Production_Rate DOUBLE,
            Consumption_Rate DOUBLE)
            """)
    
    con.execute("""CREATE TABLE IF NOT EXISTS outcomes(
            Batch_ID VARCHAR, 
            Total_Titer DOUBLE, 
            Full_Titer DOUBLE, 
            Percent_Full DOUBLE)
            """)
    
    df_cleaned = df_cleaned.astype({col: 'object' for col in df_cleaned.columns if df_cleaned[col].dtype.name == 'str'})
    df_outcomes = df_outcomes.astype({col: 'object' for col in df_outcomes.columns if df_outcomes[col].dtype.name == 'str'})

    con.execute("INSERT INTO telemetry_aligned BY NAME SELECT * FROM df_cleaned")
    con.execute("INSERT INTO outcomes BY NAME SELECT * FROM df_outcomes")

    query = """
    SELECT *
    FROM telemetry_aligned 
    """

    df_batches = con.execute(query).df()
    
    step_1.write(df_batches.tail())
    list_batches = df_batches["Batch_ID"].unique()
    return list_batches

## First Menu: Data generation
with st.expander("Step 1: generating AAV culture data") as step_1:
        if st.button(
            label="Generate data",
            key="generate_data_btn",
            help="Produces files for three moments of the culture: 75h, 100h and the end of the culture"
        ):
            st.session_state.list_batch = generate_data()
    
        list_batch = st.session_state.list_batch

## Preparing for Random forest classifier
def preparing_features(df_tel, df_out, checkpoint_hour):
    df_tel = df_tel.sort_values(by=['Batch_ID', 'Hour']).reset_index(drop=True)
    processed_features = []
    process_variables = ['VCD', 'Glucose', 'Lactate', 'Ammonia', 'Product', 'pH', 'Growth_Rate', 'Production_Rate', 'Consumption_Rate']
    
    for batch_id, batch_group in df_tel.groupby('Batch_ID'):
        # Limit data to given checkpoint hour (function argument) and put in chronological order
        group_to_hour = batch_group[batch_group['Hour'] <= checkpoint_hour].sort_values('Hour')
            
        # Creates a dataframe populated with the current batch_id of the loop
        batch_features = pd.DataFrame({'Batch_ID': [batch_id]})

        # Locate the checkpoint row
        matched_row = group_to_hour[group_to_hour['Hour'] == checkpoint_hour]

        # Extract the state and find its relative integer position in the group
        if not matched_row.empty:
            label = matched_row.index[0]
            checkpoint_index = group_to_hour.index.get_loc(label)

        # Calculating and/or storing current value, lag1 and 2, rolling average and standard deviation
        # cumulative sum
        for col in process_variables:
            batch_features[f'{col}_current'] = group_to_hour[col].iloc[checkpoint_index]
            
            batch_features[f'{col}_lag1'] = group_to_hour[col].iloc[checkpoint_index - 1]
            batch_features[f'{col}_lag2'] = group_to_hour[col].iloc[checkpoint_index - 2]
            
            short_window = group_to_hour[col].iloc[(checkpoint_index - 2):(checkpoint_index + 1)]
            batch_features[f'{col}_roll_mean3'] = short_window.mean()
            batch_features[f'{col}_roll_std3'] = short_window.std()
            
            if checkpoint_index >= 11:
                macro_window = group_to_hour.iloc[(checkpoint_index - 11):(checkpoint_index + 1)]
                hourly_rates = macro_window[col].diff().dropna() / macro_window['Hour'].diff().dropna()
                batch_features[f'{col}_avg_rate_12h'] = hourly_rates.mean()
            else:
                batch_features[f'{col}_avg_rate_12h'] = np.nan
                
            y = group_to_hour[col].values
            x = group_to_hour['Hour'].values
            if len(x) > 1:
                batch_features[f'{col}_cum_area'] = trapezoid(y, x)
            else:
                batch_features[f'{col}_cum_area'] = 0.0
                
        processed_features.append(batch_features)
              
    df_features_matrix = pd.concat(processed_features, ignore_index=True)
    df_modeling_set = pd.merge(df_features_matrix, df_out[['Batch_ID', 'Full_Titer']], on='Batch_ID', how='inner')

    
    return df_modeling_set

## Second Menu: Predictive classification for Full titer
with st.expander("Step 2: predictive classification for Full titer") as step_2:
    selected_batch = st.select_slider(label= "Batch selection", options= list_batch )
    selected_time = st.select_slider(label= "Timepoint to evaluate Full Titer (h)", options=(75, 100), value= 75, width=100)

@st.cache_data
def classifying_batch (batch, time):
    con = get_db_connection()
    
    query = """
    SELECT *
    FROM telemetry_aligned 
    WHERE "Batch_ID" = ?
    """
    
    telemetry_random_forest = con.execute(query, [batch]).df()
    
    query_2 = """
    SELECT 
        o.Batch_ID, 
        o.Full_Titer, 
    FROM outcomes o
    WHERE "Batch_ID" = ?
    """
    
    outcome_random_forest = con.execute(query_2, [batch]).df()
    
    df_checkpoint = preparing_features(telemetry_random_forest, outcome_random_forest, time)
    X = df_checkpoint.drop(columns=["Full_Titer", "Batch_ID"]).copy()
    
    if time == 100:
        classifier_100h_model_path = Path("C:\dev\AAV_Process_Modeling\models") / "Classifier_100h_model.joblib"
        classifier_100h_model = joblib.load(classifier_100h_model_path)
        y_hat = classifier_100h_model.predict(X)
    else:
        classifier_75h_model_path = Path("C:\dev\AAV_Process_Modeling\models") / "Classifier_100h_model.joblib"
        classifier_75h_model = joblib.load(classifier_75h_model_path)
        y_hat = classifier_75h_model.predict(X)
    return y_hat

if selected_batch == "No batch produced yet":
    st.write("No batch available to analyse")
else:
    full_titer_prediction = np.round(classifying_batch (selected_batch, selected_time), 2)
    st.write(f"The predicted final full titer based on data at {selected_time} hours is {full_titer_prediction} g/L")